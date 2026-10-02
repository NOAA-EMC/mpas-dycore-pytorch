"""Main entry point for MPAS-Atmosphere PyTorch dynamics core."""

import logging
import math
import os
import resource
import warnings
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from mpas_pytorch.autodiff.checkpoint import checkpoint_state_dict
from mpas_pytorch.config_defaults import (
    FORTRAN_DEFAULTS,
    get_config_value_safe,
    validate_config,
)
from mpas_pytorch.constants import CP, P0, RGAS
from mpas_pytorch.dynamics.init_coupled_diagnostics import (
    _init_coupled_connectivity_payload,
    _init_coupled_connectivity_payload_cache_key,
    atm_init_coupled_diagnostics,
)
from mpas_pytorch.dynamics.static_payloads import (
    DycoreStaticPayloads,
    ensure_static_payload_store,
)
from mpas_pytorch.dynamics.time_integration import (
    _SAVE_NOTHING_CONTEXT,
    atm_srk3,
)
from mpas_pytorch.dynamics.vector_reconstruction import (
    CELL_PADDING_OFFSET_KEY,
    _vector_reconstruction_payload,
    _vector_reconstruction_payload_cache_key,
    coeffs_reconstruct_all_zero,
    mpas_reconstruct_2d,
)
from mpas_pytorch.exceptions import MpasConfigError, MpasDeviceError
from mpas_pytorch.initialization import initialize_from_state
from mpas_pytorch.profiling import profiled_call
from mpas_pytorch.release_capabilities import validate_runtime_request
from mpas_pytorch.state import (
    dict_to_device,
    has_complete_step_boundary,
    mark_complete_step_boundary,
    require_boundary_continuation_identity,
    require_unedited_boundary_support,
)
from mpas_pytorch.utils.checks import require_consistent_device

LOGGER = logging.getLogger(__name__)
_CPU_DEVICE = torch.device("cpu")


def _validate_public_inputs(
    state: Mapping[str, Any], mesh: Mapping[str, Any], device: torch.device | str
) -> None:
    """Reject unsupported execution before preparing or converting model fields."""
    target = _check_device_available(device)
    validate_runtime_request(device=target.type, dtype="float64", physics_mode="dycore")
    for mapping in (state, mesh):
        for value in mapping.values():
            if isinstance(value, torch.Tensor):
                dtype = (
                    str(value.dtype).removeprefix("torch.")
                    if value.is_floating_point() or value.is_complex()
                    else "float64"
                )
                validate_runtime_request(
                    device=value.device.type, dtype=dtype, physics_mode="dycore"
                )


def _require_finite_model_fields(state: dict[str, Any], context: str) -> None:
    for name in (
        "u",
        "w",
        "theta",
        "rho",
        "theta_m",
        "rho_zz",
        "scalars",
        "qv",
        "rho_base",
        "theta_base",
        "zz",
        "pressure_p",
        "pressure_base",
    ):
        value = state.get(name)
        if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
            raise MpasConfigError(f"{context}: {name} contains nonfinite values")


__all__ = [
    "CheckpointPlan",
    "_run_simulation_jvp",
    "_run_simulation_vjp",
    "build_checkpoint_plan",
    "run_simulation",
    "run_timestep",
]


@dataclass(frozen=True)
class CheckpointPlan:
    """Immutable recomputation policy selected before numeric execution."""

    outer_timestep: bool
    nested_rk: bool
    outer_use_reentrant: bool
    materialize_final: bool
    nested_selection: str
    production_qualified: bool


def build_checkpoint_plan(
    *,
    differentiable: bool,
    checkpoint_timesteps: bool | None,
    checkpoint_use_reentrant: bool,
    checkpoint_materialize_final: bool,
    config: Mapping[str, Any] | None,
) -> CheckpointPlan:
    """Resolve outer and nested checkpointing without an implicit double layer."""
    if not differentiable:
        return CheckpointPlan(
            False,
            False,
            False,
            bool(checkpoint_materialize_final),
            "disabled",
            True,
        )

    outer = True if checkpoint_timesteps is None else bool(checkpoint_timesteps)
    nested_explicit = config is not None and "gradient_checkpointing" in config
    if config is not None and nested_explicit:
        nested = bool(config["gradient_checkpointing"])
        selection = "explicit_config"
    else:
        nested = not outer
        selection = "outer_only_default" if outer else "nested_only_default"
    return CheckpointPlan(
        outer_timestep=outer,
        nested_rk=nested,
        outer_use_reentrant=bool(checkpoint_use_reentrant) if outer else False,
        materialize_final=bool(checkpoint_materialize_final),
        nested_selection=selection,
        production_qualified=not (outer and nested),
    )


_INTER_STEP_PROGNOSTIC_KEYS = frozenset(
    {
        "u",
        "w",
        "theta_m",
        "rho_zz",
        "scalars",
        # Microphysics updates this dynamics tendency and the next RK step
        # consumes it in acoustic/pressure coupling, matching MPAS' tend pool.
        "rt_diabatic_tend",
    }
)
# These diagnostics are part of the discrete inter-timestep state even though
# they are not independently restartable prognostics.  ``atm_srk3`` carries
# their evolved values into the next step.  Reconstructing them from the
# prognostics changes both a handful of float64 values and, more importantly,
# the autograd graph being differentiated.  An outer checkpoint must therefore
# return these exact tensors and recompute them only as part of the checkpointed
# timestep itself.
_INTER_STEP_EXACT_DIAGNOSTIC_KEYS = frozenset(
    {
        "divergence",
        "exner",
        "exner_base",
        "h_edge",
        "ke",
        "pressure_base",
        "pressure_p",
        "pv_edge",
        "rho_edge",
        "rho_p",
        "rtheta_base",
        "rtheta_p",
        "ru",
        "rw",
        "uReconstructMeridional",
        "uReconstructX",
        "uReconstructY",
        "uReconstructZ",
        "uReconstructZonal",
        "ur_cell",
        "v",
        "vorticity",
        "vr_cell",
    }
)
_EXACT_CHECKPOINT_DIAGNOSTIC_MARKER = "_mpas_exact_checkpoint_diagnostic_keys"
_STATIC_BOUNDARY_SUPPORT_KEYS = frozenset(
    {
        "rho_base",
        "theta_base",
        "zz",
        "cellsOnEdge",
        "edgesOnCell",
        "edgesOnCell_sign",
        "nEdgesOnCell",
        "fzm",
        "fzp",
        "zb_cell",
        "zb3_cell",
        "coeffs_reconstruct",
        "latCell",
        "lonCell",
        "u_init",
        "v_init",
        "t_init",
        "qv_init",
        "ur_cell",
        "vr_cell",
        "uReconstructX",
        "uReconstructY",
        "uReconstructZ",
        "uReconstructZonal",
        "uReconstructMeridional",
    }
)
_FORWARD_INTERSTEP_STATIC_KEYS = _STATIC_BOUNDARY_SUPPORT_KEYS - frozenset(
    {
        "ur_cell",
        "vr_cell",
        "uReconstructX",
        "uReconstructY",
        "uReconstructZ",
        "uReconstructZonal",
        "uReconstructMeridional",
    }
)
_FORWARD_INTERSTEP_STATIC_KEYS = _FORWARD_INTERSTEP_STATIC_KEYS | frozenset(
    {
        "advCellsForEdge",
        "adv_coefs",
        "adv_coefs_3rd",
        "angleEdge",
        "areaCell",
        "bdyMaskCell",
        "bdyMaskEdge",
        "bdyMaskVertex",
        "cellsOnCell",
        "cellsOnEdge",
        "cellsOnVertex",
        "cf1",
        "cf2",
        "cf3",
        "dcEdge",
        "defc_a",
        "defc_b",
        "deriv_two",
        "dss",
        "dzu",
        "dzw",
        "dvEdge",
        "edgesOnCell",
        "edgesOnCell_sign",
        "edgesOnEdge",
        "edgesOnVertex",
        "edgesOnVertex_sign",
        "fEdge",
        "fVertex",
        "fnm",
        "fnp",
        "invAreaCell",
        "invAreaTriangle",
        "invDcEdge",
        "invDvEdge",
        "kiteAreasOnVertex",
        "kiteForCell",
        "latEdge",
        "meshScalingDel2",
        "meshScalingDel4",
        "nAdvCellsForEdge",
        "rdnw",
        "rdzu",
        "rdzw",
        "t_init",
        "u_init",
        "v_init",
        "verticesOnCell",
        "verticesOnEdge",
        "weightsOnEdge",
        "zb",
        "zb3",
        "zgrid",
        "zxu",
    }
)
_STATIC_BOUNDARY_SUPPORT_KEYS = (
    _STATIC_BOUNDARY_SUPPORT_KEYS | _FORWARD_INTERSTEP_STATIC_KEYS
)

# Public read-only ownership vocabulary used by benchmark runners. Keeping the
# aliases here prevents performance harnesses from inventing a second, drifting
# definition of the timestep boundary.
INTER_STEP_PROGNOSTIC_KEYS = _INTER_STEP_PROGNOSTIC_KEYS
INTER_STEP_EXACT_DIAGNOSTIC_KEYS = _INTER_STEP_EXACT_DIAGNOSTIC_KEYS
FORWARD_INTERSTEP_STATIC_KEYS = _FORWARD_INTERSTEP_STATIC_KEYS


def _checkpoint_exact_diagnostic_keys(state: Mapping[str, Any]) -> tuple[str, ...]:
    """Return and validate the exact diagnostics owned by a checkpoint boundary."""
    marker = state.get(_EXACT_CHECKPOINT_DIAGNOSTIC_MARKER)
    if marker is None:
        return tuple(
            sorted(
                key
                for key in _INTER_STEP_EXACT_DIAGNOSTIC_KEYS
                if isinstance(state.get(key), torch.Tensor)
            )
        )
    if not isinstance(marker, tuple) or marker != tuple(sorted(marker)):
        raise MpasConfigError("invalid exact checkpoint diagnostic marker")
    unknown = set(marker) - _INTER_STEP_EXACT_DIAGNOSTIC_KEYS
    if unknown:
        raise MpasConfigError(
            "exact checkpoint diagnostic marker contains unknown keys: "
            + ", ".join(sorted(unknown))
        )
    missing = [key for key in marker if not isinstance(state.get(key), torch.Tensor)]
    if missing:
        raise MpasConfigError(
            "exact checkpoint boundary lacks diagnostics: " + ", ".join(missing)
        )
    return marker


def _exact_checkpoint_boundary(
    previous_state: Mapping[str, Any],
    step_result: dict[str, Any],
) -> dict[str, Any]:
    """Return the exact tensor state consumed by the following timestep.

    The first boundary derives its diagnostic ownership from the prepared input;
    later boundaries inherit the frozen marker.  This makes deletion of any
    required evolved diagnostic a hard error instead of silently switching the
    discrete program to diagnostic rematerialization.
    """
    exact_diagnostic_keys = _checkpoint_exact_diagnostic_keys(previous_state)
    missing = [
        key
        for key in exact_diagnostic_keys
        if not isinstance(step_result.get(key), torch.Tensor)
    ]
    if missing:
        raise MpasConfigError(
            "checkpointed timestep failed to return exact diagnostics: "
            + ", ".join(missing)
        )
    boundary: dict[str, Any] = {}
    for key, value in step_result.items():
        if (
            not isinstance(value, torch.Tensor)
            or key in _INTER_STEP_PROGNOSTIC_KEYS
            or key in exact_diagnostic_keys
        ):
            boundary[key] = value
    boundary[_EXACT_CHECKPOINT_DIAGNOSTIC_MARKER] = exact_diagnostic_keys
    if "num_scalars" not in boundary and "scalars" in boundary:
        boundary["num_scalars"] = int(boundary["scalars"].shape[-1])
    return boundary


def _capture_static_boundary_support(
    prepared_state: dict[str, Any],
    mesh: dict[str, Any],
) -> dict[str, Any]:
    """Capture static tensor inputs needed to rematerialize future timesteps."""
    support: dict[str, Any] = {}
    for key in _STATIC_BOUNDARY_SUPPORT_KEYS:
        if key in prepared_state and key not in mesh:
            support[key] = prepared_state[key]
    return support


def _restore_static_boundary_support(
    state: dict[str, Any],
    static_support: dict[str, Any],
) -> None:
    """Inject missing static rematerialization inputs back into a boundary state."""
    for key, value in static_support.items():
        state.setdefault(key, value)


def _prepare_materialization_input(
    state: dict[str, Any],
    *,
    checkpoint_timesteps: bool,
    differentiable: bool,
    checkpoint_use_reentrant: bool,
    static_support: dict[str, Any],
) -> dict[str, Any]:
    """Provide a state dict with static support restored before rematerializing."""
    if not (
        checkpoint_timesteps
        and differentiable
        and not checkpoint_use_reentrant
        and static_support
    ):
        return state
    materialize_state = dict(state)
    _restore_static_boundary_support(materialize_state, static_support)
    return materialize_state


def _normalize_checkpointed_state(
    state: dict[str, Any],
    *,
    between_steps: bool,
) -> dict[str, Any]:
    """Normalize checkpointed outputs for continued differentiable use.

    Reentrant checkpoint forward executes under ``torch.no_grad()`` and the
    current dual-mode timestep path uses ``torch.inference_mode()`` for the
    non-differentiable branch. Any tensor that leaves the checkpoint boundary
    and will be used in later differentiable code must therefore be materialized
    back into a regular tensor. Between timesteps, prognostics and the evolved
    diagnostics consumed by the next step must keep autograd history. Other
    diagnostic tensors are presentation-only and may be detached.
    """
    normalized: dict[str, Any] = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            normalized[key] = value
            continue
        if key in (_INTER_STEP_PROGNOSTIC_KEYS | _INTER_STEP_EXACT_DIAGNOSTIC_KEYS):
            normalized[key] = value.clone() if value.is_inference() else value
            continue
        if between_steps:
            if value.is_inference():
                normalized[key] = value.clone().detach()
            elif value.requires_grad:
                normalized[key] = value.detach()
            else:
                normalized[key] = value
            continue
        normalized[key] = value.clone() if value.is_inference() else value
    return normalized


def _memory_probe_enabled() -> bool:
    """Return whether timestep RSS logging is enabled."""
    return os.environ.get("MPAS_MEM_PROBE") == "1"


def _log_timestep_rss(step: int, phase: str) -> None:
    """Emit the current process RSS high-water mark for the given timestep phase."""
    if not _memory_probe_enabled():
        return
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    LOGGER.info("mem_probe step=%d phase=%s ru_maxrss=%s", step, phase, rss)


def _compile_requested(config: Mapping[str, Any] | None) -> bool:
    """Return whether raw caller config requests torch.compile."""
    if os.environ.get("MPAS_DISABLE_TORCH_COMPILE", "0") == "1":
        return False
    if not config:
        return False
    for key in ("config_use_torch_compile", "use_torch_compile"):
        if key not in config:
            continue
        value = config[key]
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu":
                raise MpasConfigError(
                    f"{key} must be a Python bool or CPU tensor scalar; "
                    f"got {value.device.type} tensor"
                )
            if value.numel() != 1:
                return bool(value.any().item())
            value = value.item()
        return bool(value)
    return False


def _nondifferentiable_context(config: Mapping[str, Any] | None):
    if _compile_requested(config):
        return torch.no_grad()
    return torch.inference_mode()


def _check_device_available(device: torch.device | str) -> torch.device:
    """Reject unavailable accelerator devices at the public API boundary."""
    requested = device if isinstance(device, torch.device) else torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise MpasDeviceError(
            requested=str(device),
            message=(
                f"CUDA requested (device={str(device)!r}) but torch.cuda.is_available() is False. "
                f"Install PyTorch with CUDA support, or pass device='cpu'."
            ),
        )
    if requested.type == "mps":
        mps_available = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        if mps_available:
            raise MpasDeviceError(
                requested=str(device),
                message=(
                    "MPS requested but MPAS-PyTorch requires float64 dynamics, "
                    "which Apple MPS does not support. Pass device='cpu' or use CUDA float64."
                ),
            )
        raise MpasDeviceError(
            requested=str(device),
            message=(
                f"MPS requested (device={str(device)!r}) but torch.backends.mps.is_available() is False. "
                f"Enable Apple Metal support, or pass device='cpu'."
            ),
        )
    return requested


def _restore_padding_sentinels(state: dict[str, Any]) -> dict[str, Any]:
    """Restore fixed padding-row sentinels expected by downstream kernels/tests."""
    rho_zz = state.get("rho_zz")
    n_cells = state.get("nCells")
    if not isinstance(rho_zz, torch.Tensor) or n_cells is None:
        return state
    n_cells_int = int(n_cells)
    if rho_zz.ndim < 2 or rho_zz.shape[0] <= n_cells_int:
        return state
    # Never stamp the sentinel row in place on a caller-owned tensor. This
    # helper is used at preparation/output boundaries, so a fresh tensor here
    # preserves the public no-mutation contract for run_timestep/run_simulation
    # without relying on whole-state deepcopy guards.
    rho_zz = rho_zz.clone()
    rho_zz[n_cells_int, :] = 1.0
    state["rho_zz"] = rho_zz
    return state


def _project_checkpoint_boundary_state(
    previous_state: dict[str, Any],
    step_state: dict[str, Any],
) -> dict[str, Any]:
    """Project a full timestep result onto the exact reentrant boundary."""
    boundary = _exact_checkpoint_boundary(previous_state, step_state)
    for key, value in previous_state.items():
        if not isinstance(value, torch.Tensor) and key not in boundary:
            boundary[key] = value
    for key in _STATIC_BOUNDARY_SUPPORT_KEYS:
        if key in previous_state:
            boundary.setdefault(key, previous_state[key])
    return boundary


def _project_forward_interstep_state(
    previous_state: dict[str, Any],
    step_state: dict[str, Any],
    *,
    step_number: int,
    nsteps: int,
) -> dict[str, Any]:
    """Drop transient tensors between nondifferentiable forward timesteps."""
    if step_number >= nsteps:
        return step_state
    boundary: dict[str, Any] = {}
    for key, value in previous_state.items():
        if not isinstance(value, torch.Tensor):
            boundary[key] = value
    for key in _FORWARD_INTERSTEP_STATIC_KEYS:
        if key in previous_state:
            boundary[key] = previous_state[key]
    for key, value in step_state.items():
        if not isinstance(value, torch.Tensor):
            boundary[key] = value
    for key in _INTER_STEP_PROGNOSTIC_KEYS:
        if key in step_state:
            boundary[key] = step_state[key]
    if "num_scalars" not in boundary and "scalars" in boundary:
        boundary["num_scalars"] = int(boundary["scalars"].shape[-1])
    return _restore_padding_sentinels(boundary)


def _materialize_step_state(
    state: dict[str, Any],
    mesh: dict[str, Any],
    config: dict[str, Any],
    dt: float,
    *,
    strict: bool,
) -> dict[str, Any]:
    """Recompute diagnostics/vector outputs from a checkpoint boundary state."""
    materialized = dict(state)
    profiled_call(
        "mpas.init_coupled_diagnostics",
        config,
        _init_coupled_diagnostics,
        materialized,
        mesh,
        config=config,
        strict=strict,
    )
    profiled_call(
        "mpas.restore_padding_sentinels",
        config,
        _restore_padding_sentinels,
        materialized,
    )
    profiled_call(
        "mpas.init_solve_diagnostics",
        config,
        _init_solve_diagnostics,
        materialized,
        mesh,
        config=config,
        dt=dt,
        strict=strict,
    )
    profiled_call(
        "mpas.init_vector_reconstruction",
        config,
        _init_vector_reconstruction,
        materialized,
        mesh,
        config=config,
        strict=strict,
    )
    if "num_scalars" not in materialized and "scalars" in materialized:
        materialized["num_scalars"] = int(materialized["scalars"].shape[-1])
    return materialized


def _merge_materialized_state(
    authoritative_state: dict[str, Any],
    materialized_state: dict[str, Any],
) -> dict[str, Any]:
    """Fill missing diagnostics from a materialized state without overwriting real outputs."""
    merged = dict(materialized_state)
    merged.update(authoritative_state)
    return merged


def _rematerialize_interstep_diagnostics(
    state: dict[str, Any],
    mesh: dict[str, Any],
    config: dict[str, Any],
    dt: float,
    *,
    strict: bool,
) -> None:
    """Refresh only the diagnostics required at the next timestep boundary."""
    if "num_scalars" not in state and "scalars" in state:
        state["num_scalars"] = int(state["scalars"].shape[-1])
    profiled_call(
        "mpas.init_coupled_diagnostics",
        config,
        _init_coupled_diagnostics,
        state,
        mesh,
        config=config,
        strict=strict,
    )
    profiled_call(
        "mpas.restore_padding_sentinels", config, _restore_padding_sentinels, state
    )
    profiled_call(
        "mpas.init_solve_diagnostics",
        config,
        _init_solve_diagnostics,
        state,
        mesh,
        config=config,
        dt=dt,
        strict=strict,
    )
    profiled_call(
        "mpas.init_vector_reconstruction",
        config,
        _init_vector_reconstruction,
        state,
        mesh,
        config=config,
        strict=strict,
    )


def _rematerialize_forward_interstep_diagnostics(
    state: dict[str, Any],
    mesh: dict[str, Any],
    config: dict[str, Any],
    dt: float,
    *,
    strict: bool,
) -> None:
    """Refresh diagnostics after an opt-in forward inter-step prune."""
    if "num_scalars" not in state and "scalars" in state:
        state["num_scalars"] = int(state["scalars"].shape[-1])
    profiled_call(
        "mpas.init_coupled_diagnostics",
        config,
        _init_coupled_diagnostics,
        state,
        mesh,
        config=config,
        strict=strict,
    )
    profiled_call(
        "mpas.restore_padding_sentinels", config, _restore_padding_sentinels, state
    )
    profiled_call(
        "mpas.init_solve_diagnostics",
        config,
        _init_solve_diagnostics,
        state,
        mesh,
        config=config,
        dt=dt,
        strict=strict,
    )
    profiled_call(
        "mpas.init_vector_reconstruction",
        config,
        _init_vector_reconstruction,
        state,
        mesh,
        config=config,
        strict=strict,
    )


def _run_timestep_impl(
    state: dict[str, Any],
    mesh: dict[str, Any],
    config: dict | None = None,
    dt: float | None = None,
    device: torch.device | str = _CPU_DEVICE,
    *,
    differentiable: bool = False,
    config_validated: bool = False,
) -> dict[str, Any]:
    """Advance model state by one timestep using RK3."""
    # Isolate caller-owned mappings while still sharing the underlying tensors.
    # Production prep/rematerialization code may replace entries such as rho_zz
    # inside the working state dict; a shallow copy preserves the public
    # no-mutation contract without deep-copying every field.
    state = dict(state)
    mesh = dict(mesh)
    boundary_mesh = dict(mesh)
    state, full_config, strict = _prepare_timestep_state(
        state=state,
        mesh=mesh,
        config=config,
        dt=dt,
        device=device,
        config_validated=config_validated,
    )
    if dt is None:
        dt = float(full_config["config_dt"])

    if differentiable:
        result = atm_srk3(
            state, mesh, full_config, dt, strict=strict, differentiable=True
        )
        _require_finite_model_fields(result, "timestep result")
        return result
    result = atm_srk3(state, mesh, full_config, dt, strict=strict)
    _require_finite_model_fields(result, "timestep result")
    # A completed RK step already carries the coupled and solve diagnostics
    # needed by the next step. Re-deriving them on a later public call changes
    # the floating-point trajectory at the call boundary.
    mark_complete_step_boundary(result, mesh=boundary_mesh, config=full_config, dt=dt)
    return result


def _prepare_timestep_state(
    state: dict[str, Any],
    mesh: dict[str, Any],
    config: dict | None = None,
    dt: float | None = None,
    device: torch.device | str = _CPU_DEVICE,
    *,
    config_validated: bool = False,
) -> tuple[dict[str, Any], dict, bool]:
    """Prepare state/mesh for numerical integration (pre-atm_srk3 stage)."""
    full_config = dict(config or {}) if config_validated else validate_config(config)
    _require_finite_model_fields(state, "timestep input")
    target_device = torch.device(device)
    input_device = require_consistent_device("run_timestep inputs", state, mesh)

    if input_device != target_device:
        state = dict_to_device(state, target_device)
        mesh_on_target = dict_to_device(mesh, target_device)
        mesh.clear()
        mesh.update(mesh_on_target)

    if isinstance(config, Mapping) and "strict" in config:
        strict = bool(config["strict"])
    else:
        strict = bool(
            full_config.get("strict_initialization", full_config.get("strict", True))
        )

    use_rcm_val = full_config.get("config_mesh_reordering_rcm", False)
    if isinstance(use_rcm_val, torch.Tensor):
        if use_rcm_val.numel() != 1:
            raise MpasConfigError(
                "config_mesh_reordering_rcm must be a scalar bool/int tensor "
                f"(got shape {tuple(use_rcm_val.shape)})"
            )
        use_rcm = bool(use_rcm_val)
    else:
        use_rcm = bool(use_rcm_val)
    if use_rcm and not mesh.get("_rcm_applied", False):
        from mpas_pytorch.mesh_reordering import apply_rcm_reordering

        state_reordered, mesh_reordered = apply_rcm_reordering(state, mesh)
        state_reordered = dict_to_device(state_reordered, target_device)
        mesh_reordered = dict_to_device(mesh_reordered, target_device)
        state = state_reordered
        mesh.clear()
        mesh.update(mesh_reordered)

    require_boundary_continuation_identity(
        state,
        mesh=mesh,
        config=full_config,
        dt=dt if dt is not None else float(full_config["config_dt"]),
    )
    complete_boundary = has_complete_step_boundary(state)
    if not complete_boundary:
        require_unedited_boundary_support(state)
    state = profiled_call(
        "mpas.initialize_from_state",
        full_config,
        initialize_from_state,
        state,
        mesh,
        full_config,
        strict=strict,
    )
    target_coef = get_config_value_safe(full_config, "config_coef_3rd_order", None)
    if target_coef is None:
        target_coef = get_config_value_safe(full_config, "coef_3rd_order", None)
    if target_coef is not None:
        from mpas_pytorch.ic_loader import apply_coef_3rd_order_override_fields

        apply_coef_3rd_order_override_fields(
            state,
            mesh,
            coef_3rd_order=float(target_coef),
        )
    if complete_boundary:
        state = dict(state)
        state.pop("_restart_skip_rederivation", None)
        return state, full_config, strict
    state.pop("_restart_skip_rederivation", None)
    profiled_call(
        "mpas.init_coupled_diagnostics",
        full_config,
        _init_coupled_diagnostics,
        state,
        mesh,
        config=full_config,
        strict=strict,
    )
    profiled_call(
        "mpas.restore_padding_sentinels", full_config, _restore_padding_sentinels, state
    )
    profiled_call(
        "mpas.init_solve_diagnostics",
        full_config,
        _init_solve_diagnostics,
        state,
        mesh,
        config=full_config,
        dt=dt,
        strict=strict,
    )
    profiled_call(
        "mpas.init_vector_reconstruction",
        full_config,
        _init_vector_reconstruction,
        state,
        mesh,
        config=full_config,
        strict=strict,
    )
    if "num_scalars" not in state and "scalars" in state:
        state["num_scalars"] = int(state["scalars"].shape[-1])
    return state, full_config, strict


def run_timestep(
    state: dict[str, Any],
    mesh: dict[str, Any],
    config: dict | None = None,
    dt: float | None = None,
    device: torch.device | str = _CPU_DEVICE,
    differentiable: bool = False,
) -> dict[str, Any]:
    """Advance model state by one timestep using RK3.

    Strict-mode behavior:
    - If `config['strict']` is provided, that value is used.
    - Otherwise strictness mirrors `strict_initialization` when present.
    - Fallback remains strict (`True`) by default.
    """
    _validate_public_inputs(state, mesh, device)
    config = validate_config(config)
    effective_dt = (
        dt
        if dt is not None
        else (config or {}).get("config_dt", FORTRAN_DEFAULTS["config_dt"])
    )
    if not math.isfinite(float(effective_dt)) or effective_dt <= 0.0:
        raise MpasConfigError("dt must be finite and positive")
    config_validated = True
    if differentiable:
        checkpoint_plan = build_checkpoint_plan(
            differentiable=True,
            checkpoint_timesteps=False,
            checkpoint_use_reentrant=False,
            checkpoint_materialize_final=True,
            config=config,
        )
        effective_config = dict(config) if config else {}
        effective_config["gradient_checkpointing"] = checkpoint_plan.nested_rk
        return _run_timestep_impl(
            state=state,
            mesh=mesh,
            config=effective_config,
            dt=dt,
            device=device,
            differentiable=True,
            config_validated=config_validated,
        )
    with _nondifferentiable_context(config):
        return _run_timestep_impl(
            state=state,
            mesh=mesh,
            config=config,
            dt=dt,
            device=device,
            config_validated=config_validated,
        )


def _init_coupled_diagnostics(
    state: dict[str, Any],
    mesh: dict[str, Any],
    strict: bool = True,
    *,
    config: dict | None = None,
) -> None:
    """Compute coupled diagnostic fields from prognostic state.

    This wraps atm_init_coupled_diagnostics and is
    called once before the first RK3 timestep. Diagnostics are ALWAYS
    recomputed from the current prognostic state.

    The function handles the index convention mismatch: atm_init_coupled_diagnostics
    operates on 0-based arrays without padding, while the orchestration state uses
    1-based arrays with +1 padding. Results are embedded into padded arrays.

    Args:
        state: Model state dictionary (modified in place).
        mesh: Mesh connectivity and geometry dictionary.
        strict: If True, raise ValueError when required inputs are missing.
    """
    # Diagnostic fields produced by init_coupled_diagnostics
    diag_keys = [
        "theta_m",
        "rho_zz",
        "ru",
        "rw",
        "rho_p",
        "rtheta_base",
        "rtheta_p",
        "exner",
        "exner_base",
        "pressure_p",
        "pressure_base",
    ]

    # Always compute; do not retain externally supplied derived diagnostics.

    # Collect all sources (state and mesh may overlap)
    combined = {}
    combined.update(mesh)
    combined.update(state)

    # Also need nCells, nEdges, nVertLevels
    nCells = int(combined.get("nCells", 0))
    nEdges = int(combined.get("nEdges", 0))
    nVertLevels = int(combined.get("nVertLevels", 0))

    # Check required inputs for the kernel
    required_state = ["u", "w"]
    raw_inputs = ["theta", "rho", "qv"]
    derived_inputs = ["theta_m", "rho_zz"]
    required_base = ["rho_base", "theta_base", "zz"]
    required_mesh = [
        "cellsOnEdge",
        "edgesOnCell",
        "edgesOnCell_sign",
        "nEdgesOnCell",
        "fzm",
        "fzp",
        "zb_cell",
        "zb3_cell",
    ]

    missing = []
    for k in required_state + required_base + required_mesh:
        if k not in combined:
            missing.append(k)
    if not all(k in combined for k in raw_inputs) and not all(
        k in combined for k in derived_inputs
    ):
        missing.append("theta/rho/qv or theta_m/rho_zz")

    if nCells == 0 or nEdges == 0 or nVertLevels == 0:
        missing.append("nCells/nEdges/nVertLevels")

    if missing:
        if strict:
            raise MpasConfigError(
                f"Cannot compute init_coupled_diagnostics: missing inputs: {missing}. "
                f"Provide these from checkpoint or set strict=False to skip."
            )
        return  # Non-strict: silently skip

    # Extract arrays. State arrays use append-padding: (N+1, ...) where
    # valid data is at indices 0..N-1 and index N is the garbage/padding row.
    # The kernel expects 0-based without padding: (N, ...).
    # Slice off the trailing padding row.
    theta_m_padded = combined.get("theta_m")  # (nCells+1, nVertLevels)
    rho_zz_padded = combined.get("rho_zz")  # (nCells+1, nVertLevels)
    theta_padded = combined.get("theta")
    rho_padded = combined.get("rho")
    qv_padded = combined.get("qv")
    u_padded = combined["u"]  # (nEdges+1, nVertLevels)
    w_padded = combined["w"]  # (nCells+1, nVertLevels+1)
    rho_base_padded = combined["rho_base"]  # (nCells+1, nVertLevels)
    theta_base_padded = combined["theta_base"]  # (nCells+1, nVertLevels)
    zz_padded = combined["zz"]  # (nCells+1, nVertLevels+1) or (nCells+1, nVertLevels)

    # Convert from append-padded to unpadded: valid data at 0..N-1
    theta_m = theta_m_padded[:nCells] if theta_m_padded is not None else None
    rho_zz = rho_zz_padded[:nCells] if rho_zz_padded is not None else None
    theta = theta_padded[:nCells] if theta_padded is not None else None
    rho = rho_padded[:nCells] if rho_padded is not None else None
    qv = qv_padded[:nCells] if qv_padded is not None else None
    u = u_padded[:nEdges]  # (nEdges, nVertLevels)
    w = w_padded[:nCells]  # (nCells, nVertLevels+1)
    rho_base = rho_base_padded[:nCells]  # (nCells, nVertLevels)
    theta_base = theta_base_padded[:nCells]  # (nCells, nVertLevels)

    # zz may be (nCells+1, nVertLevels+1) or (nCells+1, nVertLevels)
    zz = zz_padded[:nCells]  # (nCells, ...)

    # Connectivity arrays stay as-is (1-based padded, kernel expects this)
    cellsOnEdge = combined["cellsOnEdge"]  # (nEdges+1, 2)
    edgesOnCell = combined["edgesOnCell"]  # (nCells+1, maxEdges)
    edgesOnCell_sign = combined["edgesOnCell_sign"]  # (nCells+1, maxEdges)
    nEdgesOnCell = combined["nEdgesOnCell"]  # (nCells+1,)

    init_connectivity_payload = None
    init_payload_kwargs = {
        "device": u.device,
        "nCells": nCells,
        "nEdges": nEdges,
        "nVertLevels": nVertLevels,
        "cellsOnEdge": cellsOnEdge,
        "edgesOnCell": edgesOnCell,
        "edgesOnCell_sign": edgesOnCell_sign,
        "nEdgesOnCell": nEdgesOnCell,
    }
    if isinstance(config, Mapping):
        init_key = _init_coupled_connectivity_payload_cache_key(**init_payload_kwargs)
        if init_key is not None:
            store = config.get("_dycore_static_payloads")
            if isinstance(store, DycoreStaticPayloads):
                init_connectivity_payload = store.get_or_build_typed(
                    "init_coupled_diagnostics",
                    init_key,
                    lambda: _init_coupled_connectivity_payload(**init_payload_kwargs),
                )
            elif isinstance(config, dict):
                init_connectivity_payload = ensure_static_payload_store(
                    config
                ).get_or_build_typed(
                    "init_coupled_diagnostics",
                    init_key,
                    lambda: _init_coupled_connectivity_payload(**init_payload_kwargs),
                )

    # fzm, fzp: may be (nVertLevels+1,) with padding or (nVertLevels,)
    fzm = combined["fzm"]
    fzp = combined["fzp"]
    # If padded (nVertLevels+1,), strip trailing padding to (nVertLevels,)
    if fzm.shape[0] == nVertLevels + 1:
        fzm = fzm[:nVertLevels]
    if fzp.shape[0] == nVertLevels + 1:
        fzp = fzp[:nVertLevels]

    zb_cell = combined["zb_cell"]  # (nCells+1, maxEdges, nVertLevels+1)
    zb3_cell = combined["zb3_cell"]  # (nCells+1, maxEdges, nVertLevels+1)

    result = atm_init_coupled_diagnostics(
        nCells=nCells,
        nEdges=nEdges,
        nVertLevels=nVertLevels,
        rgas=float(RGAS),
        cp=float(CP),
        p0=float(P0),
        theta_m=theta_m,
        rho_zz=rho_zz,
        theta=theta,
        rho=rho,
        qv=qv,
        u=u,
        w=w,
        rho_base=rho_base,
        theta_base=theta_base,
        zz=zz,
        cellsOnEdge=cellsOnEdge,
        edgesOnCell=edgesOnCell,
        edgesOnCell_sign=edgesOnCell_sign,
        nEdgesOnCell=nEdgesOnCell,
        fzm=fzm,
        fzp=fzp,
        zb_cell=zb_cell,
        zb3_cell=zb3_cell,
        _connectivity_payload=init_connectivity_payload,
    )

    # Embed results back into append-padded arrays for the state dict.
    # Valid data at indices 0..N-1, trailing zero padding at index N.
    # Only set fields that are not already present (idempotent).
    from mpas_pytorch.constants import DTYPE

    edge_idx = None
    cell_idx = None

    for key in diag_keys:
        if key not in result:
            continue
        val = result[key]
        # Determine padded shape: cell-based or edge-based
        if key == "ru":
            # ru: (nEdges, nVertLevels) -> (nEdges+1, nVertLevels)
            padded = torch.zeros(
                nEdges + 1, nVertLevels, dtype=DTYPE, device=val.device
            )
            if edge_idx is None or edge_idx.device != val.device:
                edge_idx = torch.arange(nEdges, dtype=torch.int64, device=val.device)
            padded = torch.index_copy(padded, 0, edge_idx, val)
        elif key == "rw":
            # rw: (nCells, nVertLevels+1) -> (nCells+1, nVertLevels+1)
            padded = torch.zeros(
                nCells + 1, nVertLevels + 1, dtype=DTYPE, device=val.device
            )
            if cell_idx is None or cell_idx.device != val.device:
                cell_idx = torch.arange(nCells, dtype=torch.int64, device=val.device)
            padded = torch.index_copy(padded, 0, cell_idx, val)
        else:
            # Cell-based (nCells, nVertLevels) -> (nCells+1, nVertLevels)
            padded = torch.zeros(
                nCells + 1, val.shape[-1], dtype=DTYPE, device=val.device
            )
            if cell_idx is None or cell_idx.device != val.device:
                cell_idx = torch.arange(nCells, dtype=torch.int64, device=val.device)
            padded = torch.index_copy(padded, 0, cell_idx, val)

        state[key] = padded


def _init_solve_diagnostics(
    state: dict[str, Any],
    mesh: dict[str, Any],
    config: dict | None = None,
    dt: float | None = None,
    strict: bool = True,
) -> None:
    """Compute solve_diagnostics fields from prognostic state.

    Upstream MPAS calls atm_compute_solve_diagnostics during initialization
    to compute v (tangential velocity), vorticity, divergence, ke, and pv_edge
    from the initial state.

    Always recomputes instead of retaining externally supplied diagnostics.

    Args:
        state: Model state dictionary (modified in place).
        mesh: Mesh connectivity and geometry dictionary.
        strict: If True, raise ValueError when required inputs are missing.
    """

    from mpas_pytorch.dynamics.solve_diagnostics import (
        _mark_diagnostic_solve_diagnostics_current,
        atm_compute_solve_diagnostics_work,
    )

    # Merge state and mesh for lookups
    combined = {}
    combined.update(mesh)
    combined.update(state)

    # Required inputs for solve_diagnostics
    required = [
        "u",
        "rho_zz",
        "cellsOnEdge",
        "edgesOnCell",
        "verticesOnEdge",
        "edgesOnVertex",
        "nEdgesOnCell",
        "nEdgesOnEdge",
        "edgesOnEdge",
        "verticesOnCell",
        "dvEdge",
        "dcEdge",
        "invDvEdge",
        "invDcEdge",
        "invAreaCell",
        "invAreaTriangle",
        "fVertex",
        "fEdge",
        "weightsOnEdge",
        "kiteAreasOnVertex",
        "kiteForCell",
        "edgesOnCell_sign",
        "edgesOnVertex_sign",
    ]

    nCells = int(combined.get("nCells", 0))
    nEdges = int(combined.get("nEdges", 0))
    nVertLevels = int(combined.get("nVertLevels", 0))

    if nCells == 0 or nEdges == 0 or nVertLevels == 0:
        if strict:
            raise MpasConfigError(
                "Cannot compute solve_diagnostics: nCells/nEdges/nVertLevels missing."
            )
        return

    missing = [k for k in required if k not in combined]
    if missing:
        if strict:
            raise MpasConfigError(
                f"Cannot compute solve_diagnostics: missing inputs: {missing}. "
                f"Provide these from checkpoint or set strict=False to skip."
            )
        return

    # Extract dimensions
    nVertices = int(combined.get("nVertices", 0))
    vertexDegree = int(combined.get("vertexDegree", 3))

    # Call solve_diagnostics kernel.
    # Use runtime dt/apvm defaults to match Fortran initialization behavior.
    # h = rho_zz (Jacobian-weighted density used as mass proxy)
    cfg = config or {}
    dt_diag = float(
        dt
        if dt is not None
        else cfg.get(
            "config_dt", cfg.get("dt", FORTRAN_DEFAULTS.get("config_dt", 720.0))
        )
    )
    apvm_diag = float(
        cfg.get(
            "config_apvm_upwinding",
            cfg.get(
                "apvm_upwinding", FORTRAN_DEFAULTS.get("config_apvm_upwinding", 0.5)
            ),
        )
    )
    diag_result = atm_compute_solve_diagnostics_work(
        nCells=nCells,
        nEdges=nEdges,
        nVertices=nVertices,
        nVertLevels=nVertLevels,
        vertexDegree=vertexDegree,
        dt=dt_diag,
        config_apvm_upwinding=apvm_diag,
        cellStart=1,
        cellEnd=nCells,
        vertexStart=1,
        vertexEnd=nVertices,
        edgeStart=1,
        edgeEnd=nEdges,
        fVertex=combined["fVertex"],
        fEdge=combined["fEdge"],
        invAreaTriangle=combined["invAreaTriangle"],
        invAreaCell=combined["invAreaCell"],
        dvEdge=combined["dvEdge"],
        dcEdge=combined["dcEdge"],
        invDvEdge=combined["invDvEdge"],
        invDcEdge=combined["invDcEdge"],
        weightsOnEdge=combined["weightsOnEdge"],
        kiteAreasOnVertex=combined["kiteAreasOnVertex"],
        cellsOnEdge=combined["cellsOnEdge"],
        verticesOnEdge=combined["verticesOnEdge"],
        edgesOnCell=combined["edgesOnCell"],
        edgesOnCell_sign=combined["edgesOnCell_sign"],
        edgesOnVertex=combined["edgesOnVertex"],
        edgesOnVertex_sign=combined["edgesOnVertex_sign"],
        verticesOnCell=combined["verticesOnCell"],
        kiteForCell=combined["kiteForCell"],
        nEdgesOnCell=combined["nEdgesOnCell"],
        nEdgesOnEdge=combined["nEdgesOnEdge"],
        edgesOnEdge=combined["edgesOnEdge"],
        u=combined["u"],
        h=combined["rho_zz"],
    )

    # Store computed fields in state.
    # Refresh diagnostics from the current state and replace stale file values.
    for key in ["v", "vorticity", "divergence", "ke", "pv_edge", "h_edge"]:
        if key in diag_result:
            state[key] = diag_result[key]
    # h_edge is rho_edge in the state convention
    if "h_edge" in diag_result:
        state["rho_edge"] = diag_result["h_edge"]
    _mark_diagnostic_solve_diagnostics_current(
        state, u=combined["u"], h=combined["rho_zz"]
    )


def _init_vector_reconstruction(
    state: dict[str, Any],
    mesh: dict[str, Any],
    config: dict | None = None,
    strict: bool = True,
) -> None:
    """Compute ur_cell/vr_cell from the initial edge-normal wind.

    Always recomputes — no existence guard. Prevents Fortran value injection.
    """

    combined = {}
    combined.update(mesh)
    combined.update(state)
    required = [
        "u",
        "coeffs_reconstruct",
        "edgesOnCell",
        "nEdgesOnCell",
        "latCell",
        "lonCell",
        "nCells",
        "nEdges",
        "nVertLevels",
    ]
    missing = [k for k in required if k not in combined]
    if missing:
        if strict:
            raise MpasConfigError(
                f"Cannot compute initial vector reconstruction: missing inputs: {missing}"
            )
        return

    nCells = int(combined["nCells"])
    nEdges = int(combined["nEdges"])
    nVertLevels = int(combined["nVertLevels"])
    if nCells <= 0 or nEdges <= 0 or nVertLevels <= 0:
        if strict:
            raise MpasConfigError(
                "Cannot compute initial vector reconstruction: invalid dimensions "
                f"(nCells={nCells}, nEdges={nEdges}, nVertLevels={nVertLevels})"
            )
        return

    on_a_sphere = True
    if isinstance(config, Mapping):
        on_a_sphere = bool(config.get("config_on_a_sphere", True))

    coeffs_reconstruct = combined["coeffs_reconstruct"]
    coeffs_all_zero = coeffs_reconstruct_all_zero(state, coeffs_reconstruct, nCells)
    if coeffs_all_zero and "ur_cell" in combined and "vr_cell" in combined:
        warnings.warn(
            "coeffs_reconstruct is all zeros; preserving existing ur_cell/vr_cell "
            "for uReconstruct diagnostics.",
            RuntimeWarning,
        )
        state["ur_cell"] = combined["ur_cell"]
        state["vr_cell"] = combined["vr_cell"]
        for key in ("uReconstructX", "uReconstructY", "uReconstructZ"):
            if key in combined:
                state[key] = combined[key]
        state["uReconstructZonal"] = combined.get(
            "uReconstructZonal", combined["ur_cell"].clone()
        )
        state["uReconstructMeridional"] = combined.get(
            "uReconstructMeridional", combined["vr_cell"].clone()
        )
        return

    vector_connectivity_payload = None
    if isinstance(config, Mapping):
        vector_payload_kwargs = {
            "device": combined["u"].device,
            "nCells": nCells,
            "nEdges": nEdges,
            "nVertLevels": nVertLevels,
            "u_rows": int(combined["u"].shape[0]),
            "coeffs_reconstruct": coeffs_reconstruct,
            "edgesOnCell": combined["edgesOnCell"],
            "nEdgesOnCell": combined["nEdgesOnCell"],
            "latCell": combined["latCell"],
            "lonCell": combined["lonCell"],
            "on_a_sphere": on_a_sphere,
            "cell_padding_offset": combined.get(CELL_PADDING_OFFSET_KEY),
        }
        vector_key = _vector_reconstruction_payload_cache_key(**vector_payload_kwargs)
        if vector_key is not None:
            store = config.get("_dycore_static_payloads")
            if isinstance(store, DycoreStaticPayloads):
                vector_connectivity_payload = store.get_or_build_typed(
                    "vector_reconstruction",
                    vector_key,
                    lambda: _vector_reconstruction_payload(**vector_payload_kwargs),
                )
            elif isinstance(config, dict):
                vector_connectivity_payload = ensure_static_payload_store(
                    config
                ).get_or_build_typed(
                    "vector_reconstruction",
                    vector_key,
                    lambda: _vector_reconstruction_payload(**vector_payload_kwargs),
                )

    recon = mpas_reconstruct_2d(
        nCells=nCells,
        nEdges=nEdges,
        nVertLevels=nVertLevels,
        u=combined["u"],
        coeffs_reconstruct=coeffs_reconstruct,
        edgesOnCell=combined["edgesOnCell"],
        nEdgesOnCell=combined["nEdgesOnCell"],
        latCell=combined["latCell"],
        lonCell=combined["lonCell"],
        on_a_sphere=on_a_sphere,
        cell_padding_offset=combined.get(CELL_PADDING_OFFSET_KEY),
        _connectivity_payload=vector_connectivity_payload,
    )
    state["ur_cell"] = recon["ur_cell"]
    state["vr_cell"] = recon["vr_cell"]
    state["uReconstructX"] = recon["uReconstructX"]
    state["uReconstructY"] = recon["uReconstructY"]
    state["uReconstructZ"] = recon["uReconstructZ"]
    state["uReconstructZonal"] = recon["uReconstructZonal"]
    state["uReconstructMeridional"] = recon["uReconstructMeridional"]

    if not torch.isfinite(state["ur_cell"][:nCells, :]).all():
        raise MpasConfigError(
            "Initial vector reconstruction produced non-finite ur_cell values."
        )
    if not torch.isfinite(state["vr_cell"][:nCells, :]).all():
        raise MpasConfigError(
            "Initial vector reconstruction produced non-finite vr_cell values."
        )


def run_simulation(
    initial_state: dict[str, Any],
    mesh: dict[str, Any],
    config: dict | None = None,
    dt: float | None = None,
    nsteps: int = 1,
    device: torch.device | str = _CPU_DEVICE,
    on_step: Callable[[int, dict[str, Any]], None] | None = None,
    differentiable: bool = False,
    checkpoint_timesteps: bool | None = None,
    checkpoint_use_reentrant: bool = False,
    checkpoint_materialize_final: bool = True,
    on_phase: Callable[[str], None] | None = None,
    on_initial: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run simulation for specified number of timesteps.

    Args:
        initial_state: Initial model state dictionary.
        mesh: Mesh connectivity and geometry dictionary.
        config: Optional configuration overrides. If None or partial,
            FORTRAN_DEFAULTS values are used for missing keys.
            The 'strict' key controls validation behavior (default True).
        dt: Timestep size in seconds.
        nsteps: Number of timesteps to run.
        device: Target device for state/mesh tensors (default CPU).
        on_initial: Observe the prepared state used by timestep one; must not mutate it.
        on_step: Optional callback invoked after each completed timestep as
            ``on_step(step_number, state_dict)``.
        differentiable: Enable autodiff-compatible execution.
        checkpoint_timesteps: Whether to recompute each full timestep during
            backward. ``None`` enables the current default policy: auto-enable
            for differentiable runs, including one-step TL/AD probes, so memory
            is bounded by recomputation unless callers explicitly opt out.
            Pass ``False`` to disable timestep checkpointing for exactness-
            sensitive validation runs.
        checkpoint_use_reentrant: Use reentrant outer timestep checkpoints.
            Non-reentrant checkpoints support both ``torch.autograd.grad`` and
            ``.backward()`` and should remain the default for nearly all
            workflows. Reentrant checkpoints are kept only as an opt-in
            compatibility path for experiments that explicitly want them.
        checkpoint_materialize_final: Recompute the full final diagnostic state
            before returning. Disable this for prognostic-only adjoint
            objectives to avoid rebuilding a large final diagnostic graph
            unnecessarily.
        on_phase: Optional observation callback. Production benchmark runners
            use ``forecast_integration_begin``/``forecast_integration_end`` to
            time the same post-initialization forecast interval as native MPAS.

    Returns:
        Final model state dictionary after all timesteps.
    """
    _validate_public_inputs(initial_state, mesh, device)
    config = validate_config(config)
    if dt is None:
        dt = float((config or {}).get("config_dt", FORTRAN_DEFAULTS["config_dt"]))
    if isinstance(nsteps, bool) or not isinstance(nsteps, int) or nsteps < 0:
        raise MpasConfigError("nsteps must be a nonnegative integer")
    if not math.isfinite(float(dt)) or dt <= 0.0:
        raise MpasConfigError("dt must be finite and positive")
    config_validated = True
    mesh = dict(mesh)
    boundary_mesh = dict(mesh)
    checkpoint_plan = build_checkpoint_plan(
        differentiable=differentiable,
        checkpoint_timesteps=checkpoint_timesteps,
        checkpoint_use_reentrant=checkpoint_use_reentrant,
        checkpoint_materialize_final=checkpoint_materialize_final,
        config=config,
    )
    checkpoint_timesteps = checkpoint_plan.outer_timestep
    checkpoint_use_reentrant = checkpoint_plan.outer_use_reentrant
    checkpoint_materialize_final = checkpoint_plan.materialize_final

    def _simulate() -> dict[str, Any]:
        state = dict(initial_state)
        full_config = (
            dict(config or {}) if config_validated else validate_config(config)
        )
        prune_forward_intersteps = (
            bool(full_config.get("config_prune_forward_interstep_state", False))
            and not differentiable
        )
        boundary_static_support: dict[str, Any] = {}
        if differentiable:
            full_config["gradient_checkpointing"] = checkpoint_plan.nested_rk
        required_init_dims = {"nCells", "nEdges", "nVertLevels"}
        can_prepare_once = required_init_dims.issubset(
            mesh.keys()
        ) or required_init_dims.issubset(state.keys())
        forecast_integration_started = False
        for step in range(nsteps):
            step_number = step + 1
            _log_timestep_rss(step_number, "before")
            resume_from_restart = step == 0 and has_complete_step_boundary(state)
            want_final_full_state = (
                checkpoint_timesteps
                and differentiable
                and checkpoint_materialize_final
                and step_number == nsteps
            )
            if (
                step > 0
                and differentiable
                and checkpoint_timesteps
                and not checkpoint_use_reentrant
                and boundary_static_support
            ):
                profiled_call(
                    "mpas.static_boundary_restore",
                    full_config,
                    _restore_static_boundary_support,
                    state,
                    boundary_static_support,
                )
            if can_prepare_once and step == 0:
                state, full_config, strict = _prepare_timestep_state(
                    state=state,
                    mesh=mesh,
                    config=full_config,
                    dt=dt,
                    device=device,
                    config_validated=True,
                )
                if resume_from_restart and differentiable:
                    _rematerialize_interstep_diagnostics(
                        state,
                        mesh,
                        full_config,
                        dt,
                        strict=strict,
                    )
                # Ordinary forward continuation consumes the complete boundary
                # snapshot directly. Re-deriving rho_p/rtheta_p/pressure_p here
                # was measured to change the next timestep by roundoff.
                if on_initial is not None:
                    on_initial(state)
                boundary_static_support = profiled_call(
                    "mpas.static_boundary_capture",
                    full_config,
                    _capture_static_boundary_support,
                    state,
                    mesh,
                )
                if on_phase is not None:
                    on_phase("forecast_integration_begin")
                    forecast_integration_started = True
                previous_state = state
                if differentiable:
                    if checkpoint_timesteps:

                        def _one_step_materialized(
                            sd: dict[str, Any],
                            full_config_local: dict[str, Any] = full_config,
                            strict_local: bool = strict,
                            want_final_full_state_local: bool = want_final_full_state,
                        ) -> dict[str, Any]:
                            step_result = atm_srk3(
                                sd,
                                mesh,
                                full_config_local,
                                dt,
                                strict=strict_local,
                                differentiable=(
                                    torch.is_grad_enabled()
                                    if checkpoint_use_reentrant
                                    else True
                                ),
                            )
                            if want_final_full_state_local:
                                return step_result
                            if checkpoint_use_reentrant:
                                return _project_checkpoint_boundary_state(
                                    sd, step_result
                                )
                            return _exact_checkpoint_boundary(sd, step_result)

                        state = checkpoint_state_dict(
                            _one_step_materialized,
                            state,
                            enabled=True,
                            use_reentrant=checkpoint_use_reentrant,
                            context_fn=_SAVE_NOTHING_CONTEXT,
                        )
                        state = _normalize_checkpointed_state(
                            state,
                            between_steps=not want_final_full_state,
                        )
                        if checkpoint_use_reentrant:
                            state = profiled_call(
                                "mpas.restore_padding_sentinels",
                                full_config,
                                _restore_padding_sentinels,
                                state,
                            )
                        # Exact non-reentrant outputs already carry the fixed
                        # density padding. Re-stamping adds a gradient node and
                        # changes coupled reverse accumulation unnecessarily.
                    else:
                        state = atm_srk3(
                            state,
                            mesh,
                            full_config,
                            dt,
                            strict=strict,
                            differentiable=True,
                        )
                else:
                    state = atm_srk3(state, mesh, full_config, dt, strict=strict)
                step_state = state
                need_materialized_step = (
                    checkpoint_timesteps
                    and differentiable
                    and not want_final_full_state
                    and (
                        on_step is not None
                        or (step + 1 == nsteps and checkpoint_materialize_final)
                    )
                )
                if need_materialized_step:
                    materialize_state = profiled_call(
                        "mpas.materialization_input_prepare",
                        full_config,
                        _prepare_materialization_input,
                        state,
                        checkpoint_timesteps=checkpoint_timesteps,
                        differentiable=differentiable,
                        checkpoint_use_reentrant=checkpoint_use_reentrant,
                        static_support=boundary_static_support,
                    )
                    step_state = _materialize_step_state(
                        materialize_state,
                        mesh,
                        full_config,
                        dt,
                        strict=strict,
                    )
                elif want_final_full_state and (
                    on_step is not None or checkpoint_materialize_final
                ):
                    materialize_state = profiled_call(
                        "mpas.materialization_input_prepare",
                        full_config,
                        _prepare_materialization_input,
                        state,
                        checkpoint_timesteps=checkpoint_timesteps,
                        differentiable=differentiable,
                        checkpoint_use_reentrant=checkpoint_use_reentrant,
                        static_support=boundary_static_support,
                    )
                    step_state = _merge_materialized_state(
                        state,
                        _materialize_step_state(
                            materialize_state,
                            mesh,
                            full_config,
                            dt,
                            strict=strict,
                        ),
                    )
                if on_step is not None:
                    on_step(step_number, step_state)
                if step_number == nsteps and checkpoint_materialize_final:
                    state = step_state
                elif prune_forward_intersteps:
                    state = profiled_call(
                        "mpas.forward_interstep_projection",
                        full_config,
                        _project_forward_interstep_state,
                        previous_state,
                        step_state,
                        step_number=step_number,
                        nsteps=nsteps,
                    )
                _log_timestep_rss(step_number, "after")
                continue
            if can_prepare_once and step > 0:
                strict = bool(full_config.get("strict", True))
                if differentiable:
                    if checkpoint_timesteps:

                        def _one_step_pruned(
                            sd: dict[str, Any],
                            full_config_local: dict[str, Any] = full_config,
                            strict_local: bool = strict,
                            want_final_full_state_local: bool = want_final_full_state,
                        ) -> dict[str, Any]:
                            # A checkpoint boundary is an exact discrete-model
                            # boundary, not a compact restart. Rebuilding these
                            # diagnostics changed the forward trajectory from
                            # step two onward and differentiated a different map.
                            _checkpoint_exact_diagnostic_keys(sd)
                            working_state = dict(sd)
                            step_result = atm_srk3(
                                working_state,
                                mesh,
                                full_config_local,
                                dt,
                                strict=strict_local,
                                differentiable=(
                                    torch.is_grad_enabled()
                                    if checkpoint_use_reentrant
                                    else True
                                ),
                            )
                            if want_final_full_state_local:
                                return step_result
                            if checkpoint_use_reentrant:
                                return _project_checkpoint_boundary_state(
                                    sd, step_result
                                )
                            return _exact_checkpoint_boundary(sd, step_result)

                        state = checkpoint_state_dict(
                            _one_step_pruned,
                            state,
                            enabled=True,
                            use_reentrant=checkpoint_use_reentrant,
                            context_fn=_SAVE_NOTHING_CONTEXT,
                        )
                        state = _normalize_checkpointed_state(
                            state,
                            between_steps=not want_final_full_state,
                        )
                        if checkpoint_use_reentrant:
                            state = profiled_call(
                                "mpas.restore_padding_sentinels",
                                full_config,
                                _restore_padding_sentinels,
                                state,
                            )
                    else:
                        # The uncheckpointed differentiable path carries the
                        # complete diagnostic state returned by ``atm_srk3``,
                        # just like the ordinary forward path. Rebuilding the
                        # coupled diagnostics here changes that state before
                        # every later timestep and makes the AD primal a
                        # different program. Only projected checkpoint/pruned
                        # boundaries require the rematerialization above.
                        state = atm_srk3(
                            state,
                            mesh,
                            full_config,
                            dt,
                            strict=strict,
                            differentiable=True,
                        )
                else:
                    if prune_forward_intersteps:
                        _rematerialize_forward_interstep_diagnostics(
                            state,
                            mesh,
                            full_config,
                            dt,
                            strict=strict,
                        )
                    # The ordinary forward path carries the complete diagnostic
                    # state returned by atm_srk3 into the next timestep.  Native
                    # MPAS initializes coupled diagnostics only at cold start;
                    # rebuilding rho_p/rtheta_p here from rho_zz/theta_m adds an
                    # extra floating-point round trip before every later step.
                    # Pruned/checkpoint boundaries still rematerialize above
                    # because those policies deliberately discard diagnostics.
                    previous_state = state
                    state = atm_srk3(state, mesh, full_config, dt, strict=strict)
                step_state = state
                need_materialized_step = (
                    checkpoint_timesteps
                    and differentiable
                    and not want_final_full_state
                    and (
                        on_step is not None
                        or (step + 1 == nsteps and checkpoint_materialize_final)
                    )
                )
                if need_materialized_step:
                    materialize_state = profiled_call(
                        "mpas.materialization_input_prepare",
                        full_config,
                        _prepare_materialization_input,
                        state,
                        checkpoint_timesteps=checkpoint_timesteps,
                        differentiable=differentiable,
                        checkpoint_use_reentrant=checkpoint_use_reentrant,
                        static_support=boundary_static_support,
                    )
                    step_state = _materialize_step_state(
                        materialize_state,
                        mesh,
                        full_config,
                        dt,
                        strict=strict,
                    )
                elif want_final_full_state and (
                    on_step is not None or checkpoint_materialize_final
                ):
                    materialize_state = profiled_call(
                        "mpas.materialization_input_prepare",
                        full_config,
                        _prepare_materialization_input,
                        state,
                        checkpoint_timesteps=checkpoint_timesteps,
                        differentiable=differentiable,
                        checkpoint_use_reentrant=checkpoint_use_reentrant,
                        static_support=boundary_static_support,
                    )
                    step_state = _merge_materialized_state(
                        state,
                        _materialize_step_state(
                            materialize_state,
                            mesh,
                            full_config,
                            dt,
                            strict=strict,
                        ),
                    )
                if on_step is not None:
                    on_step(step + 1, step_state)
                if step + 1 == nsteps and checkpoint_materialize_final:
                    state = step_state
                elif prune_forward_intersteps:
                    state = profiled_call(
                        "mpas.forward_interstep_projection",
                        full_config,
                        _project_forward_interstep_state,
                        previous_state,
                        step_state,
                        step_number=step + 1,
                        nsteps=nsteps,
                    )
                continue
            if differentiable:
                if checkpoint_timesteps:

                    def _one_step_checkpointed(
                        sd: dict[str, Any],
                        full_config_local: dict[str, Any] = full_config,
                        want_final_full_state_local: bool = want_final_full_state,
                    ) -> dict[str, Any]:
                        step_result = run_timestep(
                            sd,
                            mesh,
                            full_config_local,
                            dt,
                            device=device,
                            differentiable=(
                                torch.is_grad_enabled()
                                if checkpoint_use_reentrant
                                else True
                            ),
                        )
                        if want_final_full_state_local:
                            return step_result
                        if checkpoint_use_reentrant:
                            return _project_checkpoint_boundary_state(sd, step_result)
                        return _exact_checkpoint_boundary(sd, step_result)

                    state = checkpoint_state_dict(
                        _one_step_checkpointed,
                        state,
                        enabled=True,
                        use_reentrant=checkpoint_use_reentrant,
                        context_fn=_SAVE_NOTHING_CONTEXT,
                    )
                    state = _normalize_checkpointed_state(
                        state,
                        between_steps=not want_final_full_state,
                    )
                    if checkpoint_use_reentrant:
                        state = profiled_call(
                            "mpas.restore_padding_sentinels",
                            full_config,
                            _restore_padding_sentinels,
                            state,
                        )
                else:
                    state = run_timestep(
                        state, mesh, full_config, dt, device=device, differentiable=True
                    )
            else:
                previous_state = state
                state = run_timestep(state, mesh, full_config, dt, device=device)
            step_state = state
            strict = bool(full_config.get("strict", True))
            need_materialized_step = (
                checkpoint_timesteps
                and differentiable
                and not want_final_full_state
                and (
                    on_step is not None
                    or (step + 1 == nsteps and checkpoint_materialize_final)
                )
            )
            if need_materialized_step:
                materialize_state = profiled_call(
                    "mpas.materialization_input_prepare",
                    full_config,
                    _prepare_materialization_input,
                    state,
                    checkpoint_timesteps=checkpoint_timesteps,
                    differentiable=differentiable,
                    checkpoint_use_reentrant=checkpoint_use_reentrant,
                    static_support=boundary_static_support,
                )
                step_state = _materialize_step_state(
                    materialize_state,
                    mesh,
                    full_config,
                    dt,
                    strict=strict,
                )
            elif want_final_full_state and (
                on_step is not None or checkpoint_materialize_final
            ):
                materialize_state = profiled_call(
                    "mpas.materialization_input_prepare",
                    full_config,
                    _prepare_materialization_input,
                    state,
                    checkpoint_timesteps=checkpoint_timesteps,
                    differentiable=differentiable,
                    checkpoint_use_reentrant=checkpoint_use_reentrant,
                    static_support=boundary_static_support,
                )
                step_state = _merge_materialized_state(
                    state,
                    _materialize_step_state(
                        materialize_state,
                        mesh,
                        full_config,
                        dt,
                        strict=strict,
                    ),
                )
            if on_step is not None:
                on_step(step_number, step_state)
            if step_number == nsteps and checkpoint_materialize_final:
                state = step_state
            elif prune_forward_intersteps:
                state = profiled_call(
                    "mpas.forward_interstep_projection",
                    full_config,
                    _project_forward_interstep_state,
                    previous_state,
                    step_state,
                    step_number=step_number,
                    nsteps=nsteps,
                )
            _log_timestep_rss(step_number, "after")
        if forecast_integration_started and on_phase is not None:
            on_phase("forecast_integration_end")
        _require_finite_model_fields(state, "forecast result")
        if nsteps > 0 and not differentiable:
            # Make the complete returned state reusable by a later call to
            # run_simulation, just like a state loaded from write_restart.
            state = dict(state)
            mark_complete_step_boundary(
                state, mesh=boundary_mesh, config=full_config, dt=dt
            )
        return state

    if differentiable:
        return _simulate()
    with _nondifferentiable_context(config):
        return _simulate()


def _run_simulation_jvp(
    state: dict[str, Any],
    mesh: dict[str, Any],
    perturbation: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """Tangent-linear model: compute Jacobian-vector product through run_simulation.

    Args:
        state: Initial model state dictionary.
        mesh: Mesh connectivity and geometry dictionary.
        perturbation: Dict mapping prognostic keys to tangent vectors.
            Only keys present in perturbation receive a tangent;
            other state variables are treated as constants.
        **kwargs: Forwarded to run_simulation (dt, nsteps, config, device, etc.).

    Returns:
        Dict mapping state keys to their output tangent vectors (JVP).
    """
    from torch.autograd import forward_ad

    kwargs.setdefault("differentiable", True)
    with forward_ad.dual_level():
        dual_state = dict(state)
        for key, tangent in perturbation.items():
            if not isinstance(tangent, torch.Tensor):
                raise TypeError(
                    f"Perturbation '{key}' must be a tensor, got {type(tangent).__name__}"
                )
            if key not in dual_state:
                raise KeyError(f"Perturbation key '{key}' is missing from state")
            state_value = dual_state[key]
            if not isinstance(state_value, torch.Tensor):
                raise TypeError(
                    f"State entry '{key}' is not a tensor and cannot receive a tangent"
                )
            if tangent.shape != state_value.shape:
                raise MpasConfigError(
                    f"Perturbation '{key}' has shape {tuple(tangent.shape)}; "
                    f"expected {tuple(state_value.shape)}"
                )
            if tangent.dtype != state_value.dtype:
                raise MpasConfigError(
                    f"Perturbation '{key}' has dtype {tangent.dtype}; expected {state_value.dtype}"
                )
            if tangent.device != state_value.device:
                raise MpasConfigError(
                    f"Perturbation '{key}' is on {tangent.device}; expected {state_value.device}"
                )
            dual_state[key] = forward_ad.make_dual(state_value, tangent)
        out = run_simulation(dual_state, mesh, **kwargs)
        result: dict[str, Any] = {}
        for key, val in out.items():
            if isinstance(val, torch.Tensor):
                _, tangent_out = forward_ad.unpack_dual(val)
                if tangent_out is not None:
                    result[key] = tangent_out
        return result


def _run_simulation_vjp(
    state: dict[str, Any],
    mesh: dict[str, Any],
    objective: Callable[[dict[str, Any]], torch.Tensor],
    track_keys: Iterable[str] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Adjoint model: compute vector-Jacobian product through run_simulation.

    Args:
        state: Initial model state dictionary.
        mesh: Mesh connectivity and geometry dictionary.
        objective: Callable that maps the output state dict to a scalar loss.
            The adjoint computes d(objective)/d(state) for each prognostic.
        track_keys: Optional iterable of state keys to differentiate with
            respect to. Defaults to the inter-step prognostic set.
        **kwargs: Forwarded to run_simulation (dt, nsteps, config, device, etc.).

    Returns:
        Dict mapping prognostic keys to their adjoint sensitivity (gradient).

    Memory is not bounded at operational scales. This helper is intended for
    short, small-mesh gradient workflows; gradient checkpointing can reduce the
    retained graph when memory is constrained.
    """
    kwargs.setdefault("differentiable", True)
    clone_state = dict(state)
    tracked: dict[str, Any] = {}
    if track_keys is None:
        requested_keys = tuple(
            key for key in _INTER_STEP_PROGNOSTIC_KEYS if key in clone_state
        )
    else:
        requested_keys = tuple(track_keys)
    for key in requested_keys:
        if key not in clone_state:
            raise KeyError(f"Tracked VJP key '{key}' is missing from state")
        state_value = clone_state[key]
        if not isinstance(state_value, torch.Tensor):
            raise TypeError(f"Tracked VJP key '{key}' is not a tensor")
        leaf = state_value.detach().clone().requires_grad_(True)
        clone_state[key] = leaf
        tracked[key] = leaf
    out = run_simulation(clone_state, mesh, **kwargs)
    loss = objective(out)
    if not isinstance(loss, torch.Tensor):
        raise TypeError(f"objective must return a tensor, got {type(loss).__name__}")
    if loss.ndim != 0:
        raise MpasConfigError("objective must return a scalar tensor")
    grads = torch.autograd.grad(loss, list(tracked.values()), allow_unused=True)
    return {key: grad for key, grad in zip(tracked.keys(), grads) if grad is not None}


run_simulation.jvp = _run_simulation_jvp  # type: ignore[attr-defined]
run_simulation.vjp = _run_simulation_vjp  # type: ignore[attr-defined]
