"""
RK3 time integration for MPAS-Atmosphere dynamics core.

Implements split-explicit Runge-Kutta 3rd order time integration
following Klemp et al. (2007) and Skamarock & Gassmann (2011).

The split-explicit method separates fast acoustic/gravity waves from
slower advective processes, allowing different timestep sizes for each.
"""

import functools
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import torch

from mpas_pytorch.autodiff import differentiable_mode, is_grad_mode
from mpas_pytorch.autodiff.checkpoint import (
    checkpoint_if_enabled,
    checkpoint_state_dict,
)
from mpas_pytorch.autodiff.tridiagonal_scan import (
    TRIDIAG_SCAN_DEFAULT_POLICY,
    canonical_tridiag_scan_shape_keys,
    ensure_vertical_scan_plan,
    normalize_tridiag_scan_policy,
    tridiag_scan_plan_cache,
    tridiag_scan_policy,
)

try:
    from torch.utils.checkpoint import (
        CheckpointPolicy as _CheckpointPolicy,
    )
    from torch.utils.checkpoint import (
        create_selective_checkpoint_contexts as _create_selective_checkpoint_contexts,
    )
except ImportError:  # pragma: no cover - exercised by older CUDA images.
    _CheckpointPolicy = None  # type: ignore[misc,assignment]
    _create_selective_checkpoint_contexts = None  # type: ignore[assignment]


from mpas_pytorch.config_defaults import (
    ATMOSPHERE_INERT_ADVECTION_ORDER_KEYS,
    FORTRAN_DEFAULTS,
    get_config_value_safe,
    validate_time_integration_scheme,
)
from mpas_pytorch.constants import DTYPE, PhysicalConstants
from mpas_pytorch.dynamics.static_payloads import (
    DycoreStaticPayloads,
    StaticPairRelation,
    StaticSlotRelation,
    ensure_static_payload_store,
)
from mpas_pytorch.dynamics.stencil_ops import (
    _clone_for_grad,
    _compile_active,
    _replace_row_slice,
    _tensor_carries_autodiff,
    _tensor_version_cache_key,
)
from mpas_pytorch.initialization import resolve_len_disp
from mpas_pytorch.profiling import profile_section, profiled_call
from mpas_pytorch.utils.checks import require_consistent_device

# Boundary condition kernels for regional MPAS are imported lazily inside
# config_apply_lbcs guards. The current package does not ship the boundary
# module; regional/LBC readiness remains blocked until those kernels are restored.


# =============================================================================
# Integer array dtype verification
# Arrays that MUST be integer dtype (not float64). Float connectivity arrays
# will be silently truncated when .item() or int() is called during indexing,
# causing incorrect connectivity lookups.
# =============================================================================
INTEGER_ARRAYS = frozenset(
    {
        # Connectivity arrays
        "cellsOnEdge",
        "edgesOnCell",
        "verticesOnCell",
        "cellsOnCell",
        "verticesOnEdge",
        "edgesOnVertex",
        "cellsOnVertex",
        "edgesOnEdge",
        "advCellsForEdge",
        "kiteForCell",
        # Count arrays
        "nEdgesOnCell",
        "nEdgesOnEdge",
        "nAdvCellsForEdge",
        # ID and categorical/mask arrays
        "indexToCellID",
        "indexToEdgeID",
        "indexToVertexID",
        "bdyMaskVertex",
        "landmask",
        "ivgtyp",
        "isltyp",
    }
)

VALID_INTEGER_DTYPES = frozenset({torch.int32, torch.int64})

_STATIC_INPUT_VALIDATION_CACHE_VERSION = 1
_INDEXING_CONNECTIVITY_CHECKS = (
    ("cellsOnEdge", "nEdges", "nCells"),
    ("edgesOnCell", "nCells", "nEdges"),
    ("verticesOnCell", "nCells", "nVertices"),
    ("cellsOnCell", "nCells", "nCells"),
    ("verticesOnEdge", "nEdges", "nVertices"),
    ("edgesOnVertex", "nVertices", "nEdges"),
    ("cellsOnVertex", "nVertices", "nCells"),
)
_SENTINEL_PLACEMENT_CHECKS = (
    ("edgesOnCell", "nEdgesOnCell"),
    ("cellsOnCell", "nEdgesOnCell"),
    ("verticesOnCell", "nEdgesOnCell"),
)


# Inner non-reentrant checkpoints must preserve autograd.grad reachability while
# avoiding activation saves during the forward pass. MUST_RECOMPUTE is the
# PyTorch 2.7 save-nothing policy; PREFER_RECOMPUTE is equivalent to vanilla
# checkpointing and still allows the default activation-saving behavior.
def _missing_save_nothing_context():
    raise RuntimeError(
        "gradient_checkpointing with save-nothing selective checkpointing requires "
        "torch.utils.checkpoint.CheckpointPolicy/create_selective_checkpoint_contexts. "
        f"Installed PyTorch {torch.__version__} does not provide that API."
    )


if _CheckpointPolicy is not None and _create_selective_checkpoint_contexts is not None:
    _SAVE_NOTHING_CONTEXT: Callable[..., Any] = functools.partial(
        _create_selective_checkpoint_contexts,
        lambda ctx, op, *args, **kwargs: _CheckpointPolicy.MUST_RECOMPUTE,
    )
else:
    _SAVE_NOTHING_CONTEXT = _missing_save_nothing_context


# =============================================================================
# Edge-sized arrays requiring accurate nEdges
# If ANY of these keys are present in state_dict, nEdges must be explicitly
# provided and cannot be approximated from nCells * 3. The approximation
# would cause shape mismatches downstream for actual edge-sized tensors.
# =============================================================================
# Arrays that are truly edge-sized: shape (nEdges+1, ...)
EDGE_SIZED_KEYS = frozenset(
    {
        "ru",
        "ru_p",
        "ruAvg",
        "ru_save",
        "uhAvg",
        "dcEdge",
        "dvEdge",
        "invDcEdge",
        "invDvEdge",
        "tend_u",
        "v",
    }
)

# Arrays that are cell-sized but co-occur with edge data in acoustic step
CELL_SIZED_ACOUSTIC_KEYS = frozenset(
    {
        "rw",
        "rho_p",
        "rtheta_p",
        "rw_p",
        "rho_pp",
        "rtheta_pp",
        "rw_save",
    }
)

# Union: presence of ANY of these suggests edge data context
EDGE_REQUIRING_KEYS = EDGE_SIZED_KEYS | CELL_SIZED_ACOUSTIC_KEYS

# Euler's formula for hexagonal mesh: E ≈ 3V (edges ≈ 3 * cells)
_HEXAGONAL_EDGE_FACTOR = 3


def _mapping_tensor_or_zeros(
    source: Mapping[str, Any],
    key: str,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype = DTYPE,
    device: torch.device,
) -> torch.Tensor:
    value = source.get(key)
    if isinstance(value, torch.Tensor):
        return value
    return torch.zeros(shape, dtype=dtype, device=device)


def _acoustic_seed_tensor(
    source: Mapping[str, Any],
    key: str,
    shape: tuple[int, ...],
    *,
    device: torch.device,
    take_ownership: bool,
    preserve_alias: bool = False,
) -> torch.Tensor:
    """Return an owned acoustic-step seed while preserving public and AD isolation."""
    value = source.get(key)
    if not isinstance(value, torch.Tensor):
        return torch.zeros(shape, dtype=DTYPE, device=device)
    if take_ownership and not preserve_alias and not torch.is_grad_enabled():
        return value
    return value.clone()


def _first_tensor_or_zeros(
    primary: Mapping[str, Any],
    secondary: Mapping[str, Any],
    key: str,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype = DTYPE,
    device: torch.device,
) -> torch.Tensor:
    value = primary.get(key)
    if isinstance(value, torch.Tensor):
        return value
    return _mapping_tensor_or_zeros(secondary, key, shape, dtype=dtype, device=device)


def _compile_disables_workspace(config: Mapping[str, Any]) -> bool:
    """Return whether workspace scratch reuse must be disabled for compile."""
    from mpas_pytorch.dynamics.compile_utils import compile_enabled

    return compile_enabled(dict(config))


def _clone_compiled_output_mapping(outputs: Mapping[str, Any]) -> dict[str, Any]:
    """Detach compiled-region output ownership from reusable graph/workspace buffers."""
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in outputs.items()
    }


def _resolve_compile_workspace_policy(
    config: Mapping[str, Any],
    workspace_obj: Any,
    use_workspace: bool,
) -> tuple[Any, bool]:
    """Disable workspace scratch reuse for compiled execution."""
    workspace_requested = workspace_obj is not None or use_workspace
    if workspace_requested and _compile_disables_workspace(config):
        warnings.warn(
            "config_use_workspace/_dycore_workspace is disabled while torch.compile "
            "is enabled; compiled dycore kernels require functional scratch tensors.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None, False
    return workspace_obj, use_workspace


def _prebound_static_payload(
    config: Mapping[str, Any],
    payload_name: str,
    cache_key_fn: Any,
    payload_fn: Any,
    payload_kwargs: dict[str, Any],
) -> Any:
    """Return a cached static topology payload stored on the runtime config."""
    payload_key = cache_key_fn(**payload_kwargs)
    if payload_key is None:
        return None

    def build_payload() -> Any:
        # Forecast inference mode deliberately creates tensors without version
        # counters.  Static payloads outlive one call and are used in mutation-
        # sensitive plan keys, so construct them as ordinary tensors while
        # leaving the surrounding forecast math in inference mode.
        with torch.inference_mode(False):
            return payload_fn(**payload_kwargs)

    existing_store = config.get("_dycore_static_payloads")
    if isinstance(existing_store, DycoreStaticPayloads):
        return existing_store.get_or_build_typed(
            payload_name,
            payload_key,
            build_payload,
        )

    if isinstance(config, dict):
        static_payloads = ensure_static_payload_store(config)
        return static_payloads.get_or_build_typed(
            payload_name,
            payload_key,
            build_payload,
        )

    return build_payload()


def _runner_owned_monotonic_scalar_transport_numeric_plan(
    store: DycoreStaticPayloads,
    **plan_kwargs: Any,
) -> Any:
    """Build or reuse the exact monotonic-scalar-transport chunk/tile plan at the runner boundary."""
    from mpas_pytorch.dynamics.advance_scalars_mono import (
        build_monotonic_scalar_transport_numeric_plan,
        monotonic_scalar_transport_numeric_plan_owner_key,
    )

    key_kwargs = dict(plan_kwargs)
    key_kwargs.pop("connectivity_payload")
    owner_key = monotonic_scalar_transport_numeric_plan_owner_key(**key_kwargs)
    return store.get_or_build_numeric_plan(
        "monotonic_scalar_transport",
        owner_key,
        lambda: build_monotonic_scalar_transport_numeric_plan(**plan_kwargs),
    )


def _runner_owned_compiled_explicit_scalar_transport_numeric_plan(
    store: DycoreStaticPayloads,
    plan: Any,
    *,
    compile_region_names: tuple[str, ...],
    compile_kernel: Any,
    mode: str,
    backend: str,
) -> Any:
    """Compile each immutable explicit-scalar-transport section plan once at the runner boundary."""
    from mpas_pytorch.dynamics.advance_scalars import (
        compile_explicit_scalar_transport_numeric_plan,
    )

    region_names = tuple(sorted(frozenset(compile_region_names)))
    cache_key = (plan.key, region_names, str(mode), str(backend))
    return store.get_or_build_numeric_plan(
        "explicit_scalar_transport_numeric_composition_compiled",
        cache_key,
        lambda: compile_explicit_scalar_transport_numeric_plan(
            plan,
            compile_region_names=region_names,
            compile_kernel=compile_kernel,
            mode=mode,
            backend=backend,
        ),
    )


def _runner_owned_large_step_recovery_recovery_plan(
    store: DycoreStaticPayloads,
    **plan_kwargs: Any,
) -> Any:
    """Build or reuse the exact large-step-recovery recovery plan at the runner boundary."""
    from mpas_pytorch.dynamics.recover_large_step import (
        build_large_step_recovery_recovery_plan,
        large_step_recovery_recovery_plan_owner_key,
    )

    key_kwargs = dict(plan_kwargs)
    key_kwargs.pop("connectivity_payload")
    connectivity_identity = key_kwargs.pop("connectivity_key")
    owner_key = large_step_recovery_recovery_plan_owner_key(
        connectivity_identity=connectivity_identity,
        **key_kwargs,
    )
    return store.get_or_build_numeric_plan(
        "large_step_recovery_recovery",
        owner_key,
        lambda: build_large_step_recovery_recovery_plan(**plan_kwargs),
    )


def _runner_owned_small_step_perturbation_numeric_plan(
    store: DycoreStaticPayloads,
    **plan_kwargs: Any,
) -> Any:
    """Build or reuse the exact small-step-perturbation numeric plan at the runner boundary."""
    from mpas_pytorch.dynamics.smlstep_pert import (
        build_small_step_perturbation_numeric_plan,
        small_step_perturbation_numeric_plan_owner_key,
    )

    key_kwargs = dict(plan_kwargs)
    key_kwargs.pop("connectivity_payload")
    connectivity_identity = key_kwargs.pop("connectivity_key")
    owner_key = small_step_perturbation_numeric_plan_owner_key(
        connectivity_identity=connectivity_identity,
        **key_kwargs,
    )
    return store.get_or_build_numeric_plan(
        "small_step_perturbation",
        owner_key,
        lambda: build_small_step_perturbation_numeric_plan(**plan_kwargs),
    )


def _runner_owned_divergence_damping_plan(
    store: DycoreStaticPayloads,
    **plan_kwargs: Any,
) -> Any:
    """Build or reuse one divergence-damping plan per exact runner key."""
    from mpas_pytorch.dynamics.divergence_damping import (
        build_divergence_damping_plan,
        divergence_damping_plan_owner_key,
    )

    key_kwargs = dict(plan_kwargs)
    key_kwargs.pop("connectivity_payload")
    connectivity_identity = key_kwargs.pop("connectivity_key")
    owner_key = divergence_damping_plan_owner_key(
        connectivity_identity=connectivity_identity,
        **key_kwargs,
    )
    return store.get_or_build_numeric_plan(
        "divergence_damping",
        owner_key,
        lambda: build_divergence_damping_plan(**plan_kwargs),
    )


# =============================================================================
# MEMORY OPTIMIZATION: Immutable mesh arrays
# These arrays are never modified during time integration and can be referenced
# without cloning to save memory (~3x reduction in mesh tensor memory).
# =============================================================================
IMMUTABLE_MESH_KEYS = frozenset(
    {
        # Connectivity arrays (integer indices)
        "cellsOnEdge",
        "edgesOnCell",
        "verticesOnCell",
        "cellsOnCell",
        "verticesOnEdge",
        "edgesOnVertex",
        "cellsOnVertex",
        "edgesOnEdge",
        "advCellsForEdge",
        "kiteForCell",
        "nEdgesOnCell",
        "nEdgesOnEdge",
        "nAdvCellsForEdge",
        # Horizontal geometry (fixed mesh properties)
        "dvEdge",
        "dcEdge",
        "invAreaCell",
        "areaCell",
        "invDcEdge",
        "invDvEdge",
        "invAreaTriangle",
        "meshScalingDel2",
        "meshScalingDel4",
        "fEdge",
        "fVertex",
        "latCell",
        "latEdge",
        "angleEdge",
        "weightsOnEdge",
        "kiteAreasOnVertex",
        # Vertical coordinate (fixed vertical structure)
        "fzm",
        "fzp",
        "rdzw",
        "rdzu",
        "dzu",
        "dzw",
        "fnm",
        "fnp",
        "rdnw",
        "cf1",
        "cf2",
        "cf3",
        "zgrid",
        "zb",
        "zb3",
        # Scalar dimension info
        "nCells",
        "nEdges",
        "nVertices",
        "nVertLevels",
        "maxEdges",
        "vertexDegree",
        "maxEdges2",
    }
)

# =============================================================================
# MESH-AUTHORITATIVE KEYS
# Cell-associated fields computed from mesh geometry during Fortran
# initialization that may exist in the IC checkpoint with stale or zero
# values (written before the initialization routine ran). When both IC and
# mesh provide these fields, the mesh version is authoritative and must
# override the IC version.

# IMPORTANT: Only cell-associated arrays can be overridden, because the IC
# and mesh checkpoints may use different edge orderings (the edge indices
# are a permutation of each other). Cell ordering is identical in both,
# so cell-associated overrides are safe. Edge-associated arrays (zxu, zb,
# zb3, deriv_two, meshScalingDel2/Del4) must NOT be overridden here.
# =============================================================================
MESH_AUTHORITATIVE_KEYS = frozenset(
    {
        # Rayleigh damping coefficient (cell-associated: nCells+1 x nVertLevels).
        # Computed by atm_compute_damping_coefs after IC checkpoint is written.
        # IC has all zeros; mesh has correct values.
        "dss",
    }
)

# Mesh keys whose values are tied to connectivity edge ordering. When the state
# connectivity is stale/inconsistent, these must come from the same ordering
# source as cellsOnEdge/edgesOnCell.
MESH_EDGE_ORDER_KEYS = frozenset(
    {
        "cellsOnEdge",
        "edgesOnCell",
        "verticesOnCell",
        "cellsOnCell",
        "verticesOnEdge",
        "edgesOnVertex",
        "cellsOnVertex",
        "edgesOnEdge",
        "nEdgesOnCell",
        "nEdgesOnEdge",
        "advCellsForEdge",
        "nAdvCellsForEdge",
        "dvEdge",
        "dcEdge",
        "invDcEdge",
        "invDvEdge",
        "fEdge",
        "latEdge",
        "angleEdge",
        "weightsOnEdge",
        "kiteAreasOnVertex",
        "meshScalingDel2",
        "meshScalingDel4",
        "zxu",
        "zb",
        "zb3",
        "zb_cell",
        "zb3_cell",
        "deriv_two",
        "defc_a",
        "defc_b",
        "adv_coefs",
        "adv_coefs_3rd",
        "edgesOnCell_sign",
        "edgesOnVertex_sign",
    }
)


def _is_edge_sized_tensor(value: Any, nEdges: int) -> bool:
    """Return True when value is an edge-indexed tensor under MPAS conventions."""
    return (
        isinstance(value, torch.Tensor)
        and value.ndim > 0
        and value.shape[0] in (nEdges, nEdges + 1)
    )


def _normalize_connectivity_sentinels(
    arr: torch.Tensor,
    nEdges: int,
) -> torch.Tensor:
    """Treat 0 and nEdges+1 as equivalent sentinel values for comparisons."""
    if arr.dtype not in VALID_INTEGER_DTYPES:
        return arr
    sentinel = nEdges + 1
    return torch.where(arr == sentinel, torch.zeros_like(arr), arr)


def _detect_edge_ordering_mismatch(
    state_dict: dict[str, Any],
    mesh_dict: dict[str, Any],
) -> bool:
    """Detect edge-order permutations between state and mesh dictionaries.

    When this returns True, filling *missing* edge-indexed fields from mesh can
    silently mix incompatible orderings (state arrays in one edge order, mesh
    arrays in another). State values are always preferred when present.
    """
    if not isinstance(mesh_dict, dict):
        return False

    if "nEdges" not in state_dict and "nEdges" not in mesh_dict:
        return False
    nEdges = _get_int_from_dict(
        state_dict, "nEdges", _get_int_from_dict(mesh_dict, "nEdges", 0)
    )
    if nEdges <= 0:
        return False

    candidate_keys = ("edgesOnCell", "cellsOnEdge", "dvEdge", "dcEdge")
    for key in candidate_keys:
        if key not in state_dict or key not in mesh_dict:
            continue
        s_val = state_dict[key]
        m_val = mesh_dict[key]
        if not isinstance(s_val, torch.Tensor) or not isinstance(m_val, torch.Tensor):
            continue
        if s_val.shape != m_val.shape:
            continue

        if key in ("edgesOnCell", "cellsOnEdge"):
            s_cmp = _normalize_connectivity_sentinels(s_val, nEdges)
            m_cmp = _normalize_connectivity_sentinels(m_val, nEdges)
        else:
            s_cmp = s_val
            m_cmp = m_val

        if not torch.equal(s_cmp, m_cmp):
            return True

    return False


def _ru_connectivity_residual(
    state_dict: dict[str, Any],
    cells_on_edge: torch.Tensor,
) -> float | None:
    """Mean absolute residual for ru ~= 0.5*(rho_zz(c1)+rho_zz(c2))*u.

    This is used to infer which connectivity ordering is consistent with the
    prognostic fields loaded in state_dict.
    """
    required = ("u", "ru", "rho_zz", "nCells", "nEdges")
    if any(k not in state_dict for k in required):
        return None

    u = state_dict["u"]
    ru = state_dict["ru"]
    rho_zz = state_dict["rho_zz"]
    if not all(isinstance(v, torch.Tensor) for v in (u, ru, rho_zz, cells_on_edge)):
        return None
    if u.ndim != 2 or ru.ndim != 2 or rho_zz.ndim != 2:
        return None
    if cells_on_edge.ndim != 2 or cells_on_edge.shape[1] < 2:
        return None

    nCells = _get_int_from_dict(state_dict, "nCells", 0)
    nEdges = _get_int_from_dict(state_dict, "nEdges", 0)
    if nCells <= 0 or nEdges <= 0:
        return None

    edge_count = min(nEdges, u.shape[0], ru.shape[0], cells_on_edge.shape[0])
    if edge_count <= 0:
        return None
    level_count = min(u.shape[1], ru.shape[1], rho_zz.shape[1])
    if level_count <= 0:
        return None

    coe = cells_on_edge[:edge_count, :2]
    c1 = coe[:, 0] - 1
    c2 = coe[:, 1] - 1
    interior_mask = (c1 >= 0) & (c1 < nCells) & (c2 >= 0) & (c2 < nCells)
    valid_count = int(torch.count_nonzero(interior_mask).detach().cpu())
    if valid_count == 0:
        return None

    max_cell_idx = max(0, nCells - 1)
    c1_safe, c2_safe = c1.clamp(0, max_cell_idx), c2.clamp(0, max_cell_idx)
    u_sel = u[:edge_count, :level_count]
    ru_sel = ru[:edge_count, :level_count]
    rho_sel = rho_zz[:nCells, :level_count]
    rho_sum = rho_sel[c1_safe] + rho_sel[c2_safe]
    ru_pred = 0.5 * rho_sum * u_sel
    residual_sum = torch.sum(
        torch.abs(ru_pred - ru_sel) * interior_mask[:, None].to(dtype=ru.dtype)
    )
    return float((residual_sum / float(valid_count * level_count)).item())


def _should_prefer_mesh_connectivity(
    state_dict: dict[str, Any],
    mesh_dict: dict[str, Any],
) -> bool:
    """Return True when mesh connectivity is far more self-consistent than state."""
    if "cellsOnEdge" not in state_dict or "cellsOnEdge" not in mesh_dict:
        return False

    state_score = _ru_connectivity_residual(state_dict, state_dict["cellsOnEdge"])
    mesh_score = _ru_connectivity_residual(state_dict, mesh_dict["cellsOnEdge"])
    if state_score is None or mesh_score is None:
        return False

    # Prefer mesh only when there is a decisive consistency gap.
    if mesh_score <= 1.0e-12 and state_score >= 1.0e-8:
        return True
    return mesh_score * 1000.0 < state_score


def _zero_padding_row(base: torch.Tensor, row_idx: int) -> torch.Tensor:
    """Return ``base`` with the selected padding row functionally replaced by zeros."""
    if row_idx < 0 or row_idx >= base.shape[0]:
        return base
    zeros = base.new_zeros((1, *base.shape[1:]))
    return _replace_row_slice(base, row_idx, row_idx + 1, zeros)


def _backfill_edge_orientation_fields(
    state_dict: dict[str, Any],
    strict: bool = True,
) -> None:
    """Backfill edge orientation arrays from state connectivity when missing."""
    missing_signs = [
        k for k in ("edgesOnCell_sign", "edgesOnVertex_sign") if k not in state_dict
    ]
    if not missing_signs:
        return

    required_for_signs = [
        "nCells",
        "nEdges",
        "nVertices",
        "maxEdges",
        "vertexDegree",
        "cellsOnEdge",
        "edgesOnCell",
        "nEdgesOnCell",
        "edgesOnVertex",
        "verticesOnEdge",
    ]
    if all(k in state_dict for k in required_for_signs):
        from mpas_pytorch.initialization import compute_edge_signs

        sign_result = compute_edge_signs(state_dict)
        if "edgesOnCell_sign" not in state_dict:
            state_dict["edgesOnCell_sign"] = sign_result["edgesOnCell_sign"]
        if "edgesOnVertex_sign" not in state_dict:
            state_dict["edgesOnVertex_sign"] = sign_result["edgesOnVertex_sign"]
        return

    if strict:
        raise ValueError(
            "Edge ordering mismatch prevented mesh fallback for sign arrays, but "
            "state is missing connectivity needed to synthesize signs. Missing one of: "
            f"{required_for_signs}"
        )


def _apply_coef_3rd_order_override(
    state_dict: dict[str, Any],
    config: dict[str, Any],
) -> None:
    """Apply an explicit runtime override and refresh dependent transport fields."""
    target_coef = get_config_value_safe(config, "config_coef_3rd_order", None)
    if target_coef is None:
        target_coef = get_config_value_safe(config, "coef_3rd_order", None)
    if target_coef is None:
        return
    current_coef = _get_float_from_dict(state_dict, "coef_3rd_order", 0.25)
    target_coef = float(target_coef)
    if current_coef == target_coef:
        return
    from mpas_pytorch.ic_loader import apply_coef_3rd_order_override_fields

    apply_coef_3rd_order_override_fields(
        state_dict,
        state_dict,
        coef_3rd_order=target_coef,
    )


def _merge_mesh_into_state(
    state_dict: dict[str, Any],
    mesh_dict: dict[str, Any] | None,
    strict: bool = True,
) -> dict[str, Any]:
    """Merge mesh fields into state with edge-ordering safety guards."""
    merged = dict(state_dict)
    if mesh_dict is None:
        return merged

    edge_ordering_mismatch = bool(merged.get("_edge_ordering_mismatch", False))
    if not edge_ordering_mismatch:
        edge_ordering_mismatch = _detect_edge_ordering_mismatch(merged, mesh_dict)
    mesh_connectivity_preferred = False
    if edge_ordering_mismatch:
        mesh_connectivity_preferred = _should_prefer_mesh_connectivity(
            merged, mesh_dict
        )
        if mesh_connectivity_preferred:
            for key in MESH_EDGE_ORDER_KEYS:
                if key in mesh_dict:
                    merged[key] = mesh_dict[key]
            # Once static connectivity-dependent fields are aligned to mesh,
            # subsequent fallback decisions should not treat this as mismatch.
            edge_ordering_mismatch = False
    merged["_edge_ordering_mismatch"] = edge_ordering_mismatch
    merged["_mesh_connectivity_preferred"] = mesh_connectivity_preferred

    nEdges = _get_int_from_dict(
        merged, "nEdges", _get_int_from_dict(mesh_dict, "nEdges", 0)
    )
    skipped_edge_fallback = []
    edge_order_dependent_cell_keys = {"edgesOnCell_sign", "edgesOnVertex_sign"}

    for key, value in mesh_dict.items():
        if key in MESH_AUTHORITATIVE_KEYS:
            merged[key] = value
            continue
        if key in merged:
            continue

        if edge_ordering_mismatch and (
            _is_edge_sized_tensor(value, nEdges)
            or key in edge_order_dependent_cell_keys
        ):
            skipped_edge_fallback.append(key)
            continue

        merged[key] = value

    # Reconstruct signs from state connectivity if mesh fallback was blocked.
    if skipped_edge_fallback:
        _backfill_edge_orientation_fields(merged, strict=strict)

        unresolved = [k for k in skipped_edge_fallback if k not in merged]
        if unresolved and strict:
            raise ValueError(
                "State/mesh edge ordering mismatch detected. Refusing to import these "
                f"edge-indexed mesh fields into state: {sorted(unresolved)}. "
                "Provide these fields in the state checkpoint with matching edge ordering."
            )
        if unresolved and not strict:
            warnings.warn(
                "State/mesh edge ordering mismatch detected. Skipping mesh fallback "
                f"for edge-indexed fields: {sorted(unresolved)}. Subsequent kernels "
                "may fail if these fields are required.",
                RuntimeWarning,
                stacklevel=2,
            )

    return merged


def _validate_indexing_convention(
    state_dict: dict[str, Any], strict: bool = True
) -> None:
    """
    Validate that all connectivity arrays use 1-based+padded indexing.

    Kernels expect:
    - Arrays have shape (N+1, ...) where N is the dimension count
    - Values range from 0 (sentinel) to N (valid)
    - Index 0 in arrays is padding, data starts at index 1

    MESHState uses 0-based indexing (shape N, values -1 to N-1) which is
    incompatible with kernels.

    Args:
        state_dict: Dictionary of state/mesh tensors
        strict: If True, raise ValueError on mismatch. If False, warn only.

    Raises:
        ValueError: If strict=True and indexing mismatch detected.
    """
    # Define connectivity arrays and their expected dimension reference
    connectivity_checks = _INDEXING_CONNECTIVITY_CHECKS

    issues: list[str] = []

    for array_name, dim_name, value_dim_name in connectivity_checks:
        if array_name not in state_dict or dim_name not in state_dict:
            continue

        arr = state_dict[array_name]
        dim_val = state_dict[dim_name]
        n_dim = int(dim_val.item() if hasattr(dim_val, "item") else dim_val)

        # Check shape: should be (n_dim+1, ...) not (n_dim, ...)
        if arr.shape[0] == n_dim:
            issues.append(
                f"{array_name}: shape[0]={arr.shape[0]} matches {dim_name}={n_dim} "
                f"(0-based). Expected {n_dim + 1} (1-based+padded)."
            )

        # Check min value: should be 0 (sentinel) not -1 (0-based sentinel)
        min_val = arr.min().item()
        if min_val == -1:
            issues.append(
                f"{array_name}: min_val={min_val} indicates 0-based indexing. "
                f"Expected 0 (1-based sentinel)."
            )

        # Check for invalid negative values (anything < 0)
        # For 1-based indexing, only non-negative values are valid:
        # - 0 = sentinel (padding or no neighbor)
        # - 1 to N = valid indices
        negative_mask = arr < 0
        if negative_mask.any():
            unique_negatives = arr[negative_mask].unique().tolist()
            issues.append(
                f"{array_name}: contains invalid negative values {unique_negatives}. "
                f"For 1-based indexing, only non-negative values are valid (0=sentinel, 1-N=indices)."
            )

        # Check max value: should not exceed the dimension it references
        # value_dim_name indicates what dimension the values index into
        # Exception: n_value_dim + 1 is a valid boundary sentinel (edge with no neighbor)
        if value_dim_name in state_dict:
            value_dim_val = state_dict[value_dim_name]
            n_value_dim = int(
                value_dim_val.item()
                if hasattr(value_dim_val, "item")
                else value_dim_val
            )
            max_val = arr.max().item()
            valid_boundary_sentinel = n_value_dim + 1
            if max_val > n_value_dim and max_val != valid_boundary_sentinel:
                issues.append(
                    f"{array_name}: max_val={max_val} exceeds {value_dim_name}={n_value_dim}. "
                    f"Valid range is 0 (sentinel) to {n_value_dim}, or {valid_boundary_sentinel} (boundary sentinel)."
                )

    if issues:
        msg = (
            "Indexing convention mismatch detected:\n"
            + "\n".join(f"  - {issue}" for issue in issues)
            + "\n\nKernels expect 1-based+padded arrays from Fortran-style raw input dictionaries. "
            "Do not use the mesh-conversion helper that rewrites connectivity to 0-based. "
            "See MESHState docstring for details."
        )
        if strict:
            raise ValueError(msg)
        else:
            warnings.warn(msg, RuntimeWarning)


def _validate_sentinel_placement(
    arr: torch.Tensor,
    array_name: str,
    per_element_count: torch.Tensor,
    per_element_count_name: str,
) -> list:
    """Validate that sentinels (0 values) only appear in padding positions.

    For variable-size arrays like edgesOnCell[nCells+1, maxEdges], each row i
    should have valid data in columns 0..nEdgesOnCell[i]-1 and sentinels (0)
    in columns nEdgesOnCell[i]..maxEdges-1.

    This function catches a critical bug where a 0 sentinel in a valid data slot
    would cause the kernel to subtract 1 (for 1-based to 0-based conversion)
    and index into position -1, silently accessing the last element in Python.

    Args:
        arr: 2D connectivity array to validate
        array_name: Name for error messages
        per_element_count: 1D array specifying valid elements per row (e.g., nEdgesOnCell)
        per_element_count_name: Name of count array for error messages

    Returns:
        List of issue strings (empty if valid)
    """
    issues: list[str] = []

    if arr.dim() != 2:
        return issues  # Only validate 2D arrays

    n_rows, n_cols = arr.shape
    count_len = per_element_count.shape[0]

    # Validate dimensions match
    if count_len != n_rows:
        issues.append(
            f"{array_name}: shape[0]={n_rows} doesn't match "
            f"{per_element_count_name} length={count_len}"
        )
        return issues

    valid_counts = per_element_count.to(dtype=torch.int64)
    valid_rows = (valid_counts > 0) & (valid_counts <= n_cols)
    if not bool(valid_rows.any()):
        return issues

    col_idx = torch.arange(n_cols, device=arr.device, dtype=torch.int64).unsqueeze(0)
    valid_mask = valid_rows.unsqueeze(1) & (col_idx < valid_counts.unsqueeze(1))
    zero_mask = valid_mask & (arr == 0)
    row_has_zero = zero_mask.any(dim=1)
    error_count = int(torch.count_nonzero(row_has_zero).detach().cpu())
    if error_count == 0:
        return issues

    if _compile_active():
        issues.append(
            f"{array_name}: found sentinel (0) within valid connectivity region "
            f"for {error_count} rows."
        )
        return issues

    error_rows = torch.nonzero(row_has_zero, as_tuple=False).flatten()
    for i in error_rows[:10].tolist():
        zeros_in_valid = (
            torch.nonzero(zero_mask[i], as_tuple=False).flatten().cpu().tolist()
        )
        valid_count = int(valid_counts[i].item())
        issues.append(
            f"{array_name}[{i}]: found sentinel (0) at positions {zeros_in_valid} "
            f"within valid region (columns 0-{valid_count - 1} per {per_element_count_name}[{i}]={valid_count}). "
            f"This will cause -1 indexing when converted to 0-based."
        )
    if error_rows.numel() > 10:
        issues.append(f"... (truncated, more {array_name} sentinel issues exist)")

    return issues


def _static_validation_value_key(value: Any) -> tuple[object, ...] | None:
    """Return a mutation-sensitive key for static validation inputs."""
    if isinstance(value, torch.Tensor):
        if _tensor_carries_autodiff(value):
            return None
        tensor_key = _tensor_version_cache_key(value)
        if tensor_key is None:
            return None
        return ("tensor", *tensor_key)
    if isinstance(value, (bool, int, float, np.integer, np.floating)):
        return ("scalar", type(value).__name__, value)
    if value is None:
        return ("missing",)
    return None


def _static_input_validation_cache_key(
    state_dict: Mapping[str, Any],
    *,
    strict_indexing: bool,
) -> tuple[object, ...] | None:
    """Return a cache key for strict static indexing/sentinel validation.

    The key covers exactly the topology/count/dimension values read by
    _validate_indexing_convention() and _validate_sentinel_placement().  Dynamic
    prognostic fields are intentionally excluded so warmed forecast loops do not
    revalidate immutable mesh metadata at every RK stage.
    """
    names: set[str] = set()
    for array_name, dim_name, value_dim_name in _INDEXING_CONNECTIVITY_CHECKS:
        names.update((array_name, dim_name, value_dim_name))
    for array_name, count_name in _SENTINEL_PLACEMENT_CHECKS:
        names.update((array_name, count_name))

    keyed_values: list[tuple[str, tuple[object, ...]]] = []
    for name in sorted(names):
        value_key = _static_validation_value_key(state_dict.get(name))
        if value_key is None:
            return None
        keyed_values.append((name, value_key))

    return (
        "static_input_validation",
        _STATIC_INPUT_VALIDATION_CACHE_VERSION,
        bool(strict_indexing),
        tuple(keyed_values),
    )


def _static_input_validation_store(
    config: Mapping[str, Any],
) -> DycoreStaticPayloads | None:
    store = config.get("_dycore_static_payloads")
    return store if isinstance(store, DycoreStaticPayloads) else None


def _mesh_zero_based_warning_cache_key(
    mesh_dict: Mapping[str, Any],
) -> tuple[object, ...] | None:
    cells_on_edge = mesh_dict.get("cellsOnEdge")
    if not isinstance(cells_on_edge, torch.Tensor):
        return None
    if _tensor_carries_autodiff(cells_on_edge):
        return None
    tensor_key = _tensor_version_cache_key(cells_on_edge)
    if tensor_key is None:
        return None
    return (
        "mesh_zero_based_warning",
        _STATIC_INPUT_VALIDATION_CACHE_VERSION,
        *tensor_key,
    )


def _mesh_cells_on_edge_min_is_minus_one(
    mesh_dict: Mapping[str, Any],
    *,
    store: DycoreStaticPayloads | None,
) -> bool:
    key = _mesh_zero_based_warning_cache_key(mesh_dict)
    if key is not None and store is not None:
        cached = store.get_or_build(
            "mesh_zero_based_warning",
            key,
            lambda: (
                _coerce_int(
                    mesh_dict["cellsOnEdge"].min(),
                    "cellsOnEdge_min",
                    allow_negative=True,
                )
                == -1
            ),
        )
        return bool(cached)
    coe = mesh_dict["cellsOnEdge"]
    return bool(
        hasattr(coe, "min")
        and _coerce_int(coe.min(), "cellsOnEdge_min", allow_negative=True) == -1
    )


def _validate_static_indexing_inputs_cached(
    state_dict: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    strict_indexing: bool,
) -> None:
    """Validate static topology once per mutation-sensitive key."""
    cache_key = _static_input_validation_cache_key(
        state_dict,
        strict_indexing=bool(strict_indexing),
    )
    store = _static_input_validation_store(config)
    if cache_key is not None and store is not None:  # noqa: SIM102
        if store.has_validated_static_inputs(cache_key):
            return

    _validate_indexing_convention(dict(state_dict), strict=bool(strict_indexing))

    sentinel_issues = []
    for array_name, count_name in _SENTINEL_PLACEMENT_CHECKS:
        if array_name in state_dict and count_name in state_dict:
            arr = state_dict[array_name]
            counts = state_dict[count_name]
            if isinstance(arr, torch.Tensor) and isinstance(counts, torch.Tensor):
                sentinel_issues.extend(
                    _validate_sentinel_placement(arr, array_name, counts, count_name)
                )
    if sentinel_issues:
        message = "Connectivity sentinel-placement validation failed:\n" + "\n".join(
            f"  - {issue}" for issue in sentinel_issues
        )
        if bool(strict_indexing):
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return

    if cache_key is not None and store is not None:
        store.mark_validated_static_inputs(cache_key)


def _validate_state_dict(
    state_dict: dict[str, Any],
    name: str = "state_dict",
    strict: bool = True,
    skip_static_sentinel_validation: bool = False,
) -> None:
    """
    Validate that a dict-based state/mesh has correct dimensions, shapes, and dtypes.

    When state/mesh are raw dictionaries from Fortran-style inputs, this function
    ensures they have the required structure before kernels attempt to use them.
    Without this validation, errors would occur deep in kernels with cryptic messages.

    Args:
        state_dict: Dictionary of state or mesh tensors
        name: Name for error messages (e.g., "state" or "mesh")
        strict: If True, raise ValueError on validation failure.
                If False, issue warnings only.

    Raises:
        ValueError: If strict=True and validation fails.

    Validation checks:
        1. Required dimensions exist (nCells, nEdges, nVertLevels)
        2. Key array shapes match expected dimensions
        3. All tensor values are torch.float64 (DTYPE)
    """
    issues: list[str] = []

    # =========================================================================
    # Check 1: Required dimensions must be present
    # =========================================================================
    required_dims = ["nCells", "nEdges", "nVertLevels"]
    for dim in required_dims:
        if dim not in state_dict:
            issues.append(f"Missing required dimension '{dim}'")

    # If dimensions are missing, we can't validate shapes - report early
    if issues:
        msg = f"{name} validation failed:\n" + "\n".join(
            f"  - {issue}" for issue in issues
        )
        if strict:
            raise ValueError(msg)
        else:
            warnings.warn(msg, RuntimeWarning)
            return

    # =========================================================================
    # Check 2: Key array shapes match expected dimensions
    # Arrays use 1-based+padded convention: shape is (N+1, ...) not (N, ...)
    # Extended to cover connectivity, geometry, and vertical coefficient arrays.
    # =========================================================================
    shape_checks = _get_shape_checks(state_dict)

    for array_name, (shape_desc, expected_shape) in shape_checks.items():
        if array_name not in state_dict:
            continue  # Only check arrays that are present

        arr = state_dict[array_name]
        if not isinstance(arr, torch.Tensor):
            issues.append(
                f"'{array_name}' is not a torch.Tensor, got {type(arr).__name__}"
            )
            continue

        if arr.shape != expected_shape:
            issues.append(
                f"'{array_name}' has shape {tuple(arr.shape)}, "
                f"expected {shape_desc} = {expected_shape}"
            )

    # =========================================================================
    # Check 3: All tensors must be torch.float64 (DTYPE)
    # Non-float64 tensors can cause silent precision loss
    # =========================================================================
    dtype_exceptions = {
        # Integer arrays that should NOT be float64
        "nCells",
        "nEdges",
        "nVertices",
        "nVertLevels",
        "maxEdges",
        "maxEdges2",
        "vertexDegree",
        "nCellsSolve",
        "nEdgesSolve",
        "nVerticesSolve",
        "nEdgesOnCell",
        "nEdgesOnEdge",
        "nAdvCellsForEdge",
        "cellsOnEdge",
        "edgesOnCell",
        "verticesOnCell",
        "cellsOnCell",
        "verticesOnEdge",
        "edgesOnVertex",
        "cellsOnVertex",
        "edgesOnEdge",
        "advCellsForEdge",
        "kiteForCell",
        "cellStart",
        "cellEnd",
        "edgeStart",
        "edgeEnd",
        "vertexStart",
        "vertexEnd",
        "cellSolveStart",
        "cellSolveEnd",
        "edgeSolveStart",
        "edgeSolveEnd",
        "vertexSolveStart",
        "vertexSolveEnd",
        "moist_start",
        "moist_end",
        "num_scalars",
        "nRelaxZone",
        "config_time_integration_order",
        "advance_density",
        "rk_step",
        "ns",
        "small_step",
        # Boundary mask arrays (integer flags for regional MPAS)
        "bdyMaskCell",
        "bdyMaskEdge",
        "bdyMaskVertex",
        # ID and categorical/mask arrays carried from IC/mesh inputs.
        "indexToCellID",
        "indexToEdgeID",
        "indexToVertexID",
        "landmask",
        "ivgtyp",
        "isltyp",
    }

    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue  # Skip non-tensor values

        if key in dtype_exceptions:
            continue  # Skip arrays that should be integers

        if value.dtype != DTYPE:
            issues.append(
                f"'{key}' has dtype {value.dtype}, expected {DTYPE} (float64)"
            )

    # =========================================================================
    # Check 4: Integer arrays must have integer dtype
    # Float connectivity arrays will be silently truncated during indexing,
    # causing incorrect connectivity lookups when .item() or int() is called.
    # =========================================================================
    for key in INTEGER_ARRAYS:
        if key in state_dict:
            val = state_dict[key]
            if isinstance(val, torch.Tensor):  # noqa: SIM102
                if val.dtype not in VALID_INTEGER_DTYPES:
                    issues.append(
                        f"'{key}' must be integer dtype (int32 or int64), got {val.dtype}. "
                        f"Float connectivity arrays will be silently truncated during indexing."
                    )

    # =========================================================================
    # Check 5: Validate sentinel placement in variable-size arrays
    # For arrays like edgesOnCell[nCells+1, maxEdges], sentinels (0 values)
    # should only appear in padding positions, not valid data slots.
    # A 0 in a valid slot causes -1 indexing when converted to 0-based.
    # =========================================================================
    if not skip_static_sentinel_validation:
        for array_name, count_name in _SENTINEL_PLACEMENT_CHECKS:
            if array_name in state_dict and count_name in state_dict:
                arr = state_dict[array_name]
                counts = state_dict[count_name]
                if isinstance(arr, torch.Tensor) and isinstance(counts, torch.Tensor):
                    sentinel_issues = _validate_sentinel_placement(
                        arr, array_name, counts, count_name
                    )
                    issues.extend(sentinel_issues)

    # =========================================================================
    # Check 6: Validate scalars array shape if present
    # Scalars must be 3D with correct nCells and nVertLevels dimensions.
    # =========================================================================
    nCells = _get_int_from_dict(state_dict, "nCells")
    nVertLevels = _get_int_from_dict(state_dict, "nVertLevels")
    scalar_issues = _validate_scalars_shape(
        state_dict,
        nCells,
        nVertLevels,
        require_nonzero_tracers=False,  # Don't require tracers at dict validation level
    )
    issues.extend(scalar_issues)

    # =========================================================================
    # Report issues
    # =========================================================================
    if issues:
        msg = f"{name} validation failed:\n" + "\n".join(
            f"  - {issue}" for issue in issues
        )
        if strict:
            raise ValueError(msg)
        else:
            warnings.warn(msg, RuntimeWarning)


def _validate_mesh_dtypes(mesh_dict: dict[str, Any], strict: bool = True) -> None:
    """Validate that all floating-point tensors in mesh_dict are torch.float64.

    This function fills the dtype validation gap for mesh dictionaries.
    _validate_state_dict() performs dtype checks on state tensors but is not
    called on mesh_dict because mesh dicts may lack required dimension scalars
    (nCells, nEdges, nVertLevels). This function checks ONLY dtypes, without
    requiring dimension scalars.

    A float32 tensor in the mesh would cause silent precision loss when used
    in kernel computations that expect float64 throughout.

    Args:
        mesh_dict: Dictionary of mesh tensors
        strict: If True, raise ValueError on validation failure.
                If False, issue warnings only.

    Raises:
        ValueError: If strict=True and any floating-point tensor is not float64.
    """
    # Integer arrays that should NOT be float64 (same set as _validate_state_dict)
    dtype_exceptions = {
        "nCells",
        "nEdges",
        "nVertices",
        "nVertLevels",
        "maxEdges",
        "maxEdges2",
        "vertexDegree",
        "nCellsSolve",
        "nEdgesSolve",
        "nVerticesSolve",
        "nEdgesOnCell",
        "nEdgesOnEdge",
        "nAdvCellsForEdge",
        "cellsOnEdge",
        "edgesOnCell",
        "verticesOnCell",
        "cellsOnCell",
        "verticesOnEdge",
        "edgesOnVertex",
        "cellsOnVertex",
        "edgesOnEdge",
        "advCellsForEdge",
        "kiteForCell",
        "cellStart",
        "cellEnd",
        "edgeStart",
        "edgeEnd",
        "vertexStart",
        "vertexEnd",
        "cellSolveStart",
        "cellSolveEnd",
        "edgeSolveStart",
        "edgeSolveEnd",
        "vertexSolveStart",
        "vertexSolveEnd",
        "moist_start",
        "moist_end",
        "num_scalars",
        "nRelaxZone",
        "config_time_integration_order",
        "advance_density",
        "rk_step",
        "ns",
        "small_step",
        "bdyMaskCell",
        "bdyMaskEdge",
        "bdyMaskVertex",
        "indexToCellID",
        "indexToEdgeID",
        "indexToVertexID",
        "landmask",
        "ivgtyp",
        "isltyp",
    }

    issues: list[str] = []

    for key, value in mesh_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        if key in dtype_exceptions:
            continue
        if value.dtype.is_floating_point and value.dtype != DTYPE:
            issues.append(
                f"mesh '{key}' has dtype {value.dtype}, expected {DTYPE} (float64)"
            )

    if issues:
        msg = "Mesh dtype validation failed:\n" + "\n".join(
            f"  - {issue}" for issue in issues
        )
        if strict:
            raise ValueError(msg)
        else:
            warnings.warn(msg, RuntimeWarning)


def _coerce_int(value: Any, name: str, allow_negative: bool = False) -> int:
    """Coerce config values to int with validation.

    Args:
        value: Value to coerce to integer
        name: Parameter name for error messages
        allow_negative: If False (default), reject negative values

    Returns:
        Integer value

    Raises:
        ValueError: If value is non-integral float, negative when not allowed,
                   or cannot be converted to int
    """
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be a scalar, got shape {tuple(value.shape)}")
        float_val = float(value.item())
    elif isinstance(value, bool):
        raise ValueError(f"{name} must be an int, got bool")  # noqa: TRY004
    elif isinstance(value, (int, float)):
        float_val = float(value)
    else:
        raise ValueError(f"{name} must be numeric, got {type(value).__name__}")  # noqa: TRY004

    # Check for non-integral float
    int_val = int(float_val)
    if float_val != int_val:
        raise ValueError(
            f"{name} must be an integer, got {float_val} "
            f"(would be truncated to {int_val})"
        )

    # Check for negative values (dimensions should be non-negative)
    if not allow_negative and int_val < 0:
        raise ValueError(f"{name} must be non-negative, got {int_val}")

    return int_val


def _coerce_float(value: Any, name: str) -> float:
    """Coerce config values to float with clear errors."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be a scalar, got shape {tuple(value.shape)}")
        return float(value.item())
    return float(value)


def _coerce_bool(value: Any, name: str) -> bool:
    """Coerce config values to bool with explicit handling of numeric/string forms."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be a scalar, got shape {tuple(value.shape)}")
        value = value.item()

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        int_val = int(value)
        if float(value) != float(int_val):
            raise ValueError(f"{name} must be boolean/integer-like, got {value}")
        return int_val != 0

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", ".true.", "t", "1", "yes", "y"}:
            return True
        if normalized in {"false", ".false.", "f", "0", "no", "n"}:
            return False
        raise ValueError(f"{name} must be boolean-like string, got {value!r}")

    raise ValueError(f"{name} must be bool-like, got {type(value).__name__}")


_STATIC_SCALAR_MISSING = object()


def _static_scalar_cache_key(
    key: str, value: torch.Tensor
) -> tuple[object, ...] | None:
    """Return a version-sensitive cache key for immutable scalar metadata."""
    if value.numel() != 1 or _tensor_carries_autodiff(value):
        return None
    tensor_key = _tensor_version_cache_key(value)
    if tensor_key is None:
        return None
    return ("static_scalar", key, *tensor_key)


def _cached_static_scalar(
    key: str,
    value: torch.Tensor,
    store: DycoreStaticPayloads | None,
) -> Any:
    cache_key = _static_scalar_cache_key(key, value)
    if cache_key is None or store is None:
        return _STATIC_SCALAR_MISSING
    cached = store.cached_static_scalar(cache_key, _STATIC_SCALAR_MISSING)
    if cached is _STATIC_SCALAR_MISSING:
        return _STATIC_SCALAR_MISSING
    return cached


def _extract_python_scalar_from_0d_tensor(value: torch.Tensor) -> Any:
    if value.dtype == torch.bool:
        return bool(value.item())
    if value.dtype.is_floating_point:
        return float(value.item())
    if value.dtype.is_complex:
        return complex(value.item())
    return int(value.item())


def _coerce_0d_tensor_scalars_inplace(
    mapping: dict[str, Any],
    scalar_store: DycoreStaticPayloads | None = None,
) -> None:
    """Convert 0-d tensor values in a dict to Python scalars.

    Autograd-tracked scalars are preserved so callers do not silently sever
    gradient connectivity by crossing this boundary helper.
    """
    for key, value in list(mapping.items()):
        if not isinstance(value, torch.Tensor) or value.ndim != 0:
            continue
        if _tensor_carries_autodiff(value):
            continue
        cache_key = _static_scalar_cache_key(key, value)
        if cache_key is not None and scalar_store is not None:
            cached = scalar_store.cached_static_scalar(
                cache_key, _STATIC_SCALAR_MISSING
            )
            if cached is not _STATIC_SCALAR_MISSING:
                mapping[key] = cached
                continue
        scalar = _extract_python_scalar_from_0d_tensor(value)
        mapping[key] = scalar
        if cache_key is not None and scalar_store is not None:
            scalar_store.mark_static_scalar(cache_key, scalar)


@dataclass(frozen=True)
class TimestepScalars:
    """Pre-extracted scalar values reused across RK kernel orchestration."""

    # Dimensions
    nCells: int
    nEdges: int
    nVertices: int
    nVertLevels: int
    maxEdges: int
    maxEdges2: int
    vertexDegree: int

    # Config scalars frequently queried in execute_rk_step
    dt: float
    epssm: float
    smdiv: float
    smdt: float
    config_apvm_upwinding: float
    config_number_of_sub_steps: int
    config_time_integration_order: int

    # Generic scalar lookup maps (immutable)
    int_values: Mapping[str, int]
    float_values: Mapping[str, float]

    def get_int(self, source: dict[str, Any], key: str, default: Any = None) -> int:
        if key in self.int_values:
            return self.int_values[key]
        if key in source:
            return _coerce_int(source[key], key, allow_negative=True)
        if default is not None:
            return _coerce_int(default, key, allow_negative=True)
        raise KeyError(key)

    def get_float(self, source: dict[str, Any], key: str, default: Any = None) -> float:
        if key in self.float_values:
            return self.float_values[key]
        if key in source:
            return _coerce_float(source[key], key)
        if default is not None:
            return _coerce_float(default, key)
        raise KeyError(key)


def _extract_scalar_value(
    value: Any,
    *,
    key: str | None = None,
    scalar_store: DycoreStaticPayloads | None = None,
) -> Any | None:
    """Extract Python scalar from numeric values and 0-d tensors."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        if key is not None:
            cached = _cached_static_scalar(key, value, scalar_store)
            if cached is not _STATIC_SCALAR_MISSING:
                return cached
        if _tensor_carries_autodiff(value):
            return None
        scalar = _extract_python_scalar_from_0d_tensor(value)
        if key is not None and scalar_store is not None:
            cache_key = _static_scalar_cache_key(key, value)
            if cache_key is not None:
                scalar_store.mark_static_scalar(cache_key, scalar)
        return scalar
    if isinstance(value, (bool, int, float, np.integer, np.floating)):
        return value
    return None


def _build_timestep_scalars(
    state_dict: dict[str, Any],
    config: dict[str, Any],
    scalar_store: DycoreStaticPayloads | None = None,
) -> TimestepScalars:
    """Extract scalar dimensions/config values once for RK orchestration."""
    int_values: dict[str, int] = {}
    float_values: dict[str, float] = {}

    for src in (state_dict, config):
        for key, value in src.items():
            scalar_value = _extract_scalar_value(
                value,
                key=key,
                scalar_store=scalar_store,
            )
            if scalar_value is None:
                continue

            if isinstance(scalar_value, (bool, np.bool_)):
                int_values[key] = int(bool(scalar_value))
                float_values[key] = float(bool(scalar_value))
                continue

            if isinstance(scalar_value, (int, np.integer)):
                int_values[key] = int(scalar_value)
                float_values[key] = float(scalar_value)
                continue

            try:
                fval = float(scalar_value)
            except (TypeError, ValueError):
                continue
            float_values[key] = fval
            if float(fval).is_integer():
                int_values[key] = int(fval)

    nCells = _coerce_int(
        int_values.get("nCells", state_dict.get("nCells", 0)), "nCells"
    )
    if nCells > 0:
        try:
            nEdges = _resolve_nEdges(state_dict, nCells)
        except Exception:  # noqa: BLE001
            nEdges = _coerce_int(
                int_values.get("nEdges", state_dict.get("nEdges", 0)),
                "nEdges",
            )
    else:
        nEdges = _coerce_int(
            int_values.get("nEdges", state_dict.get("nEdges", 0)),
            "nEdges",
        )
    nVertices = _coerce_int(
        int_values.get("nVertices", state_dict.get("nVertices", 0)),
        "nVertices",
    )
    nVertLevels = _coerce_int(
        int_values.get("nVertLevels", state_dict.get("nVertLevels", 0)),
        "nVertLevels",
    )

    if "maxEdges" in int_values:
        maxEdges = int_values["maxEdges"]
    elif isinstance(state_dict.get("edgesOnCell"), torch.Tensor):
        maxEdges = int(state_dict["edgesOnCell"].shape[1])
    else:
        maxEdges = 10

    if "maxEdges2" in int_values:
        maxEdges2 = int_values["maxEdges2"]
    elif isinstance(state_dict.get("edgesOnEdge"), torch.Tensor):
        maxEdges2 = int(state_dict["edgesOnEdge"].shape[1])
    else:
        maxEdges2 = 2 * maxEdges

    if "vertexDegree" in int_values:
        vertexDegree = int_values["vertexDegree"]
    elif isinstance(state_dict.get("edgesOnVertex"), torch.Tensor):
        vertexDegree = int(state_dict["edgesOnVertex"].shape[1])
    else:
        vertexDegree = 3

    # Keep canonical dimensions available in maps.
    int_values["nCells"] = nCells
    int_values["nEdges"] = nEdges
    int_values["nVertices"] = nVertices
    int_values["nVertLevels"] = nVertLevels
    int_values["maxEdges"] = maxEdges
    int_values["maxEdges2"] = maxEdges2
    int_values["vertexDegree"] = vertexDegree

    return TimestepScalars(
        nCells=nCells,
        nEdges=nEdges,
        nVertices=nVertices,
        nVertLevels=nVertLevels,
        maxEdges=maxEdges,
        maxEdges2=maxEdges2,
        vertexDegree=vertexDegree,
        dt=_coerce_float(
            get_config_value_safe(
                config, "dt", get_config_value_safe(config, "config_dt", 0.0)
            ),
            "dt",
        ),
        epssm=_coerce_float(get_config_value_safe(config, "epssm", 0.1), "epssm"),
        smdiv=_coerce_float(get_config_value_safe(config, "smdiv", 0.1), "smdiv"),
        smdt=_coerce_float(get_config_value_safe(config, "smdt", 0.0), "smdt"),
        config_apvm_upwinding=_coerce_float(
            get_config_value_safe(config, "config_apvm_upwinding", 0.5),
            "config_apvm_upwinding",
        ),
        config_number_of_sub_steps=_coerce_int(
            get_config_value_safe(config, "config_number_of_sub_steps", 2),
            "config_number_of_sub_steps",
        ),
        config_time_integration_order=_coerce_int(
            get_config_value_safe(config, "config_time_integration_order", 2),
            "config_time_integration_order",
        ),
        int_values=MappingProxyType(dict(int_values)),
        float_values=MappingProxyType(dict(float_values)),
    )


def _resolve_dynamics_split(config: dict[str, Any]) -> int:
    """Resolve Fortran split-dynamics semantics from config.

    Fortran controls split transport with two settings:
    - config_split_dynamics_transport (logical)
    - config_dynamics_split_steps (integer)

    Legacy PyTorch configs may provide split_dynamics_transport as an integer.
    This function supports both forms, with Fortran-style keys taking precedence.
    """
    # Prefer Fortran-style two-parameter semantics when present.
    if (
        "config_split_dynamics_transport" in config
        or "config_dynamics_split_steps" in config
    ):
        split_enabled = _coerce_bool(
            config.get(
                "config_split_dynamics_transport",
                FORTRAN_DEFAULTS.get("config_split_dynamics_transport", True),
            ),
            "config_split_dynamics_transport",
        )
        split_steps = _coerce_int(
            config.get(
                "config_dynamics_split_steps",
                FORTRAN_DEFAULTS.get("config_dynamics_split_steps", 3),
            ),
            "config_dynamics_split_steps",
        )
        if split_steps < 0:
            raise ValueError(
                f"config_dynamics_split_steps must be >= 0, got {split_steps}"
            )
        return split_steps if split_enabled else 1

    # Backward-compatible legacy key path.
    if "split_dynamics_transport" in config:
        legacy = config["split_dynamics_transport"]
        if isinstance(legacy, bool):
            split_steps = _coerce_int(
                config.get(
                    "config_dynamics_split_steps",
                    FORTRAN_DEFAULTS.get("config_dynamics_split_steps", 3),
                ),
                "config_dynamics_split_steps",
            )
            if split_steps < 0:
                raise ValueError(
                    f"config_dynamics_split_steps must be >= 0, got {split_steps}"
                )
            return split_steps if legacy else 1

        split_steps = _coerce_int(legacy, "split_dynamics_transport")
        if split_steps < 1:
            raise ValueError(
                f"split_dynamics_transport must be >= 1, got {split_steps}"
            )
        return split_steps

    # Fortran defaults: split enabled with 3 dynamics substeps.
    split_enabled = _coerce_bool(
        FORTRAN_DEFAULTS.get("config_split_dynamics_transport", True),
        "config_split_dynamics_transport",
    )
    split_steps = _coerce_int(
        FORTRAN_DEFAULTS.get("config_dynamics_split_steps", 3),
        "config_dynamics_split_steps",
    )
    if split_steps < 0:
        raise ValueError(f"config_dynamics_split_steps must be >= 0, got {split_steps}")
    return split_steps if split_enabled else 1


def _resolve_len_disp(
    config: dict[str, Any],
    state_dict: dict[str, Any],
    strict: bool = True,
) -> float:
    """Resolve config_len_disp with Fortran fallback semantics.

    Upstream MPAS behavior:
    - If config_len_disp > 0: use config_len_disp
    - Else: fallback to nominalMinDc from mesh data
    """
    config_len_disp = _coerce_float(
        get_config_value_safe(config, "config_len_disp", 0.0),
        "config_len_disp",
    )
    try:
        return resolve_len_disp(
            config_len_disp=config_len_disp,
            nominal_min_dc=_get_float_from_dict(state_dict, "nominalMinDc")
            if "nominalMinDc" in state_dict
            else None,
        )
    except ValueError:
        pass

    # Secondary fallback: derive a representative minimum edge spacing
    # from dcEdge when nominalMinDc is not available in checkpoint data.
    if "dcEdge" in state_dict and isinstance(state_dict["dcEdge"], torch.Tensor):
        dc_edge = state_dict["dcEdge"]
        if dc_edge.numel() > 0:
            # dcEdge in kernel convention is typically padded with an unused slot.
            n_edges = None
            if "nEdges" in state_dict:
                try:
                    n_edges = _get_int_from_dict(state_dict, "nEdges")
                except Exception:  # noqa: BLE001
                    n_edges = None
            if n_edges is not None and dc_edge.shape[0] >= n_edges + 1:
                dc_slice = dc_edge[1 : n_edges + 1]
            else:
                dc_slice = dc_edge
            positive = dc_slice[dc_slice > 0.0]
            if positive.numel() > 0:
                fallback = float(positive.min().item())
                warnings.warn(
                    f"config_len_disp<=0; using dcEdge-derived fallback {fallback:.6f} "
                    "because nominalMinDc is unavailable.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return fallback

    message = (
        "config_len_disp <= 0 but no nominalMinDc/dcEdge fallback is available. "
        "Fortran uses nominalMinDc in this case; provide config_len_disp>0 or "
        "include nominalMinDc in checkpoint/mesh inputs."
    )
    if strict:
        raise ValueError(message)
    warnings.warn(message + " Proceeding with 0.0.", RuntimeWarning, stacklevel=2)
    return 0.0


def _reject_unsupported_advection_orders(config: dict[str, Any]) -> None:
    """Validate registry-only advection-order knobs without changing numerics."""
    for key in ATMOSPHERE_INERT_ADVECTION_ORDER_KEYS:
        val = get_config_value_safe(config, key, 3)
        try:
            _coerce_int(val, key)
        except Exception:  # noqa: BLE001
            raise ValueError(f"{key} must be integer-like, got {val!r}")


def _get_int_from_dict(d: dict[str, Any], key: str, default: Any = None) -> int:
    """Extract integer from dict, handling both tensor and Python int/float.

    This helper addresses the validation vulnerability where _validate_state_dict()
    accepts Python ints for dimensions, but unconditional .item() calls crash with
    AttributeError: 'int' object has no attribute 'item'.

    Args:
        d: Dictionary containing the value
        key: Key to extract
        default: Default value if key not found. If None, raises KeyError.

    Returns:
        Integer value

    Raises:
        KeyError: If key not found and no default provided
        TypeError: If value cannot be converted to int
        ValueError: If value is a non-integral float (would be truncated)
    """
    if default is not None and key not in d:
        val = default
    else:
        val = d[key]

    # Extract float value for integrality check
    if isinstance(val, torch.Tensor):
        if val.numel() != 1:
            raise TypeError(
                f"Cannot extract integer from '{key}': expected scalar tensor, got shape {tuple(val.shape)}"
            )
        float_val = float(val.item())
    elif isinstance(val, bool):
        # Reject bools explicitly (they are subclass of int in Python)
        raise TypeError(
            f"Cannot extract integer from '{key}': expected Tensor or number, got bool"
        )
    elif isinstance(val, (int, np.integer)):
        # Python int or numpy integer - always integral
        return int(val)
    elif isinstance(val, (float, np.floating)):
        float_val = float(val)
    else:
        raise TypeError(
            f"Cannot extract integer from '{key}': expected Tensor or number, got {type(val).__name__}"
        )

    # Check for non-integral float
    int_val = int(float_val)
    if float_val != int_val:
        raise ValueError(
            f"Cannot extract integer from '{key}': value {float_val} is not integral "
            f"(would be truncated to {int_val})"
        )

    return int_val


def _get_float_from_dict(d: dict[str, Any], key: str, default: Any = None) -> float:
    """Extract float from dict, handling both tensor and Python int/float.

    This helper addresses the validation vulnerability where _validate_state_dict()
    accepts Python floats for values, but unconditional .item() calls crash with
    AttributeError: 'float' object has no attribute 'item'.

    Args:
        d: Dictionary containing the value
        key: Key to extract
        default: Default value if key not found. If None, raises KeyError.

    Returns:
        Float value

    Raises:
        KeyError: If key not found and no default provided
        TypeError: If value cannot be converted to float
    """
    if default is not None and key not in d:
        val = default
    else:
        val = d[key]

    if isinstance(val, torch.Tensor):
        return float(val.item())
    elif isinstance(val, (int, float)):
        return float(val)
    else:
        raise TypeError(
            f"Cannot extract float from '{key}': expected Tensor or number, got {type(val).__name__}"
        )


def _ensure_scalars_rank3(
    arr: torch.Tensor, num_scalars: int, name: str
) -> torch.Tensor:
    """Ensure scalar arrays use (nCells+1, nVertLevels, num_scalars) layout."""
    if arr.ndim == 3:
        if arr.shape[2] != num_scalars:
            raise ValueError(
                f"{name} has shape {tuple(arr.shape)} but num_scalars={num_scalars}"
            )
        return arr
    if arr.ndim == 2 and num_scalars == 1:
        return arr.unsqueeze(-1)
    raise ValueError(
        f"{name} must be rank-3 (or rank-2 when num_scalars==1), got shape {tuple(arr.shape)}"
    )


def _alias_scalar_outputs(output_dict: dict[str, Any]) -> None:
    """Preserve 2D single-scalar aliases alongside 3D scalar arrays."""
    scalar_new = output_dict.get("scalar_new")
    if isinstance(scalar_new, torch.Tensor) and scalar_new.ndim == 3:
        output_dict["scalars_new"] = _clone_for_grad(scalar_new)
        if scalar_new.shape[2] == 1:
            output_dict["scalar_new"] = _clone_for_grad(scalar_new[:, :, 0])

    scalar_old = output_dict.get("scalar_old")
    if isinstance(scalar_old, torch.Tensor) and scalar_old.ndim == 3:
        output_dict["scalars_old"] = _clone_for_grad(scalar_old)
        if scalar_old.shape[2] == 1:
            output_dict["scalar_old"] = _clone_for_grad(scalar_old[:, :, 0])

    scalar_tend_save = output_dict.get("scalar_tend_save")
    if isinstance(scalar_tend_save, torch.Tensor) and scalar_tend_save.ndim == 3:  # noqa: SIM102
        if scalar_tend_save.shape[2] == 1:
            output_dict["scalar_tend_save"] = _clone_for_grad(scalar_tend_save[:, :, 0])


def _check_edge_data_requires_nEdges(state_dict: dict[str, Any]) -> list[str]:
    """Return list of edge-requiring keys present in state_dict.

    If any of these keys are present, nEdges must be explicitly provided
    and cannot be approximated from nCells * 3. The approximation would
    cause shape mismatches for actual edge-sized tensors.

    Comprehensive edge data detection
    replacing the incomplete 4-key acoustic check.

    Args:
        state_dict: Dictionary of state/mesh tensors

    Returns:
        List of edge-requiring keys found in state_dict (empty if none found)
    """
    return [k for k in EDGE_REQUIRING_KEYS if k in state_dict]


def _infer_nEdges_from_edge_arrays(
    state_dict: dict[str, Any],
    nCells: int,
) -> int | None:
    """Infer nEdges from truly edge-sized arrays if present.

    Some arrays in EDGE_REQUIRING_KEYS are actually cell-sized (e.g., rw_p,
    rho_pp, rtheta_pp) while others are edge-sized (e.g., ru_p, ruAvg, ru).
    This function attempts to infer nEdges from the first truly edge-sized
    array's shape.

    Edge arrays use padded indexing: shape = (nEdges+1, ...).
    Cell arrays use padded indexing: shape = (nCells+1, ...).

    Args:
        state_dict: Dictionary of state/mesh tensors
        nCells: Number of cells (used to distinguish cell vs edge arrays)

    Returns:
        Inferred nEdges if any edge-sized array is found, None otherwise
    """
    # Check only truly edge-sized keys (not cell-sized acoustic arrays)
    for key in EDGE_SIZED_KEYS:
        if key in state_dict:
            val = state_dict[key]
            if isinstance(val, torch.Tensor) and val.ndim >= 1:
                first_dim = val.shape[0] - 1  # Remove padding
                # first_dim > nCells confirms this is edge-sized
                # (nEdges ≈ 3 * nCells for hexagonal meshes)
                if first_dim > nCells:
                    return first_dim

    return None


def _resolve_nEdges(state_dict: dict[str, Any], nCells: int) -> int:
    """Resolve nEdges from state_dict, inferring from array shapes if not explicit.

    Priority:
        1. Explicit 'nEdges' key in state_dict
        2. Inferred from truly edge-sized arrays (EDGE_SIZED_KEYS)
        3. Hexagonal mesh approximation (nEdges ≈ 3 * nCells)

    Args:
        state_dict: Dictionary of state/mesh tensors
        nCells: Number of cells (for inference and fallback)

    Returns:
        nEdges as a Python int
    """
    if "nEdges" in state_dict:
        return _get_int_from_dict(state_dict, "nEdges")

    edge_keys_present = _check_edge_data_requires_nEdges(state_dict)
    if edge_keys_present:
        inferred = _infer_nEdges_from_edge_arrays(state_dict, nCells)
        if inferred is not None:
            return inferred

    return nCells * _HEXAGONAL_EDGE_FACTOR


def _validate_mesh_state_compatibility(
    state_dict: dict[str, Any], mesh_dict: dict[str, Any], strict: bool = True
) -> None:
    """Validate that state and mesh dimensions match.

    Compares critical dimension values between state and mesh to catch
    mismatched checkpoint pairs before they cause subtle errors.

    Args:
        state_dict: State dictionary (from state checkpoint)
        mesh_dict: Mesh dictionary (from mesh/init checkpoint)
        strict: If True, raise ValueError; if False, emit warning

    Raises:
        ValueError: If dimensions don't match and strict=True
    """
    dim_keys = ["nCells", "nEdges", "nVertices", "nVertLevels"]

    issues: list[str] = []
    for key in dim_keys:
        if key in state_dict and key in mesh_dict:
            state_val = _get_int_from_dict(state_dict, key)
            mesh_val = _get_int_from_dict(mesh_dict, key)
            if state_val != mesh_val:
                issues.append(f"{key}: state={state_val} != mesh={mesh_val}")

    if issues:
        msg = (
            "State/mesh dimension mismatch detected:\n"
            + "\n".join(f"  - {issue}" for issue in issues)
            + "\n\nState and mesh must come from the same simulation/mesh resolution."
        )
        if strict:
            raise ValueError(msg)
        else:
            import warnings

            warnings.warn(msg, RuntimeWarning)


def _get_shape_checks(state_dict: dict[str, Any]) -> dict[str, tuple]:
    """Build comprehensive shape checks for all known arrays.

    Returns dict mapping array_name -> (description, expected_shape).
    Only includes checks for arrays that are present in state_dict.
    Arrays with variable shapes (deriv_two, defc_a, defc_b) have None as
    expected_shape to skip validation.

    This function extends shape validation beyond the 4 core state arrays
    (u, w, rho_zz, theta_m) to cover connectivity, geometry, vertical
    coefficient arrays, dynamics-tendency state arrays, diagnostic arrays, pressure arrays,
    initialization arrays, tendency arrays, mesh geometry, and reconstruction
    arrays. This catches malformed dicts early rather than failing deep in
    kernels with cryptic errors.

    Extend shape validation to cover connectivity and geometry arrays.
    Extend to cover 50+ arrays for explicit-scalar-transport/monotonic-scalar-transport/dynamics-tendency kernels.
    """
    # Use the helper functions for safe dimension extraction
    nCells = _get_int_from_dict(state_dict, "nCells")
    nEdges = _get_int_from_dict(state_dict, "nEdges")
    nVertLevels = _get_int_from_dict(state_dict, "nVertLevels")

    # Optional dimensions with defaults
    nVertices = _get_int_from_dict(state_dict, "nVertices", default=0)

    # Get maxEdges from edgesOnCell shape if available
    maxEdges = None
    if "edgesOnCell" in state_dict and isinstance(
        state_dict["edgesOnCell"], torch.Tensor
    ):
        maxEdges = state_dict["edgesOnCell"].shape[1]

    # Get maxEdges2 from edgesOnEdge shape if available
    maxEdges2 = None
    if "edgesOnEdge" in state_dict and isinstance(
        state_dict["edgesOnEdge"], torch.Tensor
    ):
        maxEdges2 = state_dict["edgesOnEdge"].shape[1]

    # Get vertexDegree from edgesOnVertex shape if available
    vertexDegree = None
    if "edgesOnVertex" in state_dict and isinstance(
        state_dict["edgesOnVertex"], torch.Tensor
    ):
        vertexDegree = state_dict["edgesOnVertex"].shape[1]

    # Get maxAdvCells from advCellsForEdge shape if available
    maxAdvCells = None
    if "advCellsForEdge" in state_dict and isinstance(
        state_dict["advCellsForEdge"], torch.Tensor
    ):
        maxAdvCells = state_dict["advCellsForEdge"].shape[1]

    checks = {
        # === Primary state arrays (1-based+padded convention) ===
        "u": ("(nEdges+1, nVertLevels)", (nEdges + 1, nVertLevels)),
        "w": ("(nCells+1, nVertLevels+1)", (nCells + 1, nVertLevels + 1)),
        "rho_zz": ("(nCells+1, nVertLevels)", (nCells + 1, nVertLevels)),
        "theta_m": ("(nCells+1, nVertLevels)", (nCells + 1, nVertLevels)),
        # === dynamics-tendency additional state arrays ===
        "v": ("(nEdges+1, nVertLevels)", (nEdges + 1, nVertLevels)),
        "ru": ("(nEdges+1, nVertLevels)", (nEdges + 1, nVertLevels)),
        "rw": ("(nCells+1, nVertLevels+1)", (nCells + 1, nVertLevels + 1)),
        "ru_save": ("(nEdges+1, nVertLevels)", (nEdges + 1, nVertLevels)),
        "rw_save": ("(nCells+1, nVertLevels+1)", (nCells + 1, nVertLevels + 1)),
        "theta_m_save": ("(nCells+1, nVertLevels)", (nCells + 1, nVertLevels)),
        "rho_edge": ("(nEdges+1, nVertLevels)", (nEdges + 1, nVertLevels)),
        # === Diagnostic arrays ===
        "divergence": ("(nCells+1, nVertLevels)", (nCells + 1, nVertLevels)),
        "ke": ("(nCells+1, nVertLevels)", (nCells + 1, nVertLevels)),
        # === Pressure arrays ===
        "pp": ("(nCells+1, nVertLevels)", (nCells + 1, nVertLevels)),
        "rr": ("(nCells+1, nVertLevels)", (nCells + 1, nVertLevels)),
        "rr_save": ("(nCells+1, nVertLevels)", (nCells + 1, nVertLevels)),
        "rb": ("(nCells+1, nVertLevels)", (nCells + 1, nVertLevels)),
        "pressure_b": ("(nCells+1, nVertLevels)", (nCells + 1, nVertLevels)),
        "exner": ("(nCells+1, nVertLevels)", (nCells + 1, nVertLevels)),
        "pressure_p": ("(nCells+1, nVertLevels)", (nCells + 1, nVertLevels)),
        # p/pb are aliases for exner/exner_base, which are cell-centered on
        # nVertLevels, not w-level arrays.
        "p": ("(nCells+1, nVertLevels)", (nCells + 1, nVertLevels)),
        "t": ("(nCells+1, nVertLevels)", (nCells + 1, nVertLevels)),
        "rt": ("(nCells+1, nVertLevels)", (nCells + 1, nVertLevels)),
        "rtb": ("(nCells+1, nVertLevels)", (nCells + 1, nVertLevels)),
        "pb": ("(nCells+1, nVertLevels)", (nCells + 1, nVertLevels)),
        "qtot": ("(nCells+1, nVertLevels)", (nCells + 1, nVertLevels)),
        "cqw": ("(nCells+1, nVertLevels+1)", (nCells + 1, nVertLevels + 1)),
        "cqu": ("(nEdges+1, nVertLevels)", (nEdges + 1, nVertLevels)),
        "rho_base": ("(nCells+1, nVertLevels)", (nCells + 1, nVertLevels)),
        "rtheta_base": ("(nCells+1, nVertLevels)", (nCells + 1, nVertLevels)),
        "exner_base": ("(nCells+1, nVertLevels)", (nCells + 1, nVertLevels)),
        "rho_p_save": ("(nCells+1, nVertLevels)", (nCells + 1, nVertLevels)),
        "rtheta_p_save": ("(nCells+1, nVertLevels)", (nCells + 1, nVertLevels)),
        "rt_diabatic_tend": ("(nCells+1, nVertLevels)", (nCells + 1, nVertLevels)),
        # === Initialization arrays ===
        # u_init, v_init, qv_init are 1D vertical profiles (nVertLevels,)
        # used by dynamics-tendency for Rayleigh damping reference state.
        # t_init is a 2D array (nCells+1, nVertLevels) for initial temperature.
        "u_init": ("(nVertLevels,)", (nVertLevels,)),
        "v_init": ("(nVertLevels,)", (nVertLevels,)),
        "qv_init": ("(nVertLevels,)", (nVertLevels,)),
        "t_init": ("(nCells+1, nVertLevels)", (nCells + 1, nVertLevels)),
        # === Tendency arrays ===
        "tend_u": ("(nEdges+1, nVertLevels)", (nEdges + 1, nVertLevels)),
        "tend_w": ("(nCells+1, nVertLevels+1)", (nCells + 1, nVertLevels + 1)),
        "tend_theta": ("(nCells+1, nVertLevels)", (nCells + 1, nVertLevels)),
        "tend_w_pgf": ("(nCells+1, nVertLevels+1)", (nCells + 1, nVertLevels + 1)),
        "tend_w_buoy": ("(nCells+1, nVertLevels+1)", (nCells + 1, nVertLevels + 1)),
        # === Mesh geometry 1D arrays ===
        "dvEdge": ("(nEdges+1,)", (nEdges + 1,)),
        "dcEdge": ("(nEdges+1,)", (nEdges + 1,)),
        "invAreaCell": ("(nCells+1,)", (nCells + 1,)),
        "areaCell": ("(nCells+1,)", (nCells + 1,)),
        "invDcEdge": ("(nEdges+1,)", (nEdges + 1,)),
        "invDvEdge": ("(nEdges+1,)", (nEdges + 1,)),
        "meshScalingDel2": ("(nEdges+1,)", (nEdges + 1,)),
        "meshScalingDel4": ("(nEdges+1,)", (nEdges + 1,)),
        "latCell": ("(nCells+1,)", (nCells + 1,)),
        "latEdge": ("(nEdges+1,)", (nEdges + 1,)),
        "angleEdge": ("(nEdges+1,)", (nEdges + 1,)),
        "fEdge": ("(nEdges+1,)", (nEdges + 1,)),
        # === Reconstruction arrays ===
        "ur_cell": ("(nCells+1, nVertLevels)", (nCells + 1, nVertLevels)),
        "vr_cell": ("(nCells+1, nVertLevels)", (nCells + 1, nVertLevels)),
        # === Connectivity arrays (1-based+padded) ===
        "cellsOnEdge": ("(nEdges+1, 2)", (nEdges + 1, 2)),
        "verticesOnEdge": ("(nEdges+1, 2)", (nEdges + 1, 2)),
        "nEdgesOnCell": ("(nCells+1,)", (nCells + 1,)),
        "nEdgesOnEdge": ("(nEdges+1,)", (nEdges + 1,)),
        "nAdvCellsForEdge": ("(nEdges+1,)", (nEdges + 1,)),
        "bdyMaskCell": ("(nCells+1,)", (nCells + 1,)),
        "bdyMaskEdge": ("(nEdges+1,)", (nEdges + 1,)),
        # === Vertical coefficient arrays ===
        "fzm": ("(nVertLevels,)", (nVertLevels,)),
        "fzp": ("(nVertLevels,)", (nVertLevels,)),
        "rdzw": ("(nVertLevels,)", (nVertLevels,)),
        "rdzu": ("(nVertLevels,)", (nVertLevels,)),
        "dzu": ("(nVertLevels,)", (nVertLevels,)),
        "dzw": ("(nVertLevels,)", (nVertLevels,)),
        "fnm": ("(nVertLevels,)", (nVertLevels,)),
        "fnp": ("(nVertLevels,)", (nVertLevels,)),
        "rdnw": ("(nVertLevels,)", (nVertLevels,)),
        "dss": ("(nCells+1, nVertLevels)", (nCells + 1, nVertLevels)),
        # === 2D coefficient arrays ===
        "zz": ("(nCells+1, nVertLevels)", (nCells + 1, nVertLevels)),
        "zgrid": ("(nCells+1, nVertLevels+1)", (nCells + 1, nVertLevels + 1)),
        "zxu": ("(nEdges+1, nVertLevels)", (nEdges + 1, nVertLevels)),
        # === 3D reconstruction arrays for horizontal pressure gradient ===
        # zb, zb3: Edge-based, 2 cells per edge, nVertLevels+1 interfaces
        "zb": ("(nEdges+1, 2, nVertLevels+1)", (nEdges + 1, 2, nVertLevels + 1)),
        "zb3": ("(nEdges+1, 2, nVertLevels+1)", (nEdges + 1, 2, nVertLevels + 1)),
        # Note: zb_cell, zb3_cell are added conditionally below when maxEdges is known
        # === Sign arrays (variable-dimension) ===
        # These are added conditionally below with maxEdges
        # === Old/new state arrays for time stepping ===
        "rho_zz_old": ("(nCells+1, nVertLevels)", (nCells + 1, nVertLevels)),
        "rho_zz_new": ("(nCells+1, nVertLevels)", (nCells + 1, nVertLevels)),
        # === Variable-shape arrays (skip validation) ===
        # deriv_two, defc_a, defc_b have complex shapes that depend on mesh
        "deriv_two": ("variable", None),
        "defc_a": ("variable", None),
        "defc_b": ("variable", None),
    }

    # Add variable-dimension arrays if maxEdges is known
    if maxEdges is not None:
        checks["edgesOnCell"] = ("(nCells+1, maxEdges)", (nCells + 1, maxEdges))
        checks["cellsOnCell"] = ("(nCells+1, maxEdges)", (nCells + 1, maxEdges))
        checks["verticesOnCell"] = ("(nCells+1, maxEdges)", (nCells + 1, maxEdges))
        checks["edgesOnCell_sign"] = ("(nCells+1, maxEdges)", (nCells + 1, maxEdges))
        checks["kiteForCell"] = ("(nCells+1, maxEdges)", (nCells + 1, maxEdges))
        # 3D reconstruction arrays for horizontal pressure gradient
        checks["zb_cell"] = (
            "(nCells+1, maxEdges, nVertLevels+1)",
            (nCells + 1, maxEdges, nVertLevels + 1),
        )
        checks["zb3_cell"] = (
            "(nCells+1, maxEdges, nVertLevels+1)",
            (nCells + 1, maxEdges, nVertLevels + 1),
        )

    # Add edgesOnEdge arrays if maxEdges2 is known
    if maxEdges2 is not None:
        checks["edgesOnEdge"] = ("(nEdges+1, maxEdges2)", (nEdges + 1, maxEdges2))
        checks["weightsOnEdge"] = ("(nEdges+1, maxEdges2)", (nEdges + 1, maxEdges2))

    # Add vertex arrays if nVertices is known and > 0
    if nVertices > 0:
        checks["fVertex"] = ("(nVertices+1,)", (nVertices + 1,))
        checks["invAreaTriangle"] = ("(nVertices+1,)", (nVertices + 1,))
        # Vertex 2D arrays
        checks["vorticity"] = (
            "(nVertices+1, nVertLevels)",
            (nVertices + 1, nVertLevels),
        )
        checks["pv_edge"] = ("(nEdges+1, nVertLevels)", (nEdges + 1, nVertLevels))

    # Add edgesOnVertex arrays if vertexDegree is known
    if vertexDegree is not None and nVertices > 0:
        checks["edgesOnVertex"] = (
            "(nVertices+1, vertexDegree)",
            (nVertices + 1, vertexDegree),
        )
        checks["edgesOnVertex_sign"] = (
            "(nVertices+1, vertexDegree)",
            (nVertices + 1, vertexDegree),
        )
        checks["kiteAreasOnVertex"] = (
            "(nVertices+1, vertexDegree)",
            (nVertices + 1, vertexDegree),
        )

    # Add advection arrays if maxAdvCells is known
    if maxAdvCells is not None:
        checks["advCellsForEdge"] = (
            "(nEdges+1, maxAdvCells)",
            (nEdges + 1, maxAdvCells),
        )
        checks["adv_coefs"] = ("(nEdges+1, maxAdvCells)", (nEdges + 1, maxAdvCells))
        checks["adv_coefs_3rd"] = ("(nEdges+1, maxAdvCells)", (nEdges + 1, maxAdvCells))

    # Filter to only return checks for arrays present in state_dict
    # and skip arrays with None expected_shape (variable-shape arrays)
    return {
        name: spec
        for name, spec in checks.items()
        if name in state_dict and spec[1] is not None
    }


def _validate_scalars_shape(
    state_dict: dict[str, Any],
    nCells: int,
    nVertLevels: int,
    require_nonzero_tracers: bool = False,
) -> list[str]:
    """Validate scalars array shape if present.

    The scalars array has shape (nCells+1, nVertLevels, num_scalars) where
    num_scalars is the number of scalar tracers. This function validates:
    1. scalars is a torch.Tensor
    2. scalars is 3D
    3. First two dimensions match (nCells+1, nVertLevels)
    4. Optionally, that num_scalars > 0 when require_nonzero_tracers=True

    Args:
        state_dict: Dictionary containing state arrays
        nCells: Number of cells (not including padding)
        nVertLevels: Number of vertical levels
        require_nonzero_tracers: If True, reject scalars with shape[2]=0

    Returns:
        List of validation issues (empty if valid).

    Note:
        This function adds validation for the scalars array
        that was previously incomplete in _validate_state_dict().
    """
    issues: list[str] = []

    if "scalars" not in state_dict:
        return issues

    scalars = state_dict["scalars"]

    if not isinstance(scalars, torch.Tensor):
        issues.append(f"'scalars' must be a torch.Tensor, got {type(scalars).__name__}")
        return issues

    if scalars.ndim != 3:
        issues.append(
            f"'scalars' must be 3D (nCells+1, nVertLevels, num_scalars), "
            f"got {scalars.ndim}D with shape {tuple(scalars.shape)}"
        )
        return issues

    expected_shape = (nCells + 1, nVertLevels)
    if scalars.shape[:2] != expected_shape:
        issues.append(
            f"'scalars' has shape {tuple(scalars.shape)}, "
            f"expected first 2 dims to be (nCells+1={nCells + 1}, nVertLevels={nVertLevels})"
        )

    if scalars.shape[2] == 0 and require_nonzero_tracers:
        issues.append(
            "'scalars' has 0 tracers (shape[2]=0), but scalar advection is requested"
        )

    return issues


def atm_rk_integration_setup(
    state_dict: dict[str, Any],
    mesh_dict: dict[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Initialize state and tendencies for RK integration.

    Matches the upstream MPAS integration setup:
    - Copies prognostic state fields for RK substep storage
    - Initializes tendency arrays to zero

    This function is called ONCE per large timestep, BEFORE the outer
    DYNAMICS_SUBSTEPS loop (when config_split_dynamics_transport > 1).
    It prepares the state for RK time integration by:
    1. Saving initial state variables for RK substep recovery
    2. Initializing tendency arrays to zero

    Args:
        state_dict: Current state dictionary
        mesh_dict: Mesh arrays (not modified, but may be read for dimensions)
        config: Configuration parameters

    Returns:
        Updated state_dict with initialized tendencies and saved initial state
    """
    result = dict(state_dict)

    # Get dimensions - use the helper that handles Tensor/Python int
    nCells = _get_int_from_dict(state_dict, "nCells")
    nVertLevels = _get_int_from_dict(state_dict, "nVertLevels")

    # nEdges: resolve from explicit value, edge array shapes, or hexagonal approximation
    nEdges = _resolve_nEdges(state_dict, nCells)

    # Infer device from input tensors for default tensor creation
    device = None
    for key in ["u", "w", "rho_zz", "theta_m"]:
        if key in state_dict and isinstance(state_dict[key], torch.Tensor):
            device = state_dict[key].device
            break
    if device is None:
        device = torch.device("cpu")

    # =========================================================================
    # Initialize tendency arrays to zero (if not already present)
    # Initialize the upstream tendency fields to zero.
    # =========================================================================
    if "tend_ru" not in result:
        result["tend_ru"] = torch.zeros(
            nEdges + 1, nVertLevels, dtype=DTYPE, device=device
        )
    if "tend_rho" not in result:
        result["tend_rho"] = torch.zeros(
            nCells + 1, nVertLevels, dtype=DTYPE, device=device
        )
    if "tend_rt" not in result:
        result["tend_rt"] = torch.zeros(
            nCells + 1, nVertLevels, dtype=DTYPE, device=device
        )
    if "tend_rw" not in result:
        result["tend_rw"] = torch.zeros(
            nCells + 1, nVertLevels + 1, dtype=DTYPE, device=device
        )

    # GAP-H1: Preserve checkpoint-provided physics tendency fields when present.
    # Dycore-only execution leaves the optional dynamics-tendency additive physics tendencies
    # absent so dynamics-tendency can skip zero-valued tensor traffic.
    nCells_plus_1 = nCells + 1
    # Preserve physics-provided diabatic heating when present; default to zero
    # for dycore-only execution where physics is disabled.
    if "rt_diabatic_tend" not in result:
        result["rt_diabatic_tend"] = torch.zeros(
            nCells_plus_1, nVertLevels, dtype=DTYPE, device=device
        )

    # IAU (Incremental Analysis Update) is not ported.
    # Upstream calls atm_add_tend_anal_incr() when config_IAU_option /= 'off'.
    # Default: config_IAU_option = 'off' (disabled)
    # Scope: dycore-only port, IAU excluded by design.

    # =========================================================================
    # Save initial state for RK substep recovery
    # Fortran: state%u(:,:,2) = state%u(:,:,1), etc. (copy to time level 2)
    # u_rk_save (not u_init) to avoid collision with the 1D mesh field u_init
    # used by dynamics-tendency for Rayleigh damping.
    # =========================================================================
    if "u" in state_dict:
        result["u_rk_save"] = state_dict["u"].clone()
    if "w" in state_dict:
        result["w_init"] = state_dict["w"].clone()
    if "theta_m" in state_dict:
        cell_end = _get_int_from_dict(state_dict, "cellEnd", nCells)
        result["theta_m"] = _zero_padding_row(state_dict["theta_m"], cell_end)
        result["theta_m_init"] = result["theta_m"].clone()
    if "rho_zz" in state_dict:
        result["rho_zz_init"] = state_dict["rho_zz"].clone()

    # =========================================================================
    # Save diagnostic arrays for acoustic step recovery
    # ru_save, rw_save, rtheta_p_save, rho_p_save,
    # rho_zz_old_split, and scalars time level 2 copy
    # =========================================================================
    if "ru" in state_dict:
        result["ru_save"] = state_dict["ru"].clone()
    if "rw" in state_dict:
        result["rw_save"] = state_dict["rw"].clone()
    if "rtheta_p" in state_dict:
        result["rtheta_p_save"] = state_dict["rtheta_p"].clone()
    if "rho_p" in state_dict:
        result["rho_p_save"] = state_dict["rho_p"].clone()
    if "rho_zz" in state_dict:
        # rho_zz_old_split is a density snapshot for transport-step density
        # interpolation.  It is read-only (never updated after initial copy),
        # but remains a live dependency of split-transport scalar tendencies.
        # A detach here silently removed the initial-density contribution from
        # both forward- and reverse-mode derivatives.
        result["rho_zz_old_split"] = state_dict["rho_zz"].clone()

    # GAP-C2: Initialize rho_zz time levels for scalar advection
    # Upstream time levels map to rho_zz_old and rho_zz_new.
    # At setup, both start as initial rho_zz
    if "rho_zz" in state_dict:
        result["rho_zz_old"] = state_dict["rho_zz"].clone()
        result["rho_zz_new"] = state_dict["rho_zz"].clone()

    # =========================================================================
    # Split dynamics-transport accumulators (GAP-4)
    # When config_split_dynamics_transport > 1, ruAvg and wwAvg from each
    # dynamics substep are accumulated and then averaged for the transport step.
    # Initialize ruAvg_split and wwAvg_split to zero here.
    # =========================================================================
    result["ruAvg_split"] = torch.zeros(
        nEdges + 1, nVertLevels, dtype=DTYPE, device=device
    )
    result["wwAvg_split"] = torch.zeros(
        nCells + 1, nVertLevels + 1, dtype=DTYPE, device=device
    )

    if "scalars" in state_dict:
        # scalars_init is the initial scalar snapshot for diagnostics/restart.
        # Preserve its derivative as well: callers may inspect the returned
        # diagnostic state, and a read-only copy is not a constant.
        result["scalars_init"] = state_dict["scalars"].clone()

        # Initialize time-level fields for scalar advection
        # explicit-scalar-transport/monotonic-scalar-transport scalar advection kernels require:
        # - scalars_old: Scalars at beginning of advection step
        # - scalars_new: Scalars being updated during advection
        result["scalars_old"] = state_dict["scalars"].clone()
        result["scalars_new"] = state_dict["scalars"].clone()

        # For monotonic/positive-definite advection (monotonic-scalar-transport), initialize single-scalar arrays
        # These are used when config_monotonic=True or config_positive_definite=True
        # Shape: (nCells+1, nVertLevels) - extract first scalar (typically water vapor qv)
        if state_dict["scalars"].shape[2] > 0:
            result["scalar_old"] = state_dict["scalars"][:, :, 0].clone()
            result["scalar_new"] = state_dict["scalars"][:, :, 0].clone()
        else:
            # If no scalars exist, create zero arrays with correct shape
            nCells_plus_1 = state_dict["scalars"].shape[0]
            nVertLevels = state_dict["scalars"].shape[1]
            device = state_dict["scalars"].device
            result["scalar_old"] = torch.zeros(
                (nCells_plus_1, nVertLevels), dtype=DTYPE, device=device
            )
            result["scalar_new"] = torch.zeros(
                (nCells_plus_1, nVertLevels), dtype=DTYPE, device=device
            )

    # =========================================================================
    # Create kernel-internal aliases for Fortran field names.
    # The IC checkpoint uses standard MPAS names (pressure_p, rho_p, etc.)
    # while dynamics-tendency dyn_tend uses Fortran-internal abbreviations (pp, rr, rb, etc.)

    # ALWAYS overwrite aliases from canonical sources. The IC checkpoint may
    # contain stale Fortran-internal arrays (pp, rr, t_init, etc.) captured
    # from a different time level during Fortran initialization. Using these
    # stale values causes wrong perturbations and numerical blowup (e.g.,
    # pp off by 1770 Pa, rr off by 0.13, leading to 78 m/s velocity error).
    # =========================================================================
    alias_map = {
        # dynamics-tendency aliases (Fortran diag pool names -> PyTorch canonical names)
        "pp": "pressure_p",  # pressure perturbation (diag%pressure_p)
        "rr": "rho_p",  # density perturbation (diag%rho_p)
        "rb": "rho_base",  # base density (diag%rho_base)
        "pressure_b": "pressure_base",  # base pressure (diag%pressure_base)
        # vertical-implicit-coefficients aliases (Fortran local names -> PyTorch canonical names)
        # Upstream diagnostic-pool alias: exner -> p.
        "p": "exner",  # Exner function (NOT pressure_p!)
        "t": "theta_m",  # potential temperature (state%theta_m, time level 2)
        "rtb": "rtheta_base",  # base rho*theta (diag%rtheta_base)
        # Upstream diagnostic-pool alias: exner_base -> pb.
        "pb": "exner_base",  # base Exner function (NOT pressure_base!)
        "rt": "rtheta_p",  # rho*theta perturbation (diag%rtheta_p)
    }
    for alias, source in alias_map.items():
        if source in result:
            result[alias] = result[source]

    # Save-state variables: ALWAYS overwrite with current state.
    # Upstream RK setup unconditionally copies
    # state(:,:,2) = state(:,:,1) at the start of each timestep.
    # The IC checkpoint may contain stale theta_m_save / rr_save from
    # a different Fortran time level; using them would cause the RK
    # recovery to compute wrong perturbations (observed: 66 K error
    # in theta_m_save vs theta_m, leading to 78 m/s velocity error).
    if "theta_m" in state_dict:
        result["theta_m_save"] = result["theta_m"].clone()
    if "rho_p" in result:
        result["rr_save"] = result["rho_p"].clone()
    elif "rho_p_save" in result:
        result["rr_save"] = result["rho_p_save"]

    # =========================================================================
    # Initialize Rayleigh damping / eddy viscosity reference profiles.
    # Fortran retrieves t_init, u_init, v_init from the mesh pool during
    # mpas_atm_dynamics_init and NEVER updates them. They represent the
    # initial atmospheric state for computing deviations.

    # The IC checkpoint may contain STALE values (e.g., t_init = 0 instead
    # of theta_m) because it was captured before Fortran set these fields.
    # Using stale t_init (all zeros vs theta_m ~237-835 K) causes massive
    # spurious vertical diffusion in dynamics-tendency eddy viscosity.
    # =========================================================================
    # t_init is a constant reference profile (all zeros for GFS IC).
    # Never updated at runtime, never differentiated through.
    if "t_init" not in result:
        if "t_init" in state_dict:
            result["t_init"] = state_dict["t_init"].detach().clone()
        else:
            result["t_init"] = torch.zeros(
                nCells + 1, nVertLevels, dtype=DTYPE, device=device
            )

    # =========================================================================
    # Initialize acoustic perturbation variables to zero if not present.
    # These start as zero at the beginning of each timestep and accumulate
    # during acoustic substeps.
    # =========================================================================
    if "ru_p" not in result:
        result["ru_p"] = torch.zeros(
            nEdges + 1, nVertLevels, dtype=DTYPE, device=device
        )
    if "rw_p" not in result:
        result["rw_p"] = torch.zeros(
            nCells + 1, nVertLevels + 1, dtype=DTYPE, device=device
        )
    if "rho_pp" not in result:
        result["rho_pp"] = torch.zeros(
            nCells + 1, nVertLevels, dtype=DTYPE, device=device
        )
    if "rtheta_pp" not in result:
        result["rtheta_pp"] = torch.zeros(
            nCells + 1, nVertLevels, dtype=DTYPE, device=device
        )

    # =========================================================================
    # Initialize boundary mask arrays for global MPAS.
    # For global simulations (not regional), bdyMaskCell and bdyMaskEdge are
    # all zeros. These are required by large-step-recovery (recover_large_step) and vertical-velocity boundary adjustment-post-step scalar boundary setting
    # (boundary conditions). Create them if missing to avoid lookup failures.
    # =========================================================================
    if "bdyMaskCell" not in result:
        result["bdyMaskCell"] = torch.zeros(nCells + 1, dtype=DTYPE, device=device)
    if "bdyMaskEdge" not in result:
        result["bdyMaskEdge"] = torch.zeros(nEdges + 1, dtype=DTYPE, device=device)

    # Keep Fortran short-name aliases synchronized with canonical fields.
    # execute_rk_step can be called directly by tests/tools (without atm_srk3's
    # post-step alias refresh). Returning stale aliases (e.g., p, pp, rr, t, rt)
    # makes the next RK step consume old state.
    alias_refresh_map = {
        "pp": "pressure_p",
        "rr": "rho_p",
        "rb": "rho_base",
        "pressure_b": "pressure_base",
        "p": "exner",
        "pb": "exner_base",
        "t": "theta_m",
        "rt": "rtheta_p",
        "rtb": "rtheta_base",
    }
    for alias, source in alias_refresh_map.items():
        if source in result:
            result[alias] = result[source]

    return result


def compute_rk_weights(
    dt: float,
    nAcousticSteps: int,
    order: int = 3,
) -> dict[str, list]:
    """
    Compute Runge-Kutta time stepping weights.

    The split-explicit RK scheme uses different timestep sizes for
    the large (slow dynamics) and small (acoustic) steps. This function
    computes the weights for both RK2 and RK3 schemes.

    Args:
        dt: Large timestep in seconds (must be > 0)
        nAcousticSteps: Number of acoustic substeps per large step (must be >= 1)
        order: Time integration order (2 or 3)

    Returns:
        Dictionary containing:
            rk_timestep: [dt1, dt2, dt3] large step sizes for each RK stage
            rk_sub_timestep: [dts1, dts2, dts3] acoustic step sizes
            number_sub_steps: [n1, n2, n3] acoustic steps per RK stage

    Raises:
        ValueError: If nAcousticSteps < 1, dt <= 0, or order is not 2 or 3

    References:
        Klemp, J. B., W. C. Skamarock, and J. Dudhia, 2007:
            Conservative split-explicit time integration methods for
            the compressible nonhydrostatic equations. Mon. Wea. Rev.
        Skamarock, W. C., and A. Gassmann, 2011:
            Conservative transport schemes for spherical geodesic grids:
            High-order flux operators for ODE-based time integration.
            Mon. Wea. Rev.
    """
    # Input validation to prevent division by zero and invalid timesteps
    if nAcousticSteps < 1:
        raise ValueError(
            f"nAcousticSteps must be >= 1, got {nAcousticSteps}. "
            "Set nAcousticSteps >= 1 or use single-step integration."
        )
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")

    if order == 3:
        # RK3 weights: stages at t, t+dt/3, t+dt/2 -> t+dt
        rk_timestep = [dt / 3.0, dt / 2.0, dt]
        rk_sub_timestep = [
            dt / 3.0,
            dt / float(nAcousticSteps),
            dt / float(nAcousticSteps),
        ]
        number_sub_steps = [1, max(1, nAcousticSteps // 2), nAcousticSteps]
    elif order == 2:
        # RK2 weights: stages at t, t+dt/2, t+dt/2 -> t+dt
        rk_timestep = [dt / 2.0, dt / 2.0, dt]
        rk_sub_timestep = [dt / float(nAcousticSteps)] * 3
        number_sub_steps = [
            max(1, nAcousticSteps // 2),
            max(1, nAcousticSteps // 2),
            nAcousticSteps,
        ]
    else:
        raise ValueError(f"Unsupported integration order: {order}")

    return {
        "rk_timestep": rk_timestep,
        "rk_sub_timestep": rk_sub_timestep,
        "number_sub_steps": number_sub_steps,
    }


_COMPILED_STEP_PLAN_DYNAMIC_CONFIG_KEYS = frozenset(
    {
        "dt",
        "dts",
        "dt_full",
        "rk_step",
        "ns",
        "dynamics_substep",
    }
)

_COMPILED_STEP_PLAN_INTERNAL_CONFIG_KEYS = frozenset(
    {
        "_compiled_step_plan",
        "_compiled_step_plan_key",
        "_compiled_step_plan_stage_key",
        "_compiled_step_plan_stage_kind",
        "_compiled_step_specialization_key",
        "_dycore_static_payloads",
        "_dycore_workspace",
        "_timestep_scalars",
    }
)

# These controls choose how/where an already-defined numeric graph executes. They
# must not participate in the numeric specialization key: production preflight
# deliberately compiles one helper at a time, while the warmed run enables the
# cumulative target set. Backend/mode remain part of maybe_compile_kernel's own
# cache key, so excluding them here cannot reuse a wrapper across compiler modes.
_COMPILED_STEP_PLAN_EXECUTION_CONFIG_KEYS = frozenset(
    {
        "config_profile_sections",
        "config_section_timing",
        "config_use_torch_compile",
    }
)


def _compiled_plan_freeze_value(value: Any) -> Any | None:
    """Return a hashable, static representation of a plan config value."""
    if isinstance(value, torch.Tensor):
        if (
            value.numel() == 1
            and value.device.type == "cpu"
            and not value.requires_grad
        ):
            return _compiled_plan_freeze_value(value.item())
        return (
            "tensor",
            tuple(int(dim) for dim in value.shape),
            str(value.dtype),
            str(value.device),
        )
    if isinstance(value, np.generic):
        return _compiled_plan_freeze_value(value.item())
    if isinstance(value, (str, bytes, int, float, bool, type(None))):
        return value
    if isinstance(value, (tuple, list)):
        frozen_items = []
        for item in value:
            frozen = _compiled_plan_freeze_value(item)
            if frozen is None:
                return None
            frozen_items.append(frozen)
        return tuple(frozen_items)
    return None


def _compiled_step_namelist_key(
    config: Mapping[str, Any],
) -> tuple[tuple[str, Any], ...]:
    """Return only static choices that may affect compiled numeric results/shapes."""
    items: list[tuple[str, Any]] = []
    for key in sorted(str(item) for item in config):
        if key in _COMPILED_STEP_PLAN_DYNAMIC_CONFIG_KEYS:
            continue
        if key in _COMPILED_STEP_PLAN_INTERNAL_CONFIG_KEYS:
            continue
        if key in _COMPILED_STEP_PLAN_EXECUTION_CONFIG_KEYS:
            continue
        if key.startswith("config_torch_compile_"):
            continue
        if key.startswith("_"):
            continue
        value = _compiled_plan_freeze_value(config.get(key))
        if value is None:
            continue
        items.append((key, value))
    return tuple(items)


def _compiled_step_tensor_context(
    state_dict: Mapping[str, Any],
    mesh_dict: Mapping[str, Any] | None,
) -> tuple[str, str]:
    """Infer the dtype/device that should participate in compiled step keys."""
    preferred_keys = (
        "u",
        "w",
        "theta_m",
        "rho_zz",
        "scalars",
        "cellsOnEdge",
        "edgesOnCell",
    )
    for source in (state_dict, mesh_dict or {}):
        for key in preferred_keys:
            value = source.get(key)
            if isinstance(value, torch.Tensor):
                return str(value.dtype), str(value.device)
        for value in source.values():
            if isinstance(value, torch.Tensor):
                return str(value.dtype), str(value.device)
    return str(DTYPE), "cpu"


def _compiled_step_reference_tensor(
    state_dict: Mapping[str, Any],
    mesh_dict: Mapping[str, Any] | None,
) -> torch.Tensor | None:
    """Return a tensor whose device/dtype should own scan metadata."""
    preferred_keys = (
        "u",
        "w",
        "theta_m",
        "rho_zz",
        "scalars",
        "a_tri",
        "alpha_tri",
        "gamma_tri",
    )
    for source in (state_dict, mesh_dict or {}):
        for key in preferred_keys:
            value = source.get(key)
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                return value
        for value in source.values():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                return value
    return None


def _prebuild_vertical_scan_plans(
    *,
    static_payloads: DycoreStaticPayloads,
    state_dict: Mapping[str, Any],
    mesh_dict: Mapping[str, Any] | None,
    nVertLevels: int,
) -> None:
    """Populate runner-owned scan metadata before compiled numeric regions."""
    reference = _compiled_step_reference_tensor(state_dict, mesh_dict)
    if reference is None:
        return
    cache = static_payloads.vertical_scan_plan_cache()
    levels = int(nVertLevels)
    if levels <= 0:
        ensure_vertical_scan_plan(reference, 0, cache=cache)
        return
    ensure_vertical_scan_plan(reference, levels, cache=cache)
    if levels > 1:
        ensure_vertical_scan_plan(reference, levels - 1, cache=cache)


def _compiled_step_num_scalars(state_dict: Mapping[str, Any]) -> int:
    """Infer the scalar count used for stable chunk/tile compile keys."""
    value = state_dict.get("num_scalars")
    if value is not None:
        return max(int(value), 1)
    scalars = state_dict.get("scalars")
    if isinstance(scalars, torch.Tensor) and scalars.ndim >= 3:
        return max(int(scalars.shape[-1]), 1)
    return 1


def _average_split_transport_flux(
    accumulated_flux: torch.Tensor,
    dynamics_split: int,
) -> torch.Tensor:
    """Match the upstream reciprocal-then-multiply split-flux average."""
    inv_dynamics_split = 1.0 / float(dynamics_split)
    return accumulated_flux * inv_dynamics_split


def _stable_monotonic_scalar_transport_group_width(
    num_scalars: int, requested_width: int | None
) -> int:
    """Return a divisor width so compiled monotonic-scalar-transport chunks have one shape."""
    scalars = max(int(num_scalars), 1)
    if requested_width is None:
        return scalars
    requested = int(requested_width)
    if requested <= 0:
        return scalars
    width = max(1, min(requested, scalars))
    if scalars % width == 0:
        return width
    for candidate in range(width - 1, 0, -1):
        if scalars % candidate == 0:
            return candidate
    return 1


def _compiled_step_mesh_shape_key(
    scalars: TimestepScalars,
) -> tuple[tuple[str, int], ...]:
    return (
        ("nCells", scalars.nCells),
        ("nEdges", scalars.nEdges),
        ("nVertices", scalars.nVertices),
        ("nVertLevels", scalars.nVertLevels),
        ("maxEdges", scalars.maxEdges),
        ("maxEdges2", scalars.maxEdges2),
        ("vertexDegree", scalars.vertexDegree),
    )


def _compiled_step_stage_key(
    plan_key: tuple[Any, ...],
    *,
    stage_kind: str,
    dynamics_substep: int,
    rk_step: int,
    dt_value: float,
    dts_value: float | None,
    acoustic_steps: int | None,
) -> tuple[Any, ...]:
    key: tuple[Any, ...] = (
        "plan",
        plan_key,
        "stage_kind",
        stage_kind,
        "dynamics_substep",
        int(dynamics_substep),
        "rk_step",
        int(rk_step),
        "dt",
        float(dt_value),
    )
    if dts_value is not None:
        key += ("dts", float(dts_value))
    if acoustic_steps is not None:
        key += ("ns", int(acoustic_steps))
    return key


def _compiled_kernel_specialization_key(
    config: Mapping[str, Any],
    kernel_name: str,
    *items: Any,
) -> tuple[Any, ...]:
    """Specialization key rooted at the immutable stage plan when available."""
    stage_key = config.get("_compiled_step_specialization_key")
    if stage_key is None:
        return (kernel_name, *items)
    return (stage_key, "kernel", kernel_name, *items)


@dataclass(frozen=True)
class CompiledStepPlan:
    """Immutable per-timestep compile plan for SRK3 dynamics/transport stages."""

    plan_key: tuple[Any, ...]
    mesh_shape_key: tuple[tuple[str, int], ...]
    dtype: str
    device: str
    split_coupled_mode: str
    dynamics_split: int
    scalar_chunk_size: int
    monotonic_scalar_transport_reduction_slot_tile_width: int
    monotonic_scalar_transport_streaming_scalar_tile_width: int
    acoustic_step_tridiag_scan_policy: str | None
    acoustic_step_tridiag_scan_allowed_shapes: tuple[str, ...]
    namelist_key: tuple[tuple[str, Any], ...]
    rk_weights: Mapping[str, tuple[Any, ...]]
    dynamics_stage_configs: tuple[tuple[Mapping[str, Any], ...], ...]
    transport_stage_configs: tuple[Mapping[str, Any], ...]

    def dynamics_config(self, dynamics_substep: int, rk_step: int) -> Mapping[str, Any]:
        if dynamics_substep < 1 or dynamics_substep > len(self.dynamics_stage_configs):
            raise IndexError(f"dynamics_substep out of range: {dynamics_substep}")
        if rk_step < 1 or rk_step > 3:
            raise IndexError(f"rk_step out of range: {rk_step}")
        return self.dynamics_stage_configs[dynamics_substep - 1][rk_step - 1]

    def transport_config(self, rk_step: int) -> Mapping[str, Any]:
        if rk_step < 1 or rk_step > len(self.transport_stage_configs):
            raise IndexError(f"transport rk_step out of range: {rk_step}")
        return self.transport_stage_configs[rk_step - 1]

    def audit_snapshot(self) -> dict[str, Any]:
        return {
            "plan_key": repr(self.plan_key),
            "mesh_shape_key": list(self.mesh_shape_key),
            "dtype": self.dtype,
            "device": self.device,
            "split_coupled_mode": self.split_coupled_mode,
            "dynamics_split": self.dynamics_split,
            "scalar_chunk_size": self.scalar_chunk_size,
            "monotonic_scalar_transport_reduction_slot_tile_width": self.monotonic_scalar_transport_reduction_slot_tile_width,
            "monotonic_scalar_transport_streaming_scalar_tile_width": self.monotonic_scalar_transport_streaming_scalar_tile_width,
            "acoustic_step_tridiag_scan_policy": self.acoustic_step_tridiag_scan_policy,
            "acoustic_step_tridiag_scan_allowed_shapes": list(
                self.acoustic_step_tridiag_scan_allowed_shapes
            ),
            "namelist_key_count": len(self.namelist_key),
            "dynamics_stage_count": sum(
                len(stage) for stage in self.dynamics_stage_configs
            ),
            "transport_stage_count": len(self.transport_stage_configs),
            "rk_weights": {key: list(value) for key, value in self.rk_weights.items()},
        }


def build_compiled_step_plan(
    *,
    config: Mapping[str, Any],
    state_dict: Mapping[str, Any],
    mesh_dict: Mapping[str, Any] | None,
    dt: float,
    dt_dynamics: float,
    nAcousticSteps: int,
    order: int,
    dynamics_split: int,
    timestep_scalars: TimestepScalars,
    static_payloads: DycoreStaticPayloads,
    workspace_obj: Any = None,
) -> CompiledStepPlan:
    """Build immutable SRK3 stage configs with stable compile specialization keys."""
    dtype, device = _compiled_step_tensor_context(state_dict, mesh_dict)
    mesh_shape_key = _compiled_step_mesh_shape_key(timestep_scalars)
    num_scalars = _compiled_step_num_scalars(state_dict)
    requested_scalar_chunk_size = _coerce_int(
        get_config_value_safe(config, "config_scalar_transport_chunk_size", 0),
        "config_scalar_transport_chunk_size",
    )
    scalar_chunk_size = _stable_monotonic_scalar_transport_group_width(
        num_scalars,
        None if requested_scalar_chunk_size <= 0 else requested_scalar_chunk_size,
    )
    monotonic_scalar_transport_reduction_slot_tile_width = _coerce_int(
        get_config_value_safe(
            config, "config_monotonic_scalar_transport_reduction_slot_tile_width", 0
        ),
        "config_monotonic_scalar_transport_reduction_slot_tile_width",
    )
    requested_monotonic_scalar_transport_streaming_scalar_tile_width = _coerce_int(
        get_config_value_safe(
            config,
            "config_monotonic_scalar_transport_streaming_scalar_tile_width",
            monotonic_scalar_transport_reduction_slot_tile_width,
        ),
        "config_monotonic_scalar_transport_streaming_scalar_tile_width",
    )
    monotonic_scalar_transport_streaming_scalar_tile_width = (
        _stable_monotonic_scalar_transport_group_width(
            scalar_chunk_size,
            None
            if requested_monotonic_scalar_transport_streaming_scalar_tile_width <= 0
            else requested_monotonic_scalar_transport_streaming_scalar_tile_width,
        )
    )
    acoustic_step_tridiag_scan_policy = normalize_tridiag_scan_policy(
        get_config_value_safe(
            config,
            "config_acoustic_step_tridiag_scan_policy",
            TRIDIAG_SCAN_DEFAULT_POLICY,
        )
    )
    acoustic_step_tridiag_scan_allowed_shapes = tuple(
        sorted(
            canonical_tridiag_scan_shape_keys(
                get_config_value_safe(
                    config, "config_acoustic_step_tridiag_scan_allowed_shapes", ()
                )
            )
        )
    )
    normalized_config = dict(config)
    normalized_config["config_scalar_transport_chunk_size"] = int(scalar_chunk_size)
    normalized_config[
        "config_monotonic_scalar_transport_streaming_scalar_tile_width"
    ] = int(monotonic_scalar_transport_streaming_scalar_tile_width)
    normalized_config["config_acoustic_step_tridiag_scan_policy"] = (
        acoustic_step_tridiag_scan_policy
    )
    normalized_config["config_acoustic_step_tridiag_scan_allowed_shapes"] = (
        acoustic_step_tridiag_scan_allowed_shapes
    )
    split_coupled_mode = "split" if dynamics_split > 1 else "coupled"
    namelist_key = _compiled_step_namelist_key(normalized_config)

    if dynamics_split > 0:
        with profile_section("mpas.rk_weights", config):
            raw_rk_weights = compute_rk_weights(
                dt_dynamics,
                nAcousticSteps,
                order=order,
            )
        rk_weights = {
            "rk_timestep": tuple(
                float(value) for value in raw_rk_weights["rk_timestep"]
            ),
            "rk_sub_timestep": tuple(
                float(value) for value in raw_rk_weights["rk_sub_timestep"]
            ),
            "number_sub_steps": tuple(
                int(value) for value in raw_rk_weights["number_sub_steps"]
            ),
        }
    else:
        rk_weights = {
            "rk_timestep": (),
            "rk_sub_timestep": (),
            "number_sub_steps": (),
        }

    plan_key: tuple[Any, ...] = (
        "CompiledStepPlan",
        ("mesh", mesh_shape_key),
        ("dtype", dtype),
        ("device", device),
        ("mode", split_coupled_mode),
        ("order", int(order)),
        ("dynamics_split", int(dynamics_split)),
        ("dt", float(dt)),
        ("dt_dynamics", float(dt_dynamics)),
        ("nAcousticSteps", int(nAcousticSteps)),
        ("scalar_chunk_size", int(scalar_chunk_size)),
        (
            "monotonic_scalar_transport_reduction_slot_tile_width",
            int(monotonic_scalar_transport_reduction_slot_tile_width),
        ),
        (
            "monotonic_scalar_transport_streaming_scalar_tile_width",
            int(monotonic_scalar_transport_streaming_scalar_tile_width),
        ),
        ("acoustic_step_tridiag_scan_policy", acoustic_step_tridiag_scan_policy),
        (
            "acoustic_step_tridiag_scan_allowed_shapes",
            acoustic_step_tridiag_scan_allowed_shapes,
        ),
        ("namelist", namelist_key),
        (
            "rk_weights",
            tuple((key, tuple(value)) for key, value in sorted(rk_weights.items())),
        ),
    )

    base_config = dict(normalized_config)
    # ``build_compiled_step_plan`` is only used by the full SRK timestep. Keep
    # vertical-implicit-coefficients coefficients and diagnostic-solve diagnostics active-row compact across stages,
    # then publish padded compatibility fields once at timestep return.
    base_config["_defer_vertical_implicit_coefficients_public_pack"] = True
    base_config["_defer_diagnostic_solve_public_pack"] = True
    base_config["_timestep_scalars"] = timestep_scalars
    base_config["_dycore_static_payloads"] = static_payloads
    base_config["_tridiag_scan_plan_cache"] = static_payloads.vertical_scan_plan_cache()
    base_config["_compiled_step_plan_key"] = plan_key
    _prebuild_vertical_scan_plans(
        static_payloads=static_payloads,
        state_dict=state_dict,
        mesh_dict=mesh_dict,
        nVertLevels=timestep_scalars.nVertLevels,
    )
    if workspace_obj is not None:
        base_config["_dycore_workspace"] = workspace_obj
    else:
        base_config.pop("_dycore_workspace", None)

    dynamics_stage_configs: list[tuple[Mapping[str, Any], ...]] = []
    for dynamics_substep in range(1, dynamics_split + 1):
        substep_configs: list[Mapping[str, Any]] = []
        for rk_step in range(1, 4):
            dt_stage = float(rk_weights["rk_timestep"][rk_step - 1])
            dts_stage = float(rk_weights["rk_sub_timestep"][rk_step - 1])
            ns_stage = int(rk_weights["number_sub_steps"][rk_step - 1])
            validate_static_inputs = dynamics_substep == 1 and rk_step == 1
            stage_key = _compiled_step_stage_key(
                plan_key,
                stage_kind="dynamics",
                dynamics_substep=dynamics_substep,
                rk_step=rk_step,
                dt_value=dt_stage,
                dts_value=dts_stage,
                acoustic_steps=ns_stage,
            )
            stage_config = dict(base_config)
            stage_config.update(
                {
                    "dts": dts_stage,
                    "rk_step": rk_step,
                    "dt": dt_stage,
                    "ns": ns_stage,
                    "dynamics_substep": dynamics_substep,
                    "dt_full": float(dt),
                    "_rk_static_input_normalization_required": validate_static_inputs,
                    "_rk_mesh_state_compatibility_required": validate_static_inputs,
                    "_rk_mesh_merge_required": validate_static_inputs,
                    "_rk_indexing_validation_required": validate_static_inputs,
                    "_compiled_step_plan_stage_key": stage_key,
                    "_compiled_step_plan_stage_kind": "dynamics",
                    "_compiled_step_specialization_key": stage_key,
                }
            )
            substep_configs.append(MappingProxyType(stage_config))
        dynamics_stage_configs.append(tuple(substep_configs))

    transport_stage_configs: list[Mapping[str, Any]] = []
    if dynamics_split > 1:
        transport_timesteps = [float(dt) / 3.0, float(dt) / 2.0, float(dt)]
        if order == 2:
            transport_timesteps[0] = float(dt) / 2.0
        for rk_step in range(1, 4):
            dt_stage = float(transport_timesteps[rk_step - 1])
            stage_key = _compiled_step_stage_key(
                plan_key,
                stage_kind="transport",
                dynamics_substep=dynamics_split,
                rk_step=rk_step,
                dt_value=dt_stage,
                dts_value=None,
                acoustic_steps=None,
            )
            stage_config = dict(base_config)
            stage_config.update(
                {
                    "dt": dt_stage,
                    "rk_step": rk_step,
                    "config_split_dynamics_transport": True,
                    "config_dynamics_split_steps": dynamics_split,
                    "split_dynamics_transport": dynamics_split,
                    "_compiled_step_plan_stage_key": stage_key,
                    "_compiled_step_plan_stage_kind": "transport",
                    "_compiled_step_specialization_key": stage_key,
                }
            )
            transport_stage_configs.append(MappingProxyType(stage_config))

    return CompiledStepPlan(
        plan_key=plan_key,
        mesh_shape_key=mesh_shape_key,
        dtype=dtype,
        device=device,
        split_coupled_mode=split_coupled_mode,
        dynamics_split=int(dynamics_split),
        scalar_chunk_size=int(scalar_chunk_size),
        monotonic_scalar_transport_reduction_slot_tile_width=int(
            monotonic_scalar_transport_reduction_slot_tile_width
        ),
        monotonic_scalar_transport_streaming_scalar_tile_width=int(
            monotonic_scalar_transport_streaming_scalar_tile_width
        ),
        acoustic_step_tridiag_scan_policy=acoustic_step_tridiag_scan_policy,
        acoustic_step_tridiag_scan_allowed_shapes=acoustic_step_tridiag_scan_allowed_shapes,
        namelist_key=namelist_key,
        rk_weights=MappingProxyType(rk_weights),
        dynamics_stage_configs=tuple(dynamics_stage_configs),
        transport_stage_configs=tuple(transport_stage_configs),
    )


def _atm_srk3_impl(
    state: Any,
    mesh: Any,
    config: Mapping[str, Any],
    dt: float,
    strict: bool = True,
) -> dict[str, Any]:
    """
    Advance model state forward by dt using split-explicit RK3.

    This function implements the full RK3 time integration loop with optional
    dynamics/transport splitting:

        # Setup before RK integration
        call atm_rk_integration_setup(state, tend, mesh, configs)

        # Outer split loop (DYNAMICS_SUBSTEPS in Fortran)
        # Fortran semantics:
        #   config_split_dynamics_transport (logical)
        #   config_dynamics_split_steps (integer)
        # If split enabled, dynamics_substeps=config_dynamics_split_steps;
        # otherwise dynamics_substeps=1.
        for dynamics_substep in range(1, dynamics_substeps + 1):
            dt_dynamics = dt / dynamics_substeps

            # RK stages
            for rk_step in [1, 2, 3]:
                1. Compute vertical implicit coefficients (vertical-implicit-coefficients)
                2. Compute dynamic tendencies (dynamics-tendency) - BEFORE acoustic loop
                3. Set perturbation variables (small-step-perturbation)
                4. Acoustic substep loop (acoustic-step)
                5. Recover from acoustic steps (large-step-recovery)
                6. Solve diagnostics (diagnostic-solve)
                7. Scalar advection (explicit-scalar-transport/monotonic-scalar-transport)

    Args:
        state: Current model state (DycoreState or dict of tensors)
        mesh: Mesh geometry and connectivity (MESHState or dict of tensors)
        config: Configuration dictionary with model parameters:
            - nAcousticSteps: Total number of acoustic substeps per large step
            - time_integration_order: 2 or 3 (default 3)
            - config_split_dynamics_transport: Enable split dynamics/transport
            - config_dynamics_split_steps: Number of dynamics substeps when split enabled
            - split_dynamics_transport: Legacy compatibility key
            - dts: Acoustic substep size (derived from dt/nAcousticSteps)
            - epssm: Off-centering parameter for acoustic solver
        dt: Large timestep in seconds
        strict: If True (default), raise ValueError when required kernels cannot
                run due to missing inputs. If False, skip kernels with warning.

    Returns:
        Updated state dictionary after one full timestep

    Raises:
        ValueError: If strict=True and required kernel inputs are missing,
                    or state/mesh validation fails, or dynamics split settings are invalid.
    """
    # =========================================================================
    # VALIDATE STATE AND MESH
    # Call validate() if available to catch shape/dtype errors early.
    # This provides helpful error messages before cryptic index errors in kernels.
    # =========================================================================
    if hasattr(state, "validate") and callable(state.validate):
        try:
            state.validate()
        except ValueError as e:
            if strict:
                raise ValueError(f"DycoreState validation failed: {e}") from e
            else:
                warnings.warn(f"DycoreState validation warning: {e}", RuntimeWarning)

    if hasattr(mesh, "validate") and callable(mesh.validate):
        try:
            mesh.validate()
        except ValueError as e:
            if strict:
                raise ValueError(f"MESHState validation failed: {e}") from e
            else:
                warnings.warn(f"MESHState validation warning: {e}", RuntimeWarning)

    # Attach the runner-owned static payload store before the local copy so
    # warmed runs can reuse successful static-input validation across timesteps.
    if isinstance(config, dict):
        ensure_static_payload_store(config)

    # Local config copy to avoid mutating caller-owned scalar/config values.
    config = dict(config)
    static_payloads_at_boundary = ensure_static_payload_store(config)
    config["_tridiag_scan_plan_cache"] = (
        static_payloads_at_boundary.vertical_scan_plan_cache()
    )

    # Handle DycoreState vs dict input
    if hasattr(state, "__dict__"):
        # DycoreState - convert to dict
        state_dict = dict(state.__dict__)
    else:
        # Already a dict - clone to avoid mutation
        state_dict = dict(state)

    # Handle mesh dict
    if hasattr(mesh, "__dict__"):
        mesh_dict = dict(mesh.__dict__)
    else:
        mesh_dict = dict(mesh)

    # =========================================================================
    # PROPAGATE DIMENSION SCALARS FROM MESH TO STATE
    # When state and mesh are loaded from separate checkpoint files (e.g.,
    # baroclinic_wave_ic.nc for state, mesh_complete.nc for mesh), the state
    # file may store nCells/nEdges/nVertLevels only as NetCDF dimensions (not
    # variables), so they won't appear in the loaded dict. The mesh file
    # typically has them as variables. Copy missing dimension scalars from
    # mesh_dict to state_dict so downstream validation succeeds.
    # =========================================================================
    dim_keys = [
        "nCells",
        "nEdges",
        "nVertLevels",
        "nVertices",
        "maxEdges",
        "maxEdges2",
        "vertexDegree",
        "num_scalars",
    ]
    for dk in dim_keys:
        if dk not in state_dict and dk in mesh_dict:
            state_dict[dk] = mesh_dict[dk]

    static_scalar_store = _static_input_validation_store(config)

    # Normalize scalar config/dimension entries once at the dict boundary.
    _coerce_0d_tensor_scalars_inplace(config, static_scalar_store)
    _coerce_0d_tensor_scalars_inplace(state_dict, static_scalar_store)
    _coerce_0d_tensor_scalars_inplace(mesh_dict, static_scalar_store)

    # =========================================================================
    # VALIDATE DICT INPUTS
    # When state/mesh are raw dictionaries from Fortran-style inputs, no
    # validate() method exists. Call _validate_state_dict() to catch:
    # - Missing required dimensions (nCells, nEdges, nVertLevels)
    # - Wrong array shapes
    # - Wrong dtypes (not float64)
    # This prevents cryptic errors deep in kernels.
    # =========================================================================
    if not hasattr(state, "validate"):
        # state is a dict, validate it directly
        static_validation_key = _static_input_validation_cache_key(
            state_dict,
            strict_indexing=bool(strict),
        )
        static_validation_store = _static_input_validation_store(config)
        static_sentinel_prevalidated = (
            static_validation_key is not None
            and static_validation_store is not None
            and static_validation_store.has_validated_static_inputs(
                static_validation_key
            )
        )
        _validate_state_dict(
            state_dict,
            name="state",
            strict=strict,
            skip_static_sentinel_validation=static_sentinel_prevalidated,
        )

    # Note: We do NOT call _validate_state_dict on mesh_dict here because:
    # 1. Mesh dicts may be minimal (just connectivity arrays like cellsOnEdge)
    #    without the dimension scalars (nCells, nEdges, nVertLevels) that
    #    _validate_state_dict requires - those come from state_dict.
    # 2. Indexing convention validation (0-based vs 1-based) is handled by
    #    _validate_indexing_convention in execute_rk_step, which runs on the
    #    merged state+mesh dict.
    # 3. If mesh has a validate() method (MESHState), that's called above.

    # However, we DO validate mesh dtype (float64) to prevent silent precision
    # loss from float32 mesh arrays leaking into kernel computations.
    if mesh_dict is not None:
        _validate_mesh_dtypes(mesh_dict, strict=strict)

    # Enforce CPU-only execution contract early.
    require_consistent_device("atm_srk3 state", state_dict)
    if mesh_dict is not None:
        require_consistent_device("atm_srk3 mesh", mesh_dict)

    # Check for 0-based indexing (incompatible with kernels)
    if mesh_dict is not None and "cellsOnEdge" in mesh_dict:  # noqa: SIM102
        if _mesh_cells_on_edge_min_is_minus_one(
            mesh_dict,
            store=static_scalar_store,
        ):
            warnings.warn(
                "Mesh appears to use 0-based indexing (sentinel=-1). "
                "Kernels expect 1-based+padded arrays from checkpoints. "
                "Use raw Fortran-style input dictionaries for kernel data.",
                RuntimeWarning,
            )

    # Extract configuration parameters
    # Use centralized Fortran-compatible defaults
    # Use centralized config with proper alias resolution
    # config_number_of_sub_steps maps to nAcousticSteps via CONFIG_ALIASES
    # Fallback 2 matches FORTRAN_DEFAULTS (Registry.xml: config_number_of_sub_steps default_value="2")
    nAcousticSteps = _coerce_int(
        get_config_value_safe(config, "nAcousticSteps", 2), "nAcousticSteps"
    )
    # Fallback 2 matches FORTRAN_DEFAULTS (Registry.xml: config_time_integration_order default_value="2")
    order = _coerce_int(
        get_config_value_safe(config, "time_integration_order", 2),
        "time_integration_order",
    )
    dt = _coerce_float(dt, "dt")

    # Keep direct atm_srk3 callers aligned with the public config validator.
    validate_time_integration_scheme(config)

    # These registry knobs are inert in MPAS-Atmosphere core_atmosphere but
    # still need type validation for dict-based callers.
    _reject_unsupported_advection_orders(config)

    # =========================================================================
    # Resolve split dynamics/transport semantics (Fortran-compatible).
    # Fortran uses:
    #   config_split_dynamics_transport (logical)
    #   config_dynamics_split_steps (integer)
    # Legacy PyTorch key split_dynamics_transport is still supported.
    # =========================================================================
    dynamics_split = _resolve_dynamics_split(config)

    # =========================================================================
    # IAU (Incremental Analysis Update) guard
    # Upstream calls atm_add_tend_anal_incr() when config_IAU_option /= 'off'.
    # PyTorch port: dycore-only, IAU not supported
    # =========================================================================
    iau_option = config.get("config_IAU_option", "off")
    if iau_option != "off":
        raise NotImplementedError(
            f"IAU (config_IAU_option='{iau_option}') is not supported in the PyTorch port. "
            "Only dycore-only mode (config_IAU_option='off') is implemented."
        )

    # Compute dt for each dynamics substep. Upstream accepts
    # config_dynamics_split_steps=0 and executes a zero-iteration
    # DYNAMICS_SUBSTEPS loop, effectively returning the RK-setup state without
    # advancing dynamics. Mirror that do-loop semantics here instead of
    # hard-rejecting the namelist value.
    dt_dynamics = dt / dynamics_split if dynamics_split > 0 else 0.0

    # =========================================================================
    # Run RK integration setup ONCE before the outer loop
    # Upstream call: atm_rk_integration_setup(state, tend, mesh, configs)
    # This initializes tendencies and saves initial state for RK substeps.
    # =========================================================================
    current_state = profiled_call(
        "mpas.rk_integration_setup",
        config,
        atm_rk_integration_setup,
        state_dict,
        mesh_dict,
        config,
    )

    # Optional pre-allocated workspace lifecycle. A single workspace instance
    # is created once per timestep and threaded through all RK stages.
    with profile_section("mpas.workspace_setup", config):
        workspace_obj = config.get("_dycore_workspace")
        use_workspace = bool(
            get_config_value_safe(config, "config_use_workspace", False)
        )
        workspace_obj, use_workspace = _resolve_compile_workspace_policy(
            config,
            workspace_obj,
            use_workspace,
        )
        if workspace_obj is None and use_workspace:
            from mpas_pytorch.dynamics.workspace import (
                DycoreWorkspace,
                workspace_dims_from_state,
            )

            ws_dims = workspace_dims_from_state(current_state)
            if isinstance(current_state.get("u"), torch.Tensor):
                ws_device = current_state["u"].device
            elif isinstance(current_state.get("rho_zz"), torch.Tensor):
                ws_device = current_state["rho_zz"].device
            else:
                ws_device = torch.device("cpu")
            workspace_obj = DycoreWorkspace(dims=ws_dims, device=ws_device, dtype=DTYPE)
    gradient_checkpointing = _coerce_bool(
        get_config_value_safe(config, "gradient_checkpointing", False),
        "gradient_checkpointing",
    )
    rk_checkpoint_enabled = gradient_checkpointing and torch.is_grad_enabled()

    # Extract scalar dimensions/config once per timestep and pass through RK stages.
    with profile_section("mpas.timestep_static_setup", config):
        _coerce_0d_tensor_scalars_inplace(current_state, static_scalar_store)
        config_for_scalars = dict(config)
        config_for_scalars.setdefault("dt", dt)
        timestep_scalars = _build_timestep_scalars(
            current_state,
            config_for_scalars,
            static_scalar_store,
        )
        static_payloads = ensure_static_payload_store(config)
        config["_tridiag_scan_plan_cache"] = static_payloads.vertical_scan_plan_cache()

    with profile_section("mpas.compiled_step_plan", config):
        compiled_step_plan = build_compiled_step_plan(
            config=config,
            state_dict=current_state,
            mesh_dict=mesh_dict,
            dt=dt,
            dt_dynamics=dt_dynamics,
            nAcousticSteps=nAcousticSteps,
            order=order,
            dynamics_split=dynamics_split,
            timestep_scalars=timestep_scalars,
            static_payloads=static_payloads,
            workspace_obj=workspace_obj,
        )
        config["_compiled_step_plan"] = compiled_step_plan

    # =========================================================================
    # Outer split loop (DYNAMICS_SUBSTEPS in Fortran)
    # Upstream loop structure:
    #   dynamics_split = config_split_dynamics_transport
    #   DYNAMICS_SUBSTEPS : do dynamics_substep = 1, dynamics_split
    #      dt_dynamics = dt / real(dynamics_split)
    #      RK_LOOP : do rk_step = 1, config_time_integration_order
    #         call atm_advance_dynamics(...)
    #      end do RK_LOOP
    #   end do DYNAMICS_SUBSTEPS
    # =========================================================================
    for dynamics_substep in range(1, dynamics_split + 1):
        # RK3/RK2 main loop: ALWAYS execute 3 RK stages.
        # Upstream loop structure:
        #   RK3_DYNAMICS : do rk_step = 1, 3
        # The loop count is always 3 regardless of config_time_integration_order.
        # RK2 vs RK3 only affects the weights (rk_timestep, rk_sub_timestep,
        # number_sub_steps), not the number of stages.
        for rk_step in range(1, 4):
            step_config = compiled_step_plan.dynamics_config(dynamics_substep, rk_step)
            nAcousticSteps_this_step = _coerce_int(step_config["ns"], "ns")

            # Execute single RK step (vertical-implicit-coefficients -> acoustic-step loop -> large-step-recovery -> diagnostic-solve).
            # Optional RK-stage checkpointing stores only stage inputs and
            # recomputes intermediates on backward.
            def _run_rk_stage(
                state_in: dict[str, Any],
                rk_step_local: int = rk_step,
                n_acoustic_local: int = nAcousticSteps_this_step,
                step_config_local: Mapping[str, Any] = step_config,
            ) -> dict[str, Any]:
                return execute_rk_step(
                    rk_step=rk_step_local,
                    state_dict=state_in,
                    mesh_dict=mesh_dict,
                    config=step_config_local,
                    nAcousticSteps=n_acoustic_local,
                    strict=strict,
                    _take_state_ownership=True,
                    _preserve_split_average_aliases=dynamics_split > 1,
                )

            step_result = checkpoint_state_dict(
                _run_rk_stage,
                current_state,
                enabled=rk_checkpoint_enabled,
                use_reentrant=False,
                context_fn=_SAVE_NOTHING_CONTEXT if rk_checkpoint_enabled else None,
            )

            # Merge step results into current state for next iteration.
            # execute_rk_step already honors a pure-function dict contract, so
            # re-cloning every tensor here only duplicates the live grad graph
            # without providing additional alias safety.
            with profile_section("mpas.rk_state_merge_alias_refresh", config):
                current_state = dict(step_result)

                # -------------------------------------------------------------
                # Refresh mutable aliases after rebuilding current_state.

                # large-step-recovery recovery updates canonical fields but the short-name
                # aliases created in atm_rk_integration_setup are stale copies
                # from the initial condition. Refresh them every RK step.
                # -------------------------------------------------------------
                if "pressure_p" in current_state:
                    current_state["pp"] = current_state["pressure_p"]
                if "rho_p" in current_state:
                    current_state["rr"] = current_state["rho_p"]
                if "exner" in current_state:
                    current_state["p"] = current_state["exner"]
                if "exner_base" in current_state:
                    current_state["pb"] = current_state["exner_base"]
                if "theta_m" in current_state:
                    current_state["t"] = current_state["theta_m"]
                if "rtheta_p" in current_state:
                    current_state["rt"] = current_state["rtheta_p"]

        # =====================================================================
        # GAP-4: Dynamics substep finish (atm_rk_dynamics_substep_finish)

        # Part B: Accumulate ruAvg_split / wwAvg_split from each substep
        # =====================================================================
        if dynamics_split > 1:
            with profile_section("mpas.dynamics_substep_flux_accumulation", config):
                if dynamics_substep == 1:
                    # First substep: initialize accumulators from this substep's averages
                    ru_avg = current_state.get("ruAvg")
                    if not isinstance(ru_avg, torch.Tensor):
                        ru_avg = torch.zeros_like(current_state["ruAvg_split"])
                    ww_avg = current_state.get("wwAvg")
                    if not isinstance(ww_avg, torch.Tensor):
                        ww_avg = torch.zeros_like(current_state["wwAvg_split"])
                    current_state["ruAvg_split"] = _clone_for_grad(ru_avg)
                    current_state["wwAvg_split"] = _clone_for_grad(ww_avg)
                else:
                    # Subsequent substeps: accumulate
                    ru_avg = current_state.get("ruAvg")
                    if not isinstance(ru_avg, torch.Tensor):
                        ru_avg = torch.zeros_like(current_state["ruAvg_split"])
                    ww_avg = current_state.get("wwAvg")
                    if not isinstance(ww_avg, torch.Tensor):
                        ww_avg = torch.zeros_like(current_state["wwAvg_split"])
                    current_state["ruAvg_split"] = current_state["ruAvg_split"] + ru_avg
                    current_state["wwAvg_split"] = current_state["wwAvg_split"] + ww_avg

            # Part A: State reset for non-final substeps
            # Copy updated state ("time level 2") to starting state ("time level 1")
            # so the next substep begins from the end of this one.
            if dynamics_substep < dynamics_split:
                with profile_section("mpas.dynamics_substep_state_reset", config):
                    if "theta_m" in current_state:
                        cell_end = _get_int_from_dict(
                            current_state,
                            "cellEnd",
                            _get_int_from_dict(current_state, "nCells"),
                        )
                        current_state["theta_m"] = _zero_padding_row(
                            current_state["theta_m"], cell_end
                        )
                    # Save diagnostic arrays for next substep's acoustic step recovery
                    if "ru" in current_state:
                        current_state["ru_save"] = current_state["ru"].clone()
                    if "rw" in current_state:
                        current_state["rw_save"] = current_state["rw"].clone()
                    if "rtheta_p" in current_state:
                        current_state["rtheta_p_save"] = current_state[
                            "rtheta_p"
                        ].clone()
                    if "rho_p" in current_state:
                        current_state["rho_p_save"] = current_state["rho_p"].clone()

                    # Reset _init arrays so the next substep's RK recovery uses
                    # the end-of-substep state as its starting point.
                    # u_rk_save (not u_init) to avoid collision with the 1D mesh field.
                    if "u" in current_state:
                        current_state["u_rk_save"] = current_state["u"].clone()
                    if "w" in current_state:
                        current_state["w_init"] = current_state["w"].clone()
                    if "theta_m" in current_state:
                        current_state["theta_m_init"] = current_state["theta_m"].clone()
                    if "rho_zz" in current_state:
                        current_state["rho_zz_init"] = current_state["rho_zz"].clone()

                # ---------------------------------------------------------
                # Refresh aliases that were set once in atm_rk_integration_setup.

                # During the RK loop, large-step-recovery updates the canonical keys
                # (rho_p, rtheta_p, theta_m, pressure_p) but NOT their
                # short-name aliases (rr, rt, t, pp, p) or save-state
                # aliases (rr_save, theta_m_save).  When dynamics_split > 1
                # the next substep's vertical-implicit-coefficients and dynamics-tendency would read stale values
                # from the pre-timestep initial condition instead of the
                # end-of-this-substep state, causing numerical blowup.

                # The alias mapping mirrors atm_rk_integration_setup
                # .
                # ---------------------------------------------------------
                # dynamics-tendency aliases (short names for diagnostic fields)
                if "pressure_p" in current_state:
                    current_state["pp"] = current_state["pressure_p"]
                if "rho_p" in current_state:
                    current_state["rr"] = current_state["rho_p"]
                # vertical-implicit-coefficients aliases (p=exner, pb=exner_base, t=theta_m, rt=rtheta_p)
                if "exner" in current_state:
                    current_state["p"] = current_state["exner"]
                if "exner_base" in current_state:
                    current_state["pb"] = current_state["exner_base"]
                if "theta_m" in current_state:
                    current_state["t"] = current_state["theta_m"]
                if "rtheta_p" in current_state:
                    current_state["rt"] = current_state["rtheta_p"]
                # Save-state aliases used by dynamics-tendency
                if "rho_p_save" in current_state:
                    current_state["rr_save"] = current_state["rho_p_save"]
                if "theta_m" in current_state:
                    current_state["theta_m_save"] = _clone_for_grad(
                        current_state["theta_m"]
                    )

            # Part C: Final substep averaging and density restore
            if dynamics_substep == dynamics_split:
                with profile_section("mpas.dynamics_substep_final_average", config):
                    # Average the accumulated fluxes over all substeps
                    current_state["ruAvg"] = _average_split_transport_flux(
                        current_state["ruAvg_split"], dynamics_split
                    )
                    current_state["wwAvg"] = _average_split_transport_flux(
                        current_state["wwAvg_split"], dynamics_split
                    )
                    # Keep the scalar-advection alias consistent with averaged ruAvg.
                    # In split transport mode, explicit-scalar-transport/monotonic-scalar-transport consume uhAvg, which should
                    # reflect the substep-averaged horizontal mass flux.
                    current_state["uhAvg"] = _clone_for_grad(current_state["ruAvg"])
                    # For split transport, Fortran restores rho_zz time level 1
                    # (rho_zz_old) from rho_zz_old_split while keeping time level 2
                    # (rho_zz_new) as the current prognostic state.
                    if "rho_zz_old_split" in current_state:
                        current_state["rho_zz_old"] = _clone_for_grad(
                            current_state["rho_zz_old_split"]
                        )

    if bool(config.get("_return_after_dynamics_substeps", False)):
        return cast(dict[str, Any], current_state)

    # =========================================================================
    # GAP-4: Split transport RK3 loop
    # When config_split_dynamics_transport > 1, scalar advection is performed
    # in a separate RK3 loop AFTER all dynamics substeps complete, using the
    # time-averaged ruAvg/wwAvg and the full timestep dt (not dt_dynamics).

    # =========================================================================
    if dynamics_split > 1:
        from mpas_pytorch.dynamics.compile_utils import (
            compile_backend as _compile_backend_transport,
        )
        from mpas_pytorch.dynamics.compile_utils import (
            compile_enabled as _compile_enabled_transport,
        )
        from mpas_pytorch.dynamics.compile_utils import (
            compile_mode as _compile_mode_transport,
        )
        from mpas_pytorch.dynamics.compile_utils import (
            kernel_compile_enabled_for_device as _kernel_compile_enabled_for_device_transport,
        )
        from mpas_pytorch.dynamics.compile_utils import (
            maybe_compile_kernel as _maybe_compile_kernel_transport,
        )

        compile_on_transport = _compile_enabled_transport(config)
        compile_mode_transport = _compile_mode_transport(config)
        compile_backend_transport = _compile_backend_transport(config)
        advect_scalars_requested = get_config_value_safe(config, "advect_scalars", True)
        if advect_scalars_requested:
            for rk_step_t in range(1, 4):  # Always 3 steps
                transport_config = compiled_step_plan.transport_config(rk_step_t)

                # --- Scalar advection dispatch (mirrors execute_rk_step Step 6) ---
                config_monotonic = get_config_value_safe(config, "monotonic", False)
                config_positive_definite = get_config_value_safe(
                    config, "positive_definite", False
                )
                use_limiters = (
                    rk_step_t == 3  # Final stage; earlier stages use simple advection.
                    and (config_monotonic or config_positive_definite)
                )

                # Get dimensions
                nCells_t = _get_int_from_dict(current_state, "nCells")
                nEdges_t = (
                    _get_int_from_dict(current_state, "nEdges")
                    if "nEdges" in current_state
                    else nCells_t * 3
                )
                nVertLevels_t = _get_int_from_dict(current_state, "nVertLevels")
                if isinstance(current_state.get("u"), torch.Tensor):
                    device_t = current_state["u"].device
                elif isinstance(current_state.get("rho_zz"), torch.Tensor):
                    device_t = current_state["rho_zz"].device
                else:
                    device_t = torch.device("cpu")

                # advance_density = 1 for split transport (full rho_zz_int needed)
                advance_density_t = 1
                # Build full-size rho_zz_int for split transport
                rho_zz_int_t = torch.zeros(
                    nCells_t + 1, nVertLevels_t, dtype=DTYPE, device=device_t
                )

                dt_scalar_t = _coerce_float(transport_config["dt"], "transport_dt")

                if use_limiters:
                    # monotonic-scalar-transport monotonic advection (final RK step with limiters)
                    monotonic_scalar_transport_required = [
                        "scalars_old",
                        "scalars_new",
                        "uhAvg",
                        "wwAvg",
                        "rho_zz_old",
                        "rho_zz_new",
                        "fnm",
                        "fnp",
                        "rdnw",
                        "dvEdge",
                        "invAreaCell",
                        "nEdgesOnCell",
                        "nAdvCellsForEdge",
                        "bdyMaskCell",
                        "bdyMaskEdge",
                        "cellsOnEdge",
                        "cellsOnCell",
                        "edgesOnCell",
                        "advCellsForEdge",
                        "edgesOnCell_sign",
                        "adv_coefs",
                        "adv_coefs_3rd",
                        "coef_3rd_order",
                        "num_scalars",
                    ]
                    can_run = all(
                        k in current_state for k in monotonic_scalar_transport_required
                    )
                    if can_run:
                        from mpas_pytorch.dynamics.advance_scalars_mono import (
                            _MONOTONIC_SCALAR_TRANSPORT_REGION_COMPILE_TARGETS,
                            _monotonic_scalar_transport_connectivity_payload,
                            _monotonic_scalar_transport_connectivity_payload_cache_key,
                            atm_advance_scalars_mono_work,
                        )

                        num_scalars_t = _get_int_from_dict(current_state, "num_scalars")
                        coef_3rd_t = _get_float_from_dict(
                            current_state, "coef_3rd_order"
                        )
                        nCellsSolve_t = _get_int_from_dict(
                            current_state, "nCellsSolve", nCells_t
                        )
                        cellStart_t = _get_int_from_dict(current_state, "cellStart", 1)
                        cellEnd_t = _get_int_from_dict(
                            current_state, "cellEnd", nCells_t
                        )
                        edgeStart_t = _get_int_from_dict(current_state, "edgeStart", 1)
                        edgeEnd_t = _get_int_from_dict(
                            current_state, "edgeEnd", nEdges_t
                        )
                        cellSolveStart_t = _get_int_from_dict(
                            current_state, "cellSolveStart", 1
                        )
                        cellSolveEnd_t = _get_int_from_dict(
                            current_state, "cellSolveEnd", nCells_t
                        )
                        nSpecZone_t = _get_int_from_dict(
                            current_state,
                            "nSpecZone",
                            int(config.get("nSpecZone", 2)),
                        )
                        scalar_chunk_size_t = _coerce_int(
                            get_config_value_safe(
                                transport_config,
                                "config_scalar_transport_chunk_size",
                                0,
                            ),
                            "config_scalar_transport_chunk_size",
                        )
                        monotonic_scalar_transport_reduction_slot_tile_width_t = _coerce_int(
                            get_config_value_safe(
                                transport_config,
                                "config_monotonic_scalar_transport_reduction_slot_tile_width",
                                0,
                            ),
                            "config_monotonic_scalar_transport_reduction_slot_tile_width",
                        )
                        monotonic_scalar_transport_streaming_scalar_tile_width_t = _coerce_int(
                            get_config_value_safe(
                                transport_config,
                                "config_monotonic_scalar_transport_streaming_scalar_tile_width",
                                monotonic_scalar_transport_reduction_slot_tile_width_t,
                            ),
                            "config_monotonic_scalar_transport_streaming_scalar_tile_width",
                        )
                        monotonic_scalar_transport_compile_region_names_t: tuple[
                            str, ...
                        ] = ()
                        compile_whole_monotonic_scalar_transport_t = False
                        if compile_on_transport:
                            compile_whole_monotonic_scalar_transport_t = _kernel_compile_enabled_for_device_transport(
                                "monotonic_scalar_transport_advance_scalars_mono_transport",
                                transport_config,
                                device=device_t,
                            )
                            if not compile_whole_monotonic_scalar_transport_t:
                                monotonic_scalar_transport_compile_region_names_t = tuple(
                                    name
                                    for name in sorted(
                                        _MONOTONIC_SCALAR_TRANSPORT_REGION_COMPILE_TARGETS
                                    )
                                    if _kernel_compile_enabled_for_device_transport(
                                        name,
                                        transport_config,
                                        device=device_t,
                                    )
                                )
                        if (
                            compile_on_transport
                            and compile_whole_monotonic_scalar_transport_t
                        ):
                            atm_advance_scalars_mono_work = _maybe_compile_kernel_transport(
                                "monotonic_scalar_transport_advance_scalars_mono_transport",
                                atm_advance_scalars_mono_work,
                                enabled=True,
                                mode=compile_mode_transport,
                                backend=compile_backend_transport,
                                specialization_key=_compiled_kernel_specialization_key(
                                    transport_config,
                                    "monotonic_scalar_transport_advance_scalars_mono_transport",
                                    "rk_step",
                                    rk_step_t,
                                    "dt",
                                    dt_scalar_t,
                                    "advance_density",
                                    advance_density_t,
                                    "nSpecZone",
                                    nSpecZone_t,
                                    "scalar_chunk_size",
                                    scalar_chunk_size_t,
                                    "monotonic_scalar_transport_reduction_slot_tile_width",
                                    monotonic_scalar_transport_reduction_slot_tile_width_t,
                                    "monotonic_scalar_transport_streaming_scalar_tile_width",
                                    monotonic_scalar_transport_streaming_scalar_tile_width_t,
                                ),
                            )

                        monotonic_scalar_transport_connectivity_payload_t = None
                        monotonic_scalar_transport_numeric_plan_t = None
                        if not compile_whole_monotonic_scalar_transport_t:
                            monotonic_scalar_transport_payload_kwargs_t = {
                                "device": device_t,
                                "nCells": nCells_t,
                                "nEdges": nEdges_t,
                                "nCellsSolve": nCellsSolve_t,
                                "cell_start": cellStart_t - 1,
                                "cell_end": cellEnd_t,
                                "edge_start": edgeStart_t - 1,
                                "edge_end": edgeEnd_t,
                                "cell_solve_start": cellSolveStart_t - 1,
                                "cell_solve_end": cellSolveEnd_t,
                                "nSpecZone": nSpecZone_t,
                                "edgesOnCell": current_state["edgesOnCell"],
                                "nEdgesOnCell": current_state["nEdgesOnCell"],
                                "edgesOnCell_sign": current_state["edgesOnCell_sign"],
                                "cellsOnCell": current_state["cellsOnCell"],
                                "cellsOnEdge": current_state["cellsOnEdge"],
                                "bdyMaskCell": current_state["bdyMaskCell"],
                                "nAdvCellsForEdge": current_state["nAdvCellsForEdge"],
                                "advCellsForEdge": current_state["advCellsForEdge"],
                                "adv_coefs": current_state["adv_coefs"],
                                "adv_coefs_3rd": current_state["adv_coefs_3rd"],
                                "dvEdge": current_state["dvEdge"],
                                "invAreaCell": current_state["invAreaCell"],
                            }
                            monotonic_scalar_transport_connectivity_payload_t = _prebound_static_payload(
                                transport_config,
                                "monotonic_scalar_transport_transport",
                                _monotonic_scalar_transport_connectivity_payload_cache_key,
                                _monotonic_scalar_transport_connectivity_payload,
                                monotonic_scalar_transport_payload_kwargs_t,
                            )
                            if (
                                monotonic_scalar_transport_connectivity_payload_t
                                is not None
                            ):
                                monotonic_scalar_transport_numeric_plan_t = _runner_owned_monotonic_scalar_transport_numeric_plan(
                                    static_payloads_at_boundary,
                                    connectivity_payload=monotonic_scalar_transport_connectivity_payload_t,
                                    connectivity_key=_monotonic_scalar_transport_connectivity_payload_cache_key(
                                        **monotonic_scalar_transport_payload_kwargs_t
                                    ),
                                    nCells=nCells_t,
                                    nEdges=nEdges_t,
                                    nVertLevels=nVertLevels_t,
                                    num_scalars=num_scalars_t,
                                    cellStart=cellStart_t,
                                    cellEnd=cellEnd_t,
                                    edgeStart=edgeStart_t,
                                    edgeEnd=edgeEnd_t,
                                    cellSolveStart=cellSolveStart_t,
                                    cellSolveEnd=cellSolveEnd_t,
                                    dt=dt_scalar_t,
                                    soft_limiter_tau=0.01,
                                    scalar_chunk_size=scalar_chunk_size_t,
                                    streaming_scalar_tile_width=(
                                        monotonic_scalar_transport_streaming_scalar_tile_width_t
                                    ),
                                    reduction_slot_tile_width=(
                                        monotonic_scalar_transport_reduction_slot_tile_width_t
                                    ),
                                    advance_density=bool(advance_density_t),
                                )

                        scalar_result = profiled_call(
                            "mpas.monotonic_scalar_transport_advance_scalars_mono_transport",
                            transport_config,
                            atm_advance_scalars_mono_work,
                            nCells=nCells_t,
                            nEdges=nEdges_t,
                            nVertLevels=nVertLevels_t,
                            num_scalars=num_scalars_t,
                            dt=dt_scalar_t,
                            coef_3rd_order=coef_3rd_t,
                            nCellsSolve=nCellsSolve_t,
                            cellStart=cellStart_t,
                            cellEnd=cellEnd_t,
                            edgeStart=edgeStart_t,
                            edgeEnd=edgeEnd_t,
                            cellSolveStart=cellSolveStart_t,
                            cellSolveEnd=cellSolveEnd_t,
                            fnm=current_state["fnm"],
                            fnp=current_state["fnp"],
                            rdnw=current_state["rdnw"],
                            dvEdge=current_state["dvEdge"],
                            invAreaCell=current_state["invAreaCell"],
                            nEdgesOnCell=current_state["nEdgesOnCell"],
                            nAdvCellsForEdge=current_state["nAdvCellsForEdge"],
                            bdyMaskCell=current_state["bdyMaskCell"],
                            bdyMaskEdge=current_state["bdyMaskEdge"],
                            cellsOnEdge=current_state["cellsOnEdge"],
                            cellsOnCell=current_state["cellsOnCell"],
                            edgesOnCell=current_state["edgesOnCell"],
                            advCellsForEdge=current_state["advCellsForEdge"],
                            edgesOnCell_sign=current_state["edgesOnCell_sign"],
                            adv_coefs=current_state["adv_coefs"],
                            adv_coefs_3rd=current_state["adv_coefs_3rd"],
                            uhAvg=current_state["uhAvg"],
                            wwAvg=current_state["wwAvg"],
                            rho_zz_old=current_state["rho_zz_old"],
                            rho_zz_new=current_state["rho_zz_new"],
                            scalar_old=_clone_for_grad(
                                _mapping_tensor_or_zeros(
                                    current_state,
                                    "scalar_old",
                                    (nCells_t + 1, nVertLevels_t),
                                    device=device_t,
                                )
                            ),
                            scalar_new=_clone_for_grad(
                                _mapping_tensor_or_zeros(
                                    current_state,
                                    "scalar_new",
                                    (nCells_t + 1, nVertLevels_t),
                                    device=device_t,
                                )
                            ),
                            s_max=_clone_for_grad(
                                _mapping_tensor_or_zeros(
                                    current_state,
                                    "s_max",
                                    (nCells_t + 1, nVertLevels_t),
                                    device=device_t,
                                )
                            ),
                            s_min=_clone_for_grad(
                                _mapping_tensor_or_zeros(
                                    current_state,
                                    "s_min",
                                    (nCells_t + 1, nVertLevels_t),
                                    device=device_t,
                                )
                            ),
                            wdtn=_clone_for_grad(
                                _mapping_tensor_or_zeros(
                                    current_state,
                                    "wdtn",
                                    (nCells_t + 1, nVertLevels_t + 1),
                                    device=device_t,
                                )
                            ),
                            flux_arr=_clone_for_grad(
                                _mapping_tensor_or_zeros(
                                    current_state,
                                    "flux_arr",
                                    (nEdges_t + 1, nVertLevels_t),
                                    device=device_t,
                                )
                            ),
                            flux_upwind_tmp=_clone_for_grad(
                                _mapping_tensor_or_zeros(
                                    current_state,
                                    "flux_upwind_tmp",
                                    (nEdges_t + 1, nVertLevels_t),
                                    device=device_t,
                                )
                            ),
                            flux_tmp=_clone_for_grad(
                                _mapping_tensor_or_zeros(
                                    current_state,
                                    "flux_tmp",
                                    (nEdges_t + 1, nVertLevels_t),
                                    device=device_t,
                                )
                            ),
                            scalar_tend=_clone_for_grad(
                                _mapping_tensor_or_zeros(
                                    current_state,
                                    "scalar_tend",
                                    (nCells_t + 1, nVertLevels_t, num_scalars_t),
                                    device=device_t,
                                )
                            ),
                            scalars_old=_clone_for_grad(current_state["scalars_old"]),
                            scalars_new=_clone_for_grad(current_state["scalars_new"]),
                            scale_arr=_clone_for_grad(
                                _mapping_tensor_or_zeros(
                                    current_state,
                                    "scale_arr",
                                    (nCells_t + 1, 2, nVertLevels_t),
                                    device=device_t,
                                )
                            ),
                            rho_zz_int=rho_zz_int_t,
                            nSpecZone=nSpecZone_t,
                            scalar_chunk_size=scalar_chunk_size_t,
                            _compile_region_names=monotonic_scalar_transport_compile_region_names_t,
                            _compile_mode=compile_mode_transport,
                            _compile_backend=compile_backend_transport,
                            _connectivity_payload=monotonic_scalar_transport_connectivity_payload_t,
                            _profile_sections=bool(
                                transport_config.get("config_profile_sections", False)
                            )
                            and not compile_whole_monotonic_scalar_transport_t,
                            _reduction_slot_tile_width=monotonic_scalar_transport_reduction_slot_tile_width_t,
                            _streaming_scalar_tile_width=monotonic_scalar_transport_streaming_scalar_tile_width_t,
                            _numeric_plan=monotonic_scalar_transport_numeric_plan_t,
                        )
                        if compile_whole_monotonic_scalar_transport_t:
                            scalar_result = _clone_compiled_output_mapping(
                                scalar_result
                            )
                        current_state.update(scalar_result)
                        if "scalars_new" in current_state:
                            current_state["scalars"] = _clone_for_grad(
                                current_state["scalars_new"]
                            )
                        if "rho_zz_new" in current_state:
                            current_state["rho_zz"] = _clone_for_grad(
                                current_state["rho_zz_new"]
                            )
                    elif strict:
                        missing = [
                            k
                            for k in monotonic_scalar_transport_required
                            if k not in current_state
                        ]
                        raise ValueError(
                            f"Split transport: cannot run monotonic-scalar-transport monotonic advection, missing: {missing}"
                        )

                else:
                    # explicit-scalar-transport explicit advection
                    explicit_scalar_transport_required = [
                        "scalar_old",
                        "scalar_new",
                        "uhAvg",
                        "wwAvg",
                        "rho_zz_old",
                        "rho_zz_new",
                        "fnm",
                        "fnp",
                        "rdnw",
                        "dvEdge",
                        "invAreaCell",
                        "nEdgesOnCell",
                        "nAdvCellsForEdge",
                        "bdyMaskCell",
                        "bdyMaskEdge",
                        "cellsOnEdge",
                        "edgesOnCell",
                        "advCellsForEdge",
                        "edgesOnCell_sign",
                        "adv_coefs",
                        "adv_coefs_3rd",
                        "coef_3rd_order",
                        "num_scalars",
                    ]
                    can_run = all(
                        k in current_state for k in explicit_scalar_transport_required
                    )
                    if can_run:
                        from mpas_pytorch.dynamics.advance_scalars import (
                            _EXPLICIT_SCALAR_TRANSPORT_REGION_COMPILE_TARGETS,
                            _explicit_scalar_transport_connectivity_payload,
                            _explicit_scalar_transport_connectivity_payload_cache_key,
                            _explicit_scalar_transport_density_time_weights,
                            atm_advance_scalars_work,
                            build_explicit_scalar_transport_numeric_plan,
                        )

                        num_scalars_t = _get_int_from_dict(current_state, "num_scalars")
                        maxEdges_t = _get_int_from_dict(current_state, "maxEdges", 10)
                        nAdvCellsMax_t = _get_int_from_dict(
                            current_state, "nAdvCellsMax", 15
                        )
                        coef_3rd_t = _get_float_from_dict(
                            current_state, "coef_3rd_order"
                        )
                        config_time_int_order_t = _coerce_int(
                            get_config_value_safe(
                                transport_config,
                                "config_time_integration_order",
                                order,
                            ),
                            "config_time_integration_order",
                        )
                        explicit_scalar_transport_compile_region_names_t: tuple[
                            str, ...
                        ] = ()
                        compile_whole_explicit_scalar_transport_t = False
                        if compile_on_transport:
                            compile_whole_explicit_scalar_transport_t = _kernel_compile_enabled_for_device_transport(
                                "explicit_scalar_transport_advance_scalars_transport",
                                transport_config,
                                device=device_t,
                            )
                            if compile_whole_explicit_scalar_transport_t:
                                atm_advance_scalars_work = _maybe_compile_kernel_transport(
                                    "explicit_scalar_transport_advance_scalars_transport",
                                    atm_advance_scalars_work,
                                    enabled=True,
                                    mode=compile_mode_transport,
                                    backend=compile_backend_transport,
                                    specialization_key=_compiled_kernel_specialization_key(
                                        transport_config,
                                        "explicit_scalar_transport_advance_scalars_transport",
                                        "rk_step",
                                        rk_step_t,
                                        "dt",
                                        dt_scalar_t,
                                        "order",
                                        config_time_int_order_t,
                                        "advance_density",
                                        advance_density_t,
                                    ),
                                )
                            else:
                                explicit_scalar_transport_compile_region_names_t = tuple(
                                    name
                                    for name in sorted(
                                        _EXPLICIT_SCALAR_TRANSPORT_REGION_COMPILE_TARGETS
                                    )
                                    if _kernel_compile_enabled_for_device_transport(
                                        name,
                                        transport_config,
                                        device=device_t,
                                    )
                                )
                        edgeStart_t = _get_int_from_dict(current_state, "edgeStart", 1)
                        edgeEnd_t = _get_int_from_dict(
                            current_state, "edgeEnd", nEdges_t
                        )
                        cellSolveStart_t = _get_int_from_dict(
                            current_state, "cellSolveStart", 1
                        )
                        cellSolveEnd_t = _get_int_from_dict(
                            current_state, "cellSolveEnd", nCells_t
                        )
                        explicit_scalar_transport_connectivity_payload_t = None
                        if not compile_whole_explicit_scalar_transport_t:
                            explicit_scalar_transport_payload_kwargs_t = {
                                "device": device_t,
                                "nCells": nCells_t,
                                "nEdges": nEdges_t,
                                "maxEdges": maxEdges_t,
                                "nAdvCellsMax": nAdvCellsMax_t,
                                "edge_start": edgeStart_t - 1,
                                "edge_end": edgeEnd_t,
                                "cell_solve_start": cellSolveStart_t - 1,
                                "cell_solve_end": cellSolveEnd_t,
                                "edgesOnCell": current_state["edgesOnCell"],
                                "nEdgesOnCell": current_state["nEdgesOnCell"],
                                "edgesOnCell_sign": current_state["edgesOnCell_sign"],
                                "nAdvCellsForEdge": current_state["nAdvCellsForEdge"],
                                "advCellsForEdge": current_state["advCellsForEdge"],
                                "adv_coefs": current_state["adv_coefs"],
                                "adv_coefs_3rd": current_state["adv_coefs_3rd"],
                            }
                            explicit_scalar_transport_connectivity_payload_t = _prebound_static_payload(
                                transport_config,
                                "explicit_scalar_transport_transport",
                                _explicit_scalar_transport_connectivity_payload_cache_key,
                                _explicit_scalar_transport_connectivity_payload,
                                explicit_scalar_transport_payload_kwargs_t,
                            )
                        if "scalars_old" in current_state:
                            scalar_old_t = current_state["scalars_old"]
                        else:
                            scalar_old_t = current_state["scalar_old"]
                        scalar_old_t = _ensure_scalars_rank3(
                            scalar_old_t, num_scalars_t, "scalar_old"
                        )
                        if "scalars_new" in current_state:
                            scalar_new_t = current_state["scalars_new"]
                        else:
                            scalar_new_t = current_state["scalar_new"]
                        scalar_new_t = _ensure_scalars_rank3(
                            scalar_new_t, num_scalars_t, "scalar_new"
                        )
                        scalar_tend_save_t = current_state.get(
                            "scalar_tend_save",
                            torch.zeros_like(scalar_new_t),
                        )
                        scalar_tend_save_t = _ensure_scalars_rank3(
                            scalar_tend_save_t, num_scalars_t, "scalar_tend_save"
                        )
                        do_physics_t = _coerce_bool(
                            current_state.get(
                                "do_physics",
                                get_config_value_safe(
                                    transport_config,
                                    "do_physics",
                                    False,
                                ),
                            ),
                            "do_physics",
                        )

                        explicit_scalar_transport_numeric_plan_t = None
                        if explicit_scalar_transport_connectivity_payload_t is not None:
                            (
                                explicit_scalar_transport_weight_old_t,
                                explicit_scalar_transport_weight_new_t,
                            ) = _explicit_scalar_transport_density_time_weights(
                                advance_density=advance_density_t,
                                rk_step=rk_step_t,
                                config_time_integration_order=config_time_int_order_t,
                            )
                            candidate_explicit_scalar_transport_plan_t = build_explicit_scalar_transport_numeric_plan(
                                connectivity_payload=explicit_scalar_transport_connectivity_payload_t,
                                inv_area_cells=current_state["invAreaCell"][
                                    cellSolveStart_t - 1 : cellSolveEnd_t
                                ],
                                fnm=current_state["fnm"],
                                fnp=current_state["fnp"],
                                rdnw=current_state["rdnw"],
                                dt=dt_scalar_t,
                                coef_3rd_order=coef_3rd_t,
                                weight_time_old=explicit_scalar_transport_weight_old_t,
                                weight_time_new=explicit_scalar_transport_weight_new_t,
                                nVertLevels=nVertLevels_t,
                                num_scalars=num_scalars_t,
                                do_physics=do_physics_t,
                                stage_key=(
                                    "split",
                                    "rk_step",
                                    rk_step_t,
                                    "order",
                                    config_time_int_order_t,
                                    "advance_density",
                                    advance_density_t,
                                ),
                            )
                            explicit_scalar_transport_numeric_plan_t = static_payloads_at_boundary.get_or_build_numeric_plan(
                                "explicit_scalar_transport_numeric_composition",
                                candidate_explicit_scalar_transport_plan_t.key,
                                lambda plan=candidate_explicit_scalar_transport_plan_t: (  # type: ignore[misc]
                                    plan
                                ),
                            )
                            if explicit_scalar_transport_compile_region_names_t:
                                explicit_scalar_transport_numeric_plan_t = _runner_owned_compiled_explicit_scalar_transport_numeric_plan(
                                    static_payloads_at_boundary,
                                    explicit_scalar_transport_numeric_plan_t,
                                    compile_region_names=explicit_scalar_transport_compile_region_names_t,
                                    compile_kernel=_maybe_compile_kernel_transport,
                                    mode=compile_mode_transport,
                                    backend=compile_backend_transport,
                                )
                                explicit_scalar_transport_compile_region_names_t = ()

                        scalar_result = profiled_call(
                            "mpas.explicit_scalar_transport_advance_scalars_transport",
                            transport_config,
                            atm_advance_scalars_work,
                            nCells=nCells_t,
                            nEdges=nEdges_t,
                            nVertLevels=nVertLevels_t,
                            num_scalars=num_scalars_t,
                            maxEdges=maxEdges_t,
                            nAdvCellsMax=nAdvCellsMax_t,
                            dt=dt_scalar_t,
                            coef_3rd_order=coef_3rd_t,
                            edgeStart=edgeStart_t,
                            edgeEnd=edgeEnd_t,
                            cellSolveStart=cellSolveStart_t,
                            cellSolveEnd=cellSolveEnd_t,
                            rk_step=rk_step_t,
                            config_time_integration_order=config_time_int_order_t,
                            advance_density=advance_density_t,
                            scalar_old=scalar_old_t,
                            scalar_new=scalar_new_t,
                            scalar_tend_save=scalar_tend_save_t,
                            rho_zz_old=current_state["rho_zz_old"],
                            rho_zz_new=current_state["rho_zz_new"],
                            uhAvg=current_state["uhAvg"],
                            wwAvg=current_state["wwAvg"],
                            dvEdge=current_state["dvEdge"],
                            invAreaCell=current_state["invAreaCell"],
                            fnm=current_state["fnm"],
                            fnp=current_state["fnp"],
                            rdnw=current_state["rdnw"],
                            cellsOnEdge=current_state["cellsOnEdge"],
                            edgesOnCell=current_state["edgesOnCell"],
                            nEdgesOnCell=current_state["nEdgesOnCell"],
                            edgesOnCell_sign=current_state["edgesOnCell_sign"],
                            bdyMaskCell=current_state["bdyMaskCell"],
                            bdyMaskEdge=current_state["bdyMaskEdge"],
                            nAdvCellsForEdge=current_state["nAdvCellsForEdge"],
                            advCellsForEdge=current_state["advCellsForEdge"],
                            adv_coefs=current_state["adv_coefs"],
                            adv_coefs_3rd=current_state["adv_coefs_3rd"],
                            horiz_flux_arr=_mapping_tensor_or_zeros(
                                current_state,
                                "horiz_flux_arr",
                                (nEdges_t + 1, nVertLevels_t, num_scalars_t),
                                device=device_t,
                            ),
                            do_physics=do_physics_t,
                            _compile_region_names=explicit_scalar_transport_compile_region_names_t,
                            _compile_mode=compile_mode_transport,
                            _compile_backend=compile_backend_transport,
                            _connectivity_payload=explicit_scalar_transport_connectivity_payload_t,
                            _numeric_plan=explicit_scalar_transport_numeric_plan_t,
                        )
                        if compile_whole_explicit_scalar_transport_t:
                            scalar_result = _clone_compiled_output_mapping(
                                scalar_result
                            )
                        _alias_scalar_outputs(scalar_result)
                        current_state.update(scalar_result)
                        if "scalars_new" in current_state:
                            current_state["scalars"] = _clone_for_grad(
                                current_state["scalars_new"]
                            )
                        if "rho_zz_new" in current_state:
                            current_state["rho_zz"] = _clone_for_grad(
                                current_state["rho_zz_new"]
                            )
                    elif strict:
                        missing = [
                            k
                            for k in explicit_scalar_transport_required
                            if k not in current_state
                        ]
                        raise ValueError(
                            f"Split transport: cannot run explicit-scalar-transport explicit advection, missing: {missing}"
                        )

    # =========================================================================
    # Post-timestep vector reconstruction (mpas_reconstruct_2d)

    # Reconstruct cell-centered zonal/meridional velocity from the final
    # edge-normal velocity u. dynamics-tendency uses ur_cell/vr_cell on the next timestep,
    # so this refresh must happen after RK integration completes.
    # =========================================================================
    reconstruct_required = [
        "u",
        "coeffs_reconstruct",
        "edgesOnCell",
        "nEdgesOnCell",
        "latCell",
        "lonCell",
    ]
    if all(k in current_state for k in reconstruct_required):
        from mpas_pytorch.dynamics.vector_reconstruction import (
            CELL_PADDING_OFFSET_KEY,
            coeffs_reconstruct_all_zero,
            mpas_reconstruct_2d,
        )

        nCells_reconstruct = _get_int_from_dict(current_state, "nCells")
        nEdges_reconstruct = _get_int_from_dict(current_state, "nEdges")
        nVertLevels_reconstruct = _get_int_from_dict(current_state, "nVertLevels")
        on_a_sphere = _coerce_bool(
            get_config_value_safe(config, "on_a_sphere", True),
            "on_a_sphere",
        )
        coeffs_reconstruct = current_state["coeffs_reconstruct"]
        coeffs_all_zero = coeffs_reconstruct_all_zero(
            current_state,
            coeffs_reconstruct,
            nCells_reconstruct,
        )

        if (
            coeffs_all_zero
            and "ur_cell" in current_state
            and "vr_cell" in current_state
        ):
            # Some tiny-mesh init files carry all-zero reconstruction coefficients.
            # In that case, preserve the existing cell-centered winds instead of
            # overwriting them with synthetic zeros.
            warnings.warn(
                "coeffs_reconstruct is all zeros; preserving existing ur_cell/vr_cell "
                "for uReconstruct diagnostics.",
                RuntimeWarning,
            )
            current_state["uReconstructZonal"] = current_state["ur_cell"].clone()
            current_state["uReconstructMeridional"] = current_state["vr_cell"].clone()
        else:
            reconstruct_result = profiled_call(
                "mpas.vector_reconstruction",
                config,
                mpas_reconstruct_2d,
                nCells=nCells_reconstruct,
                nEdges=nEdges_reconstruct,
                nVertLevels=nVertLevels_reconstruct,
                u=current_state["u"],
                coeffs_reconstruct=coeffs_reconstruct,
                edgesOnCell=current_state["edgesOnCell"],
                nEdgesOnCell=current_state["nEdgesOnCell"],
                latCell=current_state["latCell"],
                lonCell=current_state["lonCell"],
                on_a_sphere=on_a_sphere,
                cell_padding_offset=current_state.get(CELL_PADDING_OFFSET_KEY),
            )
            current_state["uReconstructX"] = reconstruct_result["uReconstructX"]
            current_state["uReconstructY"] = reconstruct_result["uReconstructY"]
            current_state["uReconstructZ"] = reconstruct_result["uReconstructZ"]
            current_state["uReconstructZonal"] = reconstruct_result["uReconstructZonal"]
            current_state["uReconstructMeridional"] = reconstruct_result[
                "uReconstructMeridional"
            ]
            current_state["ur_cell"] = reconstruct_result["ur_cell"]
            current_state["vr_cell"] = reconstruct_result["vr_cell"]

    # =========================================================================
    # BOUNDARY SCALAR SETTING (post-step scalar boundary setting)

    # Called AFTER all RK dynamics substeps complete.
    # Directly sets scalar values in specified zone from driving data.
    # =========================================================================
    config_apply_lbcs = config.get("config_apply_lbcs", False)
    config_apply_lbcs = _coerce_bool(config_apply_lbcs, "config_apply_lbcs")

    if config_apply_lbcs:
        from mpas_pytorch.boundary.set_scalars import (  # type: ignore[import-not-found]
            atm_bdy_set_scalars,
        )

        # Check for required fields
        scalars_driving_key = (
            "scalars_driving" if "scalars_driving" in current_state else "scalars_bdy"
        )
        has_set_scalars_data = (
            scalars_driving_key in current_state
            and "bdyMaskCell" in current_state
            and "scalars" in current_state
        )

        if has_set_scalars_data:
            nRelaxZone = current_state.get("nRelaxZone", config.get("nRelaxZone", 10))
            nRelaxZone = _coerce_int(nRelaxZone, "nRelaxZone")

            # Get dimensions from scalars shape: (nCells+1, nVertLevels, nScalars)
            scalars_shape = current_state["scalars"].shape
            nCells_scalars = scalars_shape[0] - 1  # Shape is (nCells+1, ...)
            nScalars = scalars_shape[2] if len(scalars_shape) > 2 else 1

            set_scalars_result = atm_bdy_set_scalars(
                scalars=current_state["scalars"],
                scalars_bdy=current_state[scalars_driving_key],
                bdyMaskCell=current_state["bdyMaskCell"],
                nRelaxZone=nRelaxZone,
                nCells=nCells_scalars,
                nScalars=nScalars,
            )
            current_state["scalars"] = set_scalars_result["scalars"]
        elif strict:
            missing_fields = []
            if scalars_driving_key not in current_state:
                missing_fields.append("scalars_driving or scalars_bdy")
            if "bdyMaskCell" not in current_state:
                missing_fields.append("bdyMaskCell")
            if "scalars" not in current_state:
                missing_fields.append("scalars")
            raise ValueError(
                f"config_apply_lbcs=True but required fields missing for post-step scalar boundary setting set_scalars: {missing_fields}. "
                f"Either provide boundary data or set config_apply_lbcs=False."
            )
        else:
            warnings.warn(
                "Skipping boundary scalar setting (post-step scalar boundary setting): missing required fields. "
                "Regional boundary conditions will not be applied to scalars.",
                RuntimeWarning,
            )

    # =========================================================================
    # OPTIONAL: Timestep diagnostic summary
    # =========================================================================
    print_timestep_summary = config.get("print_timestep_summary", False)
    if print_timestep_summary and not is_grad_mode():
        summary = summarize_timestep(
            state_dict=current_state,
            nCells=_get_int_from_dict(current_state, "nCells"),
            nEdges=_get_int_from_dict(
                current_state,
                "nEdges",
                _get_int_from_dict(current_state, "nCells") * 3,
            ),
            nVertLevels=_get_int_from_dict(current_state, "nVertLevels"),
            num_scalars=config.get("num_scalars", 0),
            print_velocities=True,
            print_scalars=config.get("print_scalar_summary", False),
        )
        if summary["has_nan"]:
            warnings.warn(
                "NaN detected in state after RK integration — check summary "
                f"fields: {[k for k, v in summary.items() if k.endswith('_has_nan') and v]}",
                RuntimeWarning,
            )

    # Publish vertical-implicit-coefficients's padded compatibility fields exactly once.  Internally the
    # complete timestep carries only active-row patches, avoiding two full-field
    # packs and the padded RK3 handoff.  Private patches do not escape the public
    # timestep result.
    from mpas_pytorch.dynamics.vert_imp_coefs import (
        ACTIVE_VERT_IMP_COEF_KEYS,
        pack_vert_imp_coefs_public,
    )

    compact_vertical_implicit_coefficients_keys = (
        "cofrz",
        "_vertical_implicit_coefficients_active_cell_start",
        "_vertical_implicit_coefficients_active_cell_end",
        *ACTIVE_VERT_IMP_COEF_KEYS,
    )
    if all(key in current_state for key in compact_vertical_implicit_coefficients_keys):
        current_state.update(
            pack_vert_imp_coefs_public(
                current_state,
                nCells=timestep_scalars.nCells,
                nVertLevels=timestep_scalars.nVertLevels,
            )
        )
        for key in compact_vertical_implicit_coefficients_keys:
            if key != "cofrz":
                current_state.pop(key, None)

    # dynamics-tendency consumes active topology rows only.  Keep diagnostic-solve diagnostics compact
    # throughout RK orchestration and restore MPAS sentinel rows at the true
    # public/checkpoint boundary.
    from mpas_pytorch.dynamics.solve_diagnostics import (
        pack_diagnostic_solve_diagnostics_public,
    )

    diagnostic_solve_public_keys = (
        "h_edge",
        "v",
        "divergence",
        "vorticity",
        "ke",
        "pv_edge",
    )
    if all(key in current_state for key in diagnostic_solve_public_keys):
        packed_diagnostic_solve = pack_diagnostic_solve_diagnostics_public(
            current_state,
            nCells=timestep_scalars.nCells,
            nEdges=timestep_scalars.nEdges,
            nVertices=timestep_scalars.nVertices,
        )
        current_state.update(packed_diagnostic_solve)
        if "rho_edge" in current_state:
            current_state["rho_edge"] = packed_diagnostic_solve["h_edge"]

    return cast(dict[str, Any], current_state)


def atm_srk3(
    state: Mapping[str, Any],
    mesh: Mapping[str, Any],
    config: Mapping[str, Any],
    dt: float,
    strict: bool = True,
    differentiable: bool = False,
) -> dict[str, Any]:
    """Dual-mode wrapper for split-explicit RK3 integration."""
    if differentiable:
        from mpas_pytorch.dynamics.compile_utils import (
            compile_backend,
            compile_enabled,
            compile_mode,
            validate_differentiable_compile_request,
        )
        from mpas_pytorch.dynamics.dyn_tend import (
            build_dynamics_tendency_ad_compile_plan,
            resolve_dynamics_tendency_ad_compile_mode,
            use_dynamics_tendency_ad_compile_plan,
        )

        local_config = dict(config or {})
        tensor_device = next(
            (
                value.device
                for value in state.values()
                if isinstance(value, torch.Tensor)
            ),
            torch.device("cpu"),
        )
        compile_requested = compile_enabled(local_config)
        if compile_requested:
            validate_differentiable_compile_request(
                local_config,
                device=tensor_device,
                values=state,
            )
        effective_ad_mode = resolve_dynamics_tendency_ad_compile_mode(
            local_config,
            device=tensor_device,
        )
        ad_plan = build_dynamics_tendency_ad_compile_plan(
            mode=effective_ad_mode,
            device=tensor_device,
            compile_mode=compile_mode(local_config),
            compile_backend=compile_backend(local_config),
        )
        with differentiable_mode(), use_dynamics_tendency_ad_compile_plan(ad_plan):
            return _atm_srk3_impl(state, mesh, local_config, dt, strict=strict)

    from mpas_pytorch.dynamics.compile_utils import compile_enabled

    local_config = dict(config or {})
    if compile_enabled(local_config):
        with torch.no_grad():
            return _atm_srk3_impl(state, mesh, config, dt, strict=strict)

    with torch.inference_mode():
        return _atm_srk3_impl(state, mesh, config, dt, strict=strict)


def _dynamics_tendency_ordered_ad_transpose_required(
    config: Mapping[str, Any],
    *dynamic_tensors: torch.Tensor,
) -> bool:
    """Return whether the production dynamics-tendency call needs ordered transpose maps."""
    ad_mode = str(config.get("config_dynamics_tendency_ad_compile_plan", "auto"))
    ad_mode = ad_mode.strip().lower().replace("-", "_")
    return (
        ad_mode != "native_autograd"
        and bool(
            config.get("_dynamics_tendency_normal_velocity_tendency_ordered_vjp", True)
        )
        and any(_tensor_carries_autodiff(tensor) for tensor in dynamic_tensors)
    )


def _dynamics_tendency_edge_advective_ad_transpose_required(
    config: Mapping[str, Any],
    *dynamic_tensors: torch.Tensor,
) -> bool:
    """Return whether vertical-velocity-horizontal-advection/theta-horizontal-advection need their output-owned transpose maps."""
    ad_mode = str(config.get("config_dynamics_tendency_ad_compile_plan", "auto"))
    ad_mode = ad_mode.strip().lower().replace("-", "_")
    return (
        ad_mode != "native_autograd"
        and bool(config.get("_dynamics_tendency_edge_advective_ordered_vjp", True))
        and any(_tensor_carries_autodiff(tensor) for tensor in dynamic_tensors)
    )


def _resolve_pre_dynamics_tendency_diagnostic_sources(
    result: Mapping[str, Any],
    state_dict: Mapping[str, Any],
) -> tuple[object, object]:
    """Resolve the exact prognostic tensors that pre-dynamics-tendency diagnostics describe."""
    return (
        result.get("u", state_dict.get("u")),
        result.get("rho_zz", state_dict.get("rho_zz")),
    )


def execute_rk_step(
    rk_step: int,
    state_dict: dict[str, Any],
    mesh_dict: dict[str, Any],
    config: Mapping[str, Any],
    nAcousticSteps: int,
    strict: bool = True,
    _take_state_ownership: bool = False,
    _preserve_split_average_aliases: bool = False,
) -> dict[str, Any]:
    """
    Execute a single Runge-Kutta substep.

    This function orchestrates the following sequence (matching Fortran order):
    1. Compute vertical implicit coefficients (vertical-implicit-coefficients)
       - Always computed at rk_step==1
       - Recomputed at rk_step==2 for RK3 (time_integration_order==3)
       - Reused from cache at rk_step==3 (coefficients depend on dts which varies per stage)

    2. Compute dynamic tendencies (dynamics-tendency) - BEFORE acoustic loop
       Note: small-step-perturbation (set_smlstep_pert_variables) goes here but is not yet implemented
    3. Loop over acoustic substeps (acoustic-step) - uses tend_* from dynamics-tendency
    4. Recover large step variables (large-step-recovery)
    5. Compute solve diagnostics (diagnostic-solve) - computes diagnostics for next RK step
    6. Scalar advection (explicit-scalar-transport/monotonic-scalar-transport)

    Note: dynamics-tendency uses diagnostics (divergence, vorticity, ke, pv_edge) computed
    by diagnostic-solve in the PREVIOUS RK step (or from initialization for rk_step=1).

    vertical-implicit-coefficients Coefficient Caching Behavior:
        The vertical implicit coefficients (cofrz, cofwr, cofwz, coftz, cofwt,
        a_tri, alpha_tri, gamma_tri) depend on dts (small timestep), which can
        differ between RK stages. Fortran explicitly recomputes these at:
        - rk_step == 1: Always (initial computation)
        - rk_step == 2: Only for RK3 (time_integration_order == 3)
        - rk_step == 3: Reuses cached from step 2 (dts unchanged)

        When recomputation is required but inputs are missing:
        - strict=True (default): Raises ValueError because using cached
          coefficients when dts has changed violates the accuracy contract.
          The coefficients encode dts-dependent physics; stale values cause
          numerical divergence from upstream MPAS.
        - strict=False: Warns and uses cached coefficients. This is acceptable
          ONLY for testing individual kernels in isolation where we provide
          pre-computed coefficients and don't need vertical-implicit-coefficients recomputation.

    Args:
        rk_step: RK substep number (1, 2, or 3)
        state_dict: Dictionary of state tensors from checkpoint
        mesh_dict: Dictionary of mesh tensors from checkpoint
        config: Configuration parameters (dts, epssm, time_integration_order, etc.)
        nAcousticSteps: Number of acoustic substeps
        strict: If True (default), raise ValueError when required kernels cannot
                run due to missing inputs. If False, skip kernels with warning
                and continue with incomplete results. Set strict=False only for
                testing individual kernels in isolation.

    Returns:
        Updated state dictionary with vertical coefficients, acoustic outputs,
        and recovered state variables (from large-step-recovery)

    Raises:
        ValueError: If strict=True and required kernel inputs are missing.
                    This includes vertical-implicit-coefficients coefficient recomputation when inputs
                    are missing at rk_step==1 or rk_step==2 (for RK3).
    """
    from mpas_pytorch.constants import CP, GRAVITY, RGAS
    from mpas_pytorch.dynamics.acoustic_step import (
        _ACOUSTIC_STEP_REGION_COMPILE_TARGETS,
        _acoustic_step_connectivity_payload,
        _acoustic_step_connectivity_payload_cache_key,
        atm_advance_acoustic_step_work,
        build_acoustic_step_acoustic_solve_plan,
    )
    from mpas_pytorch.dynamics.compile_utils import (
        compile_backend as _compile_backend,
    )
    from mpas_pytorch.dynamics.compile_utils import (
        compile_enabled as _compile_enabled,
    )
    from mpas_pytorch.dynamics.compile_utils import (
        compile_mode as _compile_mode,
    )
    from mpas_pytorch.dynamics.compile_utils import (
        kernel_compile_enabled_for_device as _kernel_compile_enabled_for_device,
    )
    from mpas_pytorch.dynamics.compile_utils import (
        kernel_compiled_reverse_ad_enabled_for_device as _kernel_compiled_reverse_ad_enabled_for_device,
    )
    from mpas_pytorch.dynamics.compile_utils import (
        maybe_compile_kernel as _maybe_compile_kernel,
    )
    from mpas_pytorch.dynamics.divergence_damping import (
        _DIVERGENCE_DAMPING_REGION_COMPILE_TARGETS,
        _divergence_connectivity_payload,
        _divergence_connectivity_payload_cache_key,
        atm_divergence_damping_3d,
    )
    from mpas_pytorch.dynamics.moist_coefficients import (
        _moist_connectivity_payload,
        _moist_connectivity_payload_cache_key,
        atm_compute_moist_coefficients,
    )
    from mpas_pytorch.dynamics.recover_large_step import (
        _LARGE_STEP_RECOVERY_REGION_COMPILE_TARGETS,
        _large_step_recovery_connectivity_payload,
        _large_step_recovery_connectivity_payload_cache_key,
        atm_recover_large_step_variables_work,
    )
    from mpas_pytorch.dynamics.solve_diagnostics import (
        _DIAGNOSTIC_SOLVE_REGION_COMPILE_TARGETS,
        DiagnosticSolveNumericPlan,
        _diagnostic_solve_connectivity_payload,
        _diagnostic_solve_connectivity_payload_cache_key,
        _diagnostic_solve_diagnostics_are_current,
        _mark_diagnostic_solve_diagnostics_current,
        atm_compute_solve_diagnostics_work,
        build_diagnostic_solve_numeric_plan,
        compile_diagnostic_solve_numeric_plan,
    )
    from mpas_pytorch.dynamics.vert_imp_coefs import (
        ACTIVE_VERT_IMP_COEF_KEYS,
        PUBLIC_VERT_IMP_COEF_KEYS,
        atm_compute_vert_imp_coefs_work,
        pack_vert_imp_coefs_public,
    )

    # Local copies to avoid mutating caller-owned mappings.
    config = dict(config)
    if mesh_dict is not None:
        mesh_dict = dict(mesh_dict)

    # Pure-function contract: never mutate the caller's state_dict.
    # Create a shallow copy so that all subsequent reads/writes operate on
    # a local dictionary, leaving the caller's original dict untouched.
    state_dict = dict(state_dict)

    # Normalize scalar config/dimension entries once at this boundary. Runner-built
    # stage configs mark later RK stages as already-normalized to avoid repeated
    # static mesh scans and GPU scalar syncs.
    static_scalar_store = _static_input_validation_store(config)
    _coerce_0d_tensor_scalars_inplace(config, static_scalar_store)
    normalize_static_inputs = _coerce_bool(
        config.get("_rk_static_input_normalization_required", True),
        "_rk_static_input_normalization_required",
    )
    if normalize_static_inputs:
        _coerce_0d_tensor_scalars_inplace(state_dict, static_scalar_store)
        if mesh_dict is not None:
            _coerce_0d_tensor_scalars_inplace(mesh_dict, static_scalar_store)

    timestep_scalars = config.get("_timestep_scalars")
    if not isinstance(timestep_scalars, TimestepScalars):
        timestep_scalars = _build_timestep_scalars(
            state_dict,
            config,
            static_scalar_store,
        )
        config["_timestep_scalars"] = timestep_scalars

    def _scalar_int(source: dict[str, Any], key: str, default: Any = None) -> int:
        return timestep_scalars.get_int(source, key, default)

    def _scalar_float(source: dict[str, Any], key: str, default: Any = None) -> float:
        return timestep_scalars.get_float(source, key, default)

    # Optional torch.compile wrapping (vectorized kernels only).
    # Keep eager fallback for debugging and unsupported platforms.
    compile_on = _compile_enabled(config)
    compile_mode = _compile_mode(config)
    compile_backend = _compile_backend(config)

    # =========================================================================
    # VALIDATE STATE/MESH COMPATIBILITY
    # Verify that state and mesh dictionaries have matching dimensions before
    # merging. Combining state and mesh data from differently sized grids would
    # cause silent physics errors deep in kernel computations.
    # =========================================================================
    mesh_state_compatibility_required = _coerce_bool(
        config.get("_rk_mesh_state_compatibility_required", True),
        "_rk_mesh_state_compatibility_required",
    )
    if mesh_dict is not None and mesh_state_compatibility_required:
        _validate_mesh_state_compatibility(state_dict, mesh_dict, strict=strict)

    # =========================================================================
    # MERGE MESH INTO STATE
    # Mesh arrays are needed by kernels but may be loaded separately.
    # For MESH_AUTHORITATIVE_KEYS (damping coefficients, terrain-following
    # coordinates, mesh scaling), the mesh version is authoritative and
    # overrides any stale/zero version that may exist in the IC checkpoint.
    # This prevents bugs like IC dss=0 shadowing the mesh's correct Rayleigh
    # damping coefficients (computed by atm_compute_damping_coefs after IC
    # checkpoint was written).
    # For all other keys, state_dict takes precedence (prognostic variables,
    # connectivity arrays consistent with IC ordering, etc.).
    # =========================================================================
    mesh_merge_required = _coerce_bool(
        config.get("_rk_mesh_merge_required", True),
        "_rk_mesh_merge_required",
    )
    if mesh_merge_required:
        state_dict = _merge_mesh_into_state(
            state_dict=state_dict,
            mesh_dict=mesh_dict,
            strict=strict,
        )
    elif mesh_dict is not None:
        missing_mesh_keys = [key for key in mesh_dict if key not in state_dict]
        if missing_mesh_keys:
            message = (
                "_rk_mesh_merge_required=False but state_dict is missing mesh keys: "
                f"{missing_mesh_keys[:10]}"
            )
            if strict:
                raise ValueError(message)
            warnings.warn(
                message + "; falling back to mesh merge.", RuntimeWarning, stacklevel=2
            )
            state_dict = _merge_mesh_into_state(
                state_dict=state_dict,
                mesh_dict=mesh_dict,
                strict=strict,
            )
    _apply_coef_3rd_order_override(state_dict, config)

    # =========================================================================
    # Create vertical coordinate aliases for scalar advection (explicit-scalar-transport/monotonic-scalar-transport).
    # Upstream pool aliases:
    #   call mpas_pool_get_array(mesh, 'fzm', fnm)
    #   call mpas_pool_get_array(mesh, 'fzp', fnp)
    #   call mpas_pool_get_array(mesh, 'rdzw', rdnw)
    # The checkpoint stores these under their canonical MPAS names (fzm, fzp,
    # rdzw). The scalar advection kernels reference them by the upstream local
    # names (fnm, fnp, rdnw). Create aliases after mesh merge so the canonical
    # arrays are available.
    # =========================================================================
    if "fzm" in state_dict and "fnm" not in state_dict:
        state_dict["fnm"] = state_dict["fzm"]
    if "fzp" in state_dict and "fnp" not in state_dict:
        state_dict["fnp"] = state_dict["fzp"]
    if "rdzw" in state_dict and "rdnw" not in state_dict:
        state_dict["rdnw"] = state_dict["rdzw"]

    # CPU-only execution contract.
    require_consistent_device("execute_rk_step inputs", state_dict)

    # =========================================================================
    # Infer device from input tensors for default tensor creation
    # This ensures defaults match the device of input tensors (CPU-only by default).
    # =========================================================================
    device = None
    for key in ["u", "w", "rho_zz", "theta_m"]:
        if key in state_dict and isinstance(state_dict[key], torch.Tensor):
            device = state_dict[key].device
            break
    if device is None:
        device = torch.device("cpu")

    if compile_on:
        vertical_implicit_coefficients_compile_enabled = (
            not torch.is_grad_enabled()
            and _kernel_compile_enabled_for_device(
                "vertical_implicit_coefficients_vert_imp_coefs",
                config,
                device=device,
            )
        )
        atm_compute_vert_imp_coefs_work = _maybe_compile_kernel(
            "vertical_implicit_coefficients_vert_imp_coefs",
            atm_compute_vert_imp_coefs_work,
            enabled=vertical_implicit_coefficients_compile_enabled,
            mode=compile_mode,
            backend=compile_backend,
        )
        atm_compute_moist_coefficients = _maybe_compile_kernel(
            "small_step_perturbation_moist_coefficients",
            atm_compute_moist_coefficients,
            enabled=_kernel_compile_enabled_for_device(
                "small_step_perturbation_moist_coefficients",
                config,
                device=device,
            ),
            mode=compile_mode,
            backend=compile_backend,
        )

    # Optional workspace bootstrap for direct execute_rk_step callers.
    # atm_srk3 pre-creates and passes _dycore_workspace, but tests/utilities
    # may invoke execute_rk_step directly with config_use_workspace=True.
    if config.get("_dycore_workspace") is None and bool(
        get_config_value_safe(config, "config_use_workspace", False)
    ):
        with profile_section("mpas.workspace_setup.direct_rk_step", config):
            from mpas_pytorch.dynamics.workspace import (
                DycoreWorkspace,
                workspace_dims_from_state,
            )

            ws_dims = workspace_dims_from_state(state_dict)
            config["_dycore_workspace"] = DycoreWorkspace(
                dims=ws_dims,
                device=device,
                dtype=DTYPE,
            )

    # =========================================================================
    # Validate array indexing convention for ALL connectivity arrays
    # Kernels expect 1-based+padded arrays (shape N+1, values 1 to N, sentinel=0)
    # NOT 0-based arrays (shape N, values 0 to N-1, sentinel=-1)
    # Use the 'strict' parameter passed to execute_rk_step (which comes from
    # atm_srk3's strict parameter), but allow config override via 'strict_indexing'.
    # =========================================================================
    strict_indexing = config.get("strict_indexing", strict)
    if isinstance(strict_indexing, torch.Tensor):
        strict_indexing = bool(strict_indexing.item())

    # Warn when strict indexing is disabled - this can cause silent data corruption
    if not strict_indexing:
        warnings.warn(
            "strict_indexing=False: Indexing convention validation is disabled. "
            "This can cause SILENT DATA CORRUPTION if 0-based arrays are passed to "
            "kernels expecting 1-based+padded arrays. Only disable for debugging.",
            RuntimeWarning,
            stacklevel=2,
        )

    default_indexing_validation_required = int(rk_step) == 1
    indexing_validation_required = _coerce_bool(
        config.get(
            "_rk_indexing_validation_required", default_indexing_validation_required
        ),
        "_rk_indexing_validation_required",
    )
    if indexing_validation_required:
        _validate_static_indexing_inputs_cached(
            state_dict,
            config=config,
            strict_indexing=bool(strict_indexing),
        )

    # =========================================================================
    # Warn when regional boundary conditions are enabled. The public release
    # supports global meshes only, so regional boundary treatment is unsupported.
    # Only warn on rk_step==1 to avoid flooding with duplicate warnings.
    # =========================================================================
    config_apply_lbcs_early = get_config_value_safe(config, "config_apply_lbcs", False)
    if config_apply_lbcs_early and rk_step == 1:
        warnings.warn(
            "Regional boundary treatment is outside the supported global-mesh "
            "release configuration.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Extract dimensions with validation
    required_dims = ["nCells", "nVertLevels"]
    for dim in required_dims:
        if dim not in state_dict:
            raise KeyError(
                f"Required dimension '{dim}' not found in state_dict. "
                f"Available keys: {list(state_dict.keys())[:20]}..."  # First 20 keys
            )

    nCells = _scalar_int(state_dict, "nCells")
    nVertLevels = _scalar_int(state_dict, "nVertLevels")
    static_payloads_for_step = ensure_static_payload_store(config)
    config["_tridiag_scan_plan_cache"] = (
        static_payloads_for_step.vertical_scan_plan_cache()
    )
    _prebuild_vertical_scan_plans(
        static_payloads=static_payloads_for_step,
        state_dict=state_dict,
        mesh_dict=None,
        nVertLevels=nVertLevels,
    )

    # nEdges: resolve from explicit value, edge array shapes, or hexagonal approximation
    nEdges = _resolve_nEdges(state_dict, nCells)

    dts = _coerce_float(config["dts"], "dts")
    # Use centralized Fortran-compatible defaults
    epssm = get_config_value_safe(config, "epssm", 0.1)

    # Start building result from input state
    result = dict(state_dict)
    dynamics_substep = _coerce_int(
        config.get("dynamics_substep", 1), "dynamics_substep"
    )

    def sync_chain_tendency_aliases() -> None:
        # Keep upstream diagnostic aliases synchronized with the active tendency
        # storage (tend_rt/tend_rho).
        if "tend_rt" in result:
            result["tend_theta_m"] = result["tend_rt"]
        if "tend_rho" in result:
            result["tend_rho_zz"] = result["tend_rho"]

    # =========================================================================
    # Step 0: Compute moist correction coefficients (cqw, cqu)

    # These coefficients transform inverse dry density to inverse moist density
    # and are needed by vertical-implicit-coefficients (cqw) and downstream kernels (cqw, cqu).

    # Upstream computes moist coefficients once per large timestep before the
    # RK/acoustic loop (outside RK stages). In this staged Python execution,
    # refresh at the first RK stage of the first dynamics substep, or when
    # coefficients are missing.
    # =========================================================================
    moist_can_compute = "scalars" in state_dict and "cellsOnEdge" in state_dict
    dynamics_substep_moist = _coerce_int(
        config.get("dynamics_substep", 1), "dynamics_substep"
    )
    missing_moist = any(k not in result for k in ("cqw", "cqu", "qtot"))
    should_compute_moist = missing_moist or (
        rk_step == 1 and dynamics_substep_moist == 1
    )

    if moist_can_compute and should_compute_moist:
        moist_start = _scalar_int(state_dict, "moist_start", 1)
        moist_end = _scalar_int(state_dict, "moist_end", 0)
        nCellsSolve_moist = _scalar_int(state_dict, "nCellsSolve", nCells)
        cellStart_moist = _scalar_int(state_dict, "cellStart", 1)
        cellEnd_moist = _scalar_int(state_dict, "cellEnd", nCells)
        edgeStart_moist = _scalar_int(state_dict, "edgeStart", 1)
        edgeEnd_moist = _scalar_int(state_dict, "edgeEnd", nEdges)
        moist_connectivity_payload = _prebound_static_payload(
            config,
            "moist_coefficients",
            _moist_connectivity_payload_cache_key,
            _moist_connectivity_payload,
            {
                "device": device,
                "nCells": nCells,
                "nEdges": nEdges,
                "nCellsSolve": nCellsSolve_moist,
                "edge_start": edgeStart_moist - 1,
                "edge_end": edgeEnd_moist,
                "cellsOnEdge": state_dict["cellsOnEdge"],
            },
        )

        moist_result = profiled_call(
            "mpas.moist_coefficients",
            config,
            atm_compute_moist_coefficients,
            nCells=nCells,
            nEdges=nEdges,
            nVertLevels=nVertLevels,
            nCellsSolve=nCellsSolve_moist,
            cellStart=cellStart_moist,
            cellEnd=cellEnd_moist,
            edgeStart=edgeStart_moist,
            edgeEnd=edgeEnd_moist,
            moist_start=moist_start,
            moist_end=moist_end,
            cellsOnEdge=state_dict["cellsOnEdge"],
            scalars=state_dict["scalars"],
            _connectivity_payload=moist_connectivity_payload,
        )
        result["cqw"] = moist_result["cqw"]
        result["cqu"] = moist_result["cqu"]
        if "qtot" in moist_result:
            result["qtot"] = moist_result["qtot"]
    elif not moist_can_compute:
        # Strict-mode guard: if moisture species are present but we cannot compute
        # moist coefficients (cellsOnEdge missing), that is a data integrity error.
        # Default is False to allow kernel-level checkpoint testing (where scalars
        # may be present but cellsOnEdge intentionally omitted). Set strict_moist=True
        # in config for integration runs where moisture correctness is required.
        strict_moist = (
            config.get("strict_moist", False) if isinstance(config, dict) else False
        )
        if strict_moist and "scalars" in state_dict:
            moist_end_check = _scalar_int(state_dict, "moist_end", 0)
            if moist_end_check > 0:
                raise ValueError(
                    f"strict_moist=True: Moist coefficients required (moist_end={moist_end_check}) but "
                    "cannot be computed (cellsOnEdge missing from state_dict). "
                    "Either provide cellsOnEdge or set config['strict_moist']=False "
                    "for dry fallback."
                )

        # Cannot compute moist coefficients - use dry-atmosphere identity fallback
        # For dry atmosphere (no moisture tracers), cqw=1 and cqu=1 are physically
        # correct because there's no moisture correction needed. This allows kernel-level
        # checkpoints (vertical-implicit-coefficients, acoustic-step, large-step-recovery, diagnostic-solve) to run without moisture data.
        warnings.warn(
            "Moist coefficients (cqw/cqu) not provided and cannot be computed "
            "(missing scalars or cellsOnEdge). Using dry-atmosphere identity fallback "
            "(cqw=1, cqu=1, qtot=0). This is correct for dry simulations but will "
            "produce incorrect results if moisture tracers are present.",
            RuntimeWarning,
            stacklevel=2,
        )
        if "cqw" not in result:
            result["cqw"] = torch.ones(
                nCells + 1, nVertLevels + 1, dtype=DTYPE, device=device
            )
        if "cqu" not in result:
            result["cqu"] = torch.ones(
                nEdges + 1, nVertLevels, dtype=DTYPE, device=device
            )
        if "qtot" not in result:
            result["qtot"] = torch.zeros(
                nCells + 1, nVertLevels, dtype=DTYPE, device=device
            )

    # =========================================================================
    # Step 1: Compute vertical implicit coefficients (vertical-implicit-coefficients)

    # Fortran explicitly recomputes vertical-implicit-coefficients at rk_step==2 for RK3.

    #   if( (config_time_integration_order == 3) .and. (rk_step == 2)) then
    #      call atm_compute_vert_imp_coefs(state, time_lev, diag, mesh, configs, dts)
    #   end if

    # The coefficients depend on dts (small timestep), which can differ between
    # RK stages. We must recompute when:
    #   - rk_step == 1 (always compute on first step)
    #   - rk_step == 2 AND time_integration_order == 3 (recompute for RK3 step 2)
    # =========================================================================
    vert_coef_keys = list(PUBLIC_VERT_IMP_COEF_KEYS)
    compact_vert_coef_keys = [
        "cofrz",
        "_vertical_implicit_coefficients_active_cell_start",
        "_vertical_implicit_coefficients_active_cell_end",
        *ACTIVE_VERT_IMP_COEF_KEYS,
    ]
    has_public_vert_coefs = all(key in state_dict for key in vert_coef_keys)
    has_compact_vert_coefs = all(key in state_dict for key in compact_vert_coef_keys)
    has_vert_coefs = has_compact_vert_coefs or has_public_vert_coefs

    def _cached_vert_coefs() -> dict[str, Any]:
        keys = compact_vert_coef_keys if has_compact_vert_coefs else vert_coef_keys
        return {key: state_dict[key] for key in keys}

    # Check if vertical-implicit-coefficients required inputs are available
    vertical_implicit_coefficients_required = [
        "cqw",
        "p",
        "t",
        "rb",
        "rtb",
        "pb",
        "rt",
        "rdzw",
        "fzm",
        "fzp",
        "rdzu",
        "qtot",
    ]
    can_compute_vert_coefs = all(
        key in state_dict or key in result
        for key in vertical_implicit_coefficients_required
    )

    # Resolve integration order through Fortran-compatible config aliases.
    # Fortran key is config_time_integration_order (default 2).
    time_integration_order = _coerce_int(
        get_config_value_safe(config, "config_time_integration_order", 2),
        "config_time_integration_order",
    )

    # Determine if we should recompute coefficients (matching Fortran logic)
    # Fortran recomputes at rk_step==1 (initial) and rk_step==2 for RK3
    should_recompute = (
        rk_step == 1  # Always compute on first step
        or (time_integration_order == 3 and rk_step == 2)  # Recompute for RK3 step 2
    )

    vert_coefs = {}
    should_publish_deferred_vert_coefs = False
    if should_recompute and can_compute_vert_coefs:
        # Compute vertical coefficients via vertical-implicit-coefficients
        vertical_implicit_coefficients_tridiag_scan_policy = (
            normalize_tridiag_scan_policy(
                get_config_value_safe(
                    config,
                    "config_vertical_implicit_coefficients_tridiag_scan_policy",
                    None,
                )
            )
        )
        with (
            tridiag_scan_plan_cache(config.get("_tridiag_scan_plan_cache")),
            tridiag_scan_policy(vertical_implicit_coefficients_tridiag_scan_policy),
        ):
            vert_coefs = profiled_call(
                "mpas.vertical_implicit_coefficients_vert_imp_coefs",
                config,
                atm_compute_vert_imp_coefs_work,
                nCells=nCells,
                nVertLevels=nVertLevels,
                moist_start=_scalar_int(state_dict, "moist_start", 1),
                moist_end=_scalar_int(state_dict, "moist_end", 0),
                dts=dts,
                epssm=epssm,
                cellSolveStart=_scalar_int(state_dict, "cellSolveStart", 1),
                cellSolveEnd=_scalar_int(state_dict, "cellSolveEnd", nCells),
                zz=state_dict["zz"],
                cqw=result["cqw"],
                p=state_dict["p"],
                t=state_dict["t"],
                rb=state_dict["rb"],
                rtb=state_dict["rtb"],
                pb=state_dict["pb"],
                rt=state_dict["rt"],
                rdzw=state_dict["rdzw"],
                fzm=state_dict["fzm"],
                fzp=state_dict["fzp"],
                rdzu=state_dict["rdzu"],
                qtot=result["qtot"],
                gravity=GRAVITY,
                rgas=RGAS,
                cp=CP,
                pack_public=False,
            )
        should_publish_deferred_vert_coefs = True
    elif has_vert_coefs and not should_recompute:
        # Coefficients present and no recomputation needed (e.g., rk_step==3)
        # Reuse cached coefficients from previous step
        vert_coefs = _cached_vert_coefs()
    elif has_vert_coefs and should_recompute and not can_compute_vert_coefs:
        # Need to recompute but missing inputs
        # Respect strict mode - in strict mode this is an error
        # because using cached coefficients when dts has changed violates accuracy.
        if strict:
            # In strict mode, this is an error - cannot guarantee accuracy
            missing_inputs = [
                k
                for k in vertical_implicit_coefficients_required
                if k not in state_dict and k not in result
            ]
            raise ValueError(
                f"Strict mode: vertical-implicit-coefficients should recompute coefficients at rk_step={rk_step} "
                f"(time_integration_order={time_integration_order}) but missing required inputs: "
                f"{missing_inputs[:5]}{'...' if len(missing_inputs) > 5 else ''}. "
                f"Coefficients depend on dts, so using cached values violates accuracy contract. "
                f"Either provide complete checkpoint data or use strict=False for testing."
            )
        else:
            # Non-strict mode: use cached with warning (testing only)
            warnings.warn(
                f"vertical-implicit-coefficients: Should recompute coefficients at rk_step={rk_step} "
                f"(time_integration_order={time_integration_order}) but missing required inputs. "
                f"Using cached coefficients. This may cause numerical discrepancies. "
                f"This is acceptable ONLY for testing.",
                RuntimeWarning,
                stacklevel=2,
            )
            vert_coefs = _cached_vert_coefs()
    elif has_vert_coefs:
        # Coefficients present (fallback for any other case)
        vert_coefs = _cached_vert_coefs()
    # If no coefficients and can't compute, leave vert_coefs empty
    # Downstream kernels will handle missing coefficients appropriately

    # =========================================================================
    # Step 2: Dynamic Tendencies (dynamics-tendency)
    # Computes RHS of momentum and continuity equations for explicit stepping.
    # MUST be called BEFORE acoustic loop so tend_* fields are available.
    # Uses diagnostics (divergence, vorticity, ke, pv_edge) from PREVIOUS RK step.
    # =========================================================================

    # dynamics-tendency is optional - only run if compute_tendencies is requested.
    tendencies_requested = config.get("compute_tendencies", True)

    # Pre-dynamics-tendency diagnostic refresh:
    # Some checkpoints carry stale edge-diagnostic fields (v, pv_edge, ke, etc.)
    # that are inconsistent with the current u/connectivity ordering. Refresh
    # from diagnostic-solve when diagnostics are missing or v is inconsistent.
    if tendencies_requested:
        dynamics_tendency_diag_fields = [
            "v",
            "divergence",
            "vorticity",
            "ke",
            "pv_edge",
            "rho_edge",
        ]
        missing_dynamics_tendency_diag = [
            k for k in dynamics_tendency_diag_fields if k not in state_dict
        ]
        dynamics_substep = _coerce_int(
            config.get("dynamics_substep", 1), "dynamics_substep"
        )
        allow_stale_diag_refresh = rk_step == 1 and dynamics_substep == 1
        # A source receipt permits the initialization diagnostic-solve result to flow directly
        # into first-stage dynamics-tendency. Presence alone is insufficient: stale checkpoint
        # diagnostics still refresh, and tensor replacement/in-place mutation
        # invalidates the receipt through pointer/version identity.
        u_pre, h_pre = _resolve_pre_dynamics_tendency_diagnostic_sources(
            result, state_dict
        )
        diagnostics_current = bool(
            isinstance(u_pre, torch.Tensor)
            and isinstance(h_pre, torch.Tensor)
            and _diagnostic_solve_diagnostics_are_current(
                result,
                u=u_pre,
                h=h_pre,
            )
        )
        needs_diag_refresh = bool(missing_dynamics_tendency_diag) or (
            allow_stale_diag_refresh and not diagnostics_current
        )
        if needs_diag_refresh:
            if not isinstance(u_pre, torch.Tensor) or not isinstance(
                h_pre, torch.Tensor
            ):
                raise ValueError(
                    "pre-dynamics-tendency diagnostics require tensor u and rho_zz fields"
                )
            pre_diagnostic_solve_required = [
                "u",
                "rho_zz",
                "fVertex",
                "fEdge",
                "invAreaTriangle",
                "invAreaCell",
                "dvEdge",
                "dcEdge",
                "invDvEdge",
                "invDcEdge",
                "weightsOnEdge",
                "kiteAreasOnVertex",
                "cellsOnEdge",
                "verticesOnEdge",
                "edgesOnCell",
                "edgesOnCell_sign",
                "edgesOnVertex",
                "edgesOnVertex_sign",
                "verticesOnCell",
                "kiteForCell",
                "nEdgesOnCell",
                "nEdgesOnEdge",
                "edgesOnEdge",
                "nVertices",
                "vertexDegree",
            ]
            can_pre_diagnostic_solve = all(
                k in state_dict or k in result for k in pre_diagnostic_solve_required
            )

            if can_pre_diagnostic_solve:
                from mpas_pytorch.dynamics.solve_diagnostics import (
                    _diagnostic_solve_connectivity_payload,
                    _diagnostic_solve_connectivity_payload_cache_key,
                    atm_compute_solve_diagnostics_work,
                    build_diagnostic_solve_numeric_plan,
                    compile_diagnostic_solve_numeric_plan,
                )

                nVertices_pre = _scalar_int(state_dict, "nVertices", 0)
                vertexDegree_pre = _scalar_int(state_dict, "vertexDegree", 3)

                dt_diagnostic_solve_pre = config.get(
                    "dt_full", config.get("dt", state_dict.get("dt", dts))
                )
                dt_diagnostic_solve_pre = _coerce_float(
                    dt_diagnostic_solve_pre, "dt_diagnostic_solve_pre"
                )

                config_apvm_pre = config.get(
                    "config_apvm_upwinding",
                    state_dict.get("config_apvm_upwinding", 0.5),
                )
                config_apvm_pre = _coerce_float(
                    config_apvm_pre, "config_apvm_upwinding"
                )

                cellStart_pre = _scalar_int(state_dict, "cellStart", 1)
                cellEnd_pre = _scalar_int(state_dict, "cellEnd", nCells)
                vertexStart_pre = _scalar_int(state_dict, "vertexStart", 1)
                vertexEnd_pre = _scalar_int(state_dict, "vertexEnd", nVertices_pre)
                edgeStart_pre = _scalar_int(state_dict, "edgeStart", 1)
                edgeEnd_pre = _scalar_int(state_dict, "edgeEnd", nEdges)

                # Fortran diagnostic-solve reads h from the current rho_zz time level.
                # Do not reuse potentially stale state_dict['h'] aliases.
                if h_pre is not None:
                    result["h"] = h_pre
                diagnostic_solve_pre_payload_kwargs = {
                    "device": device,
                    "nCells": nCells,
                    "nEdges": nEdges,
                    "nVertices": nVertices_pre,
                    "vertexDegree": vertexDegree_pre,
                    "cellStart": cellStart_pre,
                    "cellEnd": cellEnd_pre,
                    "vertexStart": vertexStart_pre,
                    "vertexEnd": vertexEnd_pre,
                    "edgeStart": edgeStart_pre,
                    "edgeEnd": edgeEnd_pre,
                    "dvEdge": state_dict["dvEdge"],
                    "dcEdge": state_dict["dcEdge"],
                    "weightsOnEdge": state_dict["weightsOnEdge"],
                    "kiteAreasOnVertex": state_dict["kiteAreasOnVertex"],
                    "cellsOnEdge": state_dict["cellsOnEdge"],
                    "verticesOnEdge": state_dict["verticesOnEdge"],
                    "edgesOnCell": state_dict["edgesOnCell"],
                    "edgesOnCell_sign": state_dict["edgesOnCell_sign"],
                    "edgesOnVertex": state_dict["edgesOnVertex"],
                    "edgesOnVertex_sign": state_dict["edgesOnVertex_sign"],
                    "verticesOnCell": state_dict["verticesOnCell"],
                    "kiteForCell": state_dict["kiteForCell"],
                    "nEdgesOnCell": state_dict["nEdgesOnCell"],
                    "nEdgesOnEdge": state_dict["nEdgesOnEdge"],
                    "edgesOnEdge": state_dict["edgesOnEdge"],
                }
                diagnostic_solve_pre_connectivity_payload = None
                diagnostic_solve_pre_connectivity_payload = _prebound_static_payload(
                    config,
                    "diagnostic_solve_pre",
                    _diagnostic_solve_connectivity_payload_cache_key,
                    _diagnostic_solve_connectivity_payload,
                    diagnostic_solve_pre_payload_kwargs,
                )
                # This pre-dynamics-tendency refresh mirrors initialization-style diagnostic-solve
                # diagnostics to repair stale checkpoint fields. Use optional
                # rk_step semantics (rk_step omitted) so tangential velocity is
                # reconstructed from current u instead of reusing potentially
                # stale staged v.
                diagnostic_solve_pre_compile_region_names: tuple[str, ...] = ()
                if compile_on:
                    diagnostic_solve_pre_compile_region_names = tuple(
                        name
                        for name in sorted(_DIAGNOSTIC_SOLVE_REGION_COMPILE_TARGETS)
                        if _kernel_compile_enabled_for_device(
                            name,
                            config,
                            device=device,
                        )
                    )
                diagnostic_solve_pre_plan: DiagnosticSolveNumericPlan | None = None
                if diagnostic_solve_pre_connectivity_payload is not None:
                    candidate_diagnostic_solve_pre_plan = build_diagnostic_solve_numeric_plan(
                        connectivity_payload=diagnostic_solve_pre_connectivity_payload,
                        nCells=nCells,
                        nEdges=nEdges,
                        nVertices=nVertices_pre,
                        nVertLevels=nVertLevels,
                        dt=dt_diagnostic_solve_pre,
                        config_apvm_upwinding=config_apvm_pre,
                        fVertex=state_dict["fVertex"],
                        invAreaTriangle=state_dict["invAreaTriangle"],
                        invAreaCell=state_dict["invAreaCell"],
                        invDvEdge=state_dict["invDvEdge"],
                        invDcEdge=state_dict["invDcEdge"],
                        rk_step=None,
                        stage_key=("pre_dynamics_tendency",),
                    )
                    static_payload_store = config.get("_dycore_static_payloads")
                    if isinstance(static_payload_store, DycoreStaticPayloads):
                        diagnostic_solve_pre_plan = cast(
                            DiagnosticSolveNumericPlan,
                            static_payload_store.get_or_build_numeric_plan(
                                "diagnostic_solve_pre_numeric",
                                candidate_diagnostic_solve_pre_plan.key,
                                lambda: candidate_diagnostic_solve_pre_plan,
                            ),
                        )
                    else:
                        diagnostic_solve_pre_plan = candidate_diagnostic_solve_pre_plan
                    if diagnostic_solve_pre_compile_region_names:
                        if isinstance(static_payload_store, DycoreStaticPayloads):
                            plan_to_compile = diagnostic_solve_pre_plan
                            diagnostic_solve_pre_plan = cast(
                                DiagnosticSolveNumericPlan,
                                static_payload_store.get_or_build_numeric_plan(
                                    "diagnostic_solve_pre_numeric_compiled",
                                    (
                                        plan_to_compile.key,
                                        compile_mode,
                                        compile_backend,
                                    ),
                                    lambda plan=plan_to_compile: (  # type: ignore[misc]
                                        compile_diagnostic_solve_numeric_plan(
                                            plan,
                                            compile_kernel=_maybe_compile_kernel,
                                            mode=compile_mode,
                                            backend=compile_backend,
                                        )
                                    ),
                                ),
                            )
                        else:
                            diagnostic_solve_pre_plan = (
                                compile_diagnostic_solve_numeric_plan(
                                    diagnostic_solve_pre_plan,
                                    compile_kernel=_maybe_compile_kernel,
                                    mode=compile_mode,
                                    backend=compile_backend,
                                )
                            )
                        diagnostic_solve_pre_compile_region_names = ()
                diag_pre = profiled_call(
                    "mpas.diagnostic_solve_pre_dynamics_tendency_diagnostic_refresh",
                    config,
                    atm_compute_solve_diagnostics_work,
                    nCells=nCells,
                    nEdges=nEdges,
                    nVertices=nVertices_pre,
                    nVertLevels=nVertLevels,
                    vertexDegree=vertexDegree_pre,
                    dt=dt_diagnostic_solve_pre,
                    config_apvm_upwinding=config_apvm_pre,
                    cellStart=cellStart_pre,
                    cellEnd=cellEnd_pre,
                    vertexStart=vertexStart_pre,
                    vertexEnd=vertexEnd_pre,
                    edgeStart=edgeStart_pre,
                    edgeEnd=edgeEnd_pre,
                    fVertex=state_dict["fVertex"],
                    fEdge=state_dict["fEdge"],
                    invAreaTriangle=state_dict["invAreaTriangle"],
                    invAreaCell=state_dict["invAreaCell"],
                    dvEdge=state_dict["dvEdge"],
                    dcEdge=state_dict["dcEdge"],
                    invDvEdge=state_dict["invDvEdge"],
                    invDcEdge=state_dict["invDcEdge"],
                    weightsOnEdge=state_dict["weightsOnEdge"],
                    kiteAreasOnVertex=state_dict["kiteAreasOnVertex"],
                    cellsOnEdge=state_dict["cellsOnEdge"],
                    verticesOnEdge=state_dict["verticesOnEdge"],
                    edgesOnCell=state_dict["edgesOnCell"],
                    edgesOnCell_sign=state_dict["edgesOnCell_sign"],
                    edgesOnVertex=state_dict["edgesOnVertex"],
                    edgesOnVertex_sign=state_dict["edgesOnVertex_sign"],
                    verticesOnCell=state_dict["verticesOnCell"],
                    kiteForCell=state_dict["kiteForCell"],
                    nEdgesOnCell=state_dict["nEdgesOnCell"],
                    nEdgesOnEdge=state_dict["nEdgesOnEdge"],
                    edgesOnEdge=state_dict["edgesOnEdge"],
                    u=u_pre,
                    h=h_pre,
                    rk_step=None,
                    v_in=None,
                    workspace=config.get("_dycore_workspace"),
                    workspace_inplace=bool(config.get("workspace_inplace", True)),
                    pack_public=not bool(
                        config.get("_defer_diagnostic_solve_public_pack", False)
                    ),
                    _connectivity_payload=diagnostic_solve_pre_connectivity_payload,
                    _compile_region_names=diagnostic_solve_pre_compile_region_names,
                    _compile_mode=compile_mode,
                    _compile_backend=compile_backend,
                    _numeric_plan=diagnostic_solve_pre_plan,
                )
                state_dict.update(diag_pre)
                result.update(diag_pre)
                if "h_edge" in diag_pre:
                    state_dict["rho_edge"] = diag_pre["h_edge"]
                    result["rho_edge"] = diag_pre["h_edge"]
                _mark_diagnostic_solve_diagnostics_current(result, u=u_pre, h=h_pre)

    # Chain tap: state entering dynamics-tendency (after optional diagnostic refresh).
    sync_chain_tendency_aliases()

    # dynamics-tendency required inputs - mesh geometry, state variables, and diagnostic outputs
    dynamics_tendency_required = [
        # State variables
        "u",
        "v",
        "w",
        "ru",
        "rw",
        "ru_save",
        "rw_save",
        "theta_m",
        "theta_m_save",
        "rho_zz",
        "rho_edge",
        "exner",
        "pp",
        "rr",
        "rr_save",
        "rb",
        "pressure_b",
        # Diagnostic outputs (from previous RK step's diagnostic-solve, or initialization)
        "divergence",
        "vorticity",
        "ke",
        "pv_edge",
        # Mesh geometry
        "fEdge",
        "dvEdge",
        "dcEdge",
        "invDcEdge",
        "invDvEdge",
        "invAreaCell",
        "invAreaTriangle",
        "meshScalingDel2",
        "meshScalingDel4",
        "latCell",
        "latEdge",
        "angleEdge",
        # Vertical arrays
        "rdzu",
        "rdzw",
        "fzm",
        "fzp",
        "u_init",
        "v_init",
        # Connectivity
        "cellsOnEdge",
        "verticesOnEdge",
        "edgesOnCell",
        "edgesOnEdge",
        "cellsOnCell",
        "edgesOnVertex",
        "nEdgesOnCell",
        "nEdgesOnEdge",
        "advCellsForEdge",
        "nAdvCellsForEdge",
        "adv_coefs",
        "adv_coefs_3rd",
        "weightsOnEdge",
        "deriv_two",
        "defc_a",
        "defc_b",
        "edgesOnCell_sign",
        "edgesOnVertex_sign",
        # Other required arrays
        "zgrid",
        "zz",
        "zxu",
        "cqw",
        "cqu",
        "rt_diabatic_tend",
        "scalars",
        "t_init",
        "ur_cell",
        "vr_cell",
        # Dimensions
        "nVertices",
        "vertexDegree",
        "maxEdges",
        "maxEdges2",
        "num_scalars",
    ]
    can_run_dynamics_tendency = all(
        key in state_dict or key in result for key in dynamics_tendency_required
    )

    # dynamics-tendency is optional - only run if compute_tendencies is requested via config
    # or if all required inputs are available
    if tendencies_requested and not can_run_dynamics_tendency:
        missing_keys = [
            k
            for k in dynamics_tendency_required
            if k not in state_dict and k not in result
        ]
        if strict:
            raise ValueError(
                f"Cannot execute dynamic tendencies (dynamics-tendency): missing required inputs: {missing_keys}. "
                f"config['compute_tendencies']=True but required data is missing. "
                f"Ensure checkpoint contains all dynamics-tendency data or set strict=False to skip."
            )
        else:
            warnings.warn(
                f"Skipping dynamic tendencies (dynamics-tendency): missing inputs: {missing_keys}. "
                f"Tendency arrays (tend_ru, tend_rho, tend_rt, tend_rw) will not be computed.",
                RuntimeWarning,
            )

    if can_run_dynamics_tendency:
        from mpas_pytorch.dynamics.dyn_tend import (
            _DYNAMICS_TENDENCY_REGION_COMPILE_TARGETS,
            _DYNAMICS_TENDENCY_WHOLE_STAGE_COMPILE_TARGET,
            _dynamics_tendency_connectivity_payload,
            _dynamics_tendency_connectivity_payload_cache_key,
            _dynamics_tendency_whole_stage_specialization_key,
            atm_compute_dyn_tend_work,
            build_dynamics_tendency_numeric_plan,
            build_dynamics_tendency_section_ops,
            pack_dynamics_tendency_numeric_outputs,
        )

        # Get dimensions
        nVertices_dynamics_tendency = _scalar_int(state_dict, "nVertices", 0)
        nCellsSolve_dynamics_tendency = _scalar_int(state_dict, "nCellsSolve", nCells)
        nEdgesSolve_dynamics_tendency = _scalar_int(state_dict, "nEdgesSolve", nEdges)
        vertexDegree_dynamics_tendency = _scalar_int(state_dict, "vertexDegree", 3)
        maxEdges_dynamics_tendency = _scalar_int(state_dict, "maxEdges", 10)
        maxEdges2_dynamics_tendency = _scalar_int(state_dict, "maxEdges2", 20)
        num_scalars_dynamics_tendency = _scalar_int(state_dict, "num_scalars", 1)
        moist_start_dynamics_tendency = _scalar_int(state_dict, "moist_start", 1)
        moist_end_dynamics_tendency = _scalar_int(state_dict, "moist_end", 0)

        # Get dt (FULL timestep) for dynamics-tendency.
        # Fortran passes the full
        # timestep dt to atm_compute_dyn_tend, NOT the per-RK-step timestep.
        # dynamics-tendency uses 1/dt for Smagorinsky eddy viscosity limiting.
        dt_dynamics_tendency = config.get(
            "dt_full", config.get("dt", state_dict.get("dt", dts))
        )
        dt_dynamics_tendency = _coerce_float(
            dt_dynamics_tendency, "dt_dynamics_tendency"
        )

        # Get rk_step from config or use the function argument
        dynamics_tendency_rk_step = config.get("rk_step", rk_step)
        dynamics_tendency_rk_step = _coerce_int(
            dynamics_tendency_rk_step, "dynamics_tendency_rk_step"
        )

        # Configuration parameters with defaults
        # Non-config scalars from state_dict (mesh/physics parameters)
        coef_3rd_order_dynamics_tendency = _scalar_float(
            state_dict, "coef_3rd_order", 0.25
        )
        c_s_dynamics_tendency = _coerce_float(
            get_config_value_safe(config, "config_smagorinsky_coef", 0.125),
            "config_smagorinsky_coef",
        )
        config_horiz_mixing_dynamics_tendency = get_config_value_safe(
            config, "config_horiz_mixing", "2d_smagorinsky"
        )

        # Config scalars: use get_config_value_safe with Registry.xml authoritative defaults
        # These are config_ prefixed values that should come from config dict, not state_dict
        # Corrected defaults to match Registry.xml
        config_del4u_div_factor_dynamics_tendency = get_config_value_safe(
            config, "config_del4u_div_factor", 10.0
        )
        config_h_mom_eddy_visc2_dynamics_tendency = get_config_value_safe(
            config, "config_h_mom_eddy_visc2", 0.0
        )
        config_v_mom_eddy_visc2_dynamics_tendency = get_config_value_safe(
            config, "config_v_mom_eddy_visc2", 0.0
        )
        config_h_theta_eddy_visc2_dynamics_tendency = get_config_value_safe(
            config, "config_h_theta_eddy_visc2", 0.0
        )
        config_v_theta_eddy_visc2_dynamics_tendency = get_config_value_safe(
            config, "config_v_theta_eddy_visc2", 0.0
        )
        config_h_theta_eddy_visc4_dynamics_tendency = get_config_value_safe(
            config, "config_h_theta_eddy_visc4", 0.0
        )
        config_h_mom_eddy_visc4_dynamics_tendency = get_config_value_safe(
            config, "config_h_mom_eddy_visc4", 0.0
        )
        config_visc4_2dsmag_dynamics_tendency = get_config_value_safe(
            config, "config_visc4_2dsmag", 0.05
        )
        config_len_disp_dynamics_tendency = _resolve_len_disp(
            config, state_dict, strict=strict
        )

        # cf1/cf2/cf3 are legitimate state (vertical interpolation coefficients)
        # These MUST come from state_dict - validate presence in strict mode
        if strict:
            cf_missing = [k for k in ["cf1", "cf2", "cf3"] if k not in state_dict]
            if cf_missing:
                raise ValueError(
                    f"dynamics-tendency: Missing required vertical interpolation coefficients: {cf_missing}. "
                    f"These coefficients must be computed from mesh geometry (zgrid) and cannot use defaults. "
                    f"Ensure checkpoint contains cf1, cf2, cf3 or set strict=False to use fallback values."
                )
        cf1_dynamics_tendency = _scalar_float(state_dict, "cf1", 0.0)
        cf2_dynamics_tendency = _scalar_float(state_dict, "cf2", 1.0)
        cf3_dynamics_tendency = _scalar_float(state_dict, "cf3", 0.0)

        # r_earth is a physical constant, not configurable
        # Use immutable value from PhysicalConstants for exact reproducibility
        r_earth_dynamics_tendency = float(PhysicalConstants.EARTH_RADIUS)

        # Config scalars come from config dict, not state_dict
        # These are namelist options that affect model behavior
        config_mpas_cam_coef_dynamics_tendency = get_config_value_safe(
            config, "config_mpas_cam_coef", 0.0
        )
        config_mix_full_val = get_config_value_safe(config, "config_mix_full", True)
        config_mix_full_dynamics_tendency = 1 if config_mix_full_val else 0
        config_rayleigh_damp_u_val = get_config_value_safe(
            config, "config_rayleigh_damp_u", False
        )
        config_rayleigh_damp_u_dynamics_tendency = (
            1 if config_rayleigh_damp_u_val else 0
        )
        config_rayleigh_damp_u_timescale_days_dynamics_tendency = get_config_value_safe(
            config, "config_rayleigh_damp_u_timescale_days", 5.0
        )
        config_number_rayleigh_damp_u_levels_dynamics_tendency = get_config_value_safe(
            config, "config_number_rayleigh_damp_u_levels", 0
        )
        config_number_cam_damping_levels_dynamics_tendency = get_config_value_safe(
            config, "config_number_cam_damping_levels", 0
        )
        config_curvature_u_val = get_config_value_safe(
            config, "config_curvature_u", False
        )
        config_curvature_u_dynamics_tendency = bool(config_curvature_u_val)
        config_dynamics_tendency_cell_owned_vertical_velocity_horizontal_advection_theta_horizontal_advection_ab_dynamics_tendency = bool(
            get_config_value_safe(
                config,
                "config_dynamics_tendency_cell_owned_vertical_velocity_horizontal_advection_theta_horizontal_advection_ab",
                False,
            )
        )

        # Loop bounds (1-based Fortran indices)
        cellStart_dynamics_tendency = _scalar_int(state_dict, "cellStart", 1)
        cellEnd_dynamics_tendency = _scalar_int(state_dict, "cellEnd", nCells)
        vertexStart_dynamics_tendency = _scalar_int(state_dict, "vertexStart", 1)
        vertexEnd_dynamics_tendency = _scalar_int(
            state_dict, "vertexEnd", nVertices_dynamics_tendency
        )
        edgeStart_dynamics_tendency = _scalar_int(state_dict, "edgeStart", 1)
        edgeEnd_dynamics_tendency = _scalar_int(state_dict, "edgeEnd", nEdges)
        cellSolveStart_dynamics_tendency = _scalar_int(state_dict, "cellSolveStart", 1)
        cellSolveEnd_dynamics_tendency = _scalar_int(state_dict, "cellSolveEnd", nCells)
        vertexSolveStart_dynamics_tendency = _scalar_int(
            state_dict, "vertexSolveStart", 1
        )
        vertexSolveEnd_dynamics_tendency = _scalar_int(
            state_dict, "vertexSolveEnd", nVertices_dynamics_tendency
        )
        edgeSolveStart_dynamics_tendency = _scalar_int(state_dict, "edgeSolveStart", 1)
        edgeSolveEnd_dynamics_tendency = _scalar_int(state_dict, "edgeSolveEnd", nEdges)

        # Helper to get tensor from result first, then state_dict
        def get_tensor_dynamics_tendency(key):
            if key in result:
                return result[key]
            return state_dict[key]

        def get_optional_tensor_dynamics_tendency(key):
            if key in result:
                return result[key]
            return state_dict.get(key)

        dynamics_tendency_compile_region_names: tuple[str, ...] = ()
        compile_whole_dynamics_tendency = False
        if compile_on:
            theta_m_alias_dynamics_tendency = get_tensor_dynamics_tendency(
                "theta_m"
            ) is state_dict.get("theta_m_save")
            compile_whole_dynamics_tendency = _kernel_compile_enabled_for_device(
                _DYNAMICS_TENDENCY_WHOLE_STAGE_COMPILE_TARGET,
                config,
                device=device,
            )
            if compile_whole_dynamics_tendency:
                dynamics_tendency_compile_region_names = (
                    _DYNAMICS_TENDENCY_WHOLE_STAGE_COMPILE_TARGET,
                )
            else:
                dynamics_tendency_compile_region_names = tuple(
                    name
                    for name in sorted(_DYNAMICS_TENDENCY_REGION_COMPILE_TARGETS)
                    if _kernel_compile_enabled_for_device(name, config, device=device)
                )

        # Get optional arrays that persist across RK steps
        # For rk_step > 1, these should come from previous rk_step results
        kdiff_in_dynamics_tendency = result.get("kdiff", state_dict.get("kdiff"))
        tend_u_euler_in_dynamics_tendency = result.get(
            "tend_u_euler", state_dict.get("tend_u_euler")
        )
        tend_w_euler_in_dynamics_tendency = result.get(
            "tend_w_euler", state_dict.get("tend_w_euler")
        )
        tend_theta_euler_in_dynamics_tendency = result.get(
            "tend_theta_euler", state_dict.get("tend_theta_euler")
        )
        tend_rho_in_dynamics_tendency = result.get(
            "tend_rho", state_dict.get("tend_rho")
        )

        # Initialize arrays that may not exist in state_dict
        qv_init_dynamics_tendency = _mapping_tensor_or_zeros(
            state_dict,
            "qv_init",
            (nVertLevels,),
            device=device,
        )
        # These legacy dynamics-tendency arguments are not consumed by the canonical numeric
        # body. Preserve supplied tensors for API compatibility, but do not
        # fabricate two full-cell zero fields on every RK invocation.
        tend_w_pgf_dynamics_tendency = get_optional_tensor_dynamics_tendency(
            "tend_w_pgf"
        )
        tend_w_buoy_dynamics_tendency = get_optional_tensor_dynamics_tendency(
            "tend_w_buoy"
        )

        # Every execution mode enters dynamics-tendency through one runner-owned topology plan.
        # Metric products remain in the canonical Fortran operand order; substituting
        # reassociated precomputed products is a separate proven-drift optimization.
        build_cached_metric_products_dynamics_tendency = False
        build_ordered_ad_transpose_dynamics_tendency = (
            _dynamics_tendency_ordered_ad_transpose_required(
                config,
                *(
                    get_tensor_dynamics_tendency(name)
                    for name in ("u", "rw", "pv_edge")
                ),
            )
        )
        build_edge_advective_ordered_ad_transpose_dynamics_tendency = (
            _dynamics_tendency_edge_advective_ad_transpose_required(
                config,
                *(
                    get_tensor_dynamics_tendency(name)
                    for name in ("w", "ru", "theta_m")
                ),
            )
        )
        dynamics_tendency_payload_kwargs = {
            "device": device,
            "nCells": nCells,
            "nEdges": nEdges,
            "maxEdges": maxEdges_dynamics_tendency,
            "maxEdges2": maxEdges2_dynamics_tendency,
            "cell_start": cellStart_dynamics_tendency - 1,
            "cell_end": cellEnd_dynamics_tendency,
            "cell_solve_start": cellSolveStart_dynamics_tendency - 1,
            "cell_solve_end": cellSolveEnd_dynamics_tendency,
            "edge_start": edgeStart_dynamics_tendency - 1,
            "edge_end": edgeEnd_dynamics_tendency,
            "edge_solve_start": edgeSolveStart_dynamics_tendency - 1,
            "edge_solve_end": edgeSolveEnd_dynamics_tendency,
            "edgesOnCell": state_dict["edgesOnCell"],
            "nEdgesOnCell": state_dict["nEdgesOnCell"],
            "edgesOnCell_sign": state_dict["edgesOnCell_sign"],
            "cellsOnEdge": state_dict["cellsOnEdge"],
            "verticesOnEdge": state_dict["verticesOnEdge"],
            "edgesOnEdge": state_dict["edgesOnEdge"],
            "nAdvCellsForEdge": state_dict["nAdvCellsForEdge"],
            "nEdgesOnEdge": state_dict["nEdgesOnEdge"],
            "advCellsForEdge": state_dict["advCellsForEdge"],
            "adv_coefs": state_dict["adv_coefs"],
            "adv_coefs_3rd": state_dict["adv_coefs_3rd"],
            "weightsOnEdge": state_dict["weightsOnEdge"],
            "vertex_start": vertexStart_dynamics_tendency - 1,
            "vertex_end": vertexEnd_dynamics_tendency,
            "vertexDegree": vertexDegree_dynamics_tendency,
            "edgesOnVertex": state_dict["edgesOnVertex"],
            "edgesOnVertex_sign": state_dict["edgesOnVertex_sign"],
            "dvEdge": state_dict["dvEdge"],
            "dcEdge": state_dict["dcEdge"],
            "invDcEdge": state_dict["invDcEdge"]
            if build_cached_metric_products_dynamics_tendency
            else None,
            "invDvEdge": state_dict["invDvEdge"]
            if build_cached_metric_products_dynamics_tendency
            else None,
            "invAreaCell": (
                state_dict["invAreaCell"]
                if build_cached_metric_products_dynamics_tendency
                else None
            ),
            "invAreaTriangle": state_dict["invAreaTriangle"],
            "meshScalingDel2": (
                state_dict["meshScalingDel2"]
                if build_cached_metric_products_dynamics_tendency
                else None
            ),
            "meshScalingDel4": (
                state_dict["meshScalingDel4"]
                if build_cached_metric_products_dynamics_tendency
                else None
            ),
            "build_ordered_ad_transpose": build_ordered_ad_transpose_dynamics_tendency,
            "build_edge_advective_ordered_ad_transpose": (
                build_edge_advective_ordered_ad_transpose_dynamics_tendency
            ),
        }
        dynamics_tendency_connectivity_payload = _prebound_static_payload(
            config,
            "dynamics_tendency",
            _dynamics_tendency_connectivity_payload_cache_key,
            _dynamics_tendency_connectivity_payload,
            dynamics_tendency_payload_kwargs,
        )

        # Moist coefficients already produced qtot once for this large timestep,
        # matching the Fortran lifetime.  Reuse that evolving-state field instead
        # of rebuilding and repacking an identical full-cell tensor for every dynamics-tendency
        # invocation.
        scalars_dynamics_tendency = get_tensor_dynamics_tendency("scalars")
        qtot_dynamics_tendency = result.get("qtot")
        if not isinstance(qtot_dynamics_tendency, torch.Tensor):
            raise TypeError(
                "dynamics-tendency requires qtot from the moist-coefficient phase"
            )

        dynamics_tendency_call_arguments = {
            # Dimensions
            "nCells": nCells,
            "nEdges": nEdges,
            "nVertices": nVertices_dynamics_tendency,
            "nVertLevels": nVertLevels,
            "nCellsSolve": nCellsSolve_dynamics_tendency,
            "nEdgesSolve": nEdgesSolve_dynamics_tendency,
            "vertexDegree": vertexDegree_dynamics_tendency,
            "maxEdges": maxEdges_dynamics_tendency,
            "maxEdges2": maxEdges2_dynamics_tendency,
            "num_scalars": num_scalars_dynamics_tendency,
            "moist_start": moist_start_dynamics_tendency,
            "moist_end": moist_end_dynamics_tendency,
            # Time stepping
            "rk_step": dynamics_tendency_rk_step,
            "dt": dt_dynamics_tendency,
            # Configuration scalars
            "coef_3rd_order": coef_3rd_order_dynamics_tendency,
            "c_s": c_s_dynamics_tendency,
            "config_horiz_mixing": config_horiz_mixing_dynamics_tendency,
            "config_del4u_div_factor": config_del4u_div_factor_dynamics_tendency,
            "config_h_mom_eddy_visc2": config_h_mom_eddy_visc2_dynamics_tendency,
            "config_v_mom_eddy_visc2": config_v_mom_eddy_visc2_dynamics_tendency,
            "config_h_theta_eddy_visc2": config_h_theta_eddy_visc2_dynamics_tendency,
            "config_v_theta_eddy_visc2": config_v_theta_eddy_visc2_dynamics_tendency,
            "config_h_theta_eddy_visc4": config_h_theta_eddy_visc4_dynamics_tendency,
            "config_h_mom_eddy_visc4": config_h_mom_eddy_visc4_dynamics_tendency,
            "config_visc4_2dsmag": config_visc4_2dsmag_dynamics_tendency,
            "config_len_disp": config_len_disp_dynamics_tendency,
            "cf1": cf1_dynamics_tendency,
            "cf2": cf2_dynamics_tendency,
            "cf3": cf3_dynamics_tendency,
            "r_earth": r_earth_dynamics_tendency,
            "config_mpas_cam_coef": config_mpas_cam_coef_dynamics_tendency,
            "config_mix_full": config_mix_full_dynamics_tendency,
            "config_rayleigh_damp_u": config_rayleigh_damp_u_dynamics_tendency,
            "config_rayleigh_damp_u_timescale_days": config_rayleigh_damp_u_timescale_days_dynamics_tendency,
            "config_number_rayleigh_damp_u_levels": config_number_rayleigh_damp_u_levels_dynamics_tendency,
            "config_number_cam_damping_levels": config_number_cam_damping_levels_dynamics_tendency,
            # Loop bounds (1-based Fortran indices)
            "cellStart": cellStart_dynamics_tendency,
            "cellEnd": cellEnd_dynamics_tendency,
            "vertexStart": vertexStart_dynamics_tendency,
            "vertexEnd": vertexEnd_dynamics_tendency,
            "edgeStart": edgeStart_dynamics_tendency,
            "edgeEnd": edgeEnd_dynamics_tendency,
            "cellSolveStart": cellSolveStart_dynamics_tendency,
            "cellSolveEnd": cellSolveEnd_dynamics_tendency,
            "vertexSolveStart": vertexSolveStart_dynamics_tendency,
            "vertexSolveEnd": vertexSolveEnd_dynamics_tendency,
            "edgeSolveStart": edgeSolveStart_dynamics_tendency,
            "edgeSolveEnd": edgeSolveEnd_dynamics_tendency,
            # 1D arrays - mesh metric
            "fEdge": state_dict["fEdge"],
            "dvEdge": state_dict["dvEdge"],
            "dcEdge": state_dict["dcEdge"],
            "invDcEdge": state_dict["invDcEdge"],
            "invDvEdge": state_dict["invDvEdge"],
            "invAreaCell": state_dict["invAreaCell"],
            "invAreaTriangle": state_dict["invAreaTriangle"],
            "meshScalingDel2": state_dict["meshScalingDel2"],
            "meshScalingDel4": state_dict["meshScalingDel4"],
            "latCell": state_dict["latCell"],
            "latEdge": state_dict["latEdge"],
            "angleEdge": state_dict["angleEdge"],
            # 1D arrays - vertical
            "rdzu": state_dict["rdzu"],
            "rdzw": state_dict["rdzw"],
            "fzm": state_dict["fzm"],
            "fzp": state_dict["fzp"],
            "u_init": state_dict["u_init"],
            "v_init": state_dict["v_init"],
            "qv_init": qv_init_dynamics_tendency,
            # Connectivity arrays
            "cellsOnEdge": state_dict["cellsOnEdge"],
            "verticesOnEdge": state_dict["verticesOnEdge"],
            "edgesOnCell": state_dict["edgesOnCell"],
            "edgesOnEdge": state_dict["edgesOnEdge"],
            "cellsOnCell": state_dict["cellsOnCell"],
            "edgesOnVertex": state_dict["edgesOnVertex"],
            "nEdgesOnCell": state_dict["nEdgesOnCell"],
            "nEdgesOnEdge": state_dict["nEdgesOnEdge"],
            "advCellsForEdge": state_dict["advCellsForEdge"],
            "nAdvCellsForEdge": state_dict["nAdvCellsForEdge"],
            "adv_coefs": state_dict["adv_coefs"],
            "adv_coefs_3rd": state_dict["adv_coefs_3rd"],
            "weightsOnEdge": state_dict["weightsOnEdge"],
            "deriv_two": state_dict["deriv_two"],
            "defc_a": state_dict["defc_a"],
            "defc_b": state_dict["defc_b"],
            "edgesOnCell_sign": state_dict["edgesOnCell_sign"],
            "edgesOnVertex_sign": state_dict["edgesOnVertex_sign"],
            # 2D state arrays
            "u": get_tensor_dynamics_tendency("u"),
            "v": get_tensor_dynamics_tendency("v"),
            "w": get_tensor_dynamics_tendency("w"),
            "ru": get_tensor_dynamics_tendency("ru"),
            "rw": get_tensor_dynamics_tendency("rw"),
            "ru_save": state_dict["ru_save"],
            "rw_save": state_dict["rw_save"],
            "theta_m": get_tensor_dynamics_tendency("theta_m"),
            "theta_m_save": state_dict["theta_m_save"],
            "rho_zz": get_tensor_dynamics_tendency("rho_zz"),
            "rho_edge": get_tensor_dynamics_tendency("rho_edge"),
            "exner": get_tensor_dynamics_tendency("exner"),
            # Diagnostic outputs (from previous RK step's diagnostic-solve or initialization)
            "divergence": get_tensor_dynamics_tendency("divergence"),
            "vorticity": get_tensor_dynamics_tendency("vorticity"),
            "ke": get_tensor_dynamics_tendency("ke"),
            "pv_edge": get_tensor_dynamics_tendency("pv_edge"),
            # Other state arrays
            "pp": get_tensor_dynamics_tendency("pp"),
            "rr": get_tensor_dynamics_tendency("rr"),
            "rr_save": state_dict["rr_save"],
            "rb": state_dict["rb"],
            "pressure_b": state_dict["pressure_b"],
            "zgrid": state_dict["zgrid"],
            "zz": state_dict["zz"],
            "zxu": state_dict["zxu"],
            "cqw": result["cqw"],
            "cqu": result["cqu"],
            "rt_diabatic_tend": get_tensor_dynamics_tendency("rt_diabatic_tend"),
            "scalars": scalars_dynamics_tendency,
            "t_init": state_dict["t_init"],
            "tend_w_pgf": tend_w_pgf_dynamics_tendency,
            "tend_w_buoy": tend_w_buoy_dynamics_tendency,
            "ur_cell": state_dict["ur_cell"],
            "vr_cell": state_dict["vr_cell"],
            # Optional persistent arrays
            "kdiff_in": kdiff_in_dynamics_tendency,
            "config_curvature_u": config_curvature_u_dynamics_tendency,
            "config_dynamics_tendency_cell_owned_vertical_velocity_horizontal_advection_theta_horizontal_advection_ab": config_dynamics_tendency_cell_owned_vertical_velocity_horizontal_advection_theta_horizontal_advection_ab_dynamics_tendency,
            "tend_u_euler_in": tend_u_euler_in_dynamics_tendency,
            "tend_w_euler_in": tend_w_euler_in_dynamics_tendency,
            "tend_theta_euler_in": tend_theta_euler_in_dynamics_tendency,
            "tend_rho_in": tend_rho_in_dynamics_tendency,
            "tend_ru_physics": get_optional_tensor_dynamics_tendency("tend_ru_physics"),
            "tend_rtheta_physics": get_optional_tensor_dynamics_tendency(
                "tend_rtheta_physics"
            ),
            "tend_rho_physics": get_optional_tensor_dynamics_tendency(
                "tend_rho_physics"
            ),
            "qtot": qtot_dynamics_tendency,
            "_connectivity_payload": dynamics_tendency_connectivity_payload,
        }
        dynamics_tendency_section_plan_key = _compiled_kernel_specialization_key(
            config,
            "dynamics_tendency_section_plan",
            *_dynamics_tendency_whole_stage_specialization_key(
                dynamics_tendency_call_arguments
            ),
            "dt",
            dt_dynamics_tendency,
            "split",
            _resolve_dynamics_split(config),
        )
        dynamics_tendency_call_arguments["_section_ops"] = (
            build_dynamics_tendency_section_ops(
                compile_region_names=(
                    ()
                    if compile_whole_dynamics_tendency
                    else dynamics_tendency_compile_region_names
                ),
                compile_mode=compile_mode,
                compile_backend=compile_backend,
                stage_key=dynamics_tendency_section_plan_key,
                profile_sections=(
                    bool(config.get("config_profile_sections", False))
                    and not compile_whole_dynamics_tendency
                ),
                section_observer=config.get("_dynamics_tendency_section_observer"),
            )
        )
        debug_tap_sink = config.get("_dynamics_tendency_debug_tap_sink")
        if debug_tap_sink is not None:
            if not callable(debug_tap_sink):
                raise TypeError("_dynamics_tendency_debug_tap_sink must be callable")
            if (
                compile_whole_dynamics_tendency
                or dynamics_tendency_compile_region_names
            ):
                raise ValueError(
                    "dynamics-tendency debug taps are an eager-only diagnostic and cannot be "
                    "combined with compiled dynamics-tendency targets"
                )
            debug_taps: dict[str, dict[str, Any]] = {}
            debug_call_arguments = dict(dynamics_tendency_call_arguments)
            debug_call_arguments["_debug_taps"] = debug_taps
            tend_result = profiled_call(
                "mpas.dynamics_tendency_dyn_tend",
                config,
                atm_compute_dyn_tend_work,
                **debug_call_arguments,
            )
            debug_tap_sink(dynamics_tendency_rk_step, debug_taps)
        else:
            numeric_stage_key = dynamics_tendency_section_plan_key
            if compile_whole_dynamics_tendency:
                numeric_stage_key = _compiled_kernel_specialization_key(
                    config,
                    _DYNAMICS_TENDENCY_WHOLE_STAGE_COMPILE_TARGET,
                    *_dynamics_tendency_whole_stage_specialization_key(
                        dynamics_tendency_call_arguments
                    ),
                    "dt",
                    dt_dynamics_tendency,
                    "split",
                    _resolve_dynamics_split(config),
                    "theta_m_alias",
                    theta_m_alias_dynamics_tendency,
                )
            static_payload_store = config.get("_dycore_static_payloads")
            candidate_dynamics_tendency_numeric_plan = (
                build_dynamics_tendency_numeric_plan(
                    dynamics_tendency_call_arguments,
                    stage_key=numeric_stage_key,
                    require_canonical_section_ops=compile_whole_dynamics_tendency,
                )
            )
            if isinstance(static_payload_store, DycoreStaticPayloads):
                dynamics_tendency_numeric_plan = (
                    static_payload_store.get_or_build_numeric_plan(
                        "dynamics_tendency_numeric_composition",
                        candidate_dynamics_tendency_numeric_plan.key,
                        lambda: candidate_dynamics_tendency_numeric_plan,
                    )
                )
            else:
                dynamics_tendency_numeric_plan = (
                    candidate_dynamics_tendency_numeric_plan
                )
            dynamics_tendency_callable = dynamics_tendency_numeric_plan.numeric_callable
            if compile_whole_dynamics_tendency:
                dynamics_tendency_callable = _maybe_compile_kernel(
                    _DYNAMICS_TENDENCY_WHOLE_STAGE_COMPILE_TARGET,
                    dynamics_tendency_numeric_plan.numeric_callable,
                    enabled=True,
                    mode=compile_mode,
                    backend=compile_backend,
                    specialization_key=dynamics_tendency_numeric_plan.key,
                    allow_reverse_autodiff=_kernel_compiled_reverse_ad_enabled_for_device(
                        _DYNAMICS_TENDENCY_WHOLE_STAGE_COMPILE_TARGET,
                        config,
                        device=device,
                    ),
                )
            compact_tend_result = profiled_call(
                "mpas.dynamics_tendency_dyn_tend",
                config,
                dynamics_tendency_callable,
                *dynamics_tendency_numeric_plan.dynamic_values(
                    dynamics_tendency_call_arguments
                ),
            )
            tend_result = pack_dynamics_tendency_numeric_outputs(
                compact_tend_result,
                dynamics_tendency_call_arguments,
                workspace=config.get("_dycore_workspace"),
                take_unchanged_ownership=True,
            )
        # Update result with tendency outputs
        # Map dynamics-tendency outputs to the expected result keys for acoustic step
        result["tend_ru"] = tend_result["tend_u"]  # Horizontal momentum tendency
        result["tend_rho"] = tend_result["tend_rho"]  # Density tendency
        result["tend_rt"] = tend_result["tend_theta"]  # Potential temperature tendency
        result["tend_rw"] = tend_result["tend_w"]  # Vertical momentum tendency
        # Also store the internal names for completeness
        result["tend_u"] = tend_result["tend_u"]
        result["tend_theta"] = tend_result["tend_theta"]
        result["tend_w"] = tend_result["tend_w"]
        result["tend_u_euler"] = tend_result["tend_u_euler"]
        result["tend_w_euler"] = tend_result["tend_w_euler"]
        result["tend_theta_euler"] = tend_result["tend_theta_euler"]
        result["h_divergence"] = tend_result["h_divergence"]
        result["kdiff"] = tend_result["kdiff"]
        result["rthdynten"] = tend_result["rthdynten"]

    # Chain tap: dynamics-tendency outputs (tendencies) are now available in result.
    sync_chain_tendency_aliases()

    # =========================================================================
    # Step 2.5: Set perturbation variables (small-step-perturbation)
    # Converts w_tend to omega for the acoustic solver.
    # MUST be called AFTER dynamics-tendency (which provides tend_rw/w_tend) and BEFORE
    # the acoustic loop (acoustic-step) which uses the modified tend_rw.
    # =========================================================================
    from mpas_pytorch.dynamics.smlstep_pert import (
        _SMALL_STEP_PERTURBATION_REGION_COMPILE_TARGETS,
        _small_step_perturbation_connectivity_payload,
        _small_step_perturbation_connectivity_payload_cache_key,
        atm_set_smlstep_pert_variables_work,
    )

    # small-step-perturbation required inputs - mesh connectivity and vertical coordinate arrays
    small_step_perturbation_required = [
        "zb",
        "zb3",
        "zb_cell",
        "zb3_cell",
        "zz",
        "nEdgesOnCell",
        "cellsOnEdge",
        "edgesOnCell",
        "edgesOnCell_sign",
        "fzm",
        "fzp",
    ]
    can_run_small_step_perturbation = all(
        key in state_dict or key in result for key in small_step_perturbation_required
    )

    # small-step-perturbation requires tend_rw and tend_ru from dynamics-tendency
    has_tend_inputs = "tend_rw" in result and "tend_ru" in result

    if has_tend_inputs and not can_run_small_step_perturbation:
        missing_keys = [
            k
            for k in small_step_perturbation_required
            if k not in state_dict and k not in result
        ]
        if strict:
            raise ValueError(
                f"Cannot execute perturbation variables (small-step-perturbation): missing required inputs: {missing_keys}. "
                f"small-step-perturbation is required after dynamics-tendency to prepare tend_rw for acoustic step. "
                f"Ensure checkpoint contains all small-step-perturbation data or set strict=False to skip."
            )
        else:
            warnings.warn(
                f"Skipping perturbation variables (small-step-perturbation): missing inputs: {missing_keys}. "
                f"tend_rw will be used unmodified in acoustic step.",
                RuntimeWarning,
            )

    if has_tend_inputs and can_run_small_step_perturbation:
        # Get dimensions
        maxEdges_small_step_perturbation = _scalar_int(state_dict, "maxEdges", 10)
        nRelaxZone_small_step_perturbation = _scalar_int(state_dict, "nRelaxZone", 0)
        cellSolveStart_small_step_perturbation = _scalar_int(
            state_dict, "cellSolveStart", 1
        )
        cellSolveEnd_small_step_perturbation = _scalar_int(
            state_dict, "cellSolveEnd", nCells
        )

        # Get bdyMaskCell with default (zeros for global MPAS)
        bdyMaskCell_small_step_perturbation = _mapping_tensor_or_zeros(
            state_dict,
            "bdyMaskCell",
            (nCells + 1,),
            dtype=torch.int32,
            device=device,
        )
        small_step_perturbation_compile_region_names: tuple[str, ...] = ()
        if compile_on:
            small_step_perturbation_compile_region_names = tuple(
                name
                for name in sorted(_SMALL_STEP_PERTURBATION_REGION_COMPILE_TARGETS)
                if _kernel_compile_enabled_for_device(name, config, device=device)
            )
        small_step_perturbation_payload_kwargs = {
            "device": device,
            "nCells": nCells,
            "nEdges": nEdges,
            "maxEdges": maxEdges_small_step_perturbation,
            "nRelaxZone": nRelaxZone_small_step_perturbation,
            "cell_start": cellSolveStart_small_step_perturbation - 1,
            "cell_end": cellSolveEnd_small_step_perturbation,
            "nEdgesOnCell": state_dict["nEdgesOnCell"],
            "edgesOnCell": state_dict["edgesOnCell"],
            "edgesOnCell_sign": state_dict["edgesOnCell_sign"],
            "bdyMaskCell": bdyMaskCell_small_step_perturbation,
        }
        small_step_perturbation_connectivity_payload = _prebound_static_payload(
            config,
            "small_step_perturbation",
            _small_step_perturbation_connectivity_payload_cache_key,
            _small_step_perturbation_connectivity_payload,
            small_step_perturbation_payload_kwargs,
        )
        small_step_perturbation_connectivity_key = (
            _small_step_perturbation_connectivity_payload_cache_key(
                **small_step_perturbation_payload_kwargs
            )
        )
        small_step_perturbation_numeric_plan = (
            _runner_owned_small_step_perturbation_numeric_plan(
                static_payloads_for_step,
                connectivity_payload=small_step_perturbation_connectivity_payload,
                connectivity_key=small_step_perturbation_connectivity_key,
                nVertLevels=nVertLevels,
                fzm=state_dict["fzm"],
                fzp=state_dict["fzp"],
                stage_key=("rk_step", int(rk_step)),
            )
        )

        # Call small-step-perturbation to modify w_tend for acoustic solver
        # Input: tend_rw from dynamics-tendency (stored as result['tend_rw'])
        # Output: modified w_tend to be used as tend_rw in acoustic step
        small_step_perturbation_result = profiled_call(
            "mpas.small_step_perturbation_smlstep_pert",
            config,
            atm_set_smlstep_pert_variables_work,
            nCells=nCells,
            nEdges=nEdges,
            nVertLevels=nVertLevels,
            maxEdges=maxEdges_small_step_perturbation,
            cellSolveStart=cellSolveStart_small_step_perturbation,
            cellSolveEnd=cellSolveEnd_small_step_perturbation,
            nRelaxZone=nRelaxZone_small_step_perturbation,
            nEdgesOnCell=state_dict["nEdgesOnCell"],
            cellsOnEdge=state_dict["cellsOnEdge"],
            edgesOnCell=state_dict["edgesOnCell"],
            edgesOnCell_sign=state_dict["edgesOnCell_sign"],
            bdyMaskCell=bdyMaskCell_small_step_perturbation,
            fzm=state_dict["fzm"],
            fzp=state_dict["fzp"],
            zb=state_dict["zb"],
            zb3=state_dict["zb3"],
            zb_cell=state_dict["zb_cell"],
            zb3_cell=state_dict["zb3_cell"],
            zz=state_dict["zz"],
            w_tend=result["tend_rw"],  # Input from dynamics-tendency
            u_tend=result["tend_ru"],  # Input from dynamics-tendency (read-only)
            _connectivity_payload=small_step_perturbation_connectivity_payload,
            _compile_region_names=small_step_perturbation_compile_region_names,
            _compile_mode=compile_mode,
            _compile_backend=compile_backend,
            _numeric_plan=small_step_perturbation_numeric_plan,
            _take_w_tend_ownership=True,
        )

        # Update tend_rw with the modified w_tend from small-step-perturbation
        # This will be used by the acoustic step (acoustic-step)
        result["tend_rw"] = small_step_perturbation_result["w_tend"]
        result["tend_w"] = small_step_perturbation_result["w_tend"]  # Keep in sync

    # Chain tap: small-step-perturbation-adjusted tendencies entering acoustic loop.
    sync_chain_tendency_aliases()

    # =========================================================================
    # Step 2.6: Boundary Tendency Adjustments (Regional MPAS only)

    # MUST happen BEFORE acoustic step loop for regional runs.
    # 1. adjust_speczone_tend: Zeros tendencies in specified zone
    # 2. adjust_relaxzone_tend: Applies Rayleigh damping in relaxation zone
    # =========================================================================
    config_apply_lbcs = config.get("config_apply_lbcs", False)
    config_apply_lbcs = _coerce_bool(config_apply_lbcs, "config_apply_lbcs")

    if config_apply_lbcs:
        from mpas_pytorch.boundary.boundary_tendencies import (  # type: ignore[import-not-found]
            adjust_relaxzone_tend,
            adjust_speczone_tend,
        )

        # Check for required boundary tendency fields
        required_tend_bdy_fields = [
            "bdyMaskEdge",
            "bdyMaskCell",
            "ru_driving",
            "rtheta_m_driving",
            "rho_zz_driving",
            "rw_driving",
            "meshScalingRegionalEdge",
            "meshScalingRegionalCell",
        ]
        missing_tend_bdy = [f for f in required_tend_bdy_fields if f not in state_dict]
        has_tend_bdy_data = len(missing_tend_bdy) == 0

        # Also need tendencies from dynamics-tendency
        has_tend_outputs = all(
            k in result for k in ["tend_ru", "tend_rtheta_m", "tend_rho_zz", "tend_rw"]
        )
        # Fallback: tend_rtheta_m might be called tend_rt, tend_rho_zz might be called tend_rho
        if not has_tend_outputs:
            has_tend_outputs = all(
                k in result for k in ["tend_ru", "tend_rt", "tend_rho", "tend_rw"]
            )

        if has_tend_bdy_data and has_tend_outputs:
            nRelaxZone = _scalar_int(state_dict, "nRelaxZone", 10)

            # Get dt values
            dts_val = config.get(
                "dts",
                config.get("dt", 720.0) / nAcousticSteps if nAcousticSteps > 0 else 1.0,
            )
            dt_full = config.get(
                "dt", dts_val * nAcousticSteps if nAcousticSteps > 0 else 720.0
            )
            dt_rk = dts_val

            # Map tendency names (dynamics-tendency uses tend_rt, tend_rho; boundary uses tend_rtheta_m, tend_rho_zz)
            tend_ru = result.get("tend_ru")
            tend_rtheta_m = result.get("tend_rtheta_m", result.get("tend_rt"))
            tend_rho_zz = result.get("tend_rho_zz", result.get("tend_rho"))
            tend_rw = result.get("tend_rw")

            # 1. Zero tendencies in specified zone (innermost boundary ring)
            speczone_result = adjust_speczone_tend(
                tend_ru=tend_ru,
                tend_rtheta_m=tend_rtheta_m,
                tend_rho_zz=tend_rho_zz,
                tend_rw=tend_rw,
                bdyMaskEdge=state_dict["bdyMaskEdge"],
                bdyMaskCell=state_dict["bdyMaskCell"],
                nRelaxZone=nRelaxZone,
                nEdges=nEdges,
                nCells=nCells,
                nVertLevels=nVertLevels,
                nVertLevelsP1=nVertLevels + 1,
            )

            # Update result with speczone-adjusted tendencies
            result["tend_ru"] = speczone_result["tend_ru"]
            if "tend_rtheta_m" in result:
                result["tend_rtheta_m"] = speczone_result["tend_rtheta_m"]
            else:
                result["tend_rt"] = speczone_result["tend_rtheta_m"]
            if "tend_rho_zz" in result:
                result["tend_rho_zz"] = speczone_result["tend_rho_zz"]
            else:
                result["tend_rho"] = speczone_result["tend_rho_zz"]
            result["tend_rw"] = speczone_result["tend_rw"]

            # 2. Apply Rayleigh damping in relaxation zone
            relaxzone_result = adjust_relaxzone_tend(
                tend_ru=result["tend_ru"],
                tend_rtheta_m=result.get("tend_rtheta_m", result.get("tend_rt")),
                tend_rho_zz=result.get("tend_rho_zz", result.get("tend_rho")),
                tend_rw=result["tend_rw"],
                ru=state_dict["ru"],
                rtheta_m=state_dict.get(
                    "rtheta_m", state_dict["theta_m"] * state_dict["rho_zz"]
                ),
                rho_zz=state_dict["rho_zz"],
                rw=state_dict["rw"],
                ru_driving=state_dict["ru_driving"],
                rtheta_m_driving=state_dict["rtheta_m_driving"],
                rho_zz_driving=state_dict["rho_zz_driving"],
                rw_driving=state_dict["rw_driving"],
                bdyMaskEdge=state_dict["bdyMaskEdge"],
                bdyMaskCell=state_dict["bdyMaskCell"],
                nRelaxZone=nRelaxZone,
                dt=dt_full,
                dt_rk=dt_rk,
                meshScalingRegionalEdge=state_dict["meshScalingRegionalEdge"],
                meshScalingRegionalCell=state_dict["meshScalingRegionalCell"],
                nEdges=nEdges,
                nCells=nCells,
                nVertLevels=nVertLevels,
                nVertLevelsP1=nVertLevels + 1,
            )

            # Update result with relaxzone-adjusted tendencies
            result["tend_ru"] = relaxzone_result["tend_ru"]
            if "tend_rtheta_m" in result:
                result["tend_rtheta_m"] = relaxzone_result["tend_rtheta_m"]
            else:
                result["tend_rt"] = relaxzone_result["tend_rtheta_m"]
            if "tend_rho_zz" in result:
                result["tend_rho_zz"] = relaxzone_result["tend_rho_zz"]
            else:
                result["tend_rho"] = relaxzone_result["tend_rho_zz"]
            result["tend_rw"] = relaxzone_result["tend_rw"]

        elif config_apply_lbcs and not has_tend_bdy_data:
            if strict:
                raise ValueError(
                    f"config_apply_lbcs=True but required boundary tendency fields missing: {missing_tend_bdy}. "
                    f"Either provide boundary data or set config_apply_lbcs=False."
                )
            else:
                warnings.warn(
                    f"Skipping boundary tendency adjustment: missing fields {missing_tend_bdy}. "
                    f"Regional boundary conditions will not be applied to tendencies.",
                    RuntimeWarning,
                )

    # =========================================================================
    # Step 3: Acoustic substep loop (acoustic-step)
    # Only execute if acoustic step data is available
    # Uses tend_ru, tend_rho, tend_rt, tend_rw from dynamics-tendency (Step 2) via small-step-perturbation (Step 2.5)
    # =========================================================================

    # Check if required acoustic step data is available
    acoustic_required_keys = [
        "dcEdge",
        "invDcEdge",
        "dvEdge",
        "invAreaCell",
        "nEdgesOnCell",
        "cellsOnEdge",
        "edgesOnCell",
        "edgesOnCell_sign",
        "rho_zz",
        "theta_m",
        "exner",
        "cqu",
        "zxu",
        "ru",
        "ru_save",
        "dss",
        "zgrid",
        "w",
        "rw",
        "rw_save",
    ]
    has_acoustic_data = all(
        key in state_dict or key in result for key in acoustic_required_keys
    )

    if nAcousticSteps > 0 and not has_acoustic_data:
        missing_keys = [
            k for k in acoustic_required_keys if k not in state_dict and k not in result
        ]
        if strict:
            raise ValueError(
                f"Cannot execute acoustic step (acoustic-step): missing required inputs: {missing_keys}. "
                f"Ensure checkpoint contains all acoustic step data or set strict=False to skip."
            )
        else:
            warnings.warn(
                f"Skipping acoustic step (acoustic-step): missing inputs: {missing_keys}. "
                f"Results will be incomplete.",
                RuntimeWarning,
            )

    if has_acoustic_data and nAcousticSteps > 0 and not vert_coefs:
        missing_coefs = [k for k in vert_coef_keys if k not in state_dict]
        msg = (
            "Cannot execute acoustic step (acoustic-step): vertical coefficients missing. "
            f"Missing: {missing_coefs}. "
            "Provide vertical-implicit-coefficients inputs so coefficients can be computed, or preload them in state_dict."
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg + " Skipping acoustic step.", RuntimeWarning)
        has_acoustic_data = False

    if has_acoustic_data and nAcousticSteps > 0:
        # Get additional dimensions needed for acoustic step
        nCellsSolve = _scalar_int(state_dict, "nCellsSolve", nCells)
        maxEdges = _scalar_int(state_dict, "maxEdges", 10)

        # Seed perturbation arrays from the current diagnostic state (Fortran
        # passes diag%ru_p/rw_p/rho_pp/rtheta_pp/ruAvg/wwAvg directly).
        # The small_step==1 branch in acoustic-step reinitializes interior solve ranges,
        # but untouched/sentinel entries can still retain incoming values.
        # Carrying the diagnostic arrays preserves that behavior across
        # RK stages and dynamics substeps.
        transfer_acoustic_seed_ownership = bool(
            _take_state_ownership and not torch.is_grad_enabled()
        )

        def acoustic_seed(key: str, shape: tuple[int, ...]) -> torch.Tensor:
            return _acoustic_seed_tensor(
                state_dict,
                key,
                shape,
                device=device,
                take_ownership=transfer_acoustic_seed_ownership,
                preserve_alias=(
                    _preserve_split_average_aliases and key in {"ruAvg", "wwAvg"}
                ),
            )

        ru_p = acoustic_seed("ru_p", (nEdges + 1, nVertLevels))
        rw_p = acoustic_seed("rw_p", (nCells + 1, nVertLevels + 1))
        rho_pp = acoustic_seed("rho_pp", (nCells + 1, nVertLevels))
        rtheta_pp = acoustic_seed("rtheta_pp", (nCells + 1, nVertLevels))
        rtheta_pp_old = acoustic_seed(
            "rtheta_pp_old",
            (nCells + 1, nVertLevels),
        )
        ruAvg = acoustic_seed("ruAvg", (nEdges + 1, nVertLevels))
        wwAvg = acoustic_seed("wwAvg", (nCells + 1, nVertLevels + 1))

        # Get scalar coefficients from state_dict (pre-computed in checkpoint or use defaults)
        cf1 = _scalar_float(state_dict, "cf1", 0.0)
        cf2 = _scalar_float(state_dict, "cf2", 1.0)
        cf3 = _scalar_float(state_dict, "cf3", 0.0)
        c2 = _scalar_float(state_dict, "c2", CP * RGAS / (CP - RGAS))
        rcv = _scalar_float(state_dict, "rcv", RGAS / (CP - RGAS))
        resm = _scalar_float(state_dict, "resm", (1.0 - epssm) / (1.0 + epssm))
        rdts = _scalar_float(state_dict, "rdts", 1.0 / dts)

        # Loop bounds (1-based Fortran indices)
        cellStart = _scalar_int(state_dict, "cellStart", 1)
        cellEnd = _scalar_int(state_dict, "cellEnd", nCells)
        edgeStart = _scalar_int(state_dict, "edgeStart", 1)
        edgeEnd = _scalar_int(state_dict, "edgeEnd", nEdges)
        cellSolveStart = _scalar_int(state_dict, "cellSolveStart", 1)
        cellSolveEnd = _scalar_int(state_dict, "cellSolveEnd", nCells)
        acoustic_checkpoint_enabled = (
            _coerce_bool(
                get_config_value_safe(config, "gradient_checkpointing", False),
                "gradient_checkpointing",
            )
            and torch.is_grad_enabled()
        )
        spec_zone_mask_cell = state_dict.get("specZoneMaskCell")
        spec_zone_mask_cell_known_all_zero = spec_zone_mask_cell is None
        if spec_zone_mask_cell is None:
            spec_zone_mask_cell = torch.zeros(nCells + 1, dtype=DTYPE, device=device)
        spec_zone_mask_edge = state_dict.get("specZoneMaskEdge")
        spec_zone_mask_edge_known_all_zero = spec_zone_mask_edge is None
        if spec_zone_mask_edge is None:
            spec_zone_mask_edge = torch.zeros(nEdges + 1, dtype=DTYPE, device=device)
        tend_ru_fallback = state_dict.get("tend_ru")
        if tend_ru_fallback is None:
            tend_ru_fallback = torch.zeros(
                nEdges + 1, nVertLevels, dtype=DTYPE, device=device
            )
        tend_rho_fallback = state_dict.get("tend_rho")
        if tend_rho_fallback is None:
            tend_rho_fallback = torch.zeros(
                nCells + 1, nVertLevels, dtype=DTYPE, device=device
            )
        tend_rt_fallback = state_dict.get("tend_rt")
        if tend_rt_fallback is None:
            tend_rt_fallback = torch.zeros(
                nCells + 1, nVertLevels, dtype=DTYPE, device=device
            )
        tend_rw_fallback = state_dict.get("tend_rw")
        if tend_rw_fallback is None:
            tend_rw_fallback = torch.zeros(
                nCells + 1, nVertLevels + 1, dtype=DTYPE, device=device
            )
        workspace_local = config.get("_dycore_workspace")
        acoustic_step_tridiag_scan_policy = normalize_tridiag_scan_policy(
            get_config_value_safe(
                config,
                "config_acoustic_step_tridiag_scan_policy",
                TRIDIAG_SCAN_DEFAULT_POLICY,
            )
        )
        acoustic_step_tridiag_scan_allowed_shapes = tuple(
            sorted(
                canonical_tridiag_scan_shape_keys(
                    get_config_value_safe(
                        config, "config_acoustic_step_tridiag_scan_allowed_shapes", ()
                    )
                )
            )
        )

        acoustic_step_compile_region_names: tuple[str, ...] = ()
        compile_whole_acoustic_step = False
        if compile_on:
            compile_whole_acoustic_step = _kernel_compile_enabled_for_device(
                "acoustic_step_acoustic_step",
                config,
                device=device,
            )
            if not compile_whole_acoustic_step:
                acoustic_step_compile_region_names = tuple(
                    name
                    for name in sorted(_ACOUSTIC_STEP_REGION_COMPILE_TARGETS)
                    if _kernel_compile_enabled_for_device(name, config, device=device)
                )

        # Topology ownership is independent of compile grouping. Build and cache
        # this payload even when a diagnostic forces the blocked whole wrapper;
        # no path may fall back to deriving relations inside repeated substeps.
        acoustic_step_payload_kwargs = {
            "device": device,
            "nCells": nCells,
            "nEdges": nEdges,
            "nCellsSolve": nCellsSolve,
            "maxEdges": maxEdges,
            "edge_start": edgeStart - 1,
            "edge_end": edgeEnd,
            "cell_solve_start": cellSolveStart - 1,
            "cell_solve_end": cellSolveEnd,
            "cellsOnEdge": state_dict["cellsOnEdge"],
            "edgesOnCell": state_dict["edgesOnCell"],
            "nEdgesOnCell": state_dict["nEdgesOnCell"],
            "edgesOnCell_sign": state_dict["edgesOnCell_sign"],
            "specZoneMaskCell": spec_zone_mask_cell,
            "specZoneMaskEdge": spec_zone_mask_edge,
            "specZoneMaskCellKnownAllZero": spec_zone_mask_cell_known_all_zero,
            "specZoneMaskEdgeKnownAllZero": spec_zone_mask_edge_known_all_zero,
            "dvEdge": state_dict["dvEdge"],
            "invAreaCell": state_dict["invAreaCell"],
        }
        acoustic_step_connectivity_payload = _prebound_static_payload(
            config,
            "acoustic_step",
            _acoustic_step_connectivity_payload_cache_key,
            _acoustic_step_connectivity_payload,
            acoustic_step_payload_kwargs,
        )

        acoustic_step_acoustic_solve_plan = None
        if acoustic_step_connectivity_payload is not None:
            active_cell_edge_relation = acoustic_step_connectivity_payload.get(
                "active_cell_edge_relation"
            )
            active_edge_cell_relation = acoustic_step_connectivity_payload.get(
                "active_edge_cell_relation"
            )
            if not isinstance(
                active_cell_edge_relation, StaticSlotRelation
            ) or not isinstance(
                active_edge_cell_relation,
                StaticPairRelation,
            ):
                raise TypeError(
                    "acoustic-step numeric plan requires typed active topology relations"
                )
            candidate_acoustic_step_plan = build_acoustic_step_acoustic_solve_plan(
                nVertLevels=nVertLevels,
                dts=dts,
                resm=resm,
                epssm=epssm,
                cell_edge_relation=active_cell_edge_relation,
                edge_cell_relation=active_edge_cell_relation,
                full_solve_compile_path=bool(
                    acoustic_step_connectivity_payload["full_solve_compile_path"]
                ),
                solve_active_mask=acoustic_step_connectivity_payload[
                    "solve_active_mask"
                ],
                stage_key=(
                    "rk_step",
                    int(rk_step),
                    "scan_policy",
                    acoustic_step_tridiag_scan_policy,
                    "scan_shapes",
                    acoustic_step_tridiag_scan_allowed_shapes,
                ),
            )
            acoustic_step_acoustic_solve_plan = (
                static_payloads_for_step.get_or_build_numeric_plan(
                    "acoustic_step_acoustic_solve",
                    candidate_acoustic_step_plan.key,
                    lambda: candidate_acoustic_step_plan,
                )
            )
            if (
                "acoustic_step_acoustic_flux_solve_region"
                in acoustic_step_compile_region_names
            ):
                compiled_acoustic_step_solve = _maybe_compile_kernel(
                    "acoustic_step_acoustic_flux_solve_region",
                    acoustic_step_acoustic_solve_plan.numeric_callable,
                    enabled=True,
                    mode=compile_mode,
                    backend=compile_backend,
                    specialization_key=acoustic_step_acoustic_solve_plan.key,
                )
                acoustic_step_acoustic_solve_plan = (
                    acoustic_step_acoustic_solve_plan.with_numeric_callable(
                        compiled_acoustic_step_solve
                    )
                )

        config_smdiv_divergence = get_config_value_safe(config, "config_smdiv", 0.1)
        config_len_disp_divergence = _resolve_len_disp(
            config, state_dict, strict=strict
        )
        divergence_compile_region_names: tuple[str, ...] = ()
        if compile_on:
            divergence_compile_region_names = tuple(
                name
                for name in sorted(_DIVERGENCE_DAMPING_REGION_COMPILE_TARGETS)
                if _kernel_compile_enabled_for_device(name, config, device=device)
            )
        divergence_enabled = (
            config_smdiv_divergence > 0.0
            and config_len_disp_divergence > 0.0
            and dts > 0.0
        )
        divergence_connectivity_payload = None
        divergence_numeric_plan = None
        if divergence_enabled:
            divergence_payload_kwargs = {
                "device": device,
                "nCells": nCells,
                "nEdges": nEdges,
                "nCellsSolve": nCellsSolve,
                "edge_start": edgeStart - 1,
                "edge_end": edgeEnd,
                "cellsOnEdge": state_dict["cellsOnEdge"],
                "specZoneMaskEdge": spec_zone_mask_edge,
                "specZoneMaskEdgeKnownAllZero": spec_zone_mask_edge_known_all_zero,
            }
            divergence_connectivity_payload = _prebound_static_payload(
                config,
                "divergence_damping",
                _divergence_connectivity_payload_cache_key,
                _divergence_connectivity_payload,
                divergence_payload_kwargs,
            )
            divergence_connectivity_key = _divergence_connectivity_payload_cache_key(
                **divergence_payload_kwargs
            )
            divergence_numeric_plan = _runner_owned_divergence_damping_plan(
                static_payloads_for_step,
                connectivity_payload=divergence_connectivity_payload,
                connectivity_key=divergence_connectivity_key,
                nVertLevels=nVertLevels,
                dts=dts,
                config_smdiv=config_smdiv_divergence,
                config_len_disp=config_len_disp_divergence,
                stage_key=("rk_step", int(rk_step)),
            )

        # Acoustic substep loop
        for small_step in range(1, nAcousticSteps + 1):
            tend_ru_local = result.get(
                "tend_ru",
                tend_ru_fallback,
            )
            tend_rho_local = result.get(
                "tend_rho",
                tend_rho_fallback,
            )
            tend_rt_local = result.get(
                "tend_rt",
                tend_rt_fallback,
            )
            tend_rw_local = result.get(
                "tend_rw",
                tend_rw_fallback,
            )
            acoustic_kernel = atm_advance_acoustic_step_work
            if compile_on and compile_whole_acoustic_step:
                acoustic_kernel = _maybe_compile_kernel(
                    "acoustic_step_acoustic_step",
                    atm_advance_acoustic_step_work,
                    enabled=True,
                    mode=compile_mode,
                    backend=compile_backend,
                    specialization_key=_compiled_kernel_specialization_key(
                        config,
                        "acoustic_step_acoustic_step",
                        "rk_step",
                        rk_step,
                        "small_step",
                        small_step,
                        "nAcousticSteps",
                        nAcousticSteps,
                        "dts",
                        dts,
                        "tridiag_scan_policy",
                        acoustic_step_tridiag_scan_policy,
                        "tridiag_scan_shapes",
                        acoustic_step_tridiag_scan_allowed_shapes,
                    ),
                )

            def _run_acoustic_substep(
                ru_p_in: torch.Tensor,
                rw_p_in: torch.Tensor,
                rho_pp_in: torch.Tensor,
                rtheta_pp_in: torch.Tensor,
                rtheta_pp_old_in: torch.Tensor,
                ruavg_in: torch.Tensor,
                wwavg_in: torch.Tensor,
                small_step_local: int = small_step,
                spec_zone_mask_cell_local: torch.Tensor | None = spec_zone_mask_cell,
                spec_zone_mask_edge_local: torch.Tensor | None = spec_zone_mask_edge,
                acoustic_kernel_local: Callable[..., dict[str, Any]] = acoustic_kernel,
                tend_ru_bound: torch.Tensor = tend_ru_local,
                tend_rho_bound: torch.Tensor = tend_rho_local,
                tend_rt_bound: torch.Tensor = tend_rt_local,
                tend_rw_bound: torch.Tensor = tend_rw_local,
            ) -> dict[str, Any]:
                # Keep non-checkpoint tensor inputs closure-captured and immutable:
                # mesh/state geometry and dynamics-tendency tendencies are fixed over the
                # acoustic substep loop and are not mutated between forward and
                # checkpoint recompute.

                def _acoustic_kernel_with_scan_plan_cache(
                    **kwargs: Any,
                ) -> dict[str, Any]:
                    with tridiag_scan_plan_cache(
                        config.get("_tridiag_scan_plan_cache")
                    ):
                        return acoustic_kernel_local(**kwargs)

                return cast(
                    dict[str, Any],
                    profiled_call(
                        "mpas.acoustic_step_acoustic_step",
                        config,
                        _acoustic_kernel_with_scan_plan_cache,
                        # Dimensions
                        nCells=nCells,
                        nEdges=nEdges,
                        nCellsSolve=nCellsSolve,
                        nVertLevels=nVertLevels,
                        maxEdges=maxEdges,
                        # Loop bounds (1-based Fortran indices)
                        cellStart=cellStart,
                        cellEnd=cellEnd,
                        edgeStart=edgeStart,
                        edgeEnd=edgeEnd,
                        cellSolveStart=cellSolveStart,
                        cellSolveEnd=cellSolveEnd,
                        # Time stepping
                        small_step=small_step_local,
                        dts=dts,
                        epssm=epssm,
                        # Pre-computed scalar coefficients
                        cf1=cf1,
                        cf2=cf2,
                        cf3=cf3,
                        c2=c2,
                        rcv=rcv,
                        resm=resm,
                        rdts=rdts,
                        # 1D arrays
                        fzm=state_dict["fzm"],
                        fzp=state_dict["fzp"],
                        rdzw=state_dict["rdzw"],
                        cofrz=vert_coefs["cofrz"],
                        # Edge/cell geometry arrays
                        dcEdge=state_dict["dcEdge"],
                        invDcEdge=state_dict["invDcEdge"],
                        dvEdge=state_dict["dvEdge"],
                        invAreaCell=state_dict["invAreaCell"],
                        specZoneMaskCell=spec_zone_mask_cell_local,
                        specZoneMaskEdge=spec_zone_mask_edge_local,
                        # Connectivity
                        nEdgesOnCell=state_dict["nEdgesOnCell"],
                        cellsOnEdge=state_dict["cellsOnEdge"],
                        edgesOnCell=state_dict["edgesOnCell"],
                        edgesOnCell_sign=state_dict["edgesOnCell_sign"],
                        # 2D cell-based arrays
                        rho_zz=state_dict["rho_zz"],
                        # Fortran reads theta_m from time level 1 (initial state) in the
                        # acoustic step:
                        #   call mpas_pool_get_array(state, 'theta_m', theta_m, 1)
                        # Time level 1 is NEVER modified during the RK loop, so the
                        # acoustic step always sees the beginning-of-timestep theta_m.
                        # In Python, theta_m_init holds the time-level-1 value (set once
                        # in atm_rk_integration_setup), while theta_m is updated by large-step-recovery
                        # after each RK stage.
                        theta_m=state_dict.get("theta_m_init", state_dict["theta_m"]),
                        zz=state_dict["zz"],
                        exner=state_dict["exner"],
                        rho_pp=rho_pp_in,
                        rtheta_pp=rtheta_pp_in,
                        rtheta_pp_old=rtheta_pp_old_in,
                        a_tri=vert_coefs.get("a_tri"),
                        alpha_tri=vert_coefs.get("alpha_tri"),
                        gamma_tri=vert_coefs.get("gamma_tri"),
                        cofwt=vert_coefs.get("cofwt"),
                        cofwr=vert_coefs.get("cofwr"),
                        cofwz=vert_coefs.get("cofwz"),
                        dss=state_dict["dss"],
                        # Edge-based arrays
                        cqu=result["cqu"],
                        zxu=state_dict["zxu"],
                        ru=state_dict["ru"],
                        ru_save=state_dict["ru_save"],
                        ru_p=ru_p_in,
                        ruAvg=ruavg_in,
                        # Use tend_* from dynamics-tendency (Step 2) if computed, otherwise from state_dict, else zeros
                        tend_ru=tend_ru_bound,
                        tend_rho=tend_rho_bound,
                        tend_rt=tend_rt_bound,
                        # Vertical velocity arrays (nVertLevels+1)
                        coftz=vert_coefs.get("coftz"),
                        zgrid=state_dict["zgrid"],
                        w=state_dict["w"],
                        rw=state_dict["rw"],
                        rw_save=state_dict["rw_save"],
                        rw_p=rw_p_in,
                        wwAvg=wwavg_in,
                        # Use tend_rw from dynamics-tendency (Step 2) if computed, otherwise from state_dict, else zeros
                        tend_rw=tend_rw_bound,
                        workspace=workspace_local,
                        _compile_region_names=acoustic_step_compile_region_names,
                        _compile_mode=compile_mode,
                        _compile_backend=compile_backend,
                        _connectivity_payload=acoustic_step_connectivity_payload,
                        _acoustic_solve_plan=acoustic_step_acoustic_solve_plan,
                        _tridiag_scan_policy=acoustic_step_tridiag_scan_policy,
                        _tridiag_scan_allowed_shapes=acoustic_step_tridiag_scan_allowed_shapes,
                        _vertical_implicit_coefficients_active_cell_start=vert_coefs.get(
                            "_vertical_implicit_coefficients_active_cell_start"
                        ),
                        _vertical_implicit_coefficients_active_cell_end=vert_coefs.get(
                            "_vertical_implicit_coefficients_active_cell_end"
                        ),
                        _vertical_implicit_coefficients_cofwt_active=vert_coefs.get(
                            "_vertical_implicit_coefficients_cofwt_active"
                        ),
                        _vertical_implicit_coefficients_cofwr_active=vert_coefs.get(
                            "_vertical_implicit_coefficients_cofwr_active"
                        ),
                        _vertical_implicit_coefficients_cofwz_active=vert_coefs.get(
                            "_vertical_implicit_coefficients_cofwz_active"
                        ),
                        _vertical_implicit_coefficients_coftz_active=vert_coefs.get(
                            "_vertical_implicit_coefficients_coftz_active"
                        ),
                        _vertical_implicit_coefficients_a_tri_active=vert_coefs.get(
                            "_vertical_implicit_coefficients_a_tri_active"
                        ),
                        _vertical_implicit_coefficients_alpha_tri_active=vert_coefs.get(
                            "_vertical_implicit_coefficients_alpha_tri_active"
                        ),
                        _vertical_implicit_coefficients_gamma_tri_active=vert_coefs.get(
                            "_vertical_implicit_coefficients_gamma_tri_active"
                        ),
                    ),
                )

            acoustic_result = checkpoint_if_enabled(
                _run_acoustic_substep,
                ru_p,
                rw_p,
                rho_pp,
                rtheta_pp,
                rtheta_pp_old,
                ruAvg,
                wwAvg,
                enabled=acoustic_checkpoint_enabled,
                use_reentrant=False,
                context_fn=_SAVE_NOTHING_CONTEXT
                if acoustic_checkpoint_enabled
                else None,
            )
            if compile_whole_acoustic_step:
                acoustic_result = _clone_compiled_output_mapping(acoustic_result)

            # Update state with acoustic step outputs for next iteration
            ru_p = acoustic_result["ru_p"]
            rw_p = acoustic_result["rw_p"]
            rho_pp = acoustic_result["rho_pp"]
            rtheta_pp = acoustic_result["rtheta_pp"]
            rtheta_pp_old = acoustic_result["rtheta_pp_old"]
            ruAvg = acoustic_result["ruAvg"]
            wwAvg = acoustic_result["wwAvg"]

            # -----------------------------------------------------------------
            # Apply 3D divergence damping (if enabled)
            # Upstream calls atm_divergence_damping_3d after each acoustic step.

            # This damps acoustic noise based on divergence gradient between cells.
            # Fortran calls this UNCONDITIONALLY; damping is controlled by smdiv and
            # len_disp: coef_divdamp = 2.0 * smdiv * len_disp * rdts
            # If smdiv=0 or len_disp=0, the damping coefficient naturally becomes zero.
            # Removed incorrect gating on apply_lbcs and 3d_div_damp.
            # -----------------------------------------------------------------
            # Gate on smdiv and len_disp, NOT on apply_lbcs or 3d_div_damp
            # Damping runs when the coefficient is non-zero.
            if divergence_enabled:
                div_damp_result = profiled_call(
                    "mpas.divergence_damping",
                    config,
                    atm_divergence_damping_3d,
                    nCells=nCells,
                    nEdges=nEdges,
                    nCellsSolve=nCellsSolve,
                    nVertLevels=nVertLevels,
                    edgeStart=edgeStart,
                    edgeEnd=edgeEnd,
                    dts=dts,
                    config_smdiv=config_smdiv_divergence,
                    config_len_disp=config_len_disp_divergence,
                    specZoneMaskEdge=spec_zone_mask_edge,
                    cellsOnEdge=state_dict["cellsOnEdge"],
                    # Fortran reads theta_m from time level 1 (initial state)
                    # in divergence damping:
                    #   call mpas_pool_get_array(state, 'theta_m', theta_m, 1)
                    theta_m=state_dict.get("theta_m_init", state_dict["theta_m"]),
                    rtheta_pp=rtheta_pp,
                    rtheta_pp_old=rtheta_pp_old,
                    ru_p=ru_p,
                    _compile_region_names=divergence_compile_region_names,
                    _compile_mode=compile_mode,
                    _compile_backend=compile_backend,
                    _connectivity_payload=divergence_connectivity_payload,
                    _allow_inplace_update=True,
                    _numeric_plan=divergence_numeric_plan,
                )
                ru_p = div_damp_result["ru_p"]

        # Update result with final acoustic state
        result["ru_p"] = ru_p
        result["rw_p"] = rw_p
        result["rho_pp"] = rho_pp
        result["rtheta_pp"] = rtheta_pp
        result["ruAvg"] = ruAvg
        result["wwAvg"] = wwAvg
        # Alias uhAvg = ruAvg for scalar advection.
        # Upstream diagnostic-pool alias: ruAvg -> uhAvg.
        result["uhAvg"] = result["ruAvg"]

    if should_publish_deferred_vert_coefs:
        if bool(config.get("_defer_vertical_implicit_coefficients_public_pack", False)):
            # Do not keep stale padded fields alive beside the new compact
            # coefficients.  acoustic-step consumes the active patches directly and the
            # full timestep publishes compatibility fields at its true boundary.
            for key in PUBLIC_VERT_IMP_COEF_KEYS:
                if key != "cofrz":
                    result.pop(key, None)
            result.update(vert_coefs)
        else:
            packed_vert_coefs = pack_vert_imp_coefs_public(
                vert_coefs,
                nCells=nCells,
                nVertLevels=nVertLevels,
                device=device,
            )
            result.update(packed_vert_coefs)
            vert_coefs.update(packed_vert_coefs)

    # Chain tap: acoustic loop outputs (with final perturbations/averages).
    sync_chain_tendency_aliases()

    # =========================================================================
    # Step 4: Recover large step variables (large-step-recovery)
    # Called after acoustic loop completes to recover full state from perturbations
    # =========================================================================

    # Check if large-step-recovery required inputs are available
    large_step_recovery_required = [
        "ru_p",
        "rw_p",
        "rho_pp",
        "rtheta_pp",
        "ruAvg",
        "wwAvg",
        "rho_p_save",
        "rho_base",
        "rtheta_p_save",
        "rtheta_base",
        "ru_save",
        "rw_save",
        "rt_diabatic_tend",
        "exner_base",
        "pressure_b",
        "bdyMaskCell",
        "zb_cell",
        "zb3_cell",
        "zb",
        "zb3",
        "cellsOnEdge",
        "edgesOnCell",
        "nEdgesOnCell",
        "edgesOnCell_sign",
        "fzm",
        "fzp",
        "zz",
        "scalars",
    ]
    can_run_large_step_recovery = all(
        key in state_dict or key in result for key in large_step_recovery_required
    )

    # If acoustic step ran, large-step-recovery is required to recover state
    acoustic_ran = has_acoustic_data and nAcousticSteps > 0
    if acoustic_ran and not can_run_large_step_recovery:
        missing_keys = [
            k
            for k in large_step_recovery_required
            if k not in state_dict and k not in result
        ]
        if strict:
            raise ValueError(
                f"Cannot execute large step recovery (large-step-recovery): missing required inputs: {missing_keys}. "
                f"large-step-recovery is required after acoustic step to recover full state variables. "
                f"Ensure checkpoint contains all large-step-recovery data or set strict=False to skip."
            )
        else:
            warnings.warn(
                f"Skipping large step recovery (large-step-recovery): missing inputs: {missing_keys}. "
                f"Results will be incomplete - state variables not recovered from perturbations.",
                RuntimeWarning,
            )

    if can_run_large_step_recovery:
        # Get dimensions
        nCellsSolve = _scalar_int(state_dict, "nCellsSolve", nCells)
        nEdgesSolve = _scalar_int(state_dict, "nEdgesSolve", nEdges)
        maxEdges = _scalar_int(state_dict, "maxEdges", 10)

        # Get ns (number of acoustic substeps) from config or state_dict
        ns = config.get("ns", nAcousticSteps)
        ns = _coerce_int(ns, "ns")

        # Get rk_step from config or use the function argument
        recovery_rk_step = config.get("rk_step", rk_step)
        recovery_rk_step = _coerce_int(recovery_rk_step, "recovery_rk_step")

        # Get dt (large timestep) from config or state_dict
        dt = config.get("dt", state_dict.get("dt", dts * ns))
        dt = _coerce_float(dt, "dt")

        # Get pre-computed coefficients
        cf1 = _scalar_float(state_dict, "cf1", 0.0)
        cf2 = _scalar_float(state_dict, "cf2", 1.0)
        cf3 = _scalar_float(state_dict, "cf3", 0.0)

        # Loop bounds (1-based Fortran indices)
        cellStart = _scalar_int(state_dict, "cellStart", 1)
        cellEnd = _scalar_int(state_dict, "cellEnd", nCells)
        nVertices = _scalar_int(state_dict, "nVertices", nCells)
        vertexStart = _scalar_int(state_dict, "vertexStart", 1)
        vertexEnd = _scalar_int(state_dict, "vertexEnd", nVertices)
        edgeStart = _scalar_int(state_dict, "edgeStart", 1)
        edgeEnd = _scalar_int(state_dict, "edgeEnd", nEdges)
        cellSolveStart = _scalar_int(state_dict, "cellSolveStart", 1)
        cellSolveEnd = _scalar_int(state_dict, "cellSolveEnd", nCells)
        vertexSolveStart = _scalar_int(state_dict, "vertexSolveStart", 1)
        vertexSolveEnd = _scalar_int(state_dict, "vertexSolveEnd", nVertices)
        edgeSolveStart = _scalar_int(state_dict, "edgeSolveStart", 1)
        edgeSolveEnd = _scalar_int(state_dict, "edgeSolveEnd", nEdges)
        nRelaxZone = _scalar_int(state_dict, "nRelaxZone", 0)

        # Helper to get tensor from result first, then state_dict
        def get_tensor(key):
            if key in result:
                return result[key]
            return state_dict[key]

        large_step_recovery_compile_region_names: tuple[str, ...] = ()
        if compile_on:
            large_step_recovery_compile_region_names = tuple(
                name
                for name in sorted(_LARGE_STEP_RECOVERY_REGION_COMPILE_TARGETS)
                if _kernel_compile_enabled_for_device(name, config, device=device)
            )

        large_step_recovery_connectivity_payload = None
        large_step_recovery_payload_kwargs = {
            "device": device,
            "nCells": nCells,
            "nEdges": nEdges,
            "maxEdges": maxEdges,
            "nRelaxZone": nRelaxZone,
            "cell_start": cellStart - 1,
            "cell_end": cellEnd,
            "edge_start": edgeStart - 1,
            "edge_end": edgeEnd,
            "cellsOnEdge": state_dict["cellsOnEdge"],
            "edgesOnCell": state_dict["edgesOnCell"],
            "nEdgesOnCell": state_dict["nEdgesOnCell"],
            "edgesOnCell_sign": state_dict["edgesOnCell_sign"],
            "bdyMaskCell": state_dict["bdyMaskCell"],
        }
        large_step_recovery_connectivity_payload = _prebound_static_payload(
            config,
            "large_step_recovery",
            _large_step_recovery_connectivity_payload_cache_key,
            _large_step_recovery_connectivity_payload,
            large_step_recovery_payload_kwargs,
        )
        large_step_recovery_recovery_plan = None
        if large_step_recovery_connectivity_payload is not None:
            large_step_recovery_stage_key = ("rk_step", int(recovery_rk_step))
            large_step_recovery_connectivity_key = (
                _large_step_recovery_connectivity_payload_cache_key(
                    **large_step_recovery_payload_kwargs
                )
            )
            large_step_recovery_plan_kwargs = {
                "connectivity_payload": large_step_recovery_connectivity_payload,
                "connectivity_key": large_step_recovery_connectivity_key,
                "nVertLevels": nVertLevels,
                "ns": ns,
                "rk_step": recovery_rk_step,
                "dt": dt,
                "cf1": cf1,
                "cf2": cf2,
                "cf3": cf3,
                "fzm": state_dict["fzm"],
                "fzp": state_dict["fzp"],
                "stage_key": large_step_recovery_stage_key,
            }
            large_step_recovery_recovery_plan = (
                _runner_owned_large_step_recovery_recovery_plan(
                    static_payloads_for_step,
                    **large_step_recovery_plan_kwargs,
                )
            )
            if (
                "large_step_recovery_recovery_region"
                in large_step_recovery_compile_region_names
            ):
                compiled_large_step_recovery_recovery = _maybe_compile_kernel(
                    "large_step_recovery_recovery_region",
                    large_step_recovery_recovery_plan.numeric_callable,
                    enabled=True,
                    mode=compile_mode,
                    backend=compile_backend,
                    specialization_key=large_step_recovery_recovery_plan.key,
                )
                large_step_recovery_recovery_plan = (
                    large_step_recovery_recovery_plan.with_numeric_callable(
                        compiled_large_step_recovery_recovery
                    )
                )

        recovery_result = profiled_call(
            "mpas.large_step_recovery_recover_large_step",
            config,
            atm_recover_large_step_variables_work,
            # Dimensions
            nCells=nCells,
            nEdges=nEdges,
            nCellsSolve=nCellsSolve,
            nEdgesSolve=nEdgesSolve,
            nVertLevels=nVertLevels,
            maxEdges=maxEdges,
            # Time stepping
            ns=ns,
            rk_step=recovery_rk_step,
            dt=dt,
            # Loop bounds (1-based Fortran indices)
            cellStart=cellStart,
            cellEnd=cellEnd,
            vertexStart=vertexStart,
            vertexEnd=vertexEnd,
            edgeStart=edgeStart,
            edgeEnd=edgeEnd,
            cellSolveStart=cellSolveStart,
            cellSolveEnd=cellSolveEnd,
            vertexSolveStart=vertexSolveStart,
            vertexSolveEnd=vertexSolveEnd,
            edgeSolveStart=edgeSolveStart,
            edgeSolveEnd=edgeSolveEnd,
            # Regional MPAS
            nRelaxZone=nRelaxZone,
            # Pre-computed coefficients
            cf1=cf1,
            cf2=cf2,
            cf3=cf3,
            # 1D arrays
            fzm=state_dict["fzm"],
            fzp=state_dict["fzp"],
            # Connectivity
            cellsOnEdge=state_dict["cellsOnEdge"],
            edgesOnCell=state_dict["edgesOnCell"],
            nEdgesOnCell=state_dict["nEdgesOnCell"],
            edgesOnCell_sign=state_dict["edgesOnCell_sign"],
            bdyMaskCell=state_dict["bdyMaskCell"],
            # Cell-based 2D arrays (from state_dict)
            rho_p_save=state_dict["rho_p_save"],
            rho_pp=get_tensor("rho_pp"),
            rho_base=state_dict["rho_base"],
            rtheta_p_save=state_dict["rtheta_p_save"],
            rtheta_pp=get_tensor("rtheta_pp"),
            rtheta_base=state_dict["rtheta_base"],
            rt_diabatic_tend=get_tensor("rt_diabatic_tend"),
            exner_base=state_dict["exner_base"],
            zz=state_dict["zz"],
            pressure_b=state_dict["pressure_b"],
            # Cell-based 2D arrays (nVertLevels+1)
            wwAvg=get_tensor("wwAvg"),
            rw_save=state_dict["rw_save"],
            rw_p=get_tensor("rw_p"),
            # Edge-based 2D arrays
            ruAvg=get_tensor("ruAvg"),
            ru_save=state_dict["ru_save"],
            ru_p=get_tensor("ru_p"),
            # 3D terrain coefficient arrays
            zb_cell=state_dict["zb_cell"],
            zb3_cell=state_dict["zb3_cell"],
            zb=state_dict["zb"],
            zb3=state_dict["zb3"],
            # Scalars
            scalars=state_dict["scalars"],
            _connectivity_payload=large_step_recovery_connectivity_payload,
            _compile_region_names=large_step_recovery_compile_region_names,
            _compile_mode=compile_mode,
            _compile_backend=compile_backend,
            _recovery_plan=large_step_recovery_recovery_plan,
        )

        # Update result with recovery outputs
        result.update(recovery_result)

        # Keep recovery outputs MPAS-consistent: valid data in rows 0..N-1,
        # garbage row at index N untouched (except rho_zz[nCells]=1 set inside large-step-recovery).
        # GAP-C2: Update rho_zz_new from large-step-recovery recovery output
        # Fortran: large-step-recovery writes into rho_zz(:,:,2), which scalar advection reads as rho_zz_new
        if "rho_zz" in result:
            result["rho_zz_new"] = _clone_for_grad(result["rho_zz"])
            # Keep diagnostic-solve mass alias in sync with latest recovered density.
            result["h"] = result["rho_zz"]
        # Keep scalar-advection horizontal flux alias synchronized with the
        # recovered ruAvg produced by large-step-recovery.
        if "ruAvg" in result:
            result["uhAvg"] = _clone_for_grad(result["ruAvg"])

    # Chain tap: recovered prognostic state after large-step-recovery.
    sync_chain_tendency_aliases()

    # =========================================================================
    # Step 5: Scalar Advection (explicit-scalar-transport explicit or monotonic-scalar-transport monotonic)
    # In coupled mode with split transport disabled,
    # scalar advection runs BEFORE solve diagnostics within each RK step.

    # Upstream MPAS behavior:
    # - Simple advection (explicit-scalar-transport) for rk_step < 3
    # - Monotonic/positive-definite advection (monotonic-scalar-transport) ONLY on final RK step (rk_step == 3)
    #   and only if config_monotonic or config_positive_definite is enabled

    # GAP-4: When split dynamics-transport is enabled, scalar advection is
    # SKIPPED inside the dynamics RK loop and done in a separate transport
    # RK3 loop after all dynamics substeps complete.
    # =========================================================================
    # execute_rk_step is used both by atm_srk3 (full timestep orchestration)
    # and by direct kernel-level tests with minimal config dicts.

    # For full runs, split-dynamics/transport intent must come from explicit
    # config keys. For minimal direct calls that omit split keys, default to
    # coupled mode so explicit-scalar-transport/monotonic-scalar-transport behavior can be validated in isolation.
    split_keys_present = any(
        key in config
        for key in (
            "config_split_dynamics_transport",
            "split_dynamics_transport",
            "config_dynamics_split_steps",
        )
    )
    config_split_dynamics_transport = (
        _resolve_dynamics_split(config) > 1 if split_keys_present else False
    )
    config_apply_lbcs = get_config_value_safe(config, "config_apply_lbcs", False)

    # Check if explicit-scalar-transport/monotonic-scalar-transport required inputs are available
    # explicit-scalar-transport (explicit) requires: scalar_old, scalar_new, uhAvg, wwAvg, etc.
    # monotonic-scalar-transport (monotonic) requires: scalars_old, scalars_new, uhAvg, wwAvg, etc.
    # Use centralized Fortran-compatible defaults for limiter flags
    config_monotonic = get_config_value_safe(config, "monotonic", False)
    config_positive_definite = get_config_value_safe(config, "positive_definite", False)

    # Scalar advection is only strictly required if explicitly requested via config.
    # The presence of 'num_scalars' in state_dict doesn't mean advection must run.
    # Use centralized Fortran-compatible defaults
    advect_scalars_requested = get_config_value_safe(config, "advect_scalars", True)

    # Determine which advection method to use (matching Fortran logic)
    # Upstream condition: rk_step < 3 or neither limiter is enabled.
    #   -> use simple advection (atm_advance_scalars)
    # else
    #   -> use monotonic or positive-definite advection

    # In other words: use limiters ONLY on final RK step (rk_step == 3) AND only if configured
    use_limiters = (
        rk_step == 3  # Only on final RK step
        and (config_monotonic or config_positive_definite)
    )

    # Honor config_scalar_advection/config['advect_scalars'] gate exactly.
    # Fortran only calls scalar transport when config_scalar_advection is true.
    # When false, both explicit-scalar-transport/monotonic-scalar-transport (and downstream scalar boundary adjustment)
    # must be skipped in the coupled RK loop.
    if not advect_scalars_requested:
        pass
    # GAP-4: When split dynamics-transport is enabled, scalar advection
    # is deferred to a separate transport RK3 loop after all dynamics
    # substeps complete. Skip it here inside the dynamics RK loop.
    elif config_split_dynamics_transport:
        # Scalar advection (explicit-scalar-transport/monotonic-scalar-transport) and boundary adjustment (transported-scalar boundary adjustment)
        # will be handled in the split transport path in atm_srk3.
        pass
    elif use_limiters:
        # Monotonic/positive-definite advection for final RK step only (rk_step == 3)
        # This branch is reached only when: rk_step == 3 AND (config_monotonic OR config_positive_definite)
        # Check for monotonic-scalar-transport monotonic advection required inputs
        monotonic_scalar_transport_required = [
            "scalars_old",
            "scalars_new",
            "uhAvg",
            "wwAvg",
            "rho_zz_old",
            "rho_zz_new",
            "fnm",
            "fnp",
            "rdnw",
            "dvEdge",
            "invAreaCell",
            "nEdgesOnCell",
            "nAdvCellsForEdge",
            "bdyMaskCell",
            "bdyMaskEdge",
            "cellsOnEdge",
            "cellsOnCell",
            "edgesOnCell",
            "advCellsForEdge",
            "edgesOnCell_sign",
            "adv_coefs",
            "adv_coefs_3rd",
            "coef_3rd_order",
            "num_scalars",
        ]
        can_run_monotonic_scalar_transport = all(
            key in state_dict or key in result
            for key in monotonic_scalar_transport_required
        )

        # If scalar advection is explicitly requested but data is missing, handle strict mode
        if advect_scalars_requested and not can_run_monotonic_scalar_transport:
            missing_keys = [
                k
                for k in monotonic_scalar_transport_required
                if k not in state_dict and k not in result
            ]
            if strict:
                raise ValueError(
                    f"Cannot execute monotonic scalar advection (monotonic-scalar-transport): missing required inputs: {missing_keys}. "
                    f"config['advect_scalars']=True and config['monotonic']=True but required data is missing. "
                    f"Ensure checkpoint contains all monotonic-scalar-transport data or set strict=False to skip."
                )
            else:
                warnings.warn(
                    f"Skipping monotonic scalar advection (monotonic-scalar-transport): missing inputs: {missing_keys}. "
                    f"Scalar fields will not be advected.",
                    RuntimeWarning,
                )

        if can_run_monotonic_scalar_transport:
            from mpas_pytorch.dynamics.advance_scalars_mono import (
                _MONOTONIC_SCALAR_TRANSPORT_REGION_COMPILE_TARGETS,
                _monotonic_scalar_transport_connectivity_payload,
                _monotonic_scalar_transport_connectivity_payload_cache_key,
                atm_advance_scalars_mono_work,
            )

            # Get dimensions
            nCellsSolve = _scalar_int(state_dict, "nCellsSolve", nCells)
            num_scalars = _scalar_int(state_dict, "num_scalars")

            # Get dt from config or state_dict
            dt_scalar = config.get("dt", state_dict.get("dt", dts))
            dt_scalar = _coerce_float(dt_scalar, "dt_scalar_monotonic_scalar_transport")

            # Get coef_3rd_order
            coef_3rd = _scalar_float(state_dict, "coef_3rd_order")

            # Loop bounds (1-based Fortran indices)
            cellStart = _scalar_int(state_dict, "cellStart", 1)
            cellEnd = _scalar_int(state_dict, "cellEnd", nCells)
            edgeStart = _scalar_int(state_dict, "edgeStart", 1)
            edgeEnd = _scalar_int(state_dict, "edgeEnd", nEdges)
            cellSolveStart = _scalar_int(state_dict, "cellSolveStart", 1)
            cellSolveEnd = _scalar_int(state_dict, "cellSolveEnd", nCells)

            # Helper to get tensor from result first, then state_dict
            def get_tensor_monotonic_scalar_transport(key):
                if key in result:
                    return result[key]
                return state_dict[key]

            # Get working arrays - clone to avoid modifying inputs
            # GAP-C3: Read from result (populated in setup) so live integration sees evolving state
            scalars_old = _clone_for_grad(
                get_tensor_monotonic_scalar_transport("scalars_old")
            )
            scalars_new = _clone_for_grad(
                get_tensor_monotonic_scalar_transport("scalars_new")
            )
            scalar_old = _clone_for_grad(
                _first_tensor_or_zeros(
                    result,
                    state_dict,
                    "scalar_old",
                    (nCells + 1, nVertLevels),
                    device=device,
                )
            )
            scalar_new = _clone_for_grad(
                _first_tensor_or_zeros(
                    result,
                    state_dict,
                    "scalar_new",
                    (nCells + 1, nVertLevels),
                    device=device,
                )
            )
            s_max = _clone_for_grad(
                _mapping_tensor_or_zeros(
                    state_dict, "s_max", (nCells + 1, nVertLevels), device=device
                )
            )
            s_min = _clone_for_grad(
                _mapping_tensor_or_zeros(
                    state_dict, "s_min", (nCells + 1, nVertLevels), device=device
                )
            )
            wdtn = _clone_for_grad(
                _mapping_tensor_or_zeros(
                    state_dict, "wdtn", (nCells + 1, nVertLevels + 1), device=device
                )
            )
            flux_arr = _clone_for_grad(
                _mapping_tensor_or_zeros(
                    state_dict, "flux_arr", (nEdges + 1, nVertLevels), device=device
                )
            )
            flux_upwind_tmp = _clone_for_grad(
                _mapping_tensor_or_zeros(
                    state_dict,
                    "flux_upwind_tmp",
                    (nEdges + 1, nVertLevels),
                    device=device,
                )
            )
            flux_tmp = _clone_for_grad(
                _mapping_tensor_or_zeros(
                    state_dict, "flux_tmp", (nEdges + 1, nVertLevels), device=device
                )
            )
            scalar_tend = _clone_for_grad(
                _mapping_tensor_or_zeros(
                    state_dict,
                    "scalar_tend",
                    (nCells + 1, nVertLevels, num_scalars),
                    device=device,
                )
            )
            scale_arr = _clone_for_grad(
                _mapping_tensor_or_zeros(
                    state_dict, "scale_arr", (nCells + 1, 2, nVertLevels), device=device
                )
            )
            rho_zz_int = _clone_for_grad(
                _mapping_tensor_or_zeros(
                    state_dict, "rho_zz_int", (nCells + 1, nVertLevels), device=device
                )
            )
            # Fortran routes advance_density off split dynamics/transport semantics.
            # Coupled mode (split=1) must pass False; split transport passes True.
            advance_density = _resolve_dynamics_split(config) > 1
            scalar_chunk_size = _coerce_int(
                get_config_value_safe(config, "config_scalar_transport_chunk_size", 0),
                "config_scalar_transport_chunk_size",
            )
            monotonic_scalar_transport_reduction_slot_tile_width = _coerce_int(
                get_config_value_safe(
                    config,
                    "config_monotonic_scalar_transport_reduction_slot_tile_width",
                    0,
                ),
                "config_monotonic_scalar_transport_reduction_slot_tile_width",
            )
            monotonic_scalar_transport_streaming_scalar_tile_width = _coerce_int(
                get_config_value_safe(
                    config,
                    "config_monotonic_scalar_transport_streaming_scalar_tile_width",
                    monotonic_scalar_transport_reduction_slot_tile_width,
                ),
                "config_monotonic_scalar_transport_streaming_scalar_tile_width",
            )
            monotonic_scalar_transport_compile_region_names: tuple[str, ...] = ()
            compile_whole_monotonic_scalar_transport = False
            if compile_on:
                compile_whole_monotonic_scalar_transport = (
                    _kernel_compile_enabled_for_device(
                        "monotonic_scalar_transport_advance_scalars_mono",
                        config,
                        device=device,
                    )
                )
                if not compile_whole_monotonic_scalar_transport:
                    monotonic_scalar_transport_compile_region_names = tuple(
                        name
                        for name in sorted(
                            _MONOTONIC_SCALAR_TRANSPORT_REGION_COMPILE_TARGETS
                        )
                        if _kernel_compile_enabled_for_device(
                            name, config, device=device
                        )
                    )
            if compile_on and compile_whole_monotonic_scalar_transport:
                atm_advance_scalars_mono_work = _maybe_compile_kernel(
                    "monotonic_scalar_transport_advance_scalars_mono",
                    atm_advance_scalars_mono_work,
                    enabled=True,
                    mode=compile_mode,
                    backend=compile_backend,
                    specialization_key=_compiled_kernel_specialization_key(
                        config,
                        "monotonic_scalar_transport_advance_scalars_mono",
                        "dt",
                        dt_scalar,
                        "advance_density",
                        bool(advance_density),
                        "scalar_chunk_size",
                        scalar_chunk_size,
                        "monotonic_scalar_transport_reduction_slot_tile_width",
                        monotonic_scalar_transport_reduction_slot_tile_width,
                        "monotonic_scalar_transport_streaming_scalar_tile_width",
                        monotonic_scalar_transport_streaming_scalar_tile_width,
                    ),
                )

            nSpecZone_monotonic_scalar_transport = _get_int_from_dict(
                state_dict,
                "nSpecZone",
                int(config.get("nSpecZone", 2)),
            )
            monotonic_scalar_transport_connectivity_payload = None
            monotonic_scalar_transport_numeric_plan = None
            if not compile_whole_monotonic_scalar_transport:
                monotonic_scalar_transport_payload_kwargs = {
                    "device": device,
                    "nCells": nCells,
                    "nEdges": nEdges,
                    "nCellsSolve": nCellsSolve,
                    "cell_start": cellStart - 1,
                    "cell_end": cellEnd,
                    "edge_start": edgeStart - 1,
                    "edge_end": edgeEnd,
                    "cell_solve_start": cellSolveStart - 1,
                    "cell_solve_end": cellSolveEnd,
                    "nSpecZone": nSpecZone_monotonic_scalar_transport,
                    "edgesOnCell": state_dict["edgesOnCell"],
                    "nEdgesOnCell": state_dict["nEdgesOnCell"],
                    "edgesOnCell_sign": state_dict["edgesOnCell_sign"],
                    "cellsOnCell": state_dict["cellsOnCell"],
                    "cellsOnEdge": state_dict["cellsOnEdge"],
                    "bdyMaskCell": state_dict["bdyMaskCell"],
                    "nAdvCellsForEdge": state_dict["nAdvCellsForEdge"],
                    "advCellsForEdge": state_dict["advCellsForEdge"],
                    "adv_coefs": state_dict["adv_coefs"],
                    "adv_coefs_3rd": state_dict["adv_coefs_3rd"],
                    "dvEdge": state_dict["dvEdge"],
                    "invAreaCell": state_dict["invAreaCell"],
                }
                monotonic_scalar_transport_connectivity_payload = (
                    _prebound_static_payload(
                        config,
                        "monotonic_scalar_transport",
                        _monotonic_scalar_transport_connectivity_payload_cache_key,
                        _monotonic_scalar_transport_connectivity_payload,
                        monotonic_scalar_transport_payload_kwargs,
                    )
                )
                if monotonic_scalar_transport_connectivity_payload is not None:
                    monotonic_scalar_transport_numeric_plan = _runner_owned_monotonic_scalar_transport_numeric_plan(
                        static_payloads_for_step,
                        connectivity_payload=monotonic_scalar_transport_connectivity_payload,
                        connectivity_key=_monotonic_scalar_transport_connectivity_payload_cache_key(
                            **monotonic_scalar_transport_payload_kwargs
                        ),
                        nCells=nCells,
                        nEdges=nEdges,
                        nVertLevels=nVertLevels,
                        num_scalars=num_scalars,
                        cellStart=cellStart,
                        cellEnd=cellEnd,
                        edgeStart=edgeStart,
                        edgeEnd=edgeEnd,
                        cellSolveStart=cellSolveStart,
                        cellSolveEnd=cellSolveEnd,
                        dt=dt_scalar,
                        soft_limiter_tau=0.01,
                        scalar_chunk_size=scalar_chunk_size,
                        streaming_scalar_tile_width=monotonic_scalar_transport_streaming_scalar_tile_width,
                        reduction_slot_tile_width=monotonic_scalar_transport_reduction_slot_tile_width,
                        advance_density=bool(advance_density),
                    )

            scalar_result = profiled_call(
                "mpas.monotonic_scalar_transport_advance_scalars_mono",
                config,
                atm_advance_scalars_mono_work,
                nCells=nCells,
                nEdges=nEdges,
                nVertLevels=nVertLevels,
                num_scalars=num_scalars,
                dt=dt_scalar,
                coef_3rd_order=coef_3rd,
                nCellsSolve=nCellsSolve,
                cellStart=cellStart,
                cellEnd=cellEnd,
                edgeStart=edgeStart,
                edgeEnd=edgeEnd,
                cellSolveStart=cellSolveStart,
                cellSolveEnd=cellSolveEnd,
                fnm=state_dict["fnm"],
                fnp=state_dict["fnp"],
                rdnw=state_dict["rdnw"],
                dvEdge=state_dict["dvEdge"],
                invAreaCell=state_dict["invAreaCell"],
                nEdgesOnCell=state_dict["nEdgesOnCell"],
                nAdvCellsForEdge=state_dict["nAdvCellsForEdge"],
                bdyMaskCell=state_dict["bdyMaskCell"],
                bdyMaskEdge=state_dict["bdyMaskEdge"],
                cellsOnEdge=state_dict["cellsOnEdge"],
                cellsOnCell=state_dict["cellsOnCell"],
                edgesOnCell=state_dict["edgesOnCell"],
                advCellsForEdge=state_dict["advCellsForEdge"],
                edgesOnCell_sign=state_dict["edgesOnCell_sign"],
                adv_coefs=state_dict["adv_coefs"],
                adv_coefs_3rd=state_dict["adv_coefs_3rd"],
                uhAvg=get_tensor_monotonic_scalar_transport("uhAvg"),
                wwAvg=get_tensor_monotonic_scalar_transport("wwAvg"),
                rho_zz_old=get_tensor_monotonic_scalar_transport("rho_zz_old"),
                rho_zz_new=get_tensor_monotonic_scalar_transport("rho_zz_new"),
                scalar_old=scalar_old,
                scalar_new=scalar_new,
                s_max=s_max,
                s_min=s_min,
                wdtn=wdtn,
                flux_arr=flux_arr,
                flux_upwind_tmp=flux_upwind_tmp,
                flux_tmp=flux_tmp,
                scalar_tend=scalar_tend,
                scalars_old=scalars_old,
                scalars_new=scalars_new,
                scale_arr=scale_arr,
                rho_zz_int=rho_zz_int,
                advance_density=advance_density,
                nSpecZone=nSpecZone_monotonic_scalar_transport,
                scalar_chunk_size=scalar_chunk_size,
                _compile_region_names=monotonic_scalar_transport_compile_region_names,
                _compile_mode=compile_mode,
                _compile_backend=compile_backend,
                _connectivity_payload=monotonic_scalar_transport_connectivity_payload,
                _profile_sections=bool(config.get("config_profile_sections", False))
                and not compile_whole_monotonic_scalar_transport,
                _reduction_slot_tile_width=monotonic_scalar_transport_reduction_slot_tile_width,
                _streaming_scalar_tile_width=monotonic_scalar_transport_streaming_scalar_tile_width,
                _numeric_plan=monotonic_scalar_transport_numeric_plan,
            )
            if compile_whole_monotonic_scalar_transport:
                scalar_result = _clone_compiled_output_mapping(scalar_result)

            _alias_scalar_outputs(scalar_result)

            # Update result with scalar advection outputs
            result.update(scalar_result)
            if "scalars_new" in result:
                result["scalars"] = _clone_for_grad(result["scalars_new"])

    else:
        # Simple advection for rk_step < 3, or if no limiters configured (config_monotonic=False AND config_positive_definite=False)
        # This matches Upstream MPAS behavior: atm_advance_scalars() for steps 1-2, or if limiters disabled
        # Check for explicit-scalar-transport explicit advection required inputs
        explicit_scalar_transport_required = [
            "scalar_old",
            "scalar_new",
            "uhAvg",
            "wwAvg",
            "rho_zz_old",
            "rho_zz_new",
            "fnm",
            "fnp",
            "rdnw",
            "dvEdge",
            "invAreaCell",
            "nEdgesOnCell",
            "nAdvCellsForEdge",
            "bdyMaskCell",
            "bdyMaskEdge",
            "cellsOnEdge",
            "edgesOnCell",
            "advCellsForEdge",
            "edgesOnCell_sign",
            "adv_coefs",
            "adv_coefs_3rd",
            "coef_3rd_order",
            "num_scalars",
        ]
        can_run_explicit_scalar_transport = all(
            key in state_dict or key in result
            for key in explicit_scalar_transport_required
        )

        # If scalar advection is explicitly requested but explicit-scalar-transport cannot run, handle strict mode
        if advect_scalars_requested and not can_run_explicit_scalar_transport:
            missing_keys = [
                k
                for k in explicit_scalar_transport_required
                if k not in state_dict and k not in result
            ]
            if strict:
                raise ValueError(
                    f"Cannot execute explicit scalar advection (explicit-scalar-transport): missing required inputs: {missing_keys}. "
                    f"config['advect_scalars']=True but required advection inputs are missing. "
                    f"Ensure checkpoint contains all explicit-scalar-transport data or set strict=False to skip."
                )
            else:
                warnings.warn(
                    f"Skipping explicit scalar advection (explicit-scalar-transport): missing inputs: {missing_keys}. "
                    f"Scalar fields will not be advected.",
                    RuntimeWarning,
                )

        if can_run_explicit_scalar_transport:
            from mpas_pytorch.dynamics.advance_scalars import (
                _EXPLICIT_SCALAR_TRANSPORT_REGION_COMPILE_TARGETS,
                _explicit_scalar_transport_connectivity_payload,
                _explicit_scalar_transport_connectivity_payload_cache_key,
                _explicit_scalar_transport_density_time_weights,
                atm_advance_scalars_work,
                build_explicit_scalar_transport_numeric_plan,
            )

            # Get dimensions
            maxEdges = _scalar_int(state_dict, "maxEdges", 10)
            nAdvCellsMax = _scalar_int(state_dict, "nAdvCellsMax", 15)
            num_scalars = _scalar_int(state_dict, "num_scalars")

            # Get dt from config or state_dict
            dt_scalar = config.get("dt", state_dict.get("dt", dts))
            dt_scalar = _coerce_float(dt_scalar, "dt_scalar_explicit_scalar_transport")

            # Get coef_3rd_order
            coef_3rd = _scalar_float(state_dict, "coef_3rd_order")

            # Get time integration order from config semantics.
            # Fallback to checkpoint-provided value when config omits it.
            config_time_int_order_default = state_dict.get(
                "config_time_integration_order", 3
            )
            config_time_int_order = _coerce_int(
                get_config_value_safe(
                    config,
                    "config_time_integration_order",
                    config_time_int_order_default,
                ),
                "config_time_integration_order",
            )

            # Fortran routes advance_density based on split dynamics/transport mode.
            # In this coupled path (split loop disabled here), this resolves to 0.
            advance_density = _resolve_dynamics_split(config) > 1

            # Get rk_step from config or state_dict
            scalar_rk_step = config.get("rk_step", rk_step)
            scalar_rk_step = _coerce_int(scalar_rk_step, "scalar_rk_step")
            explicit_scalar_transport_compile_region_names: tuple[str, ...] = ()
            compile_whole_explicit_scalar_transport = False
            if compile_on:
                compile_whole_explicit_scalar_transport = (
                    _kernel_compile_enabled_for_device(
                        "explicit_scalar_transport_advance_scalars",
                        config,
                        device=device,
                    )
                )
                if compile_whole_explicit_scalar_transport:
                    atm_advance_scalars_work = _maybe_compile_kernel(
                        "explicit_scalar_transport_advance_scalars",
                        atm_advance_scalars_work,
                        enabled=True,
                        mode=compile_mode,
                        backend=compile_backend,
                        specialization_key=_compiled_kernel_specialization_key(
                            config,
                            "explicit_scalar_transport_advance_scalars",
                            "rk_step",
                            scalar_rk_step,
                            "dt",
                            dt_scalar,
                            "order",
                            config_time_int_order,
                            "advance_density",
                            advance_density,
                        ),
                    )
                else:
                    explicit_scalar_transport_compile_region_names = tuple(
                        name
                        for name in sorted(
                            _EXPLICIT_SCALAR_TRANSPORT_REGION_COMPILE_TARGETS
                        )
                        if _kernel_compile_enabled_for_device(
                            name,
                            config,
                            device=device,
                        )
                    )

            # Loop bounds (1-based Fortran indices)
            edgeStart = _scalar_int(state_dict, "edgeStart", 1)
            edgeEnd = _scalar_int(state_dict, "edgeEnd", nEdges)
            cellSolveStart = _scalar_int(state_dict, "cellSolveStart", 1)
            cellSolveEnd = _scalar_int(state_dict, "cellSolveEnd", nCells)
            explicit_scalar_transport_connectivity_payload = None
            if not compile_whole_explicit_scalar_transport:
                explicit_scalar_transport_payload_kwargs = {
                    "device": device,
                    "nCells": nCells,
                    "nEdges": nEdges,
                    "maxEdges": maxEdges,
                    "nAdvCellsMax": nAdvCellsMax,
                    "edge_start": edgeStart - 1,
                    "edge_end": edgeEnd,
                    "cell_solve_start": cellSolveStart - 1,
                    "cell_solve_end": cellSolveEnd,
                    "edgesOnCell": state_dict["edgesOnCell"],
                    "nEdgesOnCell": state_dict["nEdgesOnCell"],
                    "edgesOnCell_sign": state_dict["edgesOnCell_sign"],
                    "nAdvCellsForEdge": state_dict["nAdvCellsForEdge"],
                    "advCellsForEdge": state_dict["advCellsForEdge"],
                    "adv_coefs": state_dict["adv_coefs"],
                    "adv_coefs_3rd": state_dict["adv_coefs_3rd"],
                }
                explicit_scalar_transport_connectivity_payload = (
                    _prebound_static_payload(
                        config,
                        "explicit_scalar_transport",
                        _explicit_scalar_transport_connectivity_payload_cache_key,
                        _explicit_scalar_transport_connectivity_payload,
                        explicit_scalar_transport_payload_kwargs,
                    )
                )

            do_physics_explicit_scalar_transport = _coerce_bool(
                state_dict.get(
                    "do_physics",
                    get_config_value_safe(config, "do_physics", False),
                ),
                "do_physics",
            )
            explicit_scalar_transport_numeric_plan = None
            if explicit_scalar_transport_connectivity_payload is not None:
                (
                    explicit_scalar_transport_weight_old,
                    explicit_scalar_transport_weight_new,
                ) = _explicit_scalar_transport_density_time_weights(
                    advance_density=advance_density,
                    rk_step=scalar_rk_step,
                    config_time_integration_order=config_time_int_order,
                )
                candidate_explicit_scalar_transport_plan = build_explicit_scalar_transport_numeric_plan(
                    connectivity_payload=explicit_scalar_transport_connectivity_payload,
                    inv_area_cells=state_dict["invAreaCell"][
                        cellSolveStart - 1 : cellSolveEnd
                    ],
                    fnm=state_dict["fnm"],
                    fnp=state_dict["fnp"],
                    rdnw=state_dict["rdnw"],
                    dt=dt_scalar,
                    coef_3rd_order=coef_3rd,
                    weight_time_old=explicit_scalar_transport_weight_old,
                    weight_time_new=explicit_scalar_transport_weight_new,
                    nVertLevels=nVertLevels,
                    num_scalars=num_scalars,
                    do_physics=do_physics_explicit_scalar_transport,
                    stage_key=(
                        "coupled",
                        "rk_step",
                        scalar_rk_step,
                        "order",
                        config_time_int_order,
                        "advance_density",
                        advance_density,
                    ),
                )
                explicit_scalar_transport_numeric_plan = (
                    static_payloads_for_step.get_or_build_numeric_plan(
                        "explicit_scalar_transport_numeric_composition",
                        candidate_explicit_scalar_transport_plan.key,
                        lambda: candidate_explicit_scalar_transport_plan,
                    )
                )
                if explicit_scalar_transport_compile_region_names:
                    explicit_scalar_transport_numeric_plan = _runner_owned_compiled_explicit_scalar_transport_numeric_plan(
                        static_payloads_for_step,
                        explicit_scalar_transport_numeric_plan,
                        compile_region_names=explicit_scalar_transport_compile_region_names,
                        compile_kernel=_maybe_compile_kernel,
                        mode=compile_mode,
                        backend=compile_backend,
                    )
                    explicit_scalar_transport_compile_region_names = ()

            # Helper to get tensor from result first, then state_dict
            def get_tensor_explicit_scalar_transport(key):
                if key in result:
                    return result[key]
                return state_dict[key]

            # Get working arrays - clone to avoid modifying inputs
            # GAP-C3: Read from result (populated in setup) so live integration sees evolving state
            if "scalars_old" in result:
                scalar_old_source = result["scalars_old"]
            elif "scalars_old" in state_dict:
                scalar_old_source = state_dict["scalars_old"]
            else:
                scalar_old_source = get_tensor_explicit_scalar_transport("scalar_old")
            scalar_old = _ensure_scalars_rank3(
                scalar_old_source,
                num_scalars,
                "scalar_old",
            )
            if "scalars_new" in result:
                scalar_new_source = result["scalars_new"]
            elif "scalars_new" in state_dict:
                scalar_new_source = state_dict["scalars_new"]
            else:
                scalar_new_source = get_tensor_explicit_scalar_transport("scalar_new")
            scalar_new = _ensure_scalars_rank3(
                scalar_new_source,
                num_scalars,
                "scalar_new",
            )
            scalar_tend_save_source = state_dict.get("scalar_tend_save")
            if not isinstance(scalar_tend_save_source, torch.Tensor):
                scalar_tend_save_source = torch.zeros_like(scalar_new)
            scalar_tend_save = _ensure_scalars_rank3(
                scalar_tend_save_source,
                num_scalars,
                "scalar_tend_save",
            )
            horiz_flux_arr = _mapping_tensor_or_zeros(
                state_dict,
                "horiz_flux_arr",
                (nEdges + 1, nVertLevels, num_scalars),
                device=device,
            )

            scalar_result = profiled_call(
                "mpas.explicit_scalar_transport_advance_scalars",
                config,
                atm_advance_scalars_work,
                nCells=nCells,
                nEdges=nEdges,
                nVertLevels=nVertLevels,
                num_scalars=num_scalars,
                maxEdges=maxEdges,
                nAdvCellsMax=nAdvCellsMax,
                dt=dt_scalar,
                coef_3rd_order=coef_3rd,
                edgeStart=edgeStart,
                edgeEnd=edgeEnd,
                cellSolveStart=cellSolveStart,
                cellSolveEnd=cellSolveEnd,
                rk_step=scalar_rk_step,
                config_time_integration_order=config_time_int_order,
                advance_density=advance_density,
                scalar_old=scalar_old,
                scalar_new=scalar_new,
                scalar_tend_save=scalar_tend_save,
                rho_zz_old=get_tensor_explicit_scalar_transport("rho_zz_old"),
                rho_zz_new=get_tensor_explicit_scalar_transport("rho_zz_new"),
                uhAvg=get_tensor_explicit_scalar_transport("uhAvg"),
                wwAvg=get_tensor_explicit_scalar_transport("wwAvg"),
                dvEdge=state_dict["dvEdge"],
                invAreaCell=state_dict["invAreaCell"],
                fnm=state_dict["fnm"],
                fnp=state_dict["fnp"],
                rdnw=state_dict["rdnw"],
                cellsOnEdge=state_dict["cellsOnEdge"],
                edgesOnCell=state_dict["edgesOnCell"],
                nEdgesOnCell=state_dict["nEdgesOnCell"],
                edgesOnCell_sign=state_dict["edgesOnCell_sign"],
                bdyMaskCell=state_dict["bdyMaskCell"],
                bdyMaskEdge=state_dict["bdyMaskEdge"],
                nAdvCellsForEdge=state_dict["nAdvCellsForEdge"],
                advCellsForEdge=state_dict["advCellsForEdge"],
                adv_coefs=state_dict["adv_coefs"],
                adv_coefs_3rd=state_dict["adv_coefs_3rd"],
                horiz_flux_arr=horiz_flux_arr,
                do_physics=do_physics_explicit_scalar_transport,
                _compile_region_names=explicit_scalar_transport_compile_region_names,
                _compile_mode=compile_mode,
                _compile_backend=compile_backend,
                _connectivity_payload=explicit_scalar_transport_connectivity_payload,
                _numeric_plan=explicit_scalar_transport_numeric_plan,
            )
            if compile_whole_explicit_scalar_transport:
                scalar_result = _clone_compiled_output_mapping(scalar_result)
            _alias_scalar_outputs(scalar_result)
            # Expose rank-3 explicit-scalar-transport tensors through the plural
            # aliases consumed by downstream integration paths.
            if (
                "scalar_new" in scalar_result
                and isinstance(scalar_result["scalar_new"], torch.Tensor)
                and scalar_result["scalar_new"].ndim == 3
            ):
                scalar_result["scalars_new"] = _clone_for_grad(
                    scalar_result["scalar_new"]
                )
            if (
                "scalar_old" in scalar_result
                and isinstance(scalar_result["scalar_old"], torch.Tensor)
                and scalar_result["scalar_old"].ndim == 3
            ):
                scalar_result["scalars_old"] = _clone_for_grad(
                    scalar_result["scalar_old"]
                )

            # Update result with scalar advection outputs
            result.update(scalar_result)
            if "scalars_new" in result:
                result["scalars"] = _clone_for_grad(result["scalars_new"])

    # =========================================================================
    # Step 5.5: Scalar boundary adjustment (transported-scalar boundary adjustment - regional MPAS only)
    # Called after scalar advection to apply Davies relaxation in boundary zone.

    # Only applies when config_apply_lbcs=true (regional MPAS).
    # For global MPAS (config_apply_lbcs=false), this is a no-op.
    # GAP-4: Also skipped when split dynamics-transport is active (handled
    # in the split transport path).
    # =========================================================================

    if (
        advect_scalars_requested
        and not config_split_dynamics_transport
        and config_apply_lbcs
        and "scalars_new" in result
    ):
        # Check for required boundary data for scalar adjustment (Laplacian + Rayleigh)
        # Required mesh arrays for full implementation
        required_scalar_bdy_fields = [
            "bdyMaskCell",
            "scalars_driving",  # or fallback to 'scalars_bdy'
            "meshScalingRegionalCell",
            "cellsOnCell",
            "nEdgesOnCell",
            "areaCell",
            "dvEdge",
            "dcEdge",
            "edgesOnCell",
        ]
        missing_bdy_scalar = [
            f for f in required_scalar_bdy_fields if f not in state_dict
        ]
        # Allow fallback: scalars_bdy -> scalars_driving
        if "scalars_driving" not in state_dict and "scalars_bdy" in state_dict:
            missing_bdy_scalar = [
                f for f in missing_bdy_scalar if f != "scalars_driving"
            ]
        has_bdy_data_scalars = len(missing_bdy_scalar) == 0

        if has_bdy_data_scalars:
            nRelaxZone_scalar = _scalar_int(state_dict, "nRelaxZone", 10)
            num_scalars_bdy = result["scalars_new"].shape[
                2
            ]  # (nCells+1, nVertLevels, nScalars)

            # Loop bounds (1-based Fortran indices)
            cellStart_scalar = _scalar_int(state_dict, "cellStart", 1)
            cellEnd_scalar = _scalar_int(state_dict, "cellEnd", nCells)

            # Get dt values for Rayleigh damping calculation
            dt_full = config.get("dt", dts * nAcousticSteps)
            dt_rk = dts  # acoustic substep dt

            # Get scalars_driving (with fallback to scalars_bdy for backward compatibility)
            scalars_driving_key = (
                "scalars_driving" if "scalars_driving" in state_dict else "scalars_bdy"
            )

            from mpas_pytorch.boundary.adjust_scalars import (  # type: ignore[import-not-found]
                atm_bdy_adjust_scalars,
            )

            bdy_scalar_result = atm_bdy_adjust_scalars(
                scalars=result["scalars_new"],
                scalars_driving=state_dict[scalars_driving_key],
                bdyMaskCell=state_dict["bdyMaskCell"],
                nRelaxZone=nRelaxZone_scalar,
                dt=dt_full,
                dt_rk=dt_rk,
                meshScalingRegionalCell=state_dict["meshScalingRegionalCell"],
                cellsOnCell=state_dict["cellsOnCell"],
                nEdgesOnCell=state_dict["nEdgesOnCell"],
                areaCell=state_dict["areaCell"],
                dvEdge=state_dict["dvEdge"],
                dcEdge=state_dict["dcEdge"],
                edgesOnCell=state_dict["edgesOnCell"],
                nCells=nCells,
                nScalars=num_scalars_bdy,
                nVertLevels=nVertLevels,
                cellStart=cellStart_scalar,
                cellEnd=cellEnd_scalar,
            )
            result["scalars_new"] = bdy_scalar_result["scalars"]
        else:
            # Missing boundary data for regional MPAS
            if strict:
                raise ValueError(
                    f"config_apply_lbcs=True but required boundary fields missing for transported-scalar boundary adjustment: {missing_bdy_scalar}. "
                    f"Either provide boundary data or set config_apply_lbcs=False."
                )
            else:
                warnings.warn(
                    f"Skipping scalar boundary adjustment (transported-scalar boundary adjustment): missing fields {missing_bdy_scalar}. "
                    f"Regional boundary conditions will not be applied to scalars.",
                    RuntimeWarning,
                )

    # =========================================================================
    # Step 6: Compute solve diagnostics (diagnostic-solve)
    # Called after scalar advection to compute diagnostic fields:
    # h_edge, divergence, vorticity, ke, pv_edge
    # In coupled mode, this runs after scalar advection within
    # each RK step.
    # =========================================================================

    # atm_compute_solve_diagnostics reads 'h', which is rho_zz
    # (the Jacobian-weighted dry density serving as the mass variable).
    # Upstream call:
    #   call atm_compute_solve_diagnostics(dt, state, 2, diag, mesh, ...)
    # where state time level 2 provides rho_zz as h.
    # Map rho_zz -> h so diagnostic-solve can find it via get_tensor_diagnostic_solve('h').
    # Use the updated rho_zz from large-step-recovery (in result) if available.
    rho_zz_for_h = result.get("rho_zz", state_dict.get("rho_zz"))
    if rho_zz_for_h is not None:
        result["h"] = rho_zz_for_h

    # Check if diagnostic-solve required inputs are available
    diagnostic_solve_required = [
        "u",
        "h",
        "fVertex",
        "fEdge",
        "invAreaTriangle",
        "invAreaCell",
        "dvEdge",
        "dcEdge",
        "invDvEdge",
        "invDcEdge",
        "weightsOnEdge",
        "kiteAreasOnVertex",
        "cellsOnEdge",
        "verticesOnEdge",
        "edgesOnCell",
        "edgesOnCell_sign",
        "edgesOnVertex",
        "edgesOnVertex_sign",
        "verticesOnCell",
        "kiteForCell",
        "nEdgesOnCell",
        "nEdgesOnEdge",
        "edgesOnEdge",
        "nVertices",
        "vertexDegree",
    ]
    can_run_diagnostic_solve = all(
        key in state_dict or key in result for key in diagnostic_solve_required
    )

    # diagnostic-solve is an optional diagnostic step. Only raise strict error if:
    # 1. strict=True AND
    # 2. diagnostics are explicitly requested via config AND
    # 3. diagnostic-solve inputs are missing
    # This allows running core physics (acoustic-step/large-step-recovery) without requiring diagnostic data.
    diagnostics_requested = config.get("compute_diagnostics", False)
    if diagnostics_requested and not can_run_diagnostic_solve:
        missing_keys = [
            k
            for k in diagnostic_solve_required
            if k not in state_dict and k not in result
        ]
        if strict:
            raise ValueError(
                f"Cannot execute solve diagnostics (diagnostic-solve): missing required inputs: {missing_keys}. "
                f"config['compute_diagnostics']=True but required data is missing. "
                f"Ensure checkpoint contains all diagnostic-solve data or set strict=False to skip."
            )
        else:
            warnings.warn(
                f"Skipping solve diagnostics (diagnostic-solve): missing inputs: {missing_keys}. "
                f"Diagnostic fields (divergence, vorticity, ke, etc.) will not be computed.",
                RuntimeWarning,
            )

    if can_run_diagnostic_solve:
        # Get dimensions
        nVertices = _scalar_int(state_dict, "nVertices", 0)
        vertexDegree = _scalar_int(state_dict, "vertexDegree", 3)
        _scalar_int(state_dict, "maxEdges2", 20)

        # Get dt (FULL timestep) for diagnostic-solve.
        # Fortran passes full dt to solve diagnostics; dt_full must take
        # precedence over per-RK-step dt.
        dt_diagnostic_solve = config.get(
            "dt_full", config.get("dt", state_dict.get("dt", dts))
        )
        dt_diagnostic_solve = _coerce_float(dt_diagnostic_solve, "dt_diagnostic_solve")

        # Get APVM upwinding coefficient from config or state_dict
        config_apvm = config.get(
            "config_apvm_upwinding", state_dict.get("config_apvm_upwinding", 0.5)
        )
        config_apvm = _coerce_float(config_apvm, "config_apvm_upwinding")

        # Loop bounds (1-based Fortran indices)
        cellStart = _scalar_int(state_dict, "cellStart", 1)
        cellEnd = _scalar_int(state_dict, "cellEnd", nCells)
        vertexStart = _scalar_int(state_dict, "vertexStart", 1)
        vertexEnd = _scalar_int(state_dict, "vertexEnd", nVertices)
        edgeStart = _scalar_int(state_dict, "edgeStart", 1)
        edgeEnd = _scalar_int(state_dict, "edgeEnd", nEdges)

        # Helper to get tensor from result first, then state_dict.
        # Cannot use result.get(key, state_dict[key]) because Python
        # eagerly evaluates the default argument, causing KeyError when
        # the key exists in result but not state_dict (e.g. 'h').
        def get_tensor_diagnostic_solve(key):
            if key in result:
                return result[key]
            return state_dict[key]

        # Work checkpoints for diagnostic-solve are minted from the kernel entrypoint and
        # may not carry a prior-stage tangential velocity (v). When v is
        # absent, omit rk_step to trigger diagnostic-solve's optional-argument
        # reconstruction path.
        v_in_diagnostic_solve = (
            get_tensor_diagnostic_solve("v")
            if "v" in result or "v" in state_dict
            else None
        )
        rk_step_diagnostic_solve = (
            rk_step if v_in_diagnostic_solve is not None else None
        )
        diagnostic_solve_compile_region_names: tuple[str, ...] = ()
        if compile_on:
            diagnostic_solve_compile_region_names = tuple(
                name
                for name in sorted(_DIAGNOSTIC_SOLVE_REGION_COMPILE_TARGETS)
                if _kernel_compile_enabled_for_device(
                    name,
                    config,
                    device=device,
                )
            )

        diagnostic_solve_connectivity_payload = None
        diagnostic_solve_payload_kwargs = {
            "device": device,
            "nCells": nCells,
            "nEdges": nEdges,
            "nVertices": nVertices,
            "vertexDegree": vertexDegree,
            "cellStart": cellStart,
            "cellEnd": cellEnd,
            "vertexStart": vertexStart,
            "vertexEnd": vertexEnd,
            "edgeStart": edgeStart,
            "edgeEnd": edgeEnd,
            "dvEdge": state_dict["dvEdge"],
            "dcEdge": state_dict["dcEdge"],
            "weightsOnEdge": state_dict["weightsOnEdge"],
            "kiteAreasOnVertex": state_dict["kiteAreasOnVertex"],
            "cellsOnEdge": state_dict["cellsOnEdge"],
            "verticesOnEdge": state_dict["verticesOnEdge"],
            "edgesOnCell": state_dict["edgesOnCell"],
            "edgesOnCell_sign": state_dict["edgesOnCell_sign"],
            "edgesOnVertex": state_dict["edgesOnVertex"],
            "edgesOnVertex_sign": state_dict["edgesOnVertex_sign"],
            "verticesOnCell": state_dict["verticesOnCell"],
            "kiteForCell": state_dict["kiteForCell"],
            "nEdgesOnCell": state_dict["nEdgesOnCell"],
            "nEdgesOnEdge": state_dict["nEdgesOnEdge"],
            "edgesOnEdge": state_dict["edgesOnEdge"],
        }
        diagnostic_solve_connectivity_payload = _prebound_static_payload(
            config,
            "diagnostic_solve",
            _diagnostic_solve_connectivity_payload_cache_key,
            _diagnostic_solve_connectivity_payload,
            diagnostic_solve_payload_kwargs,
        )

        diagnostic_solve_numeric_plan: DiagnosticSolveNumericPlan | None = None
        if diagnostic_solve_connectivity_payload is not None:
            candidate_diagnostic_solve_plan = build_diagnostic_solve_numeric_plan(
                connectivity_payload=diagnostic_solve_connectivity_payload,
                nCells=nCells,
                nEdges=nEdges,
                nVertices=nVertices,
                nVertLevels=nVertLevels,
                dt=dt_diagnostic_solve,
                config_apvm_upwinding=config_apvm,
                fVertex=state_dict["fVertex"],
                invAreaTriangle=state_dict["invAreaTriangle"],
                invAreaCell=state_dict["invAreaCell"],
                invDvEdge=state_dict["invDvEdge"],
                invDcEdge=state_dict["invDcEdge"],
                rk_step=rk_step_diagnostic_solve,
                stage_key=("rk", rk_step, "has_v", v_in_diagnostic_solve is not None),
            )
            static_payload_store = config.get("_dycore_static_payloads")
            if isinstance(static_payload_store, DycoreStaticPayloads):
                diagnostic_solve_numeric_plan = cast(
                    DiagnosticSolveNumericPlan,
                    static_payload_store.get_or_build_numeric_plan(
                        "diagnostic_solve_numeric",
                        candidate_diagnostic_solve_plan.key,
                        lambda: candidate_diagnostic_solve_plan,
                    ),
                )
            else:
                diagnostic_solve_numeric_plan = candidate_diagnostic_solve_plan
            if diagnostic_solve_compile_region_names:
                if isinstance(static_payload_store, DycoreStaticPayloads):
                    plan_to_compile = diagnostic_solve_numeric_plan
                    diagnostic_solve_numeric_plan = cast(
                        DiagnosticSolveNumericPlan,
                        static_payload_store.get_or_build_numeric_plan(
                            "diagnostic_solve_numeric_compiled",
                            (plan_to_compile.key, compile_mode, compile_backend),
                            lambda plan=plan_to_compile: (  # type: ignore[misc]
                                compile_diagnostic_solve_numeric_plan(
                                    plan,
                                    compile_kernel=_maybe_compile_kernel,
                                    mode=compile_mode,
                                    backend=compile_backend,
                                )
                            ),
                        ),
                    )
                else:
                    diagnostic_solve_numeric_plan = (
                        compile_diagnostic_solve_numeric_plan(
                            diagnostic_solve_numeric_plan,
                            compile_kernel=_maybe_compile_kernel,
                            mode=compile_mode,
                            backend=compile_backend,
                        )
                    )
                diagnostic_solve_compile_region_names = ()

        diagnostics_result = profiled_call(
            "mpas.diagnostic_solve_solve_diagnostics",
            config,
            atm_compute_solve_diagnostics_work,
            nCells=nCells,
            nEdges=nEdges,
            nVertices=nVertices,
            nVertLevels=nVertLevels,
            vertexDegree=vertexDegree,
            dt=dt_diagnostic_solve,
            config_apvm_upwinding=config_apvm,
            cellStart=cellStart,
            cellEnd=cellEnd,
            vertexStart=vertexStart,
            vertexEnd=vertexEnd,
            edgeStart=edgeStart,
            edgeEnd=edgeEnd,
            # Mesh geometry
            fVertex=state_dict["fVertex"],
            fEdge=state_dict["fEdge"],
            invAreaTriangle=state_dict["invAreaTriangle"],
            invAreaCell=state_dict["invAreaCell"],
            dvEdge=state_dict["dvEdge"],
            dcEdge=state_dict["dcEdge"],
            invDvEdge=state_dict["invDvEdge"],
            invDcEdge=state_dict["invDcEdge"],
            weightsOnEdge=state_dict["weightsOnEdge"],
            kiteAreasOnVertex=state_dict["kiteAreasOnVertex"],
            # Connectivity
            cellsOnEdge=state_dict["cellsOnEdge"],
            verticesOnEdge=state_dict["verticesOnEdge"],
            edgesOnCell=state_dict["edgesOnCell"],
            edgesOnCell_sign=state_dict["edgesOnCell_sign"],
            edgesOnVertex=state_dict["edgesOnVertex"],
            edgesOnVertex_sign=state_dict["edgesOnVertex_sign"],
            verticesOnCell=state_dict["verticesOnCell"],
            kiteForCell=state_dict["kiteForCell"],
            nEdgesOnCell=state_dict["nEdgesOnCell"],
            nEdgesOnEdge=state_dict["nEdgesOnEdge"],
            edgesOnEdge=state_dict["edgesOnEdge"],
            # State variables (may come from large-step-recovery recovery)
            u=get_tensor_diagnostic_solve("u"),
            h=get_tensor_diagnostic_solve("h"),
            rk_step=rk_step_diagnostic_solve,
            v_in=v_in_diagnostic_solve,
            workspace=config.get("_dycore_workspace"),
            workspace_inplace=bool(config.get("workspace_inplace", True)),
            pack_public=not bool(
                config.get("_defer_diagnostic_solve_public_pack", False)
            ),
            _connectivity_payload=diagnostic_solve_connectivity_payload,
            _compile_region_names=diagnostic_solve_compile_region_names,
            _compile_mode=compile_mode,
            _compile_backend=compile_backend,
            _numeric_plan=diagnostic_solve_numeric_plan,
        )

        # Update result with diagnostics outputs
        result.update(diagnostics_result)

        # Map h_edge -> rho_edge so that dynamics-tendency sees the updated density at
        # edges. In the upstream diagnostic pool, 'rho_edge' is h_edge
        # computed by atm_compute_solve_diagnostics
        #   call mpas_pool_get_array(diag, 'rho_edge', h_edge)).
        # Without this mapping, subsequent RK steps / dynamics substeps
        # would read the stale initial rho_edge instead of the one
        # recomputed from the current rho_zz (h) field.
        if "h_edge" in diagnostics_result:
            result["rho_edge"] = diagnostics_result["h_edge"]
        _mark_diagnostic_solve_diagnostics_current(
            result,
            u=get_tensor_diagnostic_solve("u"),
            h=get_tensor_diagnostic_solve("h"),
        )

    # Chain tap: diagnostics ready for the next RK stage's dynamics-tendency.
    sync_chain_tendency_aliases()

    # =========================================================================
    # Step 6.5: Zero-gradient w boundary condition (vertical-velocity boundary adjustment - regional MPAS only)
    # Called after diagnostic-solve diagnostics.

    # Only applies when config_apply_lbcs=true (regional MPAS).
    # For global MPAS (config_apply_lbcs=false), this is a no-op.
    # =========================================================================

    if config_apply_lbcs and "w" in result:
        # Check for required boundary data
        has_bdy_data_w = "bdyMaskCell" in state_dict and "nRelaxZone" in state_dict
        if has_bdy_data_w:
            nRelaxZone_bdy = _scalar_int(state_dict, "nRelaxZone", 0)
            nVertLevelsP1 = nVertLevels + 1

            # Loop bounds (1-based Fortran indices)
            cellStart_bdy = _scalar_int(state_dict, "cellStart", 1)
            cellEnd_bdy = _scalar_int(state_dict, "cellEnd", nCells)

            from mpas_pytorch.boundary.zero_gradient_w import (  # type: ignore[import-not-found]
                atm_zero_gradient_w_bdy,
            )

            bdy_w_result = atm_zero_gradient_w_bdy(
                w=result["w"],
                bdyMaskCell=state_dict["bdyMaskCell"],
                nRelaxZone=nRelaxZone_bdy,
                nCells=nCells,
                nVertLevelsP1=nVertLevelsP1,
                cellStart=cellStart_bdy,
                cellEnd=cellEnd_bdy,
            )
            result["w"] = bdy_w_result["w"]
        else:
            # Missing boundary data for regional MPAS
            missing_bdy = []
            if "bdyMaskCell" not in state_dict:
                missing_bdy.append("bdyMaskCell")
            if "nRelaxZone" not in state_dict:
                missing_bdy.append("nRelaxZone")
            if strict:
                raise ValueError(
                    f"config_apply_lbcs=True but required boundary fields missing for vertical-velocity boundary adjustment: {missing_bdy}. "
                    f"Either provide boundary data or set config_apply_lbcs=False."
                )
            else:
                warnings.warn(
                    f"Skipping boundary w adjustment (vertical-velocity boundary adjustment): missing fields {missing_bdy}. "
                    f"Regional boundary conditions will not be applied to w.",
                    RuntimeWarning,
                )

    return result


# =============================================================================
# DIAGNOSTIC: Timestep summary
# Purely diagnostic — does not modify state.
# =============================================================================


def summarize_timestep(
    state_dict: dict[str, Any],
    nCells: int,
    nEdges: int,
    nVertLevels: int,
    num_scalars: int = 0,
    print_velocities: bool = True,
    print_scalars: bool = False,
) -> dict[str, Any]:
    """
    Minimal timestep diagnostic summary.

    Reports min/max of key state variables and checks for NaN.
    Matches the upstream MPAS timestep summary but only
    implements global min/max reporting, not per-level detail.

    All ranges are computed over valid cells/edges only (excludes the
    garbage padding row — row beyond the valid count).

    Returns dict with: has_nan, u_range, w_range, theta_m_range, scalar_ranges
    """
    result: dict[str, Any] = {"has_nan": False}

    def _valid_slice(tensor: torch.Tensor, n_valid: int) -> torch.Tensor:
        """Return the valid interior of a padded tensor (first n_valid rows)."""
        # Tensors are shape (n_valid+1, ...) with garbage at index n_valid
        return tensor[:n_valid]

    def _check_nan(tensor: torch.Tensor, name: str) -> bool:
        has = bool(torch.isnan(tensor).any().item())
        if has:
            warnings.warn(f"NaN detected in '{name}'", RuntimeWarning)
            result["has_nan"] = True
        result[f"{name}_has_nan"] = has
        return has

    # Key diagnostic fields: (name, valid-count)
    diag_fields = [
        ("u", nEdges),
        ("w", nCells),
        ("theta_m", nCells),
        ("rho_zz", nCells),
    ]

    for field, n_valid in diag_fields:
        if field not in state_dict:
            continue
        t = state_dict[field]
        _check_nan(t, field)
        valid = _valid_slice(t, n_valid)
        flat = valid.reshape(-1)
        result[f"{field}_range"] = {
            "min": flat.min().item(),
            "max": flat.max().item(),
        }

    # Optional per-scalar ranges
    result["scalar_ranges"] = {}
    if print_scalars and num_scalars > 0 and "scalars" in state_dict:
        scalars = state_dict["scalars"]
        _check_nan(scalars, "scalars")
        for s in range(num_scalars):
            if scalars.dim() == 3:
                sc = _valid_slice(scalars, nCells)[:, :, s]
            else:
                sc = _valid_slice(scalars, nCells)
            result["scalar_ranges"][s] = {
                "min": sc.min().item(),
                "max": sc.max().item(),
            }

    return result
