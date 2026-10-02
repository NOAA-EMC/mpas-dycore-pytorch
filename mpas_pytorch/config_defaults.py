"""MPAS-Atmosphere configuration defaults and option policy.

The canonical names follow ``namelist.atmosphere``. Short Python aliases are
accepted where listed in :data:`CONFIG_ALIASES`.
"""

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from mpas_pytorch.exceptions import MpasConfigError, MpasUnsupportedCapabilityError

# Fortran-compatible default configuration
# Values sourced from MPAS Registry.xml authoritative defaults
FORTRAN_DEFAULTS: dict[str, Any] = {
    # Physics flags
    "advect_scalars": True,
    "compute_tendencies": True,
    # Limiter flags (Registry.xml: config_monotonic default_value="true")
    "monotonic": True,
    "positive_definite": False,
    # Divergence damping
    "apply_lbcs": False,
    "smdiv": 0.1,
    "len_disp": 0.0,
    # Time integration (Registry.xml authoritative defaults)
    "config_time_integration": "SRK3",
    # Fortran split semantics are two-parameter:
    # - config_split_dynamics_transport (logical, default true)
    # - config_dynamics_split_steps (integer, default 3)
    # The legacy short key split_dynamics_transport is retained only for
    # backward compatibility with older tests/configs.
    "time_integration_order": 2,
    "config_time_integration_order": 2,
    "config_dt": 720.0,
    "config_split_dynamics_transport": True,
    "config_dynamics_split_steps": 3,
    "split_dynamics_transport": 1,
    # Acoustic settings (Registry.xml: config_number_of_sub_steps default_value="2")
    "epssm": 0.1,
    "config_epssm": 0.1,
    "nAcousticSteps": 2,
    "config_number_of_sub_steps": 2,
    # dynamics-tendency specific (from Registry.xml authoritative defaults)
    # config_del4u_div_factor: default_value="10.0"
    # config_visc4_2dsmag: default_value="0.05"
    # config_horiz_mixing: default_value="2d_smagorinsky"
    # config_coef_3rd_order: default_value="0.25"
    # config_smagorinsky_coef: default_value="0.125"
    "config_del4u_div_factor": 10.0,
    "config_h_mom_eddy_visc2": 0.0,
    "config_h_mom_eddy_visc4": 0.0,
    "config_v_mom_eddy_visc2": 0.0,
    "config_h_theta_eddy_visc2": 0.0,
    "config_h_theta_eddy_visc4": 0.0,
    "config_v_theta_eddy_visc2": 0.0,
    "config_horiz_mixing": "2d_smagorinsky",
    "config_visc4_2dsmag": 0.05,
    "config_smagorinsky_coef": 0.125,
    "coef_3rd_order": 0.25,
    "config_coef_3rd_order": 0.25,
    "c_s": 0.125,
    # Advection order controls (Registry.xml defaults).
    # Current PyTorch implementation is validated for order=3 pathways.
    "config_w_adv_order": 3,
    "config_theta_adv_order": 3,
    "config_scalar_adv_order": 3,
    "config_u_vadv_order": 3,
    "config_w_vadv_order": 3,
    "config_theta_vadv_order": 3,
    "config_scalar_vadv_order": 3,
    # diagnostic-solve diagnostics (Registry.xml: config_apvm_upwinding default_value="0.5")
    "config_apvm_upwinding": 0.5,
    "compute_diagnostics": False,
    "config_scalar_advection": True,
    # monotonic-scalar-transport memory control. 0 preserves the historical all-scalars-at-once path;
    # positive values tile monotonic scalar transport along the tracer dimension.
    "config_scalar_transport_chunk_size": 0,
    # Workspace reuse for pre-allocated scratch tensors.
    "config_use_workspace": False,
    "config_mesh_reordering_rcm": False,
    # Optional torch.compile wrapping for vectorized kernels.
    "config_use_torch_compile": False,
    "config_profile_sections": False,
    "config_torch_compile_mode": "reduce-overhead",
    "config_torch_compile_scope": "hot",
    "config_torch_compile_backend": "inductor",
    "config_torch_compile_cuda_inductor_policy": "safe",
    "config_torch_compile_stage": None,
    # Reserved accelerator controls. Non-default values fail closed unless the
    # active runtime policy explicitly supports them.
    "config_dynamics_tendency_cell_owned_vertical_velocity_horizontal_advection_theta_horizontal_advection_ab": False,
    "dynamics_tendency_cell_owned_vertical_velocity_horizontal_advection_theta_horizontal_advection_ab": False,
    "config_enable_sharding": False,
    "enable_sharding": False,
    # Optional gradient checkpointing in differentiable mode.
    "gradient_checkpointing": False,
    # dynamics-tendency Rayleigh damping (Registry.xml defaults)
    "config_rayleigh_damp_u": False,
    "config_rayleigh_damp_u_timescale_days": 5.0,
    "config_number_rayleigh_damp_u_levels": 6,
    # dynamics-tendency CAM damping (Registry.xml defaults)
    "config_mpas_cam_coef": 0.0,
    "config_number_cam_damping_levels": 4,
    "config_h_ScaleWithMesh": True,
    # Vertical damping (runtime namelist defaults for twin-readiness)
    "config_zd": 22000.0,
    "config_xnutr": 0.2,
    # dynamics-tendency mixing and curvature (Registry.xml defaults)
    "config_mix_full": True,
    "config_curvature_u": False,
    # Duplicate full-name keys used directly in namelists for concordance.
    "config_apply_lbcs": False,
    "config_smdiv": 0.1,
    "config_len_disp": 0.0,
    "config_monotonic": True,
    "config_positive_definite": False,
    # Strict mode - controls validation behavior throughout the codebase:
    # - True (default): Raise errors on missing checkpoint fields, invalid configs
    # - False: Use synthetic approximations with warnings (testing/debugging only)
    # Affects: initialize_from_state, load_full_checkpoint, vertical-implicit-coefficients fallback, etc.
    # CRITICAL: Production runs MUST use strict=True for machine precision.
    "strict": True,
    # DANGER: Setting strict_indexing=False disables indexing convention validation.
    # Kernels expect 1-based+padded arrays (shape N+1, values 1 to N, sentinel=0).
    # Passing 0-based arrays (shape N, values 0 to N-1) causes SILENT DATA CORRUPTION.
    # Only disable for debugging with extreme caution.
    "strict_indexing": True,
    # strict_initialization: Controls initialization field requirements.
    # - True (default): Raise ValueError when CHECKPOINT_REQUIRED_FIELDS are missing
    # - False: Use synthetic approximations (uniform scaling, dry adiabatic base state)
    # Can override the global 'strict' setting for initialization specifically.
    # Require complete initialization fields for strict execution.
    "strict_initialization": True,
}

ATMOSPHERE_INERT_ADVECTION_ORDER_KEYS: tuple[str, ...] = (
    "config_w_adv_order",
    "config_theta_adv_order",
    "config_scalar_adv_order",
    "config_u_vadv_order",
    "config_w_vadv_order",
    "config_theta_vadv_order",
    "config_scalar_vadv_order",
)

# Aliases mapping short names (Python-style) to full names (Fortran-style)
# This allows both 'apply_lbcs' and 'config_apply_lbcs' to work
CONFIG_ALIASES: dict[str, str] = {
    "apply_lbcs": "config_apply_lbcs",
    "smdiv": "config_smdiv",
    "len_disp": "config_len_disp",
    "time_integration_order": "config_time_integration_order",
    "monotonic": "config_monotonic",
    "positive_definite": "config_positive_definite",
    "del4u_div_factor": "config_del4u_div_factor",
    "visc4_2dsmag": "config_visc4_2dsmag",
    "horiz_mixing": "config_horiz_mixing",
    "apvm_upwinding": "config_apvm_upwinding",
    "epssm": "config_epssm",
    # Advection and turbulence coefficients
    "coef_3rd_order": "config_coef_3rd_order",
    "c_s": "config_smagorinsky_coef",
    "smagorinsky_coef": "config_smagorinsky_coef",
    # Time integration controls
    "nAcousticSteps": "config_number_of_sub_steps",
    "number_of_sub_steps": "config_number_of_sub_steps",
    "split_dynamics_transport": "config_split_dynamics_transport",
    "dynamics_split_steps": "config_dynamics_split_steps",
    "scalar_advection": "config_scalar_advection",
    "advect_scalars": "config_scalar_advection",
    "scalar_transport_chunk_size": "config_scalar_transport_chunk_size",
    "use_workspace": "config_use_workspace",
    "mesh_reordering_rcm": "config_mesh_reordering_rcm",
    "use_torch_compile": "config_use_torch_compile",
    "profile_sections": "config_profile_sections",
    "torch_compile_mode": "config_torch_compile_mode",
    "torch_compile_scope": "config_torch_compile_scope",
    "torch_compile_backend": "config_torch_compile_backend",
    "torch_compile_cuda_inductor_policy": "config_torch_compile_cuda_inductor_policy",
    "torch_compile_stage": "config_torch_compile_stage",
    "dynamics_tendency_cell_owned_vertical_velocity_horizontal_advection_theta_horizontal_advection_ab": "config_dynamics_tendency_cell_owned_vertical_velocity_horizontal_advection_theta_horizontal_advection_ab",
    "enable_sharding": "config_enable_sharding",
    "dt": "config_dt",
    # dynamics-tendency Rayleigh damping
    "rayleigh_damp_u": "config_rayleigh_damp_u",
    "rayleigh_damp_u_timescale_days": "config_rayleigh_damp_u_timescale_days",
    "number_rayleigh_damp_u_levels": "config_number_rayleigh_damp_u_levels",
    "zd": "config_zd",
    "xnutr": "config_xnutr",
    # dynamics-tendency CAM damping
    "mpas_cam_coef": "config_mpas_cam_coef",
    "number_cam_damping_levels": "config_number_cam_damping_levels",
    "h_ScaleWithMesh": "config_h_ScaleWithMesh",
    # dynamics-tendency mixing and curvature
    "mix_full": "config_mix_full",
    "curvature_u": "config_curvature_u",
}

# Reverse lookup: full name -> short name
_ALIASES_REVERSE: dict[str, str] = {v: k for k, v in CONFIG_ALIASES.items()}


@dataclass(frozen=True)
class OptionMeta:
    """Runtime support metadata for a recognized configuration option."""

    wired_in: tuple[str, ...]
    tested_at: Literal["invariant", "pointwise", "unsupported", "none"]
    identity_scope: Literal["scientific_case", "execution_plan"] = "scientific_case"
    sweep_values: tuple[Any, ...] = ()
    notes: str = ""


def canonicalize_config_key(key: str) -> str:
    """Return the canonical registry key for a config option."""
    return CONFIG_ALIASES.get(key, key)


def _meta(
    *wired_in: str,
    tested_at: Literal["invariant", "pointwise", "unsupported", "none"],
    identity_scope: Literal["scientific_case", "execution_plan"] = "scientific_case",
    sweep_values: tuple[Any, ...] = (),
    notes: str = "",
) -> OptionMeta:
    return OptionMeta(
        wired_in=tuple(wired_in),
        tested_at=tested_at,
        identity_scope=identity_scope,
        sweep_values=sweep_values,
        notes=notes,
    )


HONORED_OPTIONS: dict[str, OptionMeta] = {
    "compute_tendencies": _meta(
        "mpas_pytorch/dynamics/time_integration.py",
        tested_at="none",
        notes="Non-default use is outside the supported release configuration.",
    ),
    "compute_diagnostics": _meta(
        "mpas_pytorch/dynamics/time_integration.py",
        tested_at="none",
        notes="Non-default use is outside the supported release configuration.",
    ),
    "config_apply_lbcs": _meta(
        "mpas_pytorch/dynamics/time_integration.py",
        "mpas_pytorch/state.py",
        tested_at="unsupported",
        notes="Limited-area boundary conditions are not supported by the global release.",
    ),
    "config_smdiv": _meta(
        "mpas_pytorch/dynamics/time_integration.py",
        tested_at="pointwise",
        sweep_values=(0.05, 0.2),
        notes="The supported interaction includes config_epssm=0.05.",
    ),
    "config_len_disp": _meta(
        "mpas_pytorch/dynamics/time_integration.py",
        tested_at="pointwise",
        sweep_values=(120000.0,),
        notes="Values less than or equal to zero select the default spacing branch.",
    ),
    "config_time_integration": _meta(
        "mpas_pytorch/config_defaults.py",
        tested_at="invariant",
        sweep_values=("SRK3",),
        notes="Atmosphere Fortran dispatches only the SRK3 scheme; non-SRK3 values are rejected before execution.",
    ),
    "config_time_integration_order": _meta(
        "mpas_pytorch/dynamics/time_integration.py",
        "mpas_pytorch/ic_loader.py",
        tested_at="pointwise",
        sweep_values=(3,),
        notes="Orders 2 and 3 are supported.",
    ),
    "config_dt": _meta(
        "mpas_pytorch/atmosphere_model.py",
        "mpas_pytorch/dynamics/time_integration.py",
        "mpas_pytorch/ic_loader.py",
        "mpas_pytorch/cli/run.py",
        tested_at="pointwise",
        sweep_values=(120.0,),
    ),
    "config_split_dynamics_transport": _meta(
        "mpas_pytorch/config_defaults.py",
        "mpas_pytorch/dynamics/time_integration.py",
        "mpas_pytorch/ic_loader.py",
        tested_at="pointwise",
        sweep_values=(False,),
    ),
    "config_dynamics_split_steps": _meta(
        "mpas_pytorch/config_defaults.py",
        "mpas_pytorch/dynamics/time_integration.py",
        "mpas_pytorch/ic_loader.py",
        tested_at="pointwise",
        sweep_values=(1, 2),
    ),
    "config_number_of_sub_steps": _meta(
        "mpas_pytorch/dynamics/time_integration.py",
        "mpas_pytorch/ic_loader.py",
        tested_at="pointwise",
        sweep_values=(1,),
    ),
    "config_epssm": _meta(
        "mpas_pytorch/dynamics/time_integration.py",
        tested_at="pointwise",
        sweep_values=(0.05, 0.2),
        notes="The supported interaction includes config_smdiv=0.05.",
    ),
    "config_del4u_div_factor": _meta(
        "mpas_pytorch/dynamics/time_integration.py",
        tested_at="pointwise",
        sweep_values=(5.0, 20.0),
    ),
    "config_h_mom_eddy_visc2": _meta(
        "mpas_pytorch/dynamics/time_integration.py",
        tested_at="invariant",
        sweep_values=(0.0,),
    ),
    "config_h_mom_eddy_visc4": _meta(
        "mpas_pytorch/dynamics/time_integration.py",
        tested_at="invariant",
        sweep_values=(0.0,),
        notes="The supported fixed-mixing interaction uses config_h_mom_eddy_visc4=1e10.",
    ),
    "config_v_mom_eddy_visc2": _meta(
        "mpas_pytorch/dynamics/time_integration.py",
        tested_at="pointwise",
        sweep_values=(1.0,),
    ),
    "config_h_theta_eddy_visc2": _meta(
        "mpas_pytorch/dynamics/time_integration.py",
        tested_at="invariant",
        sweep_values=(0.0,),
    ),
    "config_h_theta_eddy_visc4": _meta(
        "mpas_pytorch/dynamics/time_integration.py",
        tested_at="invariant",
        sweep_values=(0.0,),
    ),
    "config_v_theta_eddy_visc2": _meta(
        "mpas_pytorch/dynamics/time_integration.py",
        tested_at="pointwise",
        sweep_values=(1.0,),
    ),
    "config_horiz_mixing": _meta(
        "mpas_pytorch/dynamics/time_integration.py",
        tested_at="pointwise",
        sweep_values=("2d_fixed",),
        notes="The supported fixed-mixing interaction uses config_h_mom_eddy_visc4=1e10.",
    ),
    "config_visc4_2dsmag": _meta(
        "mpas_pytorch/dynamics/time_integration.py",
        tested_at="pointwise",
        sweep_values=(0.0, 0.02),
        notes="The supported zero-viscosity interaction uses config_coef_3rd_order=0.",
    ),
    "config_coef_3rd_order": _meta(
        "mpas_pytorch/atmosphere_model.py",
        "mpas_pytorch/dynamics/time_integration.py",
        tested_at="pointwise",
        sweep_values=(0.0, 0.5),
        notes="The supported zero-coefficient interaction uses config_visc4_2dsmag=0.",
    ),
    "config_smagorinsky_coef": _meta(
        "mpas_pytorch/dynamics/time_integration.py",
        "mpas_pytorch/ic_loader.py",
        tested_at="pointwise",
        sweep_values=(0.0,),
    ),
    "config_apvm_upwinding": _meta(
        "mpas_pytorch/dynamics/time_integration.py",
        tested_at="invariant",
        sweep_values=(0.0, 1.0),
        notes="The supported full-upwinding interaction uses config_mpas_cam_coef=1.",
    ),
    "config_scalar_advection": _meta(
        "mpas_pytorch/dynamics/time_integration.py",
        tested_at="invariant",
        sweep_values=(False,),
    ),
    "config_scalar_transport_chunk_size": _meta(
        "mpas_pytorch/dynamics/time_integration.py",
        tested_at="pointwise",
        identity_scope="execution_plan",
        sweep_values=(1, 2),
        notes="Tiles tracer transport without changing the mathematical update.",
    ),
    "config_use_workspace": _meta(
        "mpas_pytorch/dynamics/time_integration.py",
        tested_at="pointwise",
        identity_scope="execution_plan",
        sweep_values=(True,),
    ),
    "config_mesh_reordering_rcm": _meta(
        "mpas_pytorch/atmosphere_model.py",
        tested_at="none",
    ),
    "config_use_torch_compile": _meta(
        "mpas_pytorch/dynamics/compile_utils.py",
        tested_at="pointwise",
        identity_scope="execution_plan",
        sweep_values=(True,),
        notes="Uses the supported hot-kernel compilation policy by default.",
    ),
    "config_profile_sections": _meta(
        "mpas_pytorch/dynamics/time_integration.py",
        "mpas_pytorch/profiling.py",
        tested_at="pointwise",
        identity_scope="execution_plan",
        sweep_values=(True,),
        notes="Adds profiler section ranges without changing numerical operations.",
    ),
    "config_torch_compile_mode": _meta(
        "mpas_pytorch/dynamics/compile_utils.py",
        tested_at="pointwise",
        identity_scope="execution_plan",
        sweep_values=("reduce-overhead",),
        notes="The supported compiled mode is reduce-overhead.",
    ),
    "config_torch_compile_scope": _meta(
        "mpas_pytorch/dynamics/compile_utils.py",
        tested_at="pointwise",
        identity_scope="execution_plan",
        sweep_values=("hot",),
        notes="The default hot scope admits only registered numerical kernels.",
    ),
    "config_torch_compile_backend": _meta(
        "mpas_pytorch/dynamics/compile_utils.py",
        tested_at="pointwise",
        identity_scope="execution_plan",
        sweep_values=("inductor",),
        notes="Inductor is the supported backend for the hot-kernel scope.",
    ),
    "config_torch_compile_cuda_inductor_policy": _meta(
        "mpas_pytorch/dynamics/compile_utils.py",
        tested_at="pointwise",
        identity_scope="execution_plan",
        sweep_values=("safe",),
        notes="Default safe policy leaves CPU compile unchanged and only compiles simple CUDA Inductor kernels; force/off are explicit experiment controls.",
    ),
    "config_torch_compile_stage": _meta(
        "mpas_pytorch/dynamics/compile_utils.py",
        tested_at="pointwise",
        identity_scope="execution_plan",
        sweep_values=("diagnostic_solve_diagnostics",),
        notes="Optional staged CUDA Inductor ladder. GPU runner defaults compiled CUDA to the first stage and requires explicit promotion for later stages.",
    ),
    "config_dynamics_tendency_cell_owned_vertical_velocity_horizontal_advection_theta_horizontal_advection_ab": _meta(
        "mpas_pytorch/dynamics/time_integration.py",
        tested_at="unsupported",
        identity_scope="execution_plan",
        notes="Reserved accelerator control; release configurations reject it.",
    ),
    "config_enable_sharding": _meta(
        "mpas_pytorch/config_defaults.py",
        tested_at="unsupported",
        identity_scope="execution_plan",
        notes="Reserved control; distributed execution is not supported.",
    ),
    "gradient_checkpointing": _meta(
        "mpas_pytorch/atmosphere_model.py",
        "mpas_pytorch/dynamics/time_integration.py",
        tested_at="none",
        identity_scope="execution_plan",
    ),
    "config_rayleigh_damp_u": _meta(
        "mpas_pytorch/dynamics/time_integration.py",
        tested_at="invariant",
        sweep_values=(True,),
    ),
    "config_rayleigh_damp_u_timescale_days": _meta(
        "mpas_pytorch/dynamics/time_integration.py",
        tested_at="invariant",
        sweep_values=(0.5,),
    ),
    "config_number_rayleigh_damp_u_levels": _meta(
        "mpas_pytorch/dynamics/time_integration.py",
        tested_at="unsupported",
        notes="Non-default values are outside the supported release configuration.",
    ),
    "config_mpas_cam_coef": _meta(
        "mpas_pytorch/dynamics/time_integration.py",
        tested_at="pointwise",
        sweep_values=(1.0,),
        notes="The supported full-damping interaction uses config_apvm_upwinding=1.",
    ),
    "config_number_cam_damping_levels": _meta(
        "mpas_pytorch/dynamics/time_integration.py",
        tested_at="unsupported",
        notes="Non-default values are outside the supported release configuration.",
    ),
    "config_h_ScaleWithMesh": _meta(
        "mpas_pytorch/initialization.py",
        tested_at="unsupported",
        notes="Non-default values are outside the supported release configuration.",
    ),
    "config_zd": _meta(
        "mpas_pytorch/ic_loader.py",
        tested_at="pointwise",
        sweep_values=(44000.0,),
    ),
    "config_xnutr": _meta(
        "mpas_pytorch/ic_loader.py",
        tested_at="pointwise",
        sweep_values=(0.4,),
    ),
    "config_mix_full": _meta(
        "mpas_pytorch/dynamics/time_integration.py",
        tested_at="invariant",
        sweep_values=(False,),
    ),
    "config_curvature_u": _meta(
        "mpas_pytorch/dynamics/time_integration.py",
        tested_at="unsupported",
        notes="Non-default values are outside the supported release configuration.",
    ),
    "config_monotonic": _meta(
        "mpas_pytorch/dynamics/time_integration.py",
        tested_at="invariant",
        sweep_values=(False,),
    ),
    "config_positive_definite": _meta(
        "mpas_pytorch/dynamics/time_integration.py",
        tested_at="invariant",
        sweep_values=(True,),
    ),
    "strict": _meta(
        "mpas_pytorch/atmosphere_model.py",
        "mpas_pytorch/ic_loader.py",
        tested_at="pointwise",
        sweep_values=(False,),
        notes="Strict-mode and fallback behavior are covered by dedicated strict-mode tests.",
    ),
    "strict_indexing": _meta(
        "mpas_pytorch/dynamics/time_integration.py",
        tested_at="none",
    ),
    "strict_initialization": _meta(
        "mpas_pytorch/atmosphere_model.py",
        "mpas_pytorch/ic_loader.py",
        "mpas_pytorch/initialization.py",
        tested_at="pointwise",
        sweep_values=(False,),
    ),
}


def _normalize_scalar_config_value(value: Any) -> Any:
    if hasattr(value, "item") and callable(getattr(value, "item", None)):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            return value
    return value


def _warn_on_untested_nondefault_config(user_config: Mapping[str, Any]) -> None:
    warned: set[str] = set()
    for raw_key in user_config:
        canonical_key = canonicalize_config_key(raw_key)
        if canonical_key in warned:
            continue
        meta = HONORED_OPTIONS.get(canonical_key)
        if meta is None or meta.tested_at != "none":
            continue
        resolved_value = _resolve_key(dict(user_config), canonical_key)
        default_value = _resolve_default(canonical_key)
        if resolved_value is None or default_value is None:
            continue
        resolved_scalar = _normalize_scalar_config_value(resolved_value)
        default_scalar = _normalize_scalar_config_value(default_value)
        if resolved_scalar == default_scalar:
            continue
        warnings.warn(
            f"Config key '{canonical_key}' is set to {resolved_scalar!r} "
            f"(default {default_scalar!r}) but this non-default path is not "
            "qualified for release use.",
            UserWarning,
            stacklevel=2,
        )
        warned.add(canonical_key)


def validate_time_integration_scheme(config: Mapping[str, Any]) -> str:
    """Return the supported atmosphere time-integration scheme or raise."""
    value = config.get(
        "config_time_integration", FORTRAN_DEFAULTS["config_time_integration"]
    )
    if hasattr(value, "item") and callable(getattr(value, "item", None)):
        value = value.item()
    scheme = str(value).strip().strip("'\"")
    if scheme != "SRK3":
        raise MpasConfigError(
            f"Unsupported config_time_integration={value!r}. "
            "The vendored MPAS-Atmosphere core dispatches only 'SRK3'."
        )
    return scheme


def _validate_inert_advection_order_types(config: Mapping[str, Any]) -> None:
    """Validate registry-only advection-order knobs without changing numerics."""
    for key in ATMOSPHERE_INERT_ADVECTION_ORDER_KEYS:
        if key not in config:
            continue
        try:
            int(config[key])
        except Exception:  # noqa: BLE001
            raise MpasConfigError(
                f"Config key '{key}' must be integer-like, got {config[key]!r}."
            )


def _config_bool(value: Any) -> bool:
    """Return a Python bool for policy-only config values."""
    if hasattr(value, "item") and callable(getattr(value, "item", None)):
        value = value.item()
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on", ".true.", ".t."}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off", "", ".false.", ".f."}:
            return False
        raise MpasConfigError(f"Expected a logical configuration value, got {value!r}")
    return bool(value)


def _resolve_key(config: Mapping[str, Any], key: str) -> Any | None:
    """
    Resolve a key in config, trying both the key and its alias.

    Priority for config dict lookup:
    1. Full Fortran name (if key is a short name, check config_xxx first)
    2. Exact key match
    3. Short name (if key is a full name via reverse lookup)

    Full names take precedence: if both 'config_apply_lbcs' and 'apply_lbcs'
    are in config, 'config_apply_lbcs' wins regardless of which name is queried.

    Args:
        config: Configuration dictionary
        key: Configuration key (short or full name)

    Returns:
        The value if found, None otherwise
    """
    # 1. If key is a short name, prefer full Fortran name first
    if key in CONFIG_ALIASES:
        full_name = CONFIG_ALIASES[key]
        if full_name in config:
            return config[full_name]

    # 2. Try exact key match
    if key in config:
        return config[key]

    # 3. If key is a full name, try the short name
    if key in _ALIASES_REVERSE:
        short_name = _ALIASES_REVERSE[key]
        if short_name in config:
            return config[short_name]

    return None


def _resolve_default(key: str) -> Any | None:
    """
    Resolve a key in FORTRAN_DEFAULTS, trying both the key and its alias.

    Args:
        key: Configuration key (short or full name)

    Returns:
        The default value if found, None otherwise
    """
    # 1. Try exact key
    if key in FORTRAN_DEFAULTS:
        return FORTRAN_DEFAULTS[key]

    # 2. If key is a short name, try the full name
    if key in CONFIG_ALIASES:
        full_name = CONFIG_ALIASES[key]
        if full_name in FORTRAN_DEFAULTS:
            return FORTRAN_DEFAULTS[full_name]

    # 3. If key is a full name, try the short name
    if key in _ALIASES_REVERSE:
        short_name = _ALIASES_REVERSE[key]
        if short_name in FORTRAN_DEFAULTS:
            return FORTRAN_DEFAULTS[short_name]

    return None


def get_config_value(
    config: Mapping[str, Any], key: str, override_default: Any = None
) -> Any:
    """
    Get config value with Fortran-compatible default.

    Accepts both Fortran-style full names (config_apply_lbcs) and
    Python-style short names (apply_lbcs). Full names take precedence
    when both are present in the config dict.

    Priority:
    1. Value from config dict (exact key or alias)
    2. override_default (if provided)
    3. FORTRAN_DEFAULTS (exact key or alias)
    4. Raises KeyError (if key unknown)

    Args:
        config: Configuration dictionary
        key: Configuration key (short or full name)
        override_default: Optional override for the default value

    Returns:
        Configuration value

    Raises:
        KeyError: If key not in config and not in FORTRAN_DEFAULTS
    """
    # Try config dict with alias resolution
    value = _resolve_key(config, key)
    if value is not None:
        return value

    # Try override_default
    if override_default is not None:
        return override_default

    # Try FORTRAN_DEFAULTS with alias resolution
    default_value = _resolve_default(key)
    if default_value is not None:
        return default_value

    raise KeyError(
        f"Unknown config key '{key}' with no default. "
        f"Known keys: {list(FORTRAN_DEFAULTS.keys())}"
    )


def get_config_value_safe(config: Mapping[str, Any], key: str, fallback: Any) -> Any:
    """
    Get config value with explicit fallback (no exception).

    Accepts both Fortran-style full names (config_apply_lbcs) and
    Python-style short names (apply_lbcs). Full names take precedence
    when both are present in the config dict.

    Use this when you want to provide a local default that may differ
    from FORTRAN_DEFAULTS.

    Args:
        config: Configuration dictionary
        key: Configuration key (short or full name)
        fallback: Default value if key not found

    Returns:
        Configuration value or fallback
    """
    # Try config dict with alias resolution
    value = _resolve_key(config, key)
    if value is not None:
        return value

    # Try FORTRAN_DEFAULTS with alias resolution
    default_value = _resolve_default(key)
    if default_value is not None:
        return default_value

    return fallback


def validate_config(user_config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """
    Validate user configuration and fill missing values with canonical defaults.

    Warns on unknown keys that are not in FORTRAN_DEFAULTS and not aliased.

    Args:
        user_config: User-provided configuration dictionary. If None, returns
            a copy of FORTRAN_DEFAULTS.

    Returns:
        Validated configuration with all defaults filled in.
    """
    # Start with defaults
    validated = dict(FORTRAN_DEFAULTS)

    user_config = {} if user_config is None else dict(user_config)
    # Import lazily: Registry metadata also consumes this module's defaults.
    from mpas_pytorch.namelist_metadata import (
        REGISTRY_OPTION_NAMES,
        active_unsupported_namelist_keys,
    )

    unsupported = active_unsupported_namelist_keys(user_config)
    if unsupported:
        raise MpasUnsupportedCapabilityError(
            f"Unsupported active MPAS settings: {unsupported}; this preview is dycore-only.",
            capability="configuration",
            requested=tuple(unsupported),
        )

    # Check for unknown keys and warn
    for key in user_config:
        # Check if key is in defaults or has an alias
        is_known = (
            key in FORTRAN_DEFAULTS
            or key in CONFIG_ALIASES
            or key in _ALIASES_REVERSE
            or key in REGISTRY_OPTION_NAMES
            or key
            in {
                "strict",
                "strict_initialization",
                "gradient_checkpointing",
                "_mpas_config_validated",
            }
        )
        if not is_known:
            warnings.warn(
                f"Unknown config key '{key}' will be ignored. "
                f"Known keys: {list(FORTRAN_DEFAULTS.keys())}",
                UserWarning,
                stacklevel=2,
            )

    # Update with user values
    validated.update(user_config)
    for key, default in FORTRAN_DEFAULTS.items():
        if isinstance(default, bool) and key in user_config:
            validated[key] = _config_bool(user_config[key])
    if "config_IAU_option" in user_config:
        validated["config_IAU_option"] = "off"
    if "config_physics_suite" in user_config:
        validated["config_physics_suite"] = "none"

    # Keep canonical full-name keys and their short aliases synchronized after
    # user overrides are applied. Full-name keys take precedence when both are
    # present, matching get_config_value_safe() semantics. The legacy
    # split_dynamics_transport alias is excluded because it uses custom
    # two-parameter projection into config_split_dynamics_transport plus
    # config_dynamics_split_steps below.
    for short_name, full_name in CONFIG_ALIASES.items():
        if short_name == "split_dynamics_transport":
            continue
        if short_name in user_config and full_name in user_config:
            short_value = user_config[short_name]
            full_value = user_config[full_name]
            if short_value != full_value:
                warnings.warn(
                    f"Conflicting config values supplied for alias pair "
                    f"'{short_name}'={short_value!r} and '{full_name}'={full_value!r}; "
                    f"using '{full_name}' per Fortran-name precedence.",
                    UserWarning,
                    stacklevel=2,
                )
        # Keep absent optional aliases absent: adding them during a second
        # validation would change a completed boundary's configuration identity.
        if full_name in user_config and (
            short_name in user_config or short_name in FORTRAN_DEFAULTS
        ):
            validated[short_name] = validated[full_name]
        elif short_name in user_config:
            validated[full_name] = validated[short_name]

    validated["config_time_integration"] = validate_time_integration_scheme(validated)
    _validate_inert_advection_order_types(validated)
    if _config_bool(get_config_value_safe(validated, "config_enable_sharding", False)):
        raise MpasUnsupportedCapabilityError(
            "config_enable_sharding is reserved and currently unsupported. "
            "This preview supports single-process CPU execution only.",
            capability="process_model",
            requested="sharding",
        )
    _warn_on_untested_nondefault_config(user_config)

    # Backward compatibility for legacy split_dynamics_transport semantics.
    # If user supplied only the legacy key, project it into Fortran-style keys
    # so runtime code can use authoritative two-parameter semantics.
    if "split_dynamics_transport" in user_config:
        legacy = validated["split_dynamics_transport"]
        if hasattr(legacy, "item"):
            legacy = legacy.item()

        if isinstance(legacy, bool):
            projected_enabled = legacy
            projected_steps = (
                1 if not legacy else int(validated["config_dynamics_split_steps"])
            )
        else:
            legacy_int = int(legacy)
            projected_enabled = legacy_int > 1
            projected_steps = max(1, legacy_int)

        explicit_enabled = user_config.get("config_split_dynamics_transport")
        explicit_steps = user_config.get("config_dynamics_split_steps")
        if explicit_enabled is not None or explicit_steps is not None:
            resolved_enabled = bool(validated["config_split_dynamics_transport"])
            resolved_steps = int(
                explicit_steps
                if explicit_steps is not None
                else validated["config_dynamics_split_steps"]
            )
            if (
                resolved_enabled != projected_enabled
                or resolved_steps != projected_steps
            ):
                raise MpasConfigError(
                    "Conflicting split_dynamics_transport specifications: "
                    f"legacy split_dynamics_transport={legacy!r} projects to "
                    f"(config_split_dynamics_transport={projected_enabled!r}, "
                    f"config_dynamics_split_steps={projected_steps}), but explicit "
                    f"config_split_dynamics_transport/config_dynamics_split_steps "
                    f"resolve to ({resolved_enabled!r}, {resolved_steps})."
                )
        else:
            validated["config_split_dynamics_transport"] = projected_enabled
            validated["config_dynamics_split_steps"] = projected_steps

    # Defaults must not manufacture an explicit legacy override. Otherwise a
    # second validation conflicts with the canonical split keys, and a caller's
    # canonical-key edit can revive an inherited coupled-mode alias.
    if "split_dynamics_transport" not in user_config:
        validated.pop("split_dynamics_transport", None)
    return validated
