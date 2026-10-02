"""Frozen release-preview capability contract.

This module is deliberately data-only.  The CLI, package metadata, release-scope
documentation test, and configuration rejection tests all consume this one
definition so that advertised support cannot drift between surfaces.
"""

from __future__ import annotations

import copy
from typing import Any

from mpas_pytorch.exceptions import MpasUnsupportedCapabilityError

UPSTREAM_TAG = "v8.3.1"
UPSTREAM_COMMIT = "b9090a1434b8ceb2a92fc985f4d97e7947cd95a6"
REGISTRY_SHA256 = "b5e090313ed9d2d1e3501bf64b7d70e998df2c0c941ba703bd9527febcc963c9"
HISTORY_CONTRACT = "MPAS-PyTorch diagnostic history schema v1"
RESTART_CONTRACT = "MPAS-PyTorch restart schema v1"

# Generated from the nhyd_model and damping records in the authoritative
# Registry.xml identified by REGISTRY_SHA256.  Keeping the generated values in
# the installed package makes capability reporting independent of a monorepo.
_GLOBAL_DYCORE_REGISTRY_DEFAULTS: dict[str, Any] = {
    "config_time_integration": "SRK3",
    "config_time_integration_order": 2,
    "config_dt": 720.0,
    "config_calendar_type": "gregorian",
    "config_start_time": "2010-10-23_00:00:00",
    "config_stop_time": "none",
    "config_run_duration": "5_00:00:00",
    "config_split_dynamics_transport": True,
    "config_number_of_sub_steps": 2,
    "config_dynamics_split_steps": 3,
    "config_h_mom_eddy_visc2": 0.0,
    "config_h_mom_eddy_visc4": 0.0,
    "config_v_mom_eddy_visc2": 0.0,
    "config_h_theta_eddy_visc2": 0.0,
    "config_h_theta_eddy_visc4": 0.0,
    "config_v_theta_eddy_visc2": 0.0,
    "config_horiz_mixing": "2d_smagorinsky",
    "config_len_disp": 0.0,
    "config_visc4_2dsmag": 0.05,
    "config_del4u_div_factor": 10.0,
    "config_w_adv_order": 3,
    "config_theta_adv_order": 3,
    "config_scalar_adv_order": 3,
    "config_u_vadv_order": 3,
    "config_w_vadv_order": 3,
    "config_theta_vadv_order": 3,
    "config_scalar_vadv_order": 3,
    "config_scalar_advection": True,
    "config_positive_definite": False,
    "config_monotonic": True,
    "config_coef_3rd_order": 0.25,
    "config_smagorinsky_coef": 0.125,
    "config_mix_full": True,
    "config_epssm": 0.1,
    "config_smdiv": 0.1,
    "config_apvm_upwinding": 0.5,
    "config_h_ScaleWithMesh": True,
    "config_num_halos": 2,
    "config_relax_zone_divdamp_coef": 6.0,
    "config_zd": 22000.0,
    "config_xnutr": 0.2,
    "config_mpas_cam_coef": 0.0,
    "config_number_cam_damping_levels": 4,
    "config_rayleigh_damp_u": False,
    "config_rayleigh_damp_u_timescale_days": 5.0,
    "config_number_rayleigh_damp_u_levels": 6,
}

_CAPABILITIES: dict[str, Any] = {
    "schema_version": 1,
    "product": "MPAS-Atmosphere PyTorch dycore release preview",
    "model": "MPAS-Atmosphere dynamics core",
    "upstream": {"tag": UPSTREAM_TAG, "peeled_commit": UPSTREAM_COMMIT},
    "support": {
        "geometry": ["global"],
        "process_model": ["single_process"],
        "devices": ["cpu"],
        "dtypes": ["float64"],
        "model_scope": ["dycore_only"],
        "python": ["3.11", "3.12"],
    },
    "unsupported": {
        "model_scope": ["full"],
        "geometry": ["regional", "limited_area", "LBC"],
        "process_model": ["distributed", "MPI"],
        "workflows": ["IAU", "DA"],
        "dtypes": ["float32"],
        "devices": ["mps", "cuda"],
    },
    "contracts": {"history": HISTORY_CONTRACT, "restart": RESTART_CONTRACT},
    "core_surface": {
        "meshes": ["x1.10242", "x1.40962", "x1.163842"],
        "transport_modes": ["split", "coupled"],
        "time_integration_orders": [2, 3],
        "split_dynamics_steps": 3,
        "mesh_timesteps_seconds": {
            "x1.10242": 720.0,
            "x1.40962": 360.0,
            "x1.163842": 180.0,
        },
        "registry_defaults": {
            "source": "MPAS-Atmosphere v8.3.1 Registry.xml",
            "sha256": REGISTRY_SHA256,
            "records": ["nhyd_model", "damping"],
            "values": _GLOBAL_DYCORE_REGISTRY_DEFAULTS,
        },
    },
    "qualified_nondefault_values": {
        "config_epssm": [0.05, 0.2],
        "config_smdiv": [0.05, 0.2],
        "config_visc4_2dsmag": [0.0, 0.02],
        "config_del4u_div_factor": [5.0, 20.0],
        "config_coef_3rd_order": [0.0, 0.5],
        "config_horiz_mixing": ["2d_fixed"],
        "config_mpas_cam_coef": [1.0],
        "config_apvm_upwinding": [0.0, 1.0],
        "config_v_mom_eddy_visc2": [1.0],
        "config_v_theta_eddy_visc2": [1.0],
        "config_xnutr": [0.4],
        "config_zd": [44000.0],
    },
    "qualified_interactions": [
        {"config_epssm": 0.05, "config_smdiv": 0.05},
        {"config_coef_3rd_order": 0.0, "config_visc4_2dsmag": 0.0},
        {"config_apvm_upwinding": 1.0, "config_mpas_cam_coef": 1.0},
        {"config_horiz_mixing": "2d_fixed", "config_h_mom_eddy_visc4": 1.0e10},
    ],
}


def get_release_capabilities() -> dict[str, Any]:
    """Return a caller-owned copy of the frozen capability contract."""

    return copy.deepcopy(_CAPABILITIES)


def validate_runtime_request(*, device: str, dtype: str, physics_mode: str) -> None:
    """Reject runtime requests outside the release-preview support contract."""

    if device != "cpu":
        raise MpasUnsupportedCapabilityError(
            f"Unsupported device {device!r}; this release preview supports only 'cpu'.",
            capability="device",
            requested=device,
        )
    if dtype != "float64":
        raise MpasUnsupportedCapabilityError(
            f"Unsupported dtype {dtype!r}; this release preview supports only 'float64'.",
            capability="dtype",
            requested=dtype,
        )
    if physics_mode != "dycore":
        raise MpasUnsupportedCapabilityError(
            f"Unsupported model scope {physics_mode!r}; this release preview is dycore-only.",
            capability="model_scope",
            requested=physics_mode,
        )


def render_release_scope_markdown() -> str:
    """Render the human release scope from the machine contract."""

    cap = get_release_capabilities()
    support = cap["support"]
    unsupported = cap["unsupported"]
    core = cap["core_surface"]
    contracts = cap["contracts"]
    candidates = cap["qualified_nondefault_values"]
    interactions = cap["qualified_interactions"]
    candidate_lines = "\n".join(
        f"- `{name}`: `{', '.join(repr(value) for value in values)}`"
        for name, values in candidates.items()
    )
    interaction_lines = "\n".join(
        "- " + ", ".join(f"`{name}={value!r}`" for name, value in values.items())
        for values in interactions
    )
    return f"""# Release scope

This preview packages the MPAS-Atmosphere dynamics core as one PyTorch implementation for evaluation. It makes the stated global dycore surface installable, inspectable, and reproducible; it is not a full atmospheric model.

## Source lineage

- Upstream: `{cap["upstream"]["tag"]}` at peeled commit `{cap["upstream"]["peeled_commit"]}`.
- The package has its own numerical and file-format contracts. Exact interchange with other MPAS implementations is not claimed.

## Supported preview surface

- Geometry: `{support["geometry"][0]}`.
- Process model: `{support["process_model"][0]}`.
- Device and precision: `{support["devices"][0]}`, `{support["dtypes"][0]}`.
- Model scope: `{support["model_scope"][0]}`.
- Python: {", ".join(support["python"])}.
- Meshes: {", ".join(core["meshes"])}.
- Transport: {", ".join(core["transport_modes"])}; RK orders {", ".join(str(value) for value in core["time_integration_orders"])}; split steps `{core["split_dynamics_steps"]}`.
- History contract: `{contracts["history"]}`.
- Restart contract: `{contracts["restart"]}`.

Registry defaults on the declared global dycore surface are required core behavior. Configurations outside the declared surface are not covered by this preview.

## Qualified non-default settings

The following non-default values have been qualified in registered configurations. This list is not a claim that every cross-product of values, mesh and integration mode has been tested:

{candidate_lines}

The registered interactions are:

{interaction_lines}

## What this preview does not claim

The following are explicitly unsupported: model scope {", ".join(unsupported["model_scope"])}; geometry {", ".join(unsupported["geometry"])}; process model {", ".join(unsupported["process_model"])}; workflows {", ".join(unsupported["workflows"])}; dtype {", ".join(unsupported["dtypes"])}; devices {", ".join(unsupported["devices"])}. Requests for these capabilities must fail at configuration time. History and restart files conform only to the named package contracts above; arbitrary MPAS postprocessor compatibility and restart interchange with other implementations are not claimed.
"""
