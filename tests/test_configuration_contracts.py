"""All public execution routes reject unsupported requests before preparation."""

from pathlib import Path

import mpas_pytorch as mpas
import pytest
import torch
from mpas_pytorch import atmosphere_model as model
from mpas_pytorch.config_defaults import validate_config

UNSUPPORTED = [
    ("config_physics_suite", "mesoscale_reference"),
    ("config_microp_scheme", "mp_wsm6"),
    ("config_IAU_option", "incremental"),
    ("config_do_DAcycling", True),
    ("config_jedi_da", "true"),
    ("config_jedi_da", ".true."),
    ("config_apply_lbcs", True),
    ("config_enable_sharding", True),
    ("config_enable_sharding", ".true."),
    ("config_explicit_proc_decomp", True),
]


@pytest.fixture
def forbid_preparation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Unsupported request reached scientific preparation")

    monkeypatch.setattr(model, "_prepare_timestep_state", forbidden)


def _call(route, state, mesh, config):
    kwargs = {"config": config, "dt": 720.0}
    if route == "timestep":
        return mpas.run_timestep(state, mesh, **kwargs)
    if route == "simulation":
        return mpas.run_simulation(state, mesh, nsteps=1, **kwargs)
    if route == "jvp":
        return mpas.run_simulation.jvp(
            state, mesh, {"u": torch.zeros_like(state["u"])}, nsteps=1, **kwargs
        )
    return mpas.run_simulation.vjp(
        state, mesh, lambda out: out["u"].sum(), track_keys=["u"], nsteps=1, **kwargs
    )


@pytest.mark.parametrize("route", ["timestep", "simulation", "jvp", "vjp"])
@pytest.mark.parametrize("origin", ["fresh", "loaded", "forged_receipt"])
@pytest.mark.parametrize("key,value", UNSUPPORTED)
def test_unsupported_configuration_never_reaches_preparation(
    forbid_preparation, route, origin, key, value
):
    config = {}
    if origin == "loaded":
        config = mpas.load_config_from_namelist(
            Path(__file__).resolve().parents[1] / "examples/namelist.atmosphere"
        )
    elif origin == "forged_receipt":
        config["_mpas_config_validated"] = True
    config[key] = value
    state = {"u": torch.ones((2, 1), dtype=torch.float64)}
    with pytest.raises(mpas.MpasUnsupportedCapabilityError):
        _call(route, state, {}, config)


@pytest.mark.parametrize("route", ["timestep", "simulation", "jvp", "vjp"])
@pytest.mark.parametrize("location", ["state", "mesh"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_unsupported_float_storage_rejected_before_preparation(
    forbid_preparation, route, location, dtype
):
    state = {"u": torch.ones((2, 1), dtype=torch.float64)}
    mesh = {"dcEdge": torch.ones(2, dtype=torch.float64)}
    target = state if location == "state" else mesh
    key = "u" if location == "state" else "dcEdge"
    target[key] = target[key].to(dtype)
    with pytest.raises(mpas.MpasUnsupportedCapabilityError, match="dtype"):
        _call(route, state, mesh, {"_mpas_config_validated": True})


@pytest.mark.parametrize("split", [False, True])
def test_loader_configuration_is_idempotent_and_canonical_override_survives(split):
    config = mpas.load_config_from_namelist(
        Path(__file__).resolve().parents[1] / "examples/namelist.atmosphere"
    )
    config["config_split_dynamics_transport"] = split
    checked = validate_config(config)
    assert checked == config
    assert checked["config_split_dynamics_transport"] is split
    assert checked["config_dynamics_split_steps"] == 3
    assert validate_config(checked) == checked


@pytest.mark.parametrize("key,value", UNSUPPORTED)
def test_input_loader_rejects_unsupported_config_before_reading_files(key, value):
    with pytest.raises(mpas.MpasUnsupportedCapabilityError):
        mpas.load_initial_state(
            "missing-init.nc", "missing-grid.nc", config={key: value}
        )


def test_inactive_logical_strings_and_iau_have_kernel_safe_values():
    config = validate_config(
        {
            "config_apply_lbcs": ".false.",
            "config_enable_sharding": "false",
            "config_IAU_option": "OFF",
        }
    )
    assert config["config_apply_lbcs"] is False
    assert config["config_enable_sharding"] is False
    assert config["config_IAU_option"] == "off"


def test_ambiguous_logical_string_is_a_typed_error():
    with pytest.raises(mpas.MpasConfigError, match="logical"):
        validate_config({"config_enable_sharding": "unexpected"})


@pytest.mark.parametrize(
    "config",
    [
        None,
        {},
        {"config_smagorinsky_coef": 0.25},
        {"config_scalar_transport_chunk_size": 2},
    ],
)
def test_first_validation_preserves_continuation_identity_on_revalidation(config):
    from mpas_pytorch.io_schema import continuation_configuration_digest
    from mpas_pytorch.state import (
        has_complete_step_boundary,
        mark_complete_step_boundary,
    )

    initial = validate_config(config)
    checked = validate_config(initial)
    assert checked == initial
    assert continuation_configuration_digest(
        checked
    ) == continuation_configuration_digest(initial)
    state = {
        key: torch.ones((2, 1), dtype=torch.float64)
        for key in ["u", "w", "rho_zz", "theta_m", "scalars"]
    }
    mesh = {"nCells": 1}
    mark_complete_step_boundary(state, mesh=mesh, config=initial, dt=720.0)
    assert has_complete_step_boundary(state, mesh=mesh, config=checked, dt=720.0)
    changed = dict(checked, config_smdiv=0.25)
    assert not has_complete_step_boundary(state, mesh=mesh, config=changed, dt=720.0)
