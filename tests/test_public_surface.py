from __future__ import annotations

import mpas_pytorch

EXPECTED = {
    "MpasConfigError",
    "MpasDeviceError",
    "MpasError",
    "MpasMeshMismatchError",
    "MpasMissingFieldError",
    "MpasNumericalAgreementError",
    "MpasProvenanceError",
    "MpasStrictModeError",
    "MpasUnsupportedCapabilityError",
    "load_config_from_namelist",
    "load_initial_state",
    "read_history",
    "read_restart",
    "run_simulation",
    "run_timestep",
    "write_history",
    "write_restart",
}


def test_public_surface_is_exact_and_resolvable() -> None:
    assert set(mpas_pytorch.__all__) == EXPECTED
    assert all(hasattr(mpas_pytorch, name) for name in EXPECTED)
