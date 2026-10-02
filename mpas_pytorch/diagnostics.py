"""Differentiable diagnostics derived from MPAS dynamics state."""

from __future__ import annotations

import torch

from mpas_pytorch.constants import GRAVITY


def physics_interface_surface_pressure(
    *,
    nCells: int,
    rho_zz: torch.Tensor,
    zz: torch.Tensor,
    qv: torch.Tensor,
    pressure_p: torch.Tensor,
    pressure_base: torch.Tensor,
    zgrid: torch.Tensor,
    gravity: float = GRAVITY,
) -> torch.Tensor:
    """Diagnose hydrostatic surface pressure using the MPAS physics interface.

    This is the evolved-state formula in ``mpas_atmphys_interface.F`` rather
    than the ad-hoc initialization extrapolation.  All inputs use cell-major
    layout and may contain either active cells only or one trailing padding
    row. The active count is explicit so a zero padding row cannot create a
    ``0/0`` hydrostatic thickness.

    Args:
        nCells: Number of active cell rows.
        rho_zz: Jacobian-weighted density, ``(nCells[+1], nVertLevels)``.
        zz: Vertical metric, same shape as ``rho_zz``.
        qv: Water-vapor mixing ratio, same shape as ``rho_zz``.
        pressure_p: Perturbation pressure, same shape as ``rho_zz``.
        pressure_base: Base pressure, same shape as ``rho_zz``.
        zgrid: W-level geometric height, ``(nCells[+1], nVertLevels+1)``.
        gravity: Gravitational acceleration in ``m s-2``.

    Returns:
        Hydrostatic surface pressure in Pa, one value per input cell row.
    """
    fields = {
        "rho_zz": rho_zz,
        "zz": zz,
        "qv": qv,
        "pressure_p": pressure_p,
        "pressure_base": pressure_base,
        "zgrid": zgrid,
    }
    for name, value in fields.items():
        if value.ndim != 2:
            raise ValueError(f"{name} must be rank-2, got shape {tuple(value.shape)}")

    if rho_zz.shape[0] not in (nCells, nCells + 1):
        raise ValueError(
            f"rho_zz first dimension must be nCells ({nCells}) or nCells+1 "
            f"({nCells + 1}), got {rho_zz.shape[0]}"
        )
    state_shape = tuple(rho_zz.shape)
    for name in ("zz", "qv", "pressure_p", "pressure_base"):
        value = fields[name]
        if tuple(value.shape) != state_shape:
            raise ValueError(
                f"{name} shape {tuple(value.shape)} does not match rho_zz {state_shape}"
            )
    if zgrid.shape[0] != rho_zz.shape[0]:
        raise ValueError(
            "zgrid cell extent must match rho_zz: "
            f"{zgrid.shape[0]} != {rho_zz.shape[0]}"
        )
    if rho_zz.shape[1] < 2 or zgrid.shape[1] < 3:
        raise ValueError("surface pressure requires two mass levels and three W levels")

    zgrid_active = zgrid[:nCells, :]
    tem1 = zgrid_active[:, 1] - zgrid_active[:, 0]
    tem2 = zgrid_active[:, 2] - zgrid_active[:, 1]
    rho1 = rho_zz[:nCells, 0] * zz[:nCells, 0] * (1.0 + qv[:nCells, 0])
    rho2 = rho_zz[:nCells, 1] * zz[:nCells, 1] * (1.0 + qv[:nCells, 1])
    hydrostatic = (
        0.5 * gravity * tem1 * (rho1 - 0.5 * (rho2 - rho1) * tem1 / (tem1 + tem2))
    )
    surface_pressure = hydrostatic + pressure_p[:nCells, 0] + pressure_base[:nCells, 0]
    if rho_zz.shape[0] == nCells + 1:
        return torch.cat((surface_pressure, torch.zeros_like(surface_pressure[:1])))
    return surface_pressure
