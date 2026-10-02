"""
Initialization of coupled diagnostic fields.

Computes diagnostic fields from prognostic state during initialization:
- ru: Edge momentum (rho_zz-weighted u)
- rw: Vertical momentum (rho_zz-weighted omega)
- rho_p: Density perturbation
- rtheta_base: Base state rho*theta
- rtheta_p: Perturbation rho*theta
- exner: Full Exner function
- exner_base: Base state Exner function
- pressure_p: Perturbation pressure
- pressure_base: Base state pressure


Subroutine: atm_init_coupled_diagnostics

Note: The full Fortran subroutine also computes theta_m and rho_zz from
theta, rho, and scalars (qv). This implementation takes theta_m and rho_zz
as inputs since the IC checkpoint stores post-initialization values.
"""

from collections.abc import Mapping
from typing import Any, cast

import torch

from mpas_pytorch.autodiff.soft_limiter import fct_sign_pm
from mpas_pytorch.constants import DTYPE, PhysicalConstants
from mpas_pytorch.dynamics.stencil_ops import (
    _active_slot_count,
    _static_connectivity_cache_allowed,
    _tensor_version_cache_key,
)

RVORD = PhysicalConstants.RVORD
_INIT_COUPLED_CONNECTIVITY_CACHE: dict[
    tuple[object, ...], dict[str, torch.Tensor | int]
] = {}
_INIT_COUPLED_CONNECTIVITY_TENSOR_NAMES = (
    "cellsOnEdge",
    "edgesOnCell",
    "edgesOnCell_sign",
    "nEdgesOnCell",
)


def _subtract_fortran_order(initial: torch.Tensor, terms: torch.Tensor) -> torch.Tensor:
    """Apply the source loop's repeated ``value = value - term`` updates."""
    result = initial
    for slot in range(terms.shape[1]):
        result = result - terms[:, slot, :]
    return result


def _init_coupled_tensor_cache_key(tensor: torch.Tensor) -> tuple[object, ...] | None:
    return _tensor_version_cache_key(tensor)


def _init_coupled_connectivity_payload_cache_key(
    **kwargs: object,
) -> tuple[object, ...] | None:
    """Return a mutation-sensitive key for init-coupled static topology."""
    tensors = tuple(
        kwargs.get(name) for name in _INIT_COUPLED_CONNECTIVITY_TENSOR_NAMES
    )
    if not _static_connectivity_cache_allowed(tensors):
        return None

    tensor_keys: list[tuple[object, ...]] = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            tensor_key = _init_coupled_tensor_cache_key(tensor)
            if tensor_key is None:
                return None
            tensor_keys.append(tensor_key)

    return (
        kwargs["device"],
        kwargs["nCells"],
        kwargs["nEdges"],
        kwargs["nVertLevels"],
        *tensor_keys,
    )


def _build_init_coupled_connectivity_payload(
    *,
    device: torch.device,
    nCells: int,
    nEdges: int,
    nVertLevels: int,
    cellsOnEdge: torch.Tensor,
    edgesOnCell: torch.Tensor,
    edgesOnCell_sign: torch.Tensor,
    nEdgesOnCell: torch.Tensor,
) -> dict[str, torch.Tensor | int]:
    """Build static edge/cell relations for init-coupled diagnostics."""
    del nVertLevels
    cells_on_edge = cellsOnEdge[:nEdges, :2] - 1
    cell1 = cells_on_edge[:, 0].to(device=device, dtype=torch.int64)
    cell2 = cells_on_edge[:, 1].to(device=device, dtype=torch.int64)

    if nCells > 0:
        max_edges = _active_slot_count(
            nEdgesOnCell[:nCells], edgesOnCell.shape[1], device
        )
    else:
        max_edges = 0
    edge_slots = torch.arange(max_edges, dtype=torch.int64, device=device)
    edges_on_cell = edgesOnCell[:nCells, :max_edges] - 1
    slot_valid = edge_slots[None, :] < nEdgesOnCell[:nCells][:, None]
    edge_valid = (edges_on_cell >= 0) & (edges_on_cell < nEdges)
    valid = slot_valid & edge_valid

    return {
        "cell1": cell1,
        "cell2": cell2,
        "max_edges": int(max_edges),
        "safe_edges": torch.where(
            valid, edges_on_cell, torch.zeros_like(edges_on_cell)
        ).to(device=device, dtype=torch.int64),
        "valid": valid,
        "edge_sign": edgesOnCell_sign[:nCells, :max_edges][:, :, None],
    }


def _init_coupled_connectivity_payload(
    **kwargs: object,
) -> dict[str, torch.Tensor | int]:
    """Return cached init-coupled topology when mesh tensors are static."""
    key = _init_coupled_connectivity_payload_cache_key(**kwargs)
    if key is None:
        return _build_init_coupled_connectivity_payload(**kwargs)  # type: ignore[arg-type]
    cached = _INIT_COUPLED_CONNECTIVITY_CACHE.get(key)
    if cached is not None:
        return cached
    payload = _build_init_coupled_connectivity_payload(**kwargs)  # type: ignore[arg-type]
    if len(_INIT_COUPLED_CONNECTIVITY_CACHE) >= 4:
        _INIT_COUPLED_CONNECTIVITY_CACHE.pop(
            next(iter(_INIT_COUPLED_CONNECTIVITY_CACHE))
        )
    _INIT_COUPLED_CONNECTIVITY_CACHE[key] = payload
    return payload


def atm_init_coupled_diagnostics(
    nCells: int,
    nEdges: int,
    nVertLevels: int,
    # Physical constants
    rgas: float,
    cp: float,
    p0: float,
    # Prognostic state (0-based, no padding)
    theta_m: torch.Tensor | None,  # (nCells, nVertLevels)
    rho_zz: torch.Tensor | None,  # (nCells, nVertLevels)
    theta: torch.Tensor | None,  # (nCells, nVertLevels)
    rho: torch.Tensor | None,  # (nCells, nVertLevels)
    qv: torch.Tensor | None,  # (nCells, nVertLevels)
    u: torch.Tensor,  # (nEdges, nVertLevels)
    w: torch.Tensor,  # (nCells, nVertLevels+1)
    # Base state (0-based, no padding)
    rho_base: torch.Tensor,  # (nCells, nVertLevels)
    theta_base: torch.Tensor,  # (nCells, nVertLevels)
    zz: torch.Tensor,  # (nCells, nVertLevels)
    # Mesh connectivity (0-based indexing, +1 garbage cell; stored values are 1-based)
    cellsOnEdge: torch.Tensor,  # (nEdges+1, 2) - 1-based cell indices
    edgesOnCell: torch.Tensor,  # (nCells+1, maxEdges) - 1-based edge indices
    edgesOnCell_sign: torch.Tensor,  # (nCells+1, maxEdges)
    nEdgesOnCell: torch.Tensor,  # (nCells+1,)
    # Vertical interpolation weights (0-based, length nVertLevels)
    fzm: torch.Tensor,  # (nVertLevels,)
    fzp: torch.Tensor,  # (nVertLevels,)
    # Metric terms (0-based indexing, +1 garbage cell)
    zb_cell: torch.Tensor,  # (nCells+1, maxEdges, nVertLevels+1)
    zb3_cell: torch.Tensor,  # (nCells+1, maxEdges, nVertLevels+1)
    _connectivity_payload: Mapping[str, Any] | None = None,
) -> dict:
    """
    Compute diagnostic fields from prognostic state during initialization.

    This implements the Fortran atm_init_coupled_diagnostics subroutine.
    When raw theta/rho/qv are provided, Step 1 (theta_m, rho_zz from
    theta, rho, qv) is recomputed internally instead of trusting any
    preexisting startup values.

    Index Convention:
        State arrays (theta_m, rho_zz, u, w, rho_base, theta_base, zz):
            0-based, no padding. Shape (nCells, nVertLevels) etc.
            Element [i, k] corresponds to upstream (k+1, iCell=i+1).

        Mesh connectivity arrays (cellsOnEdge, edgesOnCell, etc.):
            0-based indexing with an extra garbage cell at index N.
            Shape (N+1, ...). Index 0 = first valid element.
            Values stored in connectivity arrays are one-based MPAS indices.
            Subtract 1 to convert to 0-based for state array access.

        fzm, fzp:
            0-based, length nVertLevels. fzm[k] maps to upstream fzm(k+1).
            Used for k >= 1 (corresponding to upstream k >= 2).

        zb_cell, zb3_cell:
            0-based with garbage cell. Shape (nCells+1, maxEdges, nVertLevels+1).
            zb_cell[iCell, i, k] maps to upstream zb_cell(k+1, i+1, iCell+1).

    Args:
        nCells: Number of cells
        nEdges: Number of edges
        nVertLevels: Number of vertical levels
        rgas: Gas constant for dry air (J/(kg*K))
        cp: Specific heat at constant pressure (J/(kg*K))
        p0: Reference pressure (Pa), typically 1e5
        theta_m: Moist potential temperature at cells
        rho_zz: Density / zz at cells
        u: Normal velocity at edges
        w: Vertical velocity at cells (nVertLevels+1 interfaces)
        rho_base: Base state density at cells
        theta_base: Base state potential temperature at cells
        zz: Jacobian (dz/dzeta) at cells
        cellsOnEdge: Cells adjacent to each edge (values are 1-based)
        edgesOnCell: Edges of each cell (values are 1-based)
        edgesOnCell_sign: Sign of edge relative to cell
        nEdgesOnCell: Number of edges per cell
        fzm: Vertical interpolation weight (lower level)
        fzp: Vertical interpolation weight (upper level)
        zb_cell: Metric coefficient at cell edges
        zb3_cell: Third-order metric coefficient at cell edges

    Returns:
        Dictionary containing (all 0-based, no padding):
            ru: (nEdges, nVertLevels) Edge momentum
            rw: (nCells, nVertLevels+1) Vertical momentum
            rho_p: (nCells, nVertLevels) Density perturbation
            rtheta_base: (nCells, nVertLevels) Base state rho*theta
            rtheta_p: (nCells, nVertLevels) Perturbation rho*theta
            exner: (nCells, nVertLevels) Full Exner function
            exner_base: (nCells, nVertLevels) Base state Exner function
            pressure_p: (nCells, nVertLevels) Perturbation pressure
            pressure_base: (nCells, nVertLevels) Base state pressure
    """
    tensor_device = theta_m if theta_m is not None else theta
    if tensor_device is None:
        raise ValueError(
            "atm_init_coupled_diagnostics requires theta_m or theta inputs"
        )
    device = tensor_device.device

    # Constants
    rcv = rgas / (cp - rgas)
    rgas_over_p0 = rgas / p0
    # =========================================================================
    # Step 1: Use dynamics-advanced theta_m/rho_zz when present.

    # Upstream computes theta_m/rho_zz from raw theta/rho/qv only during initial
    # startup. Subsequent timesteps advance theta_m/rho_zz prognostically and
    # do not reconstruct them from the stale raw IC theta/rho fields.
    # =========================================================================
    if theta_m is not None and rho_zz is not None:
        theta_m_work = theta_m
        rho_zz_work = rho_zz
    elif theta is not None and rho is not None and qv is not None:
        theta_m_work = theta * (1.0 + RVORD * qv)
        rho_zz_work = rho / zz
    else:
        raise ValueError(
            "atm_init_coupled_diagnostics requires either derived theta_m/rho_zz "
            "or raw theta/rho/qv"
        )

    # =========================================================================
    # Step 2: Compute ru (edge momentum)
    # MPAS expression: ru(k,iEdge) = 0.5 * u(k,iEdge) * (rho_zz(k,cell1) + rho_zz(k,cell2))
    # Loop over all edges (edgeStart..edgeEnd = 1..nEdges).
    # =========================================================================
    conn: Mapping[str, Any]
    if _connectivity_payload is None:
        conn = _init_coupled_connectivity_payload(
            device=device,
            nCells=nCells,
            nEdges=nEdges,
            nVertLevels=nVertLevels,
            cellsOnEdge=cellsOnEdge,
            edgesOnCell=edgesOnCell,
            edgesOnCell_sign=edgesOnCell_sign,
            nEdgesOnCell=nEdgesOnCell,
        )
    else:
        conn = _connectivity_payload
    cell1 = conn["cell1"]
    cell2 = conn["cell2"]
    ru = 0.5 * u[:nEdges, :] * (rho_zz_work[cell1, :] + rho_zz_work[cell2, :])

    # =========================================================================
    # Step 3: Compute rw (vertical momentum)
    # Two parts: w-dependent piece and ru-dependent piece
    # Fortran loop: iCell = cellStart..cellEnd (1-based), k = 2..nVertLevels
    # Python: iCell = 0..nCells-1 (0-based), k = 1..nVertLevels-1
    # =========================================================================
    if nVertLevels > 1:
        # Build rw interior (k=1..nVertLevels-1) functionally, then pad the
        # zero-valued boundary interfaces to preserve autograd history.
        fzp_slice = fzp[1:nVertLevels][None, :]
        fzm_slice = fzm[1:nVertLevels][None, :]
        rho_interp = (
            fzp_slice * rho_zz_work[:nCells, : nVertLevels - 1]
            + fzm_slice * rho_zz_work[:nCells, 1:nVertLevels]
        )
        zz_interp = (
            fzp_slice * zz[:nCells, : nVertLevels - 1]
            + fzm_slice * zz[:nCells, 1:nVertLevels]
        )
        rw_interior = w[:nCells, 1:nVertLevels] * rho_interp * zz_interp

        max_edges = int(conn["max_edges"])
        valid = cast(torch.Tensor, conn["valid"])
        safe_edges = conn["safe_edges"]

        fzm_slice = fzm[1:nVertLevels][None, None, :]
        fzp_slice = fzp[1:nVertLevels][None, None, :]

        ru_k = ru[safe_edges, 1:nVertLevels]
        ru_km1 = ru[safe_edges, : nVertLevels - 1]
        flux = fzm_slice * ru_k + fzp_slice * ru_km1

        edge_sign = conn["edge_sign"]
        zb_val = zb_cell[:nCells, :max_edges, 1:nVertLevels]
        zb3_val = zb3_cell[:nCells, :max_edges, 1:nVertLevels]
        flux_sign = fct_sign_pm(flux)

        zz_interp_flux = (
            fzp_slice * zz[:nCells, : nVertLevels - 1][:, None, :]
            + fzm_slice * zz[:nCells, 1:nVertLevels][:, None, :]
        )

        contrib = edge_sign * (zb_val + flux_sign * zb3_val) * flux * zz_interp_flux
        masked_contrib = torch.where(
            valid[:, :, None], contrib, torch.zeros_like(contrib)
        )
        # The source does not form a sum and subtract it once.  It updates rw
        # after every edge slot, so each subtraction rounds independently.
        rw_interior = _subtract_fortran_order(rw_interior, masked_contrib)
        rw = torch.zeros((nCells, nVertLevels + 1), dtype=DTYPE, device=device)
        rw[:, 1:nVertLevels] = rw_interior
    else:
        rw = torch.zeros((nCells, nVertLevels + 1), dtype=DTYPE, device=device)

    # =========================================================================
    # Step 4: Perturbation fields

    # Preserve the upstream separate-multiply-then-add evaluation order.
    # =========================================================================
    rho_p = rho_zz_work - rho_base
    rtheta_base = theta_base * rho_base

    fma_c = rho_base * (theta_m_work - theta_base)
    rtheta_p = _source_order_rtheta_p(theta_m_work, rho_p, fma_c)

    # =========================================================================
    # Step 5: Exner function and pressure
    # exner = (zz * (rgas/p0) * (rtheta_p + rtheta_base))^rcv
    # exner_base = (zz * (rgas/p0) * rtheta_base)^rcv
    # pressure_p = zz * rgas * (exner*rtheta_p + rtheta_base*(exner - exner_base))
    # pressure_base = zz * rgas * exner_base * rtheta_base

    # In non-grad mode, use scalar math.pow in upstream cell/level loop order.
    # This avoids changing libm evaluation behavior.
    # =========================================================================
    exner = (zz * rgas_over_p0 * (rtheta_p + rtheta_base)) ** rcv
    exner_base_out = (zz * rgas_over_p0 * rtheta_base) ** rcv
    pressure_p = zz * rgas * (exner * rtheta_p + rtheta_base * (exner - exner_base_out))
    pressure_base = zz * rgas * exner_base_out * rtheta_base

    return {
        "theta_m": theta_m_work,
        "rho_zz": rho_zz_work,
        "ru": ru,
        "rw": rw,
        "rho_p": rho_p,
        "rtheta_base": rtheta_base,
        "rtheta_p": rtheta_p,
        "exner": exner,
        "exner_base": exner_base_out,
        "pressure_p": pressure_p,
        "pressure_base": pressure_base,
    }


def _source_order_rtheta_p(
    theta_m: torch.Tensor, rho_p: torch.Tensor, fma_c: torch.Tensor
) -> torch.Tensor:
    """Evaluate ``rtheta_p`` with the single PyTorch dycore expression."""
    return theta_m * rho_p + fma_c
