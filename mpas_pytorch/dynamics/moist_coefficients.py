"""
Moist correction coefficients computation.

Computes cqw and cqu, which transform inverse dry density (1/rho_d) to
inverse moist density (1/rho_m). These coefficients account for the
mass of moisture species (water vapor, cloud water, etc.) when computing
the full (moist) density from the dry density.

cqw is defined at cell interfaces (vertical levels 2..nVertLevels in Fortran,
or indices 1..nVertLevels-1 in 0-based), computed as the vertical average
of total moisture at adjacent levels.

cqu is defined at edge midpoints, computed as the horizontal average of
total moisture across the two cells sharing each edge.


Subroutine: atm_compute_moist_coefficients
"""

from typing import cast

import torch

from mpas_pytorch.constants import DTYPE
from mpas_pytorch.dynamics.stencil_ops import (
    _replace_row_slice,
    _replace_row_slice_cols,
    _static_connectivity_cache_allowed,
    _tensor_version_cache_key,
    sum_fortran_order,
)
from mpas_pytorch.utils.checks import require_consistent_device

_MOIST_CONNECTIVITY_CACHE: dict[tuple[object, ...], dict[str, torch.Tensor | int]] = {}
_MOIST_CONNECTIVITY_TENSOR_NAMES = ("cellsOnEdge",)


def _moist_tensor_cache_key(tensor: torch.Tensor) -> tuple[object, ...] | None:
    return _tensor_version_cache_key(tensor)


def _moist_connectivity_payload_cache_key(
    **kwargs: object,
) -> tuple[object, ...] | None:
    """Return a mutation-sensitive cache key for moist edge topology."""
    tensors = tuple(kwargs.get(name) for name in _MOIST_CONNECTIVITY_TENSOR_NAMES)
    if not _static_connectivity_cache_allowed(tensors):
        return None

    tensor_keys: list[tuple[object, ...]] = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            tensor_key = _moist_tensor_cache_key(tensor)
            if tensor_key is None:
                return None
            tensor_keys.append(tensor_key)

    return (
        kwargs["device"],
        kwargs["nCells"],
        kwargs["nEdges"],
        kwargs["nCellsSolve"],
        kwargs["edge_start"],
        kwargs["edge_end"],
        *tensor_keys,
    )


def _build_moist_connectivity_payload(
    *,
    device: torch.device,
    nCells: int,
    nEdges: int,
    nCellsSolve: int,
    edge_start: int,
    edge_end: int,
    cellsOnEdge: torch.Tensor,
) -> dict[str, torch.Tensor | int]:
    """Build static edge masks and safe cell indices for moist coefficients."""
    edge_start = max(int(edge_start), 0)
    edge_end = min(int(edge_end), int(nEdges))
    edge_slice = slice(edge_start, edge_end)
    cell1_1based = cellsOnEdge[edge_slice, 0]
    cell2_1based = cellsOnEdge[edge_slice, 1]

    valid_cells = (
        (cell1_1based > 0)
        & (cell1_1based <= nCells)
        & (cell2_1based > 0)
        & (cell2_1based <= nCells)
    )
    owned_edge = (cell1_1based <= nCellsSolve) | (cell2_1based <= nCellsSolve)
    edge_mask = valid_cells & owned_edge
    max_cell_idx = max(nCells - 1, 0)

    return {
        "edge_start": edge_start,
        "edge_end": edge_end,
        "edge_mask": edge_mask.to(device=device),
        "c1": torch.clamp(cell1_1based - 1, min=0, max=max_cell_idx).to(
            device=device, dtype=torch.int64
        ),
        "c2": torch.clamp(cell2_1based - 1, min=0, max=max_cell_idx).to(
            device=device, dtype=torch.int64
        ),
    }


def _moist_connectivity_payload(**kwargs: object) -> dict[str, torch.Tensor | int]:
    """Return cached moist coefficient topology when mesh tensors are static."""
    key = _moist_connectivity_payload_cache_key(**kwargs)
    if key is None:
        return _build_moist_connectivity_payload(**kwargs)  # type: ignore[arg-type]
    cached = _MOIST_CONNECTIVITY_CACHE.get(key)
    if cached is not None:
        return cached
    payload = _build_moist_connectivity_payload(**kwargs)  # type: ignore[arg-type]
    if len(_MOIST_CONNECTIVITY_CACHE) >= 4:
        _MOIST_CONNECTIVITY_CACHE.pop(next(iter(_MOIST_CONNECTIVITY_CACHE)))
    _MOIST_CONNECTIVITY_CACHE[key] = payload
    return payload


def atm_compute_moist_coefficients(
    nCells: int,
    nEdges: int,
    nVertLevels: int,
    nCellsSolve: int,
    cellStart: int,
    cellEnd: int,
    edgeStart: int,
    edgeEnd: int,
    moist_start: int,
    moist_end: int,
    # Connectivity arrays (with padding, 1-based Fortran indexing)
    cellsOnEdge: torch.Tensor,  # (nEdges+1, 2)
    # State variables (with padding)
    scalars: torch.Tensor,  # (nCells+1, nVertLevels, num_scalars)
    _connectivity_payload: dict[str, torch.Tensor | int] | None = None,
) -> dict:
    """
    Compute moist correction coefficients cqw and cqu.

    The moist coefficients transform inverse dry density (1/rho_d) into
    inverse full (moist) density (1/rho_m):
        1/rho_m = cqw * (1/rho_d)  [at cell interfaces]
        1/rho_m = cqu * (1/rho_d)  [at edge midpoints]

    The correction factor is 1/(1 + qtotal) where qtotal is the sum of all
    moisture species mixing ratios.

    Index Convention:
        Fortran uses 1-based indexing with arrays sized (N+1) where index 0
        is padding. This function expects Fortran-style arrays directly from
        checkpoint data.

        cellStart, cellEnd, edgeStart, edgeEnd are 1-based loop bounds.
        moist_start, moist_end are 1-based scalar species indices.

        Fortran scalars layout: scalars(iq, k, iCell)
        PyTorch layout after column-major transpose: scalars[iCell, k, iq]
        where iCell is 0-based (index 0 is padding for 1-based access).

        IMPORTANT: Connectivity arrays (cellsOnEdge) contain 1-based
        Fortran indices. When using these to index into data arrays, we
        subtract 1 to convert to 0-based Python indexing.

    Args:
        nCells: Number of cells in mesh
        nEdges: Number of edges in mesh
        nVertLevels: Number of vertical levels
        nCellsSolve: Number of locally owned cells (for boundary filtering)
        cellStart, cellEnd: Cell loop bounds (1-based, inclusive)
        edgeStart, edgeEnd: Edge loop bounds (1-based, inclusive)
        moist_start, moist_end: Scalar species index range (1-based, inclusive)
        cellsOnEdge: Cells adjacent to each edge (1-based indices),
                     shape (nEdges+1, 2)
        scalars: Moisture scalar fields, shape (nCells+1, nVertLevels, num_scalars).
                 The first axis is 1-based (index 0 is padding).
                 The third axis contains one-based scalar species,
                 stored zero-based in the tensor: upstream iq=1 -> index 0.

    Returns:
        Dictionary containing:
            cqw: (nCells+1, nVertLevels+1) Moist coefficient at cell interfaces.
                 Only levels k=1..nVertLevels-1 (0-based) are computed, which
                 correspond to upstream k=2..nVertLevels. Level 0 and nVertLevels
                 remain zero (top/bottom boundaries).
            cqu: (nEdges+1, nVertLevels) Moist coefficient at edge midpoints.
    """
    require_consistent_device("atm_compute_moist_coefficients", cellsOnEdge, scalars)
    device = (
        scalars.device if isinstance(scalars, torch.Tensor) else torch.device("cpu")
    )

    # Allocate output arrays (with padding dimension at index 0)
    # cqw: upstream shape is (nVertLevels+1, nCells+1), transposed here.
    cqw = torch.zeros(nCells + 1, nVertLevels + 1, dtype=DTYPE, device=device)
    # cqu: upstream shape is (nVertLevels, nEdges+1), transposed here.
    cqu = torch.zeros(nEdges + 1, nVertLevels, dtype=DTYPE, device=device)

    # Intermediate: total moisture at each cell/level
    # Upstream qtot(k, iCell) maps to qtot[iCell, k].
    qtot = torch.zeros(nCells + 1, nVertLevels, dtype=DTYPE, device=device)

    # Convert moist_start/moist_end from one-based MPAS to zero-based Python.
    moist_start_0 = moist_start - 1
    moist_end_0 = moist_end  # exclusive upper bound; upstream end is inclusive
    moist_slice = slice(moist_start_0, moist_end_0)
    has_moist = moist_end_0 > moist_start_0

    # =========================================================================
    # Step 1: Compute qtot(iCell, k) = sum of moisture scalars
    # Upstream: qtot(k, iCell) = sum over iq of scalars(iq, k, iCell)
    # PyTorch: qtot[iCell, k] = sum over iq of scalars[iCell, k, iq]
    # =========================================================================
    cell_start = cellStart - 1
    cell_end = cellEnd
    cell_slice = slice(cell_start, cell_end)
    if has_moist:
        qtot_vals = sum_fortran_order(scalars[cell_slice, :, moist_slice], dim=2)
    else:
        qtot_vals = torch.zeros(
            (cell_end - cell_start, nVertLevels), dtype=DTYPE, device=device
        )
    qtot = _replace_row_slice(qtot, cell_start, cell_end, qtot_vals)

    # =========================================================================
    # Step 2: Compute cqw at cell interfaces (upstream k=2..nVertLevels)
    # In 0-based indexing: k=1..nVertLevels-1
    # cqw[iCell, k] = 1 / (1 + 0.5*(qtot[iCell, k] + qtot[iCell, k-1]))
    # =========================================================================
    if nVertLevels > 1:
        qtotal_w = 0.5 * (
            qtot[cell_slice, 1:nVertLevels] + qtot[cell_slice, : nVertLevels - 1]
        )
        cqw_vals = 1.0 / (1.0 + qtotal_w)
        cqw = _replace_row_slice_cols(
            cqw, cell_start, cell_end, 1, nVertLevels, cqw_vals
        )

    # =========================================================================
    # Step 3: Compute cqu at edges (upstream k=1..nVertLevels)
    # In 0-based: k=0..nVertLevels-1
    # Only for edges where cell1 <= nCellsSolve or cell2 <= nCellsSolve
    # (1-based comparison)

    # cqu[iEdge, k] = 1 / (1 + sum_iq(0.5*(scalars[cell1,k,iq] + scalars[cell2,k,iq])))

    # SENTINEL HANDLING: cellsOnEdge may contain 0 (sentinel for boundary edges).
    # =========================================================================
    edge_start = edgeStart - 1
    edge_end = edgeEnd
    edge_slice = slice(edge_start, edge_end)
    if _connectivity_payload is None:
        conn = _moist_connectivity_payload(
            device=device,
            nCells=nCells,
            nEdges=nEdges,
            nCellsSolve=nCellsSolve,
            edge_start=edge_start,
            edge_end=edge_end,
            cellsOnEdge=cellsOnEdge,
        )
    else:
        conn = _connectivity_payload
    edge_start = int(conn["edge_start"])
    edge_end = int(conn["edge_end"])
    edge_slice = slice(edge_start, edge_end)
    edge_mask = cast(torch.Tensor, conn["edge_mask"])
    c1 = conn["c1"]
    c2 = conn["c2"]
    if has_moist:
        qtotal_u = sum_fortran_order(
            0.5 * (scalars[c1, :, moist_slice] + scalars[c2, :, moist_slice]),
            dim=2,
        )
    else:
        qtotal_u = torch.zeros(
            (edge_end - edge_start, nVertLevels), dtype=DTYPE, device=device
        )
    cqu_vals = 1.0 / (1.0 + qtotal_u)
    current = cqu[edge_slice]
    updated = torch.where(edge_mask[:, None], cqu_vals, current)
    cqu = _replace_row_slice(cqu, edge_start, edge_end, updated)

    return {
        "cqw": cqw,
        "cqu": cqu,
        "qtot": qtot,
    }
