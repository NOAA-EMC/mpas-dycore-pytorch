"""Vector reconstruction from edge-normal velocity to cell-centered components."""

from collections.abc import Mapping, MutableMapping
from typing import Any

import torch

from mpas_pytorch.constants import DTYPE
from mpas_pytorch.dynamics.stencil_ops import (
    _active_slot_count,
    _clone_for_grad,
    _replace_row_slice,
    _static_connectivity_cache_allowed,
    _sum_weighted_stencil_components,
    _tensor_version_cache_key,
)

_CELL_PADDING_OFFSET_CACHE: dict[tuple[int, int, int], int] = {}
_VECTOR_RECONSTRUCTION_CACHE: dict[tuple[object, ...], dict[str, Any]] = {}
_VECTOR_RECONSTRUCTION_TENSOR_NAMES = (
    "edgesOnCell",
    "nEdgesOnCell",
    "coeffs_reconstruct",
    "latCell",
    "lonCell",
)
COEFFS_RECONSTRUCT_ALL_ZERO_KEY = "_coeffs_reconstruct_all_zero"
CELL_PADDING_OFFSET_KEY = "_cell_padding_offset"


def _vector_tensor_cache_key(tensor: torch.Tensor) -> tuple[object, ...] | None:
    return _tensor_version_cache_key(tensor)


def _vector_reconstruction_payload_cache_key(
    **kwargs: object,
) -> tuple[object, ...] | None:
    """Return a mutation-sensitive key for vector reconstruction topology."""
    tensors = tuple(kwargs.get(name) for name in _VECTOR_RECONSTRUCTION_TENSOR_NAMES)
    if not _static_connectivity_cache_allowed(tensors):
        return None

    tensor_keys: list[tuple[object, ...]] = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            tensor_key = _vector_tensor_cache_key(tensor)
            if tensor_key is None:
                return None
            tensor_keys.append(tensor_key)

    return (
        kwargs["device"],
        kwargs["nCells"],
        kwargs["nEdges"],
        kwargs["nVertLevels"],
        kwargs["u_rows"],
        kwargs.get("cell_padding_offset"),
        kwargs["on_a_sphere"],
        *tensor_keys,
    )


def _build_vector_reconstruction_payload(
    *,
    device: torch.device,
    nCells: int,
    nEdges: int,
    nVertLevels: int,
    u_rows: int,
    coeffs_reconstruct: torch.Tensor,
    edgesOnCell: torch.Tensor,
    nEdgesOnCell: torch.Tensor,
    latCell: torch.Tensor,
    lonCell: torch.Tensor,
    on_a_sphere: bool,
    cell_padding_offset: int | None = None,
) -> dict[str, Any]:
    """Build static row/slot topology for edge-to-cell vector reconstruction."""
    del nVertLevels
    cell_padding_offset = _infer_cell_padding_offset(
        nCells=nCells,
        edgesOnCell=edgesOnCell,
        nEdgesOnCell=nEdgesOnCell,
        cell_padding_offset=cell_padding_offset,
    )
    edge_padding_offset = cell_padding_offset
    cell_ids0 = torch.arange(nCells, dtype=torch.int64, device=device)
    cell_rows_out = cell_ids0 + cell_padding_offset

    if edgesOnCell.shape[0] == nCells + 1:
        cell_rows_eoc = cell_ids0 + cell_padding_offset
    elif edgesOnCell.shape[0] == nCells:
        cell_rows_eoc = cell_ids0
    else:
        raise ValueError(
            f"edgesOnCell first dimension must be nCells ({nCells}) or nCells+1 ({nCells + 1}), "
            f"got {edgesOnCell.shape[0]}"
        )

    if nEdgesOnCell.shape[0] == nCells + 1:
        cell_rows_neoc = cell_ids0 + cell_padding_offset
    elif nEdgesOnCell.shape[0] == nCells:
        cell_rows_neoc = cell_ids0
    else:
        raise ValueError(
            f"nEdgesOnCell first dimension must be nCells ({nCells}) or nCells+1 ({nCells + 1}), "
            f"got {nEdgesOnCell.shape[0]}"
        )

    n_edges_this_cell = nEdgesOnCell[cell_rows_neoc]
    active_slot_count = _active_slot_count(
        n_edges_this_cell, edgesOnCell.shape[1], device
    )
    eoc = edgesOnCell[cell_rows_eoc, :active_slot_count]
    edge_slots = torch.arange(active_slot_count, dtype=torch.int64, device=device)
    slot_valid = edge_slots[None, :] < n_edges_this_cell[:, None]
    edge_id_1based = eoc
    edge_valid = (edge_id_1based > 0) & (edge_id_1based <= nEdges)
    valid = slot_valid & edge_valid
    edge_id_0based = torch.where(
        valid, edge_id_1based - 1, torch.zeros_like(edge_id_1based)
    )
    if u_rows == nEdges + 1:
        edge_rows_u = edge_id_0based + edge_padding_offset
    elif u_rows == nEdges:
        edge_rows_u = edge_id_0based
    else:
        raise ValueError(
            f"u first dimension must be nEdges ({nEdges}) or nEdges+1 ({nEdges + 1}), got {u_rows}"
        )

    s0, s1, s2 = coeffs_reconstruct.shape
    if s2 == 3 and s0 in (nCells, nCells + 1):
        coeff_rows = cell_rows_out if s0 == nCells + 1 else cell_ids0
        coeff_xyz = coeffs_reconstruct[coeff_rows, :active_slot_count, :]
    elif s0 == 3 and s2 in (nCells, nCells + 1):
        coeff_rows = cell_rows_out if s2 == nCells + 1 else cell_ids0
        coeff_xyz = coeffs_reconstruct[:, :active_slot_count, coeff_rows].permute(
            2, 1, 0
        )
    elif s1 == 3 and s0 in (nCells, nCells + 1):
        coeff_rows = cell_rows_out if s0 == nCells + 1 else cell_ids0
        coeff_xyz = coeffs_reconstruct[coeff_rows, :, :active_slot_count].permute(
            0, 2, 1
        )
    else:
        raise ValueError(
            "Unsupported coeffs_reconstruct layout. Expected one of "
            "(nCells+1,maxEdges,3), (3,maxEdges,nCells+1), or (nCells+1,3,maxEdges); "
            f"got {tuple(coeffs_reconstruct.shape)}"
        )

    clat = slat = clon = slon = None
    if on_a_sphere:
        if latCell.shape[0] == nCells + 1:
            cell_rows_lat = cell_rows_out
        elif latCell.shape[0] == nCells:
            cell_rows_lat = cell_ids0
        else:
            raise ValueError(
                f"latCell first dimension must be nCells ({nCells}) or nCells+1 ({nCells + 1}), got {latCell.shape[0]}"
            )

        if lonCell.shape[0] == nCells + 1:
            cell_rows_lon = cell_rows_out
        elif lonCell.shape[0] == nCells:
            cell_rows_lon = cell_ids0
        else:
            raise ValueError(
                f"lonCell first dimension must be nCells ({nCells}) or nCells+1 ({nCells + 1}), got {lonCell.shape[0]}"
            )
        clat = torch.cos(latCell[cell_rows_lat])[:, None]
        slat = torch.sin(latCell[cell_rows_lat])[:, None]
        clon = torch.cos(lonCell[cell_rows_lon])[:, None]
        slon = torch.sin(lonCell[cell_rows_lon])[:, None]

    return {
        "cell_padding_offset": cell_padding_offset,
        "edge_rows_u": edge_rows_u.to(dtype=torch.int64),
        "valid": valid,
        "coeff_xyz": coeff_xyz,
        "row_start": cell_padding_offset,
        "row_end": cell_padding_offset + nCells,
        "clat": clat,
        "slat": slat,
        "clon": clon,
        "slon": slon,
    }


def _vector_reconstruction_payload(**kwargs: object) -> dict[str, Any]:
    """Return cached vector reconstruction topology when inputs are static."""
    key = _vector_reconstruction_payload_cache_key(**kwargs)
    if key is None:
        return _build_vector_reconstruction_payload(**kwargs)  # type: ignore[arg-type]
    cached = _VECTOR_RECONSTRUCTION_CACHE.get(key)
    if cached is not None:
        return cached
    payload = _build_vector_reconstruction_payload(**kwargs)  # type: ignore[arg-type]
    if len(_VECTOR_RECONSTRUCTION_CACHE) >= 4:
        _VECTOR_RECONSTRUCTION_CACHE.pop(next(iter(_VECTOR_RECONSTRUCTION_CACHE)))
    _VECTOR_RECONSTRUCTION_CACHE[key] = payload
    return payload


def coeffs_reconstruct_all_zero(
    state: Mapping[str, Any],
    coeffs_reconstruct: torch.Tensor,
    n_cells: int | None = None,
) -> bool:
    """Return cached all-zero status without synchronizing accelerator tensors."""
    cached = state.get(COEFFS_RECONSTRUCT_ALL_ZERO_KEY)
    if isinstance(cached, bool):
        return cached
    if isinstance(cached, (int, float)):
        return bool(cached)

    if coeffs_reconstruct.device.type != "cpu":
        if "ur_cell" in state and "vr_cell" in state:
            raise ValueError(
                "Non-CPU vector reconstruction states with ur_cell/vr_cell fallback "
                f"must provide {COEFFS_RECONSTRUCT_ALL_ZERO_KEY!r} metadata; "
                "loader-created states record this automatically."
            )
        return False

    active = coeffs_reconstruct[:n_cells] if n_cells is not None else coeffs_reconstruct
    value = not bool(torch.count_nonzero(active))
    if isinstance(state, MutableMapping):
        state[COEFFS_RECONSTRUCT_ALL_ZERO_KEY] = value
    return value


def _infer_cell_padding_offset(
    nCells: int,
    edgesOnCell: torch.Tensor,
    nEdgesOnCell: torch.Tensor,
    *,
    cell_padding_offset: int | None = None,
) -> int:
    """Infer whether nCells+1 arrays use leading or trailing padding.

    Returns:
        0 for trailing padding (rows 0..nCells-1 valid, row nCells padding),
        1 for leading padding (row 0 padding, rows 1..nCells valid).
    """
    cache_key = (nCells, edgesOnCell.data_ptr(), nEdgesOnCell.data_ptr())
    cached = _CELL_PADDING_OFFSET_CACHE.get(cache_key)
    if cached is not None:
        return cached

    def _cache_and_return(offset: int) -> int:
        _CELL_PADDING_OFFSET_CACHE[cache_key] = int(offset)
        return int(offset)

    if cell_padding_offset is not None:
        if cell_padding_offset not in (0, 1):
            raise ValueError(
                f"cell_padding_offset must be 0 or 1, got {cell_padding_offset}"
            )
        return _cache_and_return(cell_padding_offset)

    if edgesOnCell.ndim != 2 or nEdgesOnCell.ndim != 1:
        return _cache_and_return(0)
    if edgesOnCell.shape[0] != nCells + 1 or nEdgesOnCell.shape[0] != nCells + 1:
        return _cache_and_return(0)

    if edgesOnCell.device.type != "cpu" or nEdgesOnCell.device.type != "cpu":
        raise ValueError(
            "Non-CPU vector reconstruction inputs must provide explicit "
            f"{CELL_PADDING_OFFSET_KEY!r} metadata; loader-created states use "
            "the trailing-padding convention and record this automatically."
        )

    neoc0 = int(nEdgesOnCell[0])
    neoc_last = int(nEdgesOnCell[nCells])
    if neoc0 == 0 and neoc_last > 0:
        return _cache_and_return(1)
    if neoc_last == 0 and neoc0 > 0:
        return _cache_and_return(0)

    eoc0_nonzero = bool(torch.any(edgesOnCell[0] != 0))
    eoc_last_nonzero = bool(torch.any(edgesOnCell[nCells] != 0))
    if (not eoc0_nonzero) and eoc_last_nonzero:
        return _cache_and_return(1)
    if (not eoc_last_nonzero) and eoc0_nonzero:
        return _cache_and_return(0)

    # Default to the trailing-padding convention used by baroclinic checkpoints.
    return _cache_and_return(0)


def _resolve_cell_row(
    i_cell_1based: int,
    n_cells: int,
    first_dim: int,
    name: str,
    padding_offset: int,
) -> int:
    """Map Fortran-style 1-based cell index to a valid tensor row."""
    if first_dim == n_cells + 1:
        return i_cell_1based - 1 + padding_offset
    if first_dim == n_cells:
        return i_cell_1based - 1
    raise ValueError(
        f"{name} first dimension must be nCells ({n_cells}) or nCells+1 ({n_cells + 1}), "
        f"got {first_dim}"
    )


def mpas_reconstruct_2d(
    nCells: int,
    nEdges: int,
    nVertLevels: int,
    u: torch.Tensor,
    coeffs_reconstruct: torch.Tensor,
    edgesOnCell: torch.Tensor,
    nEdgesOnCell: torch.Tensor,
    latCell: torch.Tensor,
    lonCell: torch.Tensor,
    on_a_sphere: bool = True,
    cell_padding_offset: int | None = None,
    _connectivity_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reconstruct edge-normal velocity in the upstream MPAS tensor convention."""
    if u.ndim != 2:
        raise ValueError(
            f"u must have shape (nEdges[+1], nVertLevels), got {tuple(u.shape)}"
        )
    if u.shape[1] != nVertLevels:
        raise ValueError(
            f"u second dimension must be nVertLevels={nVertLevels}, got {u.shape[1]}"
        )
    if edgesOnCell.ndim != 2:
        raise ValueError(
            f"edgesOnCell must be rank-2, got shape {tuple(edgesOnCell.shape)}"
        )
    if nEdgesOnCell.ndim != 1:
        raise ValueError(
            f"nEdgesOnCell must be rank-1, got shape {tuple(nEdgesOnCell.shape)}"
        )
    if latCell.ndim != 1 or lonCell.ndim != 1:
        raise ValueError("latCell and lonCell must be rank-1 tensors")

    device = u.device
    conn: Mapping[str, Any]
    if _connectivity_payload is None:
        conn = _vector_reconstruction_payload(
            device=device,
            nCells=nCells,
            nEdges=nEdges,
            nVertLevels=nVertLevels,
            u_rows=int(u.shape[0]),
            coeffs_reconstruct=coeffs_reconstruct,
            edgesOnCell=edgesOnCell,
            nEdgesOnCell=nEdgesOnCell,
            latCell=latCell,
            lonCell=lonCell,
            on_a_sphere=on_a_sphere,
            cell_padding_offset=cell_padding_offset,
        )
    else:
        conn = _connectivity_payload

    # Keep append-padding convention used throughout dycore orchestration.
    u_reconstruct_x = torch.zeros((nCells + 1, nVertLevels), dtype=DTYPE, device=device)
    u_reconstruct_y = torch.zeros((nCells + 1, nVertLevels), dtype=DTYPE, device=device)
    u_reconstruct_z = torch.zeros((nCells + 1, nVertLevels), dtype=DTYPE, device=device)

    edge_rows_u = conn["edge_rows_u"]
    valid = conn["valid"]
    coeff_xyz = conn["coeff_xyz"]
    u_reconstruct_active = _sum_weighted_stencil_components(
        field=u,
        indices=edge_rows_u,
        weights=coeff_xyz,
        valid_mask=valid,
    )
    u_reconstruct_x_active = u_reconstruct_active[:, :, 0]
    u_reconstruct_y_active = u_reconstruct_active[:, :, 1]
    u_reconstruct_z_active = u_reconstruct_active[:, :, 2]
    row_start = int(conn["row_start"])
    row_end = int(conn["row_end"])
    u_reconstruct_x = _replace_row_slice(
        u_reconstruct_x, row_start, row_end, u_reconstruct_x_active
    )
    u_reconstruct_y = _replace_row_slice(
        u_reconstruct_y, row_start, row_end, u_reconstruct_y_active
    )
    u_reconstruct_z = _replace_row_slice(
        u_reconstruct_z, row_start, row_end, u_reconstruct_z_active
    )

    u_reconstruct_zonal = torch.zeros(
        (nCells + 1, nVertLevels), dtype=DTYPE, device=device
    )
    u_reconstruct_meridional = torch.zeros(
        (nCells + 1, nVertLevels), dtype=DTYPE, device=device
    )

    if on_a_sphere:
        clat = conn["clat"]
        slat = conn["slat"]
        clon = conn["clon"]
        slon = conn["slon"]
        if not all(
            isinstance(value, torch.Tensor) for value in (clat, slat, clon, slon)
        ):
            raise ValueError(
                "vector reconstruction payload is missing spherical transform tensors"
            )

        x = u_reconstruct_x_active
        y = u_reconstruct_y_active
        z = u_reconstruct_z_active
        zonal_active = -x * slon + y * clon
        meridional_active = -(x * clon + y * slon) * slat + z * clat
        u_reconstruct_zonal = _replace_row_slice(
            u_reconstruct_zonal, row_start, row_end, zonal_active
        )
        u_reconstruct_meridional = _replace_row_slice(
            u_reconstruct_meridional,
            row_start,
            row_end,
            meridional_active,
        )
    else:
        u_reconstruct_zonal = _clone_for_grad(u_reconstruct_x)
        u_reconstruct_meridional = _clone_for_grad(u_reconstruct_y)

    return {
        "uReconstructX": u_reconstruct_x,
        "uReconstructY": u_reconstruct_y,
        "uReconstructZ": u_reconstruct_z,
        "uReconstructZonal": u_reconstruct_zonal,
        "uReconstructMeridional": u_reconstruct_meridional,
        "ur_cell": _clone_for_grad(u_reconstruct_zonal),
        "vr_cell": _clone_for_grad(u_reconstruct_meridional),
    }


def mpas_cell_wind_to_edge_normal(
    *,
    nCells: int,
    nEdges: int,
    zonal: torch.Tensor,
    meridional: torch.Tensor,
    cellsOnEdge: torch.Tensor,
    latCell: torch.Tensor,
    lonCell: torch.Tensor,
    edgeNormalVectors: torch.Tensor,
) -> torch.Tensor:
    """Map cell-centered zonal/meridional increments to edge-normal wind.

    This is the spherical-coordinate form of upstream
    ``mpas_vector_R3Cell_to_normalVectorEdge``.  Cell vectors are rotated into
    R3, the two cells adjoining each edge are averaged, and the result is
    dotted with the edge-normal unit vector. Connectivity values are MPAS
    one-based indices. Inputs may have active rows only or one trailing padding
    row; a trailing zero edge row is returned exactly when ``cellsOnEdge`` has
    one.
    """
    if zonal.ndim != 2 or meridional.ndim != 2:
        raise ValueError("zonal and meridional must be rank-2 cell-by-level tensors")
    if tuple(zonal.shape) != tuple(meridional.shape):
        raise ValueError(
            f"zonal and meridional shapes differ: {tuple(zonal.shape)} != "
            f"{tuple(meridional.shape)}"
        )
    if zonal.shape[0] not in (nCells, nCells + 1):
        raise ValueError(
            f"cell wind first dimension must be nCells ({nCells}) or nCells+1 "
            f"({nCells + 1}), got {zonal.shape[0]}"
        )
    if cellsOnEdge.ndim != 2 or cellsOnEdge.shape[1] < 2:
        raise ValueError("cellsOnEdge must have shape (nEdges[+1], at least 2)")
    if cellsOnEdge.shape[0] not in (nEdges, nEdges + 1):
        raise ValueError(
            f"cellsOnEdge first dimension must be nEdges ({nEdges}) or nEdges+1 "
            f"({nEdges + 1}), got {cellsOnEdge.shape[0]}"
        )
    if latCell.ndim != 1 or lonCell.ndim != 1:
        raise ValueError("latCell and lonCell must be rank-1 tensors")
    if latCell.shape[0] not in (nCells, nCells + 1):
        raise ValueError("latCell must contain nCells active rows and optional padding")
    if lonCell.shape[0] not in (nCells, nCells + 1):
        raise ValueError("lonCell must contain nCells active rows and optional padding")
    if edgeNormalVectors.ndim != 2 or tuple(edgeNormalVectors.shape[1:]) != (3,):
        raise ValueError("edgeNormalVectors must have shape (nEdges[+1], 3)")
    if edgeNormalVectors.shape[0] not in (nEdges, nEdges + 1):
        raise ValueError(
            "edgeNormalVectors must contain nEdges active rows and optional padding"
        )

    zonal_active = zonal[:nCells, :]
    meridional_active = meridional[:nCells, :]
    lat = latCell[:nCells]
    lon = lonCell[:nCells]
    sin_lat = torch.sin(lat)[:, None]
    cos_lat = torch.cos(lat)[:, None]
    sin_lon = torch.sin(lon)[:, None]
    cos_lon = torch.cos(lon)[:, None]

    vector_x = -zonal_active * sin_lon - meridional_active * sin_lat * cos_lon
    vector_y = zonal_active * cos_lon - meridional_active * sin_lat * sin_lon
    vector_z = meridional_active * cos_lat

    cells = cellsOnEdge[:nEdges, :2].to(dtype=torch.int64) - 1
    if cells.device.type == "cpu" and (
        bool(torch.any(cells < 0)) or bool(torch.any(cells >= nCells))
    ):
        raise ValueError("active cellsOnEdge entries must be in [1, nCells]")
    cell1 = cells[:, 0]
    cell2 = cells[:, 1]

    x_edge = 0.5 * (vector_x[cell1, :] + vector_x[cell2, :])
    y_edge = 0.5 * (vector_y[cell1, :] + vector_y[cell2, :])
    z_edge = 0.5 * (vector_z[cell1, :] + vector_z[cell2, :])
    normals = edgeNormalVectors[:nEdges, :]
    edge_normal = (
        x_edge * normals[:, 0, None]
        + y_edge * normals[:, 1, None]
        + z_edge * normals[:, 2, None]
    )

    if cellsOnEdge.shape[0] == nEdges + 1:
        return torch.cat((edge_normal, torch.zeros_like(edge_normal[:1, :])), dim=0)
    return edge_normal


def edge_normal_gradient_to_zonal_meridional(
    gradient: torch.Tensor,
    angleEdge: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact edge-local zonal and meridional gradient components.

    The MPAS edge wind satisfies ``u = U*cos(angleEdge) + V*sin(angleEdge)``.
    Its transpose therefore maps ``dJ/du`` to the two components returned here.
    The orthogonal tangential direction is a null direction by construction.
    """
    if gradient.ndim != 2:
        raise ValueError("gradient must be a rank-2 edge-by-level tensor")
    if angleEdge.ndim != 1 or angleEdge.shape[0] != gradient.shape[0]:
        raise ValueError(
            "angleEdge must be rank-1 with the same edge extent as gradient"
        )
    return (
        gradient * torch.cos(angleEdge)[:, None],
        gradient * torch.sin(angleEdge)[:, None],
    )
