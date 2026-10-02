"""RCM-based mesh/state reordering utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from scipy.sparse import coo_matrix  # type: ignore[import-untyped]
from scipy.sparse.csgraph import reverse_cuthill_mckee  # type: ignore[import-untyped]

INT_DTYPES = {torch.int32, torch.int64}


@dataclass(frozen=True)
class RCMPermutation:
    """Permutation maps for cells, edges, and vertices."""

    cell_new_to_old: torch.Tensor
    edge_new_to_old: torch.Tensor
    vertex_new_to_old: torch.Tensor
    cell_old_to_new: torch.Tensor
    edge_old_to_new: torch.Tensor
    vertex_old_to_new: torch.Tensor


def _as_int(v: Any) -> int:
    if isinstance(v, torch.Tensor):
        return int(v.item())
    return int(v)


def _inverse_perm(new_to_old: torch.Tensor) -> torch.Tensor:
    old_to_new = torch.empty_like(new_to_old)
    old_to_new[new_to_old] = torch.arange(
        new_to_old.numel(),
        dtype=new_to_old.dtype,
        device=new_to_old.device,
    )
    return old_to_new


def _safe_rcm(n: int, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    if n <= 0:
        return np.arange(0, dtype=np.int64)
    if rows.size == 0:
        return np.arange(n, dtype=np.int64)
    data = np.ones(rows.shape[0], dtype=np.int8)
    graph = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    graph = (graph + graph.T).tocsr()
    graph.data[:] = 1
    return cast(
        np.ndarray,
        reverse_cuthill_mckee(graph, symmetric_mode=True).astype(np.int64),
    )


def _build_cell_permutation(mesh: dict[str, Any], n_cells: int) -> torch.Tensor:
    coc = mesh["cellsOnCell"][:n_cells].cpu().numpy()
    rows = np.repeat(np.arange(n_cells, dtype=np.int64), coc.shape[1])
    cols = coc.reshape(-1) - 1
    valid = (cols >= 0) & (cols < n_cells)
    return torch.from_numpy(_safe_rcm(n_cells, rows[valid], cols[valid]))


def _build_vertex_permutation(
    mesh: dict[str, Any], n_vertices: int, n_edges: int
) -> torch.Tensor:
    voe = mesh["verticesOnEdge"][:n_edges, :2].cpu().numpy() - 1
    valid = (
        (voe[:, 0] >= 0)
        & (voe[:, 0] < n_vertices)
        & (voe[:, 1] >= 0)
        & (voe[:, 1] < n_vertices)
    )
    rows = np.concatenate([voe[valid, 0], voe[valid, 1]])
    cols = np.concatenate([voe[valid, 1], voe[valid, 0]])
    return torch.from_numpy(_safe_rcm(n_vertices, rows, cols))


def _build_edge_permutation(
    mesh: dict[str, Any],
    n_edges: int,
    n_cells: int,
    cell_old_to_new_1b: torch.Tensor,
) -> torch.Tensor:
    coe = mesh["cellsOnEdge"][:n_edges, :2]
    c1 = coe[:, 0]
    c2 = coe[:, 1]
    valid1 = (c1 >= 1) & (c1 <= n_cells)
    valid2 = (c2 >= 1) & (c2 <= n_cells)
    c1n = torch.where(
        valid1, cell_old_to_new_1b[c1 - 1], torch.full_like(c1, n_cells + 1)
    )
    c2n = torch.where(
        valid2, cell_old_to_new_1b[c2 - 1], torch.full_like(c2, n_cells + 1)
    )
    mins = torch.minimum(c1n, c2n).cpu().numpy()
    maxs = torch.maximum(c1n, c2n).cpu().numpy()
    old = np.arange(n_edges, dtype=np.int64)
    perm = np.lexsort((old, maxs, mins)).astype(np.int64)
    return torch.from_numpy(perm)


def compute_rcm_permutation(mesh: dict[str, Any]) -> RCMPermutation:
    """Compute cell/edge/vertex permutations for RCM-reordered mesh locality."""
    n_cells = _as_int(mesh["nCells"])
    n_edges = _as_int(mesh["nEdges"])
    n_vertices = _as_int(mesh["nVertices"])

    cell_new_to_old = _build_cell_permutation(mesh, n_cells)
    cell_old_to_new = _inverse_perm(cell_new_to_old)
    cell_old_to_new_1b = cell_old_to_new + 1

    vertex_new_to_old = _build_vertex_permutation(mesh, n_vertices, n_edges)
    vertex_old_to_new = _inverse_perm(vertex_new_to_old)
    vertex_old_to_new_1b = vertex_old_to_new + 1

    edge_new_to_old = _build_edge_permutation(
        mesh=mesh,
        n_edges=n_edges,
        n_cells=n_cells,
        cell_old_to_new_1b=cell_old_to_new_1b,
    )
    edge_old_to_new = _inverse_perm(edge_new_to_old)
    _ = vertex_old_to_new_1b  # reserved for future value remaps; keeps naming symmetry

    return RCMPermutation(
        cell_new_to_old=cell_new_to_old,
        edge_new_to_old=edge_new_to_old,
        vertex_new_to_old=vertex_new_to_old,
        cell_old_to_new=cell_old_to_new,
        edge_old_to_new=edge_old_to_new,
        vertex_old_to_new=vertex_old_to_new,
    )


def _permute_first_dim(value: Any, perm: torch.Tensor) -> Any:
    if not isinstance(value, torch.Tensor) or value.ndim == 0:
        return value
    n = perm.numel()
    if value.shape[0] == n:
        return value[perm]
    if value.shape[0] == n + 1:
        return torch.cat([value[:n][perm], value[n : n + 1]], dim=0)
    return value


def _remap_indices_1based(
    values: torch.Tensor, old_to_new: torch.Tensor, n: int
) -> torch.Tensor:
    out = values.clone()
    valid = (out >= 1) & (out <= n)
    if valid.any():
        remapped = old_to_new[out[valid] - 1] + 1
        out[valid] = remapped.to(out.dtype)
    return out


def _apply_row_permutations(
    data: dict[str, Any],
    perms: RCMPermutation,
    n_cells: int,
    n_edges: int,
    n_vertices: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            if value.shape[0] in {n_cells, n_cells + 1}:
                out[key] = _permute_first_dim(
                    value, perms.cell_new_to_old.to(value.device)
                )
                continue
            if value.shape[0] in {n_edges, n_edges + 1}:
                out[key] = _permute_first_dim(
                    value, perms.edge_new_to_old.to(value.device)
                )
                continue
            if value.shape[0] in {n_vertices, n_vertices + 1}:
                out[key] = _permute_first_dim(
                    value, perms.vertex_new_to_old.to(value.device)
                )
                continue
        out[key] = value
    return out


def _apply_connectivity_remaps(
    data: dict[str, Any],
    perms: RCMPermutation,
    n_cells: int,
    n_edges: int,
    n_vertices: int,
) -> dict[str, Any]:
    cell_index_keys = {"cellsOnEdge", "cellsOnCell", "cellsOnVertex", "advCellsForEdge"}
    edge_index_keys = {"edgesOnCell", "edgesOnEdge", "edgesOnVertex"}
    vertex_index_keys = {"verticesOnEdge", "verticesOnCell"}
    out = dict(data)

    for key, value in list(out.items()):
        if not isinstance(value, torch.Tensor) or value.dtype not in INT_DTYPES:
            continue
        if key in cell_index_keys:
            out[key] = _remap_indices_1based(
                value, perms.cell_old_to_new.to(value.device), n_cells
            )
        elif key in edge_index_keys:
            out[key] = _remap_indices_1based(
                value, perms.edge_old_to_new.to(value.device), n_edges
            )
        elif key in vertex_index_keys:
            out[key] = _remap_indices_1based(
                value, perms.vertex_old_to_new.to(value.device), n_vertices
            )

    return out


def _target_device_from_inputs(*dicts: dict[str, Any]) -> torch.device:
    for data in dicts:
        for value in data.values():
            if isinstance(value, torch.Tensor):
                return value.device
    return torch.device("cpu")


def _move_tensor_dict_to_device(
    data: dict[str, Any], target_device: torch.device
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            out[key] = (
                value.to(target_device) if value.device != target_device else value
            )
        else:
            out[key] = value
    return out


def apply_rcm_reordering(
    state: dict[str, Any],
    mesh: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Apply RCM mesh reordering to mesh and state dictionaries.

    Note: This uses `scipy.sparse.csgraph.reverse_cuthill_mckee` (CPU-only).
    Permutations are computed on CPU and reordered outputs are transferred back
    to the input tensor device.

    Returns reordered copies. Input dicts are not mutated.
    """
    target_device = _target_device_from_inputs(state, mesh)
    n_cells = _as_int(mesh["nCells"])
    n_edges = _as_int(mesh["nEdges"])
    n_vertices = _as_int(mesh["nVertices"])

    perms = compute_rcm_permutation(mesh)

    mesh_rows = _apply_row_permutations(mesh, perms, n_cells, n_edges, n_vertices)
    state_rows = _apply_row_permutations(state, perms, n_cells, n_edges, n_vertices)

    mesh_reordered = _apply_connectivity_remaps(
        mesh_rows, perms, n_cells, n_edges, n_vertices
    )
    state_reordered = _apply_connectivity_remaps(
        state_rows, perms, n_cells, n_edges, n_vertices
    )
    mesh_reordered = _move_tensor_dict_to_device(mesh_reordered, target_device)
    state_reordered = _move_tensor_dict_to_device(state_reordered, target_device)

    mesh_reordered["_rcm_applied"] = True
    mesh_reordered["_rcm_permutation"] = {
        "cell_new_to_old": perms.cell_new_to_old.to(target_device).clone(),
        "edge_new_to_old": perms.edge_new_to_old.to(target_device).clone(),
        "vertex_new_to_old": perms.vertex_new_to_old.to(target_device).clone(),
        "cell_old_to_new": perms.cell_old_to_new.to(target_device).clone(),
        "edge_old_to_new": perms.edge_old_to_new.to(target_device).clone(),
        "vertex_old_to_new": perms.vertex_old_to_new.to(target_device).clone(),
    }
    return state_reordered, mesh_reordered
