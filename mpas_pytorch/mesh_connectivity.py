"""
Precomputed mesh connectivity for GPU/vectorized kernels.

This module converts Fortran-style 1-based connectivity arrays (with sentinels)
into 0-based index tensors plus validity masks, and builds CSR views for sparse
operators used by dynamics kernels.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch


def _strip_leading_padding(array: torch.Tensor, expected_rows: int) -> torch.Tensor:
    """Strip trailing checkpoint padding row if present."""
    if array.shape[0] == expected_rows + 1:
        return array[:expected_rows]
    if array.shape[0] == expected_rows:
        return array
    raise ValueError(
        f"Unexpected leading dimension {array.shape[0]} (expected {expected_rows} or {expected_rows + 1})"
    )


def _require_int_tensor(name: str, value: torch.Tensor) -> None:
    if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
        raise ValueError(f"{name} must be integer, got dtype={value.dtype}")


def _to_zero_based_with_mask(
    indices_1based: torch.Tensor,
    upper_bound: int,
    active_counts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert 1-based indices to 0-based and build validity mask.

    Sentinels are invalid:
      - 0
      - values > upper_bound (e.g., nCells+1 checkpoint sentinels)
    Invalid entries are clamped to 0 in output index tensor and masked out.
    """
    _require_int_tensor("indices_1based", indices_1based)
    idx = indices_1based

    valid = (idx >= 1) & (idx <= upper_bound)
    idx0 = torch.clamp(idx - 1, min=0, max=max(0, upper_bound - 1))

    if active_counts is not None:
        counts = active_counts
        # Broadcast column indices against per-row active count.
        col = torch.arange(
            idx.shape[1], device=idx.device, dtype=torch.int64
        ).unsqueeze(0)
        valid_by_count = col < counts.unsqueeze(1)
        valid = valid & valid_by_count

    idx0 = torch.where(valid, idx0, torch.zeros_like(idx0))
    return idx0, valid


def _build_csr(
    indices0: torch.Tensor, valid_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build CSR row_ptr/col_ind from dense index+mask representation."""
    if indices0.ndim != 2 or valid_mask.ndim != 2:
        raise ValueError("CSR builder expects rank-2 tensors")
    if indices0.shape != valid_mask.shape:
        raise ValueError("indices0 and valid_mask shapes must match")

    device = indices0.device
    counts = valid_mask.sum(dim=1)
    row_ptr = torch.zeros(indices0.shape[0] + 1, dtype=torch.int64, device=device)
    # GPU NOTE: integer prefix sum (exact for int64); no FP ordering drift.
    row_ptr[1:] = torch.cumsum(counts, dim=0)
    col_ind = indices0[valid_mask]
    return row_ptr, col_ind


@dataclass(frozen=True)
class MeshConnectivity:
    """Precomputed 0-based mesh connectivity and masks."""

    nCells: int
    nEdges: int
    nVertices: int

    # Pattern A: Edge -> 2 Cells
    edge_cells_idx: torch.Tensor
    edge_cells_valid: torch.Tensor

    # Pattern B: Cell -> N Edges
    cell_edges_idx: torch.Tensor
    cell_edges_valid: torch.Tensor

    # Pattern C: Edge -> M Adv Cells
    edge_adv_cells_idx: torch.Tensor
    edge_adv_cells_valid: torch.Tensor

    # Pattern D: Vertex -> 3 Edges
    vertex_edges_idx: torch.Tensor
    vertex_edges_valid: torch.Tensor

    # Pattern E: Edge -> M Edges
    edge_edges_idx: torch.Tensor
    edge_edges_valid: torch.Tensor

    # Pattern F: Cell -> N Cells
    cell_cells_idx: torch.Tensor
    cell_cells_valid: torch.Tensor

    # Pattern G: Cell -> N Vertices
    cell_vertices_idx: torch.Tensor
    cell_vertices_valid: torch.Tensor

    # Ownership masks
    cell_owned: torch.Tensor
    edge_owned: torch.Tensor

    # CSR candidates
    divergence_row_ptr: torch.Tensor
    divergence_col_ind: torch.Tensor
    vorticity_row_ptr: torch.Tensor
    vorticity_col_ind: torch.Tensor

    @classmethod
    def from_mpas_mesh(
        cls,
        mesh: dict[str, Any],
        *,
        nCells: int,
        nEdges: int,
        nVertices: int,
        nCellsSolve: int | None = None,
        nEdgesSolve: int | None = None,
    ) -> MeshConnectivity:
        """
        Build connectivity cache from checkpoint-style (1-based+padded) tensors.

        Required keys:
          cellsOnEdge, edgesOnCell, nEdgesOnCell, advCellsForEdge, nAdvCellsForEdge,
          edgesOnVertex, edgesOnEdge, nEdgesOnEdge, cellsOnCell, verticesOnCell
        """
        required = [
            "cellsOnEdge",
            "edgesOnCell",
            "nEdgesOnCell",
            "advCellsForEdge",
            "nAdvCellsForEdge",
            "edgesOnVertex",
            "edgesOnEdge",
            "nEdgesOnEdge",
            "cellsOnCell",
            "verticesOnCell",
        ]
        missing = [k for k in required if k not in mesh]
        if missing:
            raise ValueError(f"Missing mesh connectivity fields: {missing}")

        # Strip checkpoint padding row where present.
        cellsOnEdge = _strip_leading_padding(mesh["cellsOnEdge"], nEdges)
        edgesOnCell = _strip_leading_padding(mesh["edgesOnCell"], nCells)
        nEdgesOnCell = _strip_leading_padding(mesh["nEdgesOnCell"], nCells)
        advCellsForEdge = _strip_leading_padding(mesh["advCellsForEdge"], nEdges)
        nAdvCellsForEdge = _strip_leading_padding(mesh["nAdvCellsForEdge"], nEdges)
        edgesOnVertex = _strip_leading_padding(mesh["edgesOnVertex"], nVertices)
        edgesOnEdge = _strip_leading_padding(mesh["edgesOnEdge"], nEdges)
        nEdgesOnEdge = _strip_leading_padding(mesh["nEdgesOnEdge"], nEdges)
        cellsOnCell = _strip_leading_padding(mesh["cellsOnCell"], nCells)
        verticesOnCell = _strip_leading_padding(mesh["verticesOnCell"], nCells)

        # Pattern A
        edge_cells_idx, edge_cells_valid = _to_zero_based_with_mask(cellsOnEdge, nCells)
        # Pattern B
        cell_edges_idx, cell_edges_valid = _to_zero_based_with_mask(
            edgesOnCell, nEdges, active_counts=nEdgesOnCell
        )
        # Pattern C
        edge_adv_cells_idx, edge_adv_cells_valid = _to_zero_based_with_mask(
            advCellsForEdge, nCells, active_counts=nAdvCellsForEdge
        )
        # Pattern D
        vertex_edges_idx, vertex_edges_valid = _to_zero_based_with_mask(
            edgesOnVertex, nEdges
        )
        # Pattern E
        edge_edges_idx, edge_edges_valid = _to_zero_based_with_mask(
            edgesOnEdge, nEdges, active_counts=nEdgesOnEdge
        )
        # Pattern F
        cell_cells_idx, cell_cells_valid = _to_zero_based_with_mask(
            cellsOnCell, nCells, active_counts=nEdgesOnCell
        )
        # Pattern G
        cell_vertices_idx, cell_vertices_valid = _to_zero_based_with_mask(
            verticesOnCell, nVertices, active_counts=nEdgesOnCell
        )

        # Ownership masks for local domain bounds.
        cell_owned = torch.zeros(nCells, dtype=torch.bool, device=cellsOnEdge.device)
        edge_owned = torch.zeros(nEdges, dtype=torch.bool, device=cellsOnEdge.device)
        n_cells_owned = (
            nCells if nCellsSolve is None else min(max(int(nCellsSolve), 0), nCells)
        )
        n_edges_owned = (
            nEdges if nEdgesSolve is None else min(max(int(nEdgesSolve), 0), nEdges)
        )
        cell_owned[:n_cells_owned] = True
        edge_owned[:n_edges_owned] = True

        # CSR views for SpMM candidates:
        # divergence: rows=cells, cols=edges (from cell->edges pattern)
        divergence_row_ptr, divergence_col_ind = _build_csr(
            cell_edges_idx, cell_edges_valid
        )
        # vorticity: rows=vertices, cols=edges (from vertex->edges pattern)
        vorticity_row_ptr, vorticity_col_ind = _build_csr(
            vertex_edges_idx, vertex_edges_valid
        )

        return cls(
            nCells=nCells,
            nEdges=nEdges,
            nVertices=nVertices,
            edge_cells_idx=edge_cells_idx,
            edge_cells_valid=edge_cells_valid,
            cell_edges_idx=cell_edges_idx,
            cell_edges_valid=cell_edges_valid,
            edge_adv_cells_idx=edge_adv_cells_idx,
            edge_adv_cells_valid=edge_adv_cells_valid,
            vertex_edges_idx=vertex_edges_idx,
            vertex_edges_valid=vertex_edges_valid,
            edge_edges_idx=edge_edges_idx,
            edge_edges_valid=edge_edges_valid,
            cell_cells_idx=cell_cells_idx,
            cell_cells_valid=cell_cells_valid,
            cell_vertices_idx=cell_vertices_idx,
            cell_vertices_valid=cell_vertices_valid,
            cell_owned=cell_owned,
            edge_owned=edge_owned,
            divergence_row_ptr=divergence_row_ptr,
            divergence_col_ind=divergence_col_ind,
            vorticity_row_ptr=vorticity_row_ptr,
            vorticity_col_ind=vorticity_col_ind,
        )

    def memory_bytes(self) -> int:
        """Return total memory footprint of cached tensors."""
        total = 0
        for value in self.__dict__.values():
            if isinstance(value, torch.Tensor):
                total += value.numel() * value.element_size()
        return total

    def memory_megabytes(self) -> float:
        return self.memory_bytes() / (1024.0 * 1024.0)

    def to(self, device: torch.device | str) -> MeshConnectivity:
        """
        Return a new MeshConnectivity with all tensor fields moved to ``device``.

        MeshConnectivity is a frozen dataclass, so this method never mutates
        the current instance in place.
        """
        target = torch.device(device)
        kwargs = {}
        for field in fields(self):
            value = getattr(self, field.name)
            kwargs[field.name] = (
                value.to(target) if isinstance(value, torch.Tensor) else value
            )
        return MeshConnectivity(**kwargs)  # type: ignore[arg-type]
