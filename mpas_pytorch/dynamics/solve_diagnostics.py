"""
Diagnostic variable computation (diagnostic-solve).

Computes diagnostic variables including:
- h_edge: Height at edge locations (average of adjacent cells)
- divergence: Horizontal divergence at cell centers
- vorticity: Relative vorticity at vertices
- ke: Kinetic energy at cell centers (Ringler et al. JCP 2009, Hollingsworth correction)
- pv_edge: Potential vorticity at edges (with optional APVM upwinding)


Subroutine: atm_compute_solve_diagnostics_work
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

import torch

from mpas_pytorch.autodiff import is_grad_mode
from mpas_pytorch.dynamics.static_payloads import (
    StaticPairRelation,
    StaticRange,
    StaticSlotRelation,
    static_pair_gather,
    static_slot_reduce,
)
from mpas_pytorch.dynamics.stencil_ops import (
    _active_slot_count,
    _compile_active,
    _static_connectivity_cache_allowed,
    _static_mask_all_true,
    _static_slot_all_true,
    _tensor_version_cache_key,
    gather_rows,
)
from mpas_pytorch.utils.checks import require_consistent_device

DIAGNOSTIC_SOLVE_SOURCE_RECEIPT_KEY = "_diagnostic_solve_diagnostics_source_receipt"
_DIAGNOSTIC_SOLVE_REQUIRED_DIAGNOSTIC_KEYS = (
    "v",
    "vorticity",
    "divergence",
    "ke",
    "pv_edge",
    "rho_edge",
)
_DIAGNOSTIC_SOLVE_REGION_COMPILE_TARGETS = frozenset(
    {
        "diagnostic_solve_solve_diagnostics",
    }
)
_DIAGNOSTIC_SOLVE_CONNECTIVITY_TENSOR_NAMES = (
    "dvEdge",
    "dcEdge",
    "weightsOnEdge",
    "kiteAreasOnVertex",
    "cellsOnEdge",
    "verticesOnEdge",
    "edgesOnCell",
    "edgesOnCell_sign",
    "edgesOnVertex",
    "edgesOnVertex_sign",
    "verticesOnCell",
    "kiteForCell",
    "nEdgesOnCell",
    "nEdgesOnEdge",
    "edgesOnEdge",
)
_DIAGNOSTIC_SOLVE_DYNAMIC_ARGUMENT_NAMES = ("u", "h", "v_in")


@dataclass(frozen=True)
class DiagnosticSolveNumericPlan:
    """Immutable diagnostic-solve topology, metrics, and namelist specialization."""

    key: tuple[object, ...]
    dynamic_argument_names: tuple[str, ...]
    static_arguments: Mapping[str, Any]
    numeric_callable: Callable[..., tuple[torch.Tensor, ...]]

    def dynamic_values(self, call_arguments: Mapping[str, Any]) -> tuple[object, ...]:
        missing = [
            name for name in self.dynamic_argument_names if name not in call_arguments
        ]
        if missing:
            raise KeyError(
                f"diagnostic-solve numeric call is missing dynamic arguments: {missing}"
            )
        return tuple(call_arguments[name] for name in self.dynamic_argument_names)

    def with_numeric_callable(
        self,
        numeric_callable: Callable[..., tuple[torch.Tensor, ...]],
    ) -> "DiagnosticSolveNumericPlan":
        return replace(self, numeric_callable=numeric_callable)


def _diagnostic_solve_tensor_source_receipt(tensor: torch.Tensor) -> tuple[object, ...]:
    """Describe the exact tensor storage/version used to derive diagnostics."""
    primal = tensor
    try:
        unpacked_primal, _tangent = torch.autograd.forward_ad.unpack_dual(tensor)
        primal = unpacked_primal
    except RuntimeError:
        pass
    try:
        version: int | None = int(primal._version)
    except RuntimeError:
        # Inference tensors intentionally omit version counters. The receipt is
        # consumed only across the immediate init-to-first-dynamics-tendency boundary, where
        # the runner performs no in-place prognostic updates.
        version = None
    return (
        primal.device.type,
        primal.device.index,
        str(primal.dtype),
        tuple(primal.shape),
        tuple(primal.stride()),
        int(primal.storage_offset()),
        int(primal.data_ptr()),
        version,
    )


def _mark_diagnostic_solve_diagnostics_current(
    mapping: dict[str, Any],
    *,
    u: torch.Tensor,
    h: torch.Tensor,
) -> None:
    """Stamp diagnostics with the exact prognostic sources they were derived from."""
    mapping[DIAGNOSTIC_SOLVE_SOURCE_RECEIPT_KEY] = (
        _diagnostic_solve_tensor_source_receipt(u),
        _diagnostic_solve_tensor_source_receipt(h),
    )


def _diagnostic_solve_diagnostics_are_current(
    mapping: Mapping[str, Any],
    *,
    u: torch.Tensor,
    h: torch.Tensor,
) -> bool:
    """Return true only for complete diagnostics derived from these exact tensors."""
    if any(key not in mapping for key in _DIAGNOSTIC_SOLVE_REQUIRED_DIAGNOSTIC_KEYS):
        return False
    expected = (
        _diagnostic_solve_tensor_source_receipt(u),
        _diagnostic_solve_tensor_source_receipt(h),
    )
    return mapping.get(DIAGNOSTIC_SOLVE_SOURCE_RECEIPT_KEY) == expected


def _diagnostic_solve_tensor_cache_key(
    tensor: torch.Tensor,
) -> tuple[object, ...] | None:
    return _tensor_version_cache_key(tensor)


def _diagnostic_solve_connectivity_payload_cache_key(
    **kwargs: object,
) -> tuple[object, ...] | None:
    """Return the mutation-sensitive cache key for a diagnostic-solve static payload."""
    tensors = tuple(
        kwargs.get(name) for name in _DIAGNOSTIC_SOLVE_CONNECTIVITY_TENSOR_NAMES
    )
    if not _static_connectivity_cache_allowed(tensors):
        return None

    tensor_keys: list[tuple[object, ...]] = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            tensor_key = _diagnostic_solve_tensor_cache_key(tensor)
            if tensor_key is None:
                return None
            tensor_keys.append(tensor_key)

    return (
        kwargs["device"],
        kwargs["nCells"],
        kwargs["nEdges"],
        kwargs["nVertices"],
        kwargs["vertexDegree"],
        kwargs["cellStart"],
        kwargs["cellEnd"],
        kwargs["vertexStart"],
        kwargs["vertexEnd"],
        kwargs["edgeStart"],
        kwargs["edgeEnd"],
        *tensor_keys,
    )


def _diagnostic_solve_static_all_true(
    mask: torch.Tensor, *, allow_cuda_sync: bool = False
) -> bool:
    """Return setup-time all-true metadata for static topology masks."""
    return bool(_static_mask_all_true(mask, allow_cuda_sync=allow_cuda_sync))


def _diagnostic_solve_slot_all_true(
    mask: torch.Tensor,
    *,
    allow_cuda_sync: bool = False,
) -> tuple[bool, ...]:
    metadata = _static_slot_all_true(mask, allow_cuda_sync=allow_cuda_sync)
    if metadata is None:
        return tuple(False for _ in range(mask.shape[1]))
    return metadata


def _build_diagnostic_solve_connectivity_payload(
    *,
    device: torch.device,
    nCells: int,
    nEdges: int,
    nVertices: int,
    vertexDegree: int,
    cellStart: int,
    cellEnd: int,
    vertexStart: int,
    vertexEnd: int,
    edgeStart: int,
    edgeEnd: int,
    dvEdge: torch.Tensor,
    dcEdge: torch.Tensor,
    weightsOnEdge: torch.Tensor,
    kiteAreasOnVertex: torch.Tensor,
    cellsOnEdge: torch.Tensor,
    verticesOnEdge: torch.Tensor,
    edgesOnCell: torch.Tensor,
    edgesOnCell_sign: torch.Tensor,
    edgesOnVertex: torch.Tensor,
    edgesOnVertex_sign: torch.Tensor,
    verticesOnCell: torch.Tensor,
    kiteForCell: torch.Tensor,
    nEdgesOnCell: torch.Tensor,
    nEdgesOnEdge: torch.Tensor,
    edgesOnEdge: torch.Tensor,
) -> dict[str, Any]:
    """Build reusable diagnostic-solve mesh topology and metric stencil payloads."""
    source_topology_key = tuple(
        _diagnostic_solve_tensor_cache_key(tensor)
        or (
            "tensor_identity",
            id(tensor),
            tuple(tensor.shape),
            str(tensor.dtype),
            str(tensor.device),
        )
        for tensor in (
            dvEdge,
            dcEdge,
            weightsOnEdge,
            kiteAreasOnVertex,
            cellsOnEdge,
            verticesOnEdge,
            edgesOnCell,
            edgesOnCell_sign,
            edgesOnVertex,
            edgesOnVertex_sign,
            verticesOnCell,
            kiteForCell,
            nEdgesOnCell,
            nEdgesOnEdge,
            edgesOnEdge,
        )
    )
    edge_ids = torch.arange(nEdges, device=device)
    cell_ids = torch.arange(nCells, device=device)
    vertex_ids = torch.arange(nVertices, device=device)
    edge_active = (edge_ids >= edgeStart - 1) & (edge_ids < edgeEnd)
    cell_active = (cell_ids >= cellStart - 1) & (cell_ids < cellEnd)
    vertex_active = (vertex_ids >= vertexStart - 1) & (vertex_ids < vertexEnd)

    cells_on_edge = cellsOnEdge[:nEdges]
    vertices_on_edge = verticesOnEdge[:nEdges]
    edges_on_vertex = edgesOnVertex[:nVertices, :vertexDegree]
    edges_on_vertex_sign = edgesOnVertex_sign[:nVertices, :vertexDegree]
    n_edges_on_cell = nEdgesOnCell[:nCells]
    cell_slot_count = _active_slot_count(n_edges_on_cell, edgesOnCell.shape[1], device)
    edges_on_cell = edgesOnCell[:nCells, :cell_slot_count]
    edges_on_cell_sign = edgesOnCell_sign[:nCells, :cell_slot_count]
    vertices_on_cell = verticesOnCell[:nCells, :cell_slot_count]
    kite_for_cell = kiteForCell[:nCells, :cell_slot_count]
    n_edges_on_edge = nEdgesOnEdge[:nEdges]
    edges_on_edge = edgesOnEdge[:nEdges]
    max_edges = cell_slot_count
    full_max_edges2 = edges_on_edge.shape[1]

    coe_valid = (cells_on_edge > 0) & (cells_on_edge <= nCells)
    valid_h_edges = edge_active & coe_valid[:, 0] & coe_valid[:, 1]
    coe_idx = torch.clamp(cells_on_edge - 1, min=0, max=max(0, nCells - 1))
    efac = dcEdge[:nEdges] * dvEdge[:nEdges]

    eov_valid = (edges_on_vertex > 0) & (edges_on_vertex <= nEdges)
    eov_idx = torch.clamp(edges_on_vertex - 1, min=0, max=max(0, nEdges - 1))
    eov_valid_all_by_slot = _diagnostic_solve_slot_all_true(
        eov_valid, allow_cuda_sync=True
    )
    eov_weight = edges_on_vertex_sign * dcEdge[eov_idx]
    eov_weight = torch.where(eov_valid, eov_weight, torch.zeros_like(eov_weight))

    eoc_pos = torch.arange(max_edges, device=device)[None, :] < n_edges_on_cell[:, None]
    eoc_valid = (edges_on_cell > 0) & (edges_on_cell <= nEdges)
    eoc_mask = eoc_pos & eoc_valid
    eoc_idx = torch.clamp(edges_on_cell - 1, min=0, max=max(0, nEdges - 1))
    eoc_valid_all_by_slot = _diagnostic_solve_slot_all_true(
        eoc_mask, allow_cuda_sync=True
    )
    div_weight = edges_on_cell_sign * dvEdge[eoc_idx]
    div_weight = torch.where(eoc_mask, div_weight, torch.zeros_like(div_weight))
    ke_weight = torch.where(
        eoc_mask,
        torch.full_like(edges_on_cell_sign, 0.25),
        torch.zeros_like(edges_on_cell_sign),
    )

    voc_valid = (vertices_on_cell > 0) & (vertices_on_cell <= nVertices)
    kite_valid = (kite_for_cell > 0) & (kite_for_cell <= kiteAreasOnVertex.shape[1])
    vok_mask = eoc_pos & voc_valid & kite_valid
    voc_idx = torch.clamp(vertices_on_cell - 1, min=0, max=max(0, nVertices - 1))
    kite_idx = torch.clamp(
        kite_for_cell - 1, min=0, max=max(0, kiteAreasOnVertex.shape[1] - 1)
    )
    vok_valid_all_by_slot = _diagnostic_solve_slot_all_true(
        vok_mask, allow_cuda_sync=True
    )
    kite_area = kiteAreasOnVertex[voc_idx, kite_idx]
    kite_area = torch.where(vok_mask, kite_area, torch.zeros_like(kite_area))

    voe_valid = (vertices_on_edge > 0) & (vertices_on_edge <= nVertices)
    valid_pv_edges = edge_active & voe_valid[:, 0] & voe_valid[:, 1]
    voe_idx = torch.clamp(vertices_on_edge - 1, min=0, max=max(0, nVertices - 1))
    apvm_edge_valid = valid_pv_edges & coe_valid[:, 0] & coe_valid[:, 1]

    if n_edges_on_edge.numel() > 0 and device.type == "cpu" and not _compile_active():
        max_edges2 = min(full_max_edges2, int(n_edges_on_edge.max().detach().cpu()))
    else:
        max_edges2 = full_max_edges2
    eoe_pos = (
        torch.arange(max_edges2, device=device)[None, :] < n_edges_on_edge[:, None]
    )
    edges_on_edge_trimmed = edges_on_edge[:, :max_edges2]
    eoe_valid = (edges_on_edge_trimmed > 0) & (edges_on_edge_trimmed <= nEdges)
    eoe_mask = eoe_pos & eoe_valid
    eoe_idx = torch.clamp(edges_on_edge_trimmed - 1, min=0, max=max(0, nEdges - 1))
    weights_on_edge = weightsOnEdge[:nEdges, :max_edges2]
    eoe_valid_all_by_slot = _diagnostic_solve_slot_all_true(
        eoe_mask, allow_cuda_sync=True
    )
    v_weight = torch.where(eoe_mask, weights_on_edge, torch.zeros_like(weights_on_edge))

    edge_prefix_range = StaticRange("diagnostic_solve.edge_prefix", 0, nEdges)
    cell_prefix_range = StaticRange("diagnostic_solve.cell_prefix", 0, nCells)
    vertex_prefix_range = StaticRange("diagnostic_solve.vertex_prefix", 0, nVertices)
    h_edge_cell_relation = StaticPairRelation(
        "diagnostic_solve.h_edge_cells",
        first=coe_idx[:, 0],
        second=coe_idx[:, 1],
        valid=valid_h_edges,
        valid_all=_diagnostic_solve_static_all_true(
            valid_h_edges, allow_cuda_sync=True
        ),
    )
    vertex_edge_relation = StaticSlotRelation(
        "diagnostic_solve.vertex_edges",
        indices=eov_idx,
        valid=eov_valid,
        weights=eov_weight,
        valid_all_by_slot=eov_valid_all_by_slot,
    )
    cell_divergence_edge_relation = StaticSlotRelation(
        "diagnostic_solve.cell_edges_divergence",
        indices=eoc_idx,
        valid=eoc_mask,
        weights=div_weight,
        valid_all_by_slot=eoc_valid_all_by_slot,
    )
    cell_ke_edge_relation = StaticSlotRelation(
        "diagnostic_solve.cell_edges_ke",
        indices=eoc_idx,
        valid=eoc_mask,
        weights=ke_weight,
        valid_all_by_slot=eoc_valid_all_by_slot,
    )
    cell_vertex_relation = StaticSlotRelation(
        "diagnostic_solve.cell_vertices",
        indices=voc_idx,
        valid=vok_mask,
        weights=kite_area,
        valid_all_by_slot=vok_valid_all_by_slot,
    )
    edge_vertex_relation = StaticPairRelation(
        "diagnostic_solve.edge_vertices",
        first=voe_idx[:, 0],
        second=voe_idx[:, 1],
        valid=valid_pv_edges,
        valid_all=_diagnostic_solve_static_all_true(
            valid_pv_edges, allow_cuda_sync=True
        ),
    )
    edge_neighbor_relation = StaticSlotRelation(
        "diagnostic_solve.edge_neighbors",
        indices=eoe_idx,
        valid=eoe_mask,
        weights=v_weight,
        valid_all_by_slot=eoe_valid_all_by_slot,
    )

    return {
        "source_topology_key": source_topology_key,
        "edge_prefix_range": edge_prefix_range,
        "cell_prefix_range": cell_prefix_range,
        "vertex_prefix_range": vertex_prefix_range,
        "edge_active": edge_active,
        "edge_active_all": _diagnostic_solve_static_all_true(
            edge_active, allow_cuda_sync=True
        ),
        "cell_active": cell_active,
        "cell_active_all": _diagnostic_solve_static_all_true(
            cell_active, allow_cuda_sync=True
        ),
        "vertex_active": vertex_active,
        "vertex_active_all": _diagnostic_solve_static_all_true(
            vertex_active, allow_cuda_sync=True
        ),
        "coe_idx": coe_idx,
        "valid_h_edges": valid_h_edges,
        "valid_h_edges_all": _diagnostic_solve_static_all_true(
            valid_h_edges, allow_cuda_sync=True
        ),
        "h_edge_cell_relation": h_edge_cell_relation,
        "efac": efac,
        "eov_valid": eov_valid,
        "eov_valid_all": _diagnostic_solve_static_all_true(
            eov_valid, allow_cuda_sync=True
        ),
        "eov_valid_all_by_slot": eov_valid_all_by_slot,
        "eov_idx": eov_idx,
        "eov_weight": eov_weight,
        "vertex_edge_relation": vertex_edge_relation,
        "eoc_idx": eoc_idx,
        "eoc_valid_all_by_slot": eoc_valid_all_by_slot,
        "div_weight": div_weight,
        "ke_weight": ke_weight,
        "cell_divergence_edge_relation": cell_divergence_edge_relation,
        "cell_ke_edge_relation": cell_ke_edge_relation,
        "voc_idx": voc_idx,
        "vok_valid_all_by_slot": vok_valid_all_by_slot,
        "kite_area": kite_area,
        "cell_vertex_relation": cell_vertex_relation,
        "voe_idx": voe_idx,
        "valid_pv_edges": valid_pv_edges,
        "valid_pv_edges_all": _diagnostic_solve_static_all_true(
            valid_pv_edges, allow_cuda_sync=True
        ),
        "edge_vertex_relation": edge_vertex_relation,
        "apvm_edge_valid": apvm_edge_valid,
        "apvm_edge_valid_all": _diagnostic_solve_static_all_true(
            apvm_edge_valid,
            allow_cuda_sync=True,
        ),
        "eoe_idx": eoe_idx,
        "eoe_valid_all_by_slot": eoe_valid_all_by_slot,
        "v_weight": v_weight,
        "edge_neighbor_relation": edge_neighbor_relation,
    }


def _diagnostic_solve_connectivity_payload(**kwargs: object) -> dict[str, Any]:
    """Build diagnostic-solve topology for a direct call without retaining module state.

    Production callers retain this payload in ``DycoreStaticPayloads``.  Keeping
    this fallback stateless prevents a second, invisible owner whose lifetime and
    invalidation policy can diverge from the runner-owned store.
    """
    return _build_diagnostic_solve_connectivity_payload(**kwargs)  # type: ignore[arg-type]


def _diagnostic_solve_masked_prefix_values(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    mask_all: bool = False,
) -> torch.Tensor:
    if mask_all:
        return values
    return torch.where(mask[:, None], values, torch.zeros_like(values))


def _diagnostic_solve_tensor_has_forward_ad(tensor: object) -> bool:
    if not isinstance(tensor, torch.Tensor):
        return False
    try:
        _primal, tangent = torch.autograd.forward_ad.unpack_dual(tensor)
    except Exception:  # noqa: BLE001
        return False
    return tangent is not None


def _diagnostic_solve_add_fortran_ordered_slots(
    base: torch.Tensor,
    field: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    valid: torch.Tensor | None,
    valid_all_by_slot: tuple[bool, ...] | None,
    *,
    leading_factor: float,
    row_factor: torch.Tensor,
) -> torch.Tensor:
    """Add weighted slots to ``base`` in the Fortran expression order."""
    total = base
    for slot in range(indices.shape[1]):
        contribution = leading_factor * weights[:, slot, None]
        contribution = contribution * gather_rows(field, indices[:, slot])
        contribution = contribution * row_factor
        valid_all = valid_all_by_slot is not None and valid_all_by_slot[slot]
        if valid is not None and not valid_all:
            contribution = torch.where(
                valid[:, slot, None], contribution, torch.zeros_like(contribution)
            )
        total = total + contribution
    return total


def _diagnostic_solve_numeric_composition(
    *,
    nCells: int,
    nEdges: int,
    nVertices: int,
    nVertLevels: int,
    dt: float,
    config_apvm_upwinding: float,
    fVertex: torch.Tensor,
    invAreaTriangle: torch.Tensor,
    invAreaCell: torch.Tensor,
    invDvEdge: torch.Tensor,
    invDcEdge: torch.Tensor,
    u: torch.Tensor,
    h: torch.Tensor,
    h_edge_first: torch.Tensor,
    h_edge_second: torch.Tensor,
    h_edge_valid: torch.Tensor | None,
    h_edge_valid_all: bool | None,
    vertex_edge_indices: torch.Tensor,
    vertex_edge_weights: torch.Tensor | None,
    vertex_edge_valid: torch.Tensor | None,
    vertex_edge_valid_all_by_slot: tuple[bool, ...] | None,
    cell_div_indices: torch.Tensor,
    cell_div_weights: torch.Tensor | None,
    cell_div_valid: torch.Tensor | None,
    cell_div_valid_all_by_slot: tuple[bool, ...] | None,
    cell_ke_indices: torch.Tensor,
    cell_ke_weights: torch.Tensor | None,
    cell_ke_valid: torch.Tensor | None,
    cell_ke_valid_all_by_slot: tuple[bool, ...] | None,
    cell_vertex_indices: torch.Tensor,
    cell_vertex_weights: torch.Tensor | None,
    cell_vertex_valid: torch.Tensor | None,
    cell_vertex_valid_all_by_slot: tuple[bool, ...] | None,
    edge_vertex_first: torch.Tensor,
    edge_vertex_second: torch.Tensor,
    edge_vertex_valid: torch.Tensor | None,
    edge_vertex_valid_all: bool | None,
    edge_neighbor_indices: torch.Tensor,
    edge_neighbor_weights: torch.Tensor | None,
    edge_neighbor_valid: torch.Tensor | None,
    edge_neighbor_valid_all_by_slot: tuple[bool, ...] | None,
    edge_active: torch.Tensor,
    edge_active_all: bool,
    cell_active: torch.Tensor,
    cell_active_all: bool,
    vertex_active: torch.Tensor,
    vertex_active_all: bool,
    valid_h_edges: torch.Tensor,
    valid_h_edges_all: bool,
    valid_pv_edges: torch.Tensor,
    valid_pv_edges_all: bool,
    apvm_edge_valid: torch.Tensor,
    apvm_edge_valid_all: bool,
    efac: torch.Tensor,
    rk_step: int | None = None,
    v_in: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Canonical diagnostic-solve composition returning active rows only."""
    del nVertLevels
    h_first, h_second = static_pair_gather(
        h, h_edge_first, h_edge_second, h_edge_valid, h_edge_valid_all
    )
    h_edge_values = _diagnostic_solve_masked_prefix_values(
        0.5 * (h_first + h_second), valid_h_edges, mask_all=valid_h_edges_all
    )
    ke_edge_values = _diagnostic_solve_masked_prefix_values(
        efac[:, None] * (u[:nEdges, :] ** 2),
        valid_h_edges,
        mask_all=valid_h_edges_all,
    )
    vorticity_values = _diagnostic_solve_masked_prefix_values(
        static_slot_reduce(
            u,
            vertex_edge_indices,
            vertex_edge_weights,
            vertex_edge_valid,
            vertex_edge_valid_all_by_slot,
        )
        * invAreaTriangle[:nVertices, None],
        vertex_active,
        mask_all=vertex_active_all,
    )
    divergence_values = _diagnostic_solve_masked_prefix_values(
        static_slot_reduce(
            u,
            cell_div_indices,
            cell_div_weights,
            cell_div_valid,
            cell_div_valid_all_by_slot,
        )
        * invAreaCell[:nCells, None],
        cell_active,
        mask_all=cell_active_all,
    )
    ke_values = (
        static_slot_reduce(
            ke_edge_values,
            cell_ke_indices,
            cell_ke_weights,
            cell_ke_valid,
            cell_ke_valid_all_by_slot,
        )
        * invAreaCell[:nCells, None]
    )
    ke_vertex_sum = static_slot_reduce(
        ke_edge_values,
        vertex_edge_indices,
        None,
        vertex_edge_valid,
        vertex_edge_valid_all_by_slot,
    )
    ke_vertex_values = _diagnostic_solve_masked_prefix_values(
        0.25 * invAreaTriangle[:nVertices, None] * ke_vertex_sum,
        vertex_active,
        mask_all=vertex_active_all,
    )
    ke_fact = 1.0 - 0.375
    if cell_vertex_weights is None:
        raise ValueError("diagnostic-solve cell-vertex relation requires weights")
    ke_values = _diagnostic_solve_masked_prefix_values(
        _diagnostic_solve_add_fortran_ordered_slots(
            ke_fact * ke_values,
            ke_vertex_values,
            cell_vertex_indices,
            cell_vertex_weights,
            cell_vertex_valid,
            cell_vertex_valid_all_by_slot,
            leading_factor=1.0 - ke_fact,
            row_factor=invAreaCell[:nCells, None],
        ),
        cell_active,
        mask_all=cell_active_all,
    )

    pv_vertex_values = _diagnostic_solve_masked_prefix_values(
        fVertex[:nVertices, None] + vorticity_values,
        vertex_active,
        mask_all=vertex_active_all,
    )
    pv_vertex_1, pv_vertex_2 = static_pair_gather(
        pv_vertex_values,
        edge_vertex_first,
        edge_vertex_second,
        edge_vertex_valid,
        edge_vertex_valid_all,
    )
    pv_edge_values = _diagnostic_solve_masked_prefix_values(
        0.5 * (pv_vertex_1 + pv_vertex_2),
        valid_pv_edges,
        mask_all=valid_pv_edges_all,
    )

    reconstruct_v = rk_step is None or int(rk_step) == 3
    if reconstruct_v:
        v_values = _diagnostic_solve_masked_prefix_values(
            static_slot_reduce(
                u,
                edge_neighbor_indices,
                edge_neighbor_weights,
                edge_neighbor_valid,
                edge_neighbor_valid_all_by_slot,
            ),
            edge_active,
            mask_all=edge_active_all,
        )
    else:
        if v_in is None:
            raise ValueError(
                "diagnostic-solve called with rk_step != 3 but no v_in provided. "
                "Pass the previous-step tangential velocity for this RK stage."
            )
        v_values = v_in[:nEdges, :]

    if config_apvm_upwinding > 0.0:
        pv_cell_values = _diagnostic_solve_masked_prefix_values(
            static_slot_reduce(
                pv_vertex_values,
                cell_vertex_indices,
                cell_vertex_weights,
                cell_vertex_valid,
                cell_vertex_valid_all_by_slot,
                slot_factor=invAreaCell[:nCells, None],
            ),
            cell_active,
            mask_all=cell_active_all,
        )
        grad_pvt = (pv_vertex_2 - pv_vertex_1) * invDvEdge[:nEdges, None]
        pv_cell_1, pv_cell_2 = static_pair_gather(
            pv_cell_values,
            h_edge_first,
            h_edge_second,
            h_edge_valid,
            h_edge_valid_all,
        )
        grad_pvn = (pv_cell_2 - pv_cell_1) * invDcEdge[:nEdges, None]
        delta = (config_apvm_upwinding * dt) * (
            v_values * grad_pvt + u[:nEdges, :] * grad_pvn
        )
        if not apvm_edge_valid_all:
            delta = torch.where(
                apvm_edge_valid[:, None], delta, torch.zeros_like(delta)
            )
        pv_edge_values = _diagnostic_solve_masked_prefix_values(
            pv_edge_values - delta, edge_active, mask_all=edge_active_all
        )

    return (
        h_edge_values,
        v_values,
        divergence_values,
        vorticity_values,
        ke_values,
        pv_edge_values,
    )


_DIAGNOSTIC_SOLVE_OUTPUT_ROW_COUNTS = {
    "h_edge": "nEdges",
    "v": "nEdges",
    "divergence": "nCells",
    "vorticity": "nVertices",
    "ke": "nCells",
    "pv_edge": "nEdges",
}


def pack_diagnostic_solve_diagnostics_public(
    diagnostics: Mapping[str, torch.Tensor],
    *,
    nCells: int,
    nEdges: int,
    nVertices: int,
    v_padding_source: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Pack active diagnostic-solve rows at a public/checkpoint boundary."""
    row_counts = {
        "nCells": int(nCells),
        "nEdges": int(nEdges),
        "nVertices": int(nVertices),
    }
    packed: dict[str, Any] = {}
    for name, count_name in _DIAGNOSTIC_SOLVE_OUTPUT_ROW_COUNTS.items():
        if name not in diagnostics:
            continue
        values = diagnostics[name]
        active_rows = row_counts[count_name]
        if values.shape[0] == active_rows + 1:
            packed[name] = values
            continue
        if values.shape[0] != active_rows:
            raise ValueError(
                f"diagnostic-solve {name} row count mismatch: expected {active_rows} active "
                f"or {active_rows + 1} padded rows, got {values.shape[0]}"
            )
        if (
            name == "v"
            and isinstance(v_padding_source, torch.Tensor)
            and v_padding_source.shape[0] > active_rows
        ):
            padding = v_padding_source[active_rows : active_rows + 1]
        else:
            padding = values.new_zeros((1, *values.shape[1:]))
        packed[name] = torch.cat((values, padding), dim=0)
    return packed


def _diagnostic_solve_plan_tensor_fingerprint(
    tensor: torch.Tensor,
) -> tuple[object, ...]:
    return _diagnostic_solve_tensor_cache_key(tensor) or (
        "tensor_identity",
        id(tensor),
        tuple(tensor.shape),
        str(tensor.dtype),
        str(tensor.device),
    )


def build_diagnostic_solve_numeric_plan(
    *,
    connectivity_payload: Mapping[str, Any],
    nCells: int,
    nEdges: int,
    nVertices: int,
    nVertLevels: int,
    dt: float,
    config_apvm_upwinding: float,
    fVertex: torch.Tensor,
    invAreaTriangle: torch.Tensor,
    invAreaCell: torch.Tensor,
    invDvEdge: torch.Tensor,
    invDcEdge: torch.Tensor,
    rk_step: int | None,
    stage_key: tuple[object, ...] = (),
) -> DiagnosticSolveNumericPlan:
    """Bind diagnostic-solve topology and static choices into a dynamic-field callable."""
    relation_types = {
        "h_edge_cell_op": ("h_edge_cell_relation", StaticPairRelation),
        "vertex_edge_op": ("vertex_edge_relation", StaticSlotRelation),
        "cell_divergence_edge_op": (
            "cell_divergence_edge_relation",
            StaticSlotRelation,
        ),
        "cell_ke_edge_op": ("cell_ke_edge_relation", StaticSlotRelation),
        "cell_vertex_op": ("cell_vertex_relation", StaticSlotRelation),
        "edge_vertex_op": ("edge_vertex_relation", StaticPairRelation),
        "edge_neighbor_op": ("edge_neighbor_relation", StaticSlotRelation),
    }
    relations: dict[str, Any] = {}
    for target_name, (source_name, expected_type) in relation_types.items():
        relation = connectivity_payload.get(source_name)
        if not isinstance(relation, expected_type):
            raise TypeError(
                f"diagnostic-solve plan requires {source_name} as {expected_type.__name__}"
            )
        relations[target_name] = relation

    static_arguments = MappingProxyType(
        {
            "nCells": int(nCells),
            "nEdges": int(nEdges),
            "nVertices": int(nVertices),
            "nVertLevels": int(nVertLevels),
            "dt": float(dt),
            "config_apvm_upwinding": float(config_apvm_upwinding),
            "fVertex": fVertex,
            "invAreaTriangle": invAreaTriangle,
            "invAreaCell": invAreaCell,
            "invDvEdge": invDvEdge,
            "invDcEdge": invDcEdge,
            **relations,
            "edge_active": connectivity_payload["edge_active"],
            "edge_active_all": bool(connectivity_payload["edge_active_all"]),
            "cell_active": connectivity_payload["cell_active"],
            "cell_active_all": bool(connectivity_payload["cell_active_all"]),
            "vertex_active": connectivity_payload["vertex_active"],
            "vertex_active_all": bool(connectivity_payload["vertex_active_all"]),
            "valid_h_edges": connectivity_payload["valid_h_edges"],
            "valid_h_edges_all": bool(connectivity_payload["valid_h_edges_all"]),
            "valid_pv_edges": connectivity_payload["valid_pv_edges"],
            "valid_pv_edges_all": bool(connectivity_payload["valid_pv_edges_all"]),
            "apvm_edge_valid": connectivity_payload["apvm_edge_valid"],
            "apvm_edge_valid_all": bool(connectivity_payload["apvm_edge_valid_all"]),
            "efac": connectivity_payload["efac"],
            "rk_step": None if rk_step is None else int(rk_step),
        }
    )
    key = (
        "DiagnosticSolveNumericPlan",
        tuple(stage_key),
        connectivity_payload.get("source_topology_key"),
        tuple(
            (name, _diagnostic_solve_plan_tensor_fingerprint(value))
            for name, value in static_arguments.items()
            if isinstance(value, torch.Tensor)
        ),
        int(nCells),
        int(nEdges),
        int(nVertices),
        int(nVertLevels),
        float(dt),
        float(config_apvm_upwinding),
        -1 if rk_step is None else int(rk_step),
    )

    n_cells_v = int(static_arguments["nCells"])
    n_edges_v = int(static_arguments["nEdges"])
    n_vertices_v = int(static_arguments["nVertices"])
    n_levels_v = int(static_arguments["nVertLevels"])
    dt_v = float(static_arguments["dt"])
    apvm_v = float(static_arguments["config_apvm_upwinding"])
    f_vertex_v = static_arguments["fVertex"]
    inv_area_triangle_v = static_arguments["invAreaTriangle"]
    inv_area_cell_v = static_arguments["invAreaCell"]
    inv_dv_edge_v = static_arguments["invDvEdge"]
    inv_dc_edge_v = static_arguments["invDcEdge"]
    h_edge_cell_op_v = relations["h_edge_cell_op"]
    vertex_edge_op_v = relations["vertex_edge_op"]
    cell_divergence_edge_op_v = relations["cell_divergence_edge_op"]
    cell_ke_edge_op_v = relations["cell_ke_edge_op"]
    cell_vertex_op_v = relations["cell_vertex_op"]
    edge_vertex_op_v = relations["edge_vertex_op"]
    edge_neighbor_op_v = relations["edge_neighbor_op"]
    h_edge_first_v = h_edge_cell_op_v.first
    h_edge_second_v = h_edge_cell_op_v.second
    h_edge_valid_v = h_edge_cell_op_v.valid
    h_edge_valid_all_relation_v = h_edge_cell_op_v.valid_all
    vertex_edge_indices_v = vertex_edge_op_v.indices
    vertex_edge_weights_v = vertex_edge_op_v.weights
    vertex_edge_valid_v = vertex_edge_op_v.valid
    vertex_edge_valid_all_by_slot_v = vertex_edge_op_v.valid_all_by_slot
    cell_div_indices_v = cell_divergence_edge_op_v.indices
    cell_div_weights_v = cell_divergence_edge_op_v.weights
    cell_div_valid_v = cell_divergence_edge_op_v.valid
    cell_div_valid_all_by_slot_v = cell_divergence_edge_op_v.valid_all_by_slot
    cell_ke_indices_v = cell_ke_edge_op_v.indices
    cell_ke_weights_v = cell_ke_edge_op_v.weights
    cell_ke_valid_v = cell_ke_edge_op_v.valid
    cell_ke_valid_all_by_slot_v = cell_ke_edge_op_v.valid_all_by_slot
    cell_vertex_indices_v = cell_vertex_op_v.indices
    cell_vertex_weights_v = cell_vertex_op_v.weights
    cell_vertex_valid_v = cell_vertex_op_v.valid
    cell_vertex_valid_all_by_slot_v = cell_vertex_op_v.valid_all_by_slot
    edge_vertex_first_v = edge_vertex_op_v.first
    edge_vertex_second_v = edge_vertex_op_v.second
    edge_vertex_valid_v = edge_vertex_op_v.valid
    edge_vertex_valid_all_v = edge_vertex_op_v.valid_all
    edge_neighbor_indices_v = edge_neighbor_op_v.indices
    edge_neighbor_weights_v = edge_neighbor_op_v.weights
    edge_neighbor_valid_v = edge_neighbor_op_v.valid
    edge_neighbor_valid_all_by_slot_v = edge_neighbor_op_v.valid_all_by_slot
    edge_active_v = static_arguments["edge_active"]
    edge_active_all_v = bool(static_arguments["edge_active_all"])
    cell_active_v = static_arguments["cell_active"]
    cell_active_all_v = bool(static_arguments["cell_active_all"])
    vertex_active_v = static_arguments["vertex_active"]
    vertex_active_all_v = bool(static_arguments["vertex_active_all"])
    valid_h_edges_v = static_arguments["valid_h_edges"]
    valid_h_edges_all_v = bool(static_arguments["valid_h_edges_all"])
    valid_pv_edges_v = static_arguments["valid_pv_edges"]
    valid_pv_edges_all_v = bool(static_arguments["valid_pv_edges_all"])
    apvm_edge_valid_v = static_arguments["apvm_edge_valid"]
    apvm_edge_valid_all_v = bool(static_arguments["apvm_edge_valid_all"])
    efac_v = static_arguments["efac"]
    rk_step_v = static_arguments["rk_step"]

    def _diagnostic_solve_bound_numeric(
        u: torch.Tensor,
        h: torch.Tensor,
        v_in: torch.Tensor | None,
    ) -> tuple[torch.Tensor, ...]:
        return _diagnostic_solve_numeric_composition(
            nCells=n_cells_v,
            nEdges=n_edges_v,
            nVertices=n_vertices_v,
            nVertLevels=n_levels_v,
            dt=dt_v,
            config_apvm_upwinding=apvm_v,
            fVertex=f_vertex_v,
            invAreaTriangle=inv_area_triangle_v,
            invAreaCell=inv_area_cell_v,
            invDvEdge=inv_dv_edge_v,
            invDcEdge=inv_dc_edge_v,
            u=u,
            h=h,
            h_edge_first=h_edge_first_v,
            h_edge_second=h_edge_second_v,
            h_edge_valid=h_edge_valid_v,
            h_edge_valid_all=h_edge_valid_all_relation_v,
            vertex_edge_indices=vertex_edge_indices_v,
            vertex_edge_weights=vertex_edge_weights_v,
            vertex_edge_valid=vertex_edge_valid_v,
            vertex_edge_valid_all_by_slot=vertex_edge_valid_all_by_slot_v,
            cell_div_indices=cell_div_indices_v,
            cell_div_weights=cell_div_weights_v,
            cell_div_valid=cell_div_valid_v,
            cell_div_valid_all_by_slot=cell_div_valid_all_by_slot_v,
            cell_ke_indices=cell_ke_indices_v,
            cell_ke_weights=cell_ke_weights_v,
            cell_ke_valid=cell_ke_valid_v,
            cell_ke_valid_all_by_slot=cell_ke_valid_all_by_slot_v,
            cell_vertex_indices=cell_vertex_indices_v,
            cell_vertex_weights=cell_vertex_weights_v,
            cell_vertex_valid=cell_vertex_valid_v,
            cell_vertex_valid_all_by_slot=cell_vertex_valid_all_by_slot_v,
            edge_vertex_first=edge_vertex_first_v,
            edge_vertex_second=edge_vertex_second_v,
            edge_vertex_valid=edge_vertex_valid_v,
            edge_vertex_valid_all=edge_vertex_valid_all_v,
            edge_neighbor_indices=edge_neighbor_indices_v,
            edge_neighbor_weights=edge_neighbor_weights_v,
            edge_neighbor_valid=edge_neighbor_valid_v,
            edge_neighbor_valid_all_by_slot=edge_neighbor_valid_all_by_slot_v,
            edge_active=edge_active_v,
            edge_active_all=edge_active_all_v,
            cell_active=cell_active_v,
            cell_active_all=cell_active_all_v,
            vertex_active=vertex_active_v,
            vertex_active_all=vertex_active_all_v,
            valid_h_edges=valid_h_edges_v,
            valid_h_edges_all=valid_h_edges_all_v,
            valid_pv_edges=valid_pv_edges_v,
            valid_pv_edges_all=valid_pv_edges_all_v,
            apvm_edge_valid=apvm_edge_valid_v,
            apvm_edge_valid_all=apvm_edge_valid_all_v,
            efac=efac_v,
            rk_step=rk_step_v,
            v_in=v_in,
        )

    _diagnostic_solve_bound_numeric.__name__ = "_diagnostic_solve_bound_numeric"
    _diagnostic_solve_bound_numeric.__qualname__ = "_diagnostic_solve_bound_numeric"
    return DiagnosticSolveNumericPlan(
        key=key,
        dynamic_argument_names=_DIAGNOSTIC_SOLVE_DYNAMIC_ARGUMENT_NAMES,
        static_arguments=static_arguments,
        numeric_callable=_diagnostic_solve_bound_numeric,
    )


def compile_diagnostic_solve_numeric_plan(
    plan: DiagnosticSolveNumericPlan,
    *,
    compile_kernel: Callable[..., Callable[..., Any]],
    mode: str,
    backend: str,
) -> DiagnosticSolveNumericPlan:
    """Compile the prebound compact-scratch diagnostic-solve body outside orchestration."""
    compiled = compile_kernel(
        "diagnostic_solve_solve_diagnostics",
        plan.numeric_callable,
        enabled=True,
        mode=mode,
        backend=backend,
        specialization_key=plan.key,
    )
    return plan.with_numeric_callable(compiled)


def atm_compute_solve_diagnostics_work(
    nCells: int,
    nEdges: int,
    nVertices: int,
    nVertLevels: int,
    vertexDegree: int,
    dt: float,
    config_apvm_upwinding: float,
    cellStart: int,
    cellEnd: int,
    vertexStart: int,
    vertexEnd: int,
    edgeStart: int,
    edgeEnd: int,
    # Mesh geometry (1D, with padding)
    fVertex: torch.Tensor,  # (nVertices+1,)
    fEdge: torch.Tensor,  # (nEdges+1,) - unused in current implementation
    invAreaTriangle: torch.Tensor,  # (nVertices+1,)
    invAreaCell: torch.Tensor,  # (nCells+1,)
    dvEdge: torch.Tensor,  # (nEdges+1,)
    dcEdge: torch.Tensor,  # (nEdges+1,)
    invDvEdge: torch.Tensor,  # (nEdges+1,)
    invDcEdge: torch.Tensor,  # (nEdges+1,)
    weightsOnEdge: torch.Tensor,  # (nEdges+1, maxEdges2)
    kiteAreasOnVertex: torch.Tensor,  # (nVertices+1, vertexDegree)
    # Connectivity arrays (with padding, 1-based Fortran indexing)
    cellsOnEdge: torch.Tensor,  # (nEdges+1, 2)
    verticesOnEdge: torch.Tensor,  # (nEdges+1, 2)
    edgesOnCell: torch.Tensor,  # (nCells+1, maxEdges)
    edgesOnCell_sign: torch.Tensor,  # (nCells+1, maxEdges)
    edgesOnVertex: torch.Tensor,  # (nVertices+1, vertexDegree)
    edgesOnVertex_sign: torch.Tensor,  # (nVertices+1, vertexDegree)
    verticesOnCell: torch.Tensor,  # (nCells+1, maxEdges)
    kiteForCell: torch.Tensor,  # (nCells+1, maxEdges)
    nEdgesOnCell: torch.Tensor,  # (nCells+1,)
    nEdgesOnEdge: torch.Tensor,  # (nEdges+1,)
    edgesOnEdge: torch.Tensor,  # (nEdges+1, maxEdges2)
    # State variables (with padding)
    u: torch.Tensor,  # (nEdges+1, nVertLevels)
    h: torch.Tensor,  # (nCells+1, nVertLevels)
    rk_step: int | None = None,  # Fortran: optional rk_step controls v reconstruction
    v_in: torch.Tensor | None = None,  # Previous-step v when rk_step != 3
    workspace=None,  # Optional DycoreWorkspace with pre-allocated diagnostic_solve tensors
    workspace_inplace: bool = True,
    pack_public: bool = True,
    _connectivity_payload: dict[str, Any] | None = None,
    _compile_region_names: tuple[str, ...] | frozenset[str] | None = None,
    _compile_mode: str = "reduce-overhead",
    _compile_backend: str = "inductor",
    _numeric_plan: DiagnosticSolveNumericPlan | None = None,
) -> dict:
    """
    Compute diagnostic variables from prognostic state.

    This routine computes several diagnostic fields used in the dynamics:
    1. h_edge: Height interpolated to edge midpoints
    2. divergence: Horizontal divergence at cell centers
    3. vorticity: Relative vorticity at vertices (circulation / area)
    4. ke: Kinetic energy at cell centers with Hollingsworth correction
    5. pv_edge: Potential vorticity at edges (with APVM upwinding if enabled)

    Index Convention:
        Fortran uses 1-based indexing with arrays sized (N+1) where index N+1 is padding.
        This function expects Fortran-style arrays directly from checkpoint.
        cellStart, cellEnd, etc. are 1-based loop bounds from Fortran.

        IMPORTANT: Connectivity arrays (cellsOnEdge, edgesOnCell, etc.) contain 1-based
        Fortran indices. When using these to index into data arrays (h, u, etc.), we
        must subtract 1 to convert to 0-based Python indexing.

        Data array layout (after column-major to row-major conversion):
        - h[iCell-1, k] corresponds to Fortran h(k+1, iCell)
        - u[iEdge-1, k] corresponds to Fortran u(k+1, iEdge)
        where iCell, iEdge are 1-based Fortran indices.

    Args:
        nCells: Number of cells in mesh
        nEdges: Number of edges in mesh
        nVertices: Number of vertices in mesh
        nVertLevels: Number of vertical levels
        vertexDegree: Number of edges per vertex (typically 3)
        dt: Timestep in seconds (used for APVM upwinding)
        config_apvm_upwinding: APVM upwinding coefficient (0.0 to disable)
        cellStart, cellEnd: Cell loop bounds (1-based)
        vertexStart, vertexEnd: Vertex loop bounds (1-based)
        edgeStart, edgeEnd: Edge loop bounds (1-based)
        fVertex: Coriolis parameter at vertices
        fEdge: Coriolis parameter at edges (unused)
        invAreaTriangle: Inverse of triangle (dual cell) area
        invAreaCell: Inverse of cell area
        dvEdge: Edge length in normal direction
        dcEdge: Edge length in tangential direction
        invDvEdge: Inverse of dvEdge
        invDcEdge: Inverse of dcEdge
        weightsOnEdge: Reconstruction weights for tangential velocity
        kiteAreasOnVertex: Kite areas at each vertex
        cellsOnEdge: Cells adjacent to each edge (1-based indices)
        verticesOnEdge: Vertices of each edge (1-based indices)
        edgesOnCell: Edges of each cell (1-based indices)
        edgesOnCell_sign: Sign of edge relative to cell
        edgesOnVertex: Edges meeting at each vertex (1-based indices)
        edgesOnVertex_sign: Sign of edge relative to vertex
        verticesOnCell: Vertices of each cell (1-based indices)
        kiteForCell: Kite index at each cell vertex
        nEdgesOnCell: Number of edges for each cell
        nEdgesOnEdge: Number of edges for edge reconstruction
        edgesOnEdge: Edges used for tangential velocity reconstruction
        u: Normal velocity at edges
        h: Layer thickness at cells

    Returns:
        Dictionary containing:
            h_edge: (nEdges+1, nVertLevels) Height at edges
            divergence: (nCells+1, nVertLevels) Horizontal divergence
            vorticity: (nVertices+1, nVertLevels) Relative vorticity
            ke: (nCells+1, nVertLevels) Kinetic energy
            pv_edge: (nEdges+1, nVertLevels) Potential vorticity at edges
    """
    require_consistent_device(
        "atm_compute_solve_diagnostics_work",
        {
            "u": u,
            "h": h,
            "fVertex": fVertex,
            "invAreaTriangle": invAreaTriangle,
            "invAreaCell": invAreaCell,
            "dvEdge": dvEdge,
            "dcEdge": dcEdge,
            "weightsOnEdge": weightsOnEdge,
        },
    )
    device = u.device if isinstance(u, torch.Tensor) else torch.device("cpu")

    compile_region_names = (
        frozenset(_compile_region_names or ())
        & _DIAGNOSTIC_SOLVE_REGION_COMPILE_TARGETS
    )
    autodiff_active = bool(
        is_grad_mode()
        or getattr(u, "requires_grad", False)
        or getattr(h, "requires_grad", False)
        or _diagnostic_solve_tensor_has_forward_ad(u)
        or _diagnostic_solve_tensor_has_forward_ad(h)
    )
    compiling_region = (
        "diagnostic_solve_solve_diagnostics" in compile_region_names
        and not autodiff_active
    )

    ws_diagnostic_solve = getattr(workspace, "diagnostic_solve", None)
    workspace_inplace = bool(
        pack_public
        and workspace_inplace
        and ws_diagnostic_solve is not None
        and not is_grad_mode()
        and not compiling_region
    )

    if _connectivity_payload is None:
        conn = _diagnostic_solve_connectivity_payload(
            device=device,
            nCells=nCells,
            nEdges=nEdges,
            nVertices=nVertices,
            vertexDegree=vertexDegree,
            cellStart=cellStart,
            cellEnd=cellEnd,
            vertexStart=vertexStart,
            vertexEnd=vertexEnd,
            edgeStart=edgeStart,
            edgeEnd=edgeEnd,
            dvEdge=dvEdge,
            dcEdge=dcEdge,
            weightsOnEdge=weightsOnEdge,
            kiteAreasOnVertex=kiteAreasOnVertex,
            cellsOnEdge=cellsOnEdge,
            verticesOnEdge=verticesOnEdge,
            edgesOnCell=edgesOnCell,
            edgesOnCell_sign=edgesOnCell_sign,
            edgesOnVertex=edgesOnVertex,
            edgesOnVertex_sign=edgesOnVertex_sign,
            verticesOnCell=verticesOnCell,
            kiteForCell=kiteForCell,
            nEdgesOnCell=nEdgesOnCell,
            nEdgesOnEdge=nEdgesOnEdge,
            edgesOnEdge=edgesOnEdge,
        )
    else:
        conn = _connectivity_payload

    if _numeric_plan is None:
        numeric_plan = build_diagnostic_solve_numeric_plan(
            connectivity_payload=conn,
            nCells=nCells,
            nEdges=nEdges,
            nVertices=nVertices,
            nVertLevels=nVertLevels,
            dt=dt,
            config_apvm_upwinding=config_apvm_upwinding,
            fVertex=fVertex,
            invAreaTriangle=invAreaTriangle,
            invAreaCell=invAreaCell,
            invDvEdge=invDvEdge,
            invDcEdge=invDcEdge,
            rk_step=rk_step,
            stage_key=(cellStart, cellEnd, vertexStart, vertexEnd, edgeStart, edgeEnd),
        )
    else:
        numeric_plan = _numeric_plan

    numeric_callable = numeric_plan.numeric_callable
    if compiling_region:
        from mpas_pytorch.dynamics.compile_utils import maybe_compile_kernel

        numeric_plan = compile_diagnostic_solve_numeric_plan(
            numeric_plan,
            compile_kernel=maybe_compile_kernel,
            mode=_compile_mode,
            backend=_compile_backend,
        )
        numeric_callable = numeric_plan.numeric_callable
    names = ("h_edge", "v", "divergence", "vorticity", "ke", "pv_edge")
    numeric_outputs = numeric_callable(u, h, v_in)
    active_outputs = dict(zip(names, numeric_outputs))
    boundary_outputs = (
        pack_diagnostic_solve_diagnostics_public(
            active_outputs,
            nCells=nCells,
            nEdges=nEdges,
            nVertices=nVertices,
            v_padding_source=v_in
            if rk_step is not None and int(rk_step) != 3
            else None,
        )
        if pack_public
        else active_outputs
    )
    outputs: dict[str, Any] = {}
    for name in names:
        values = boundary_outputs[name]
        workspace_tensor = (
            getattr(ws_diagnostic_solve, name, None)
            if ws_diagnostic_solve is not None
            else None
        )
        if workspace_inplace and workspace_tensor is not None:
            workspace_tensor.copy_(values)
            outputs[name] = workspace_tensor
        else:
            outputs[name] = values
    return outputs
