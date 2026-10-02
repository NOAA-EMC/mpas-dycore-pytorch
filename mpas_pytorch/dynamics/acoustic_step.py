"""
Acoustic mode solver (acoustic-step).

Solves the acoustic (vertically integrated) equations for the fast modes.
This is the innermost loop of split-explicit time integration.


Subroutine: atm_advance_acoustic_step_work
"""

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, cast

import torch

from mpas_pytorch.autodiff.tridiagonal_scan import (
    canonical_tridiag_scan_shape_keys,
    normalize_tridiag_scan_policy,
    tridiag_scan_policy,
)
from mpas_pytorch.autodiff.tridiagonal_solve import TridiagonalSolve
from mpas_pytorch.constants import DTYPE, GRAVITY  # noqa: F401
from mpas_pytorch.dynamics.static_payloads import (
    StaticPairRelation,
    StaticRange,
    StaticSlotRelation,
)
from mpas_pytorch.dynamics.stencil_ops import (
    _accumulate_ordered_slot,
    _active_slot_count,
    _clone_for_grad,
    _compile_active,
    _gather_slot_batch,
    _replace_all_rows_cols,
    _replace_row_slice,
    _slot_batches,
    _slot_gather_batch,
    _static_connectivity_cache_allowed,
    _static_mask_all_true,
    _static_slot_all_true,
    _tensor_version_cache_key,
    gather_rows,
)
from mpas_pytorch.utils.checks import require_consistent_device

_MPAS_NO_INPLACE = os.environ.get("MPAS_NO_INPLACE", "0") == "1"

_ACOUSTIC_STEP_REGION_COMPILE_TARGETS = frozenset(
    {
        "acoustic_step_acoustic_flux_solve_region",
        "acoustic_step_horizontal_momentum_update",
    }
)

_ACOUSTIC_STEP_CONNECTIVITY_CACHE: dict[tuple[object, ...], dict[str, Any]] = {}
_ACOUSTIC_STEP_CONNECTIVITY_TENSOR_NAMES = (
    "cellsOnEdge",
    "edgesOnCell",
    "nEdgesOnCell",
    "edgesOnCell_sign",
    "specZoneMaskCell",
    "specZoneMaskEdge",
    "dvEdge",
    "invAreaCell",
)

_ACOUSTIC_STEP_ACOUSTIC_SOLVE_DYNAMIC_NAMES = (
    "cofrz",
    "rdzw",
    "fzm",
    "fzp",
    "ru_p",
    "theta_m",
    "rho_pp_active",
    "rtheta_pp_active",
    "rw_p_active",
    "ww_avg_active",
    "tend_rho_active",
    "tend_rt_active",
    "tend_rw_active",
    "coftz_active",
    "cofwz_active",
    "cofwr_active",
    "cofwt_active",
    "zz_active",
    "a_tri_active",
    "alpha_tri_active",
    "gamma_tri_active",
    "rho_zz_active",
    "rw_save_active",
    "rw_base_active",
    "dss_active",
    "w_active",
)


@dataclass(frozen=True)
class AcousticSolvePlan:
    """Immutable topology/scalar binding for the repeated acoustic-step solve region."""

    key: tuple[object, ...]
    static_argument_names: tuple[str, ...]
    dynamic_argument_names: tuple[str, ...]
    static_arguments: Mapping[str, Any]
    numeric_callable: Callable[
        ..., tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]

    def dynamic_values(self, call_arguments: Mapping[str, Any]) -> tuple[object, ...]:
        missing = [
            name for name in self.dynamic_argument_names if name not in call_arguments
        ]
        if missing:
            raise KeyError(
                f"acoustic-step numeric call is missing dynamic arguments: {missing}"
            )
        return tuple(call_arguments[name] for name in self.dynamic_argument_names)

    def with_numeric_callable(
        self,
        numeric_callable: Callable[
            ..., tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ],
    ) -> "AcousticSolvePlan":
        return replace(self, numeric_callable=numeric_callable)


def _acoustic_step_tensor_cache_key(tensor: torch.Tensor) -> tuple[object, ...] | None:
    return _tensor_version_cache_key(tensor)


def _acoustic_step_select_active_coef_rows(
    field: torch.Tensor | None,
    active_patch: torch.Tensor | None,
    *,
    active_cell_slice: slice,
    active_cells_are_slice: bool,
    active_cells: torch.Tensor,
    patch_cell_start: int | None,
    patch_cell_end: int | None,
) -> torch.Tensor:
    """Prefer vertical-implicit-coefficients compact coefficient rows when they match acoustic-step's active range."""
    if (
        active_patch is not None
        and active_cells_are_slice
        and isinstance(patch_cell_start, int)
        and isinstance(patch_cell_end, int)
        and patch_cell_start == active_cell_slice.start
        and patch_cell_end == active_cell_slice.stop
    ):
        expected_rows = int(active_cell_slice.stop - active_cell_slice.start)
        if active_patch.shape[0] != expected_rows:
            raise ValueError(
                "vertical-implicit-coefficients active coefficient patch row count mismatch: "
                f"got {active_patch.shape[0]}, expected {expected_rows}"
            )
        return active_patch
    if field is None:
        raise ValueError(
            "acoustic-step requires a public coefficient field when no matching vertical-implicit-coefficients "
            "active coefficient patch is available"
        )
    if active_cells_are_slice:
        return field[active_cell_slice]
    return gather_rows(field, active_cells)


_ACOUSTIC_STEP_PAYLOAD_KEY_ONLY_KWARGS = frozenset(
    {
        "specZoneMaskCellKnownAllZero",
        "specZoneMaskEdgeKnownAllZero",
    }
)


def _acoustic_step_spec_zone_mask_cache_key(
    tensor: torch.Tensor,
    *,
    known_all_zero: bool = False,
) -> tuple[object, ...] | None:
    if known_all_zero:
        return (
            "specZoneMask",
            "all_zero",
            str(tensor.device),
            str(tensor.dtype),
            tuple(tensor.shape),
        )
    if tensor.device.type == "cpu" and not _compile_active():
        is_all_zero = bool(torch.all(tensor == 0.0).detach().cpu())
        if is_all_zero:
            return (
                "specZoneMask",
                "all_zero",
                str(tensor.device),
                str(tensor.dtype),
                tuple(tensor.shape),
            )
    return _acoustic_step_tensor_cache_key(tensor)


def _acoustic_step_connectivity_payload_cache_key(
    **kwargs: object,
) -> tuple[object, ...] | None:
    """Return the mutation-sensitive cache key for a acoustic-step static payload."""
    tensors = tuple(
        kwargs.get(name) for name in _ACOUSTIC_STEP_CONNECTIVITY_TENSOR_NAMES
    )
    if not _static_connectivity_cache_allowed(tensors):
        return None

    tensor_keys: list[tuple[object, ...]] = []
    for name in _ACOUSTIC_STEP_CONNECTIVITY_TENSOR_NAMES:
        tensor = kwargs.get(name)
        if isinstance(tensor, torch.Tensor):
            if name in {"specZoneMaskCell", "specZoneMaskEdge"}:
                tensor_key = _acoustic_step_spec_zone_mask_cache_key(
                    tensor,
                    known_all_zero=bool(kwargs.get(f"{name}KnownAllZero", False)),
                )
            else:
                tensor_key = _acoustic_step_tensor_cache_key(tensor)
            if tensor_key is None:
                return None
            tensor_keys.append(tensor_key)

    return (
        kwargs["device"],
        kwargs["nCells"],
        kwargs["nEdges"],
        kwargs["nCellsSolve"],
        kwargs["maxEdges"],
        kwargs["edge_start"],
        kwargs["edge_end"],
        kwargs["cell_solve_start"],
        kwargs["cell_solve_end"],
        *tensor_keys,
    )


def _acoustic_step_static_all_true(
    mask: torch.Tensor, *, allow_cuda_sync: bool = False
) -> bool:
    """Return setup-time all-true metadata for static topology masks."""
    return bool(_static_mask_all_true(mask, allow_cuda_sync=allow_cuda_sync))


def _acoustic_step_slot_all_true(
    mask: torch.Tensor,
    *,
    allow_cuda_sync: bool = False,
) -> tuple[bool, ...] | None:
    """Return setup-time per-slot all-true metadata for static topology masks."""
    return _static_slot_all_true(mask, allow_cuda_sync=allow_cuda_sync)


def _acoustic_step_long_index(index: torch.Tensor) -> torch.Tensor:
    return index if index.dtype == torch.int64 else index.to(dtype=torch.int64)


def _build_acoustic_step_connectivity_payload(
    *,
    device: torch.device,
    nCells: int,
    nEdges: int,
    nCellsSolve: int,
    maxEdges: int,
    edge_start: int,
    edge_end: int,
    cell_solve_start: int,
    cell_solve_end: int,
    cellsOnEdge: torch.Tensor,
    edgesOnCell: torch.Tensor,
    nEdgesOnCell: torch.Tensor,
    edgesOnCell_sign: torch.Tensor,
    specZoneMaskCell: torch.Tensor,
    specZoneMaskEdge: torch.Tensor,
    dvEdge: torch.Tensor | None = None,
    invAreaCell: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Build reusable acoustic-step edge/cell topology for acoustic substeps."""
    edge_idx = torch.arange(edge_start, edge_end, device=device, dtype=torch.int64)
    cell1_1based = cellsOnEdge[edge_idx, 0]
    cell2_1based = cellsOnEdge[edge_idx, 1]
    edge_valid_cells = (
        (cell1_1based > 0)
        & (cell1_1based <= nCells)
        & (cell2_1based > 0)
        & (cell2_1based <= nCells)
    )
    edge_owned = (cell1_1based <= nCellsSolve) | (cell2_1based <= nCellsSolve)
    edge_valid = edge_valid_cells & edge_owned
    edge_valid_2d = edge_valid[:, None]
    edge_c1 = cell1_1based - 1
    edge_c2 = cell2_1based - 1
    edge_c1 = edge_c1.masked_fill(~edge_valid_cells, 0)
    edge_c2 = edge_c2.masked_fill(~edge_valid_cells, 0)
    edge_c1 = _acoustic_step_long_index(edge_c1)
    edge_c2 = _acoustic_step_long_index(edge_c2)
    cell_solve_idx = torch.arange(
        cell_solve_start, cell_solve_end, device=device, dtype=torch.int64
    )
    solve_active_mask = specZoneMaskCell[cell_solve_idx] == 0.0
    solve_active_all = _acoustic_step_static_all_true(
        solve_active_mask, allow_cuda_sync=True
    )
    full_solve_compile_path = _compile_active() or not solve_active_all
    if full_solve_compile_path:
        active_cells = cell_solve_idx
        active_cells_are_slice = True
    else:
        active_cells = cell_solve_idx[solve_active_mask]
        active_cells_are_slice = _acoustic_step_static_all_true(
            solve_active_mask,
            allow_cuda_sync=True,
        )

    active_slot_count = _active_slot_count(nEdgesOnCell[active_cells], maxEdges, device)
    edge_slots = torch.arange(active_slot_count, device=device, dtype=torch.int64)
    active_edge_1based = edgesOnCell[active_cells, :active_slot_count]
    active_edge0 = active_edge_1based - 1
    active_slot_valid = edge_slots[None, :] < nEdgesOnCell[active_cells][:, None]
    active_edge_idx_valid = active_edge0 >= 0
    active_connected = active_slot_valid & active_edge_idx_valid
    active_safe_edge_idx = _acoustic_step_long_index(
        torch.where(active_connected, active_edge0, torch.zeros_like(active_edge0))
    )

    edge_cells = gather_rows(cellsOnEdge, active_safe_edge_idx)
    edge_cell1 = edge_cells[:, :, 0]
    edge_cell2 = edge_cells[:, :, 1]
    edge_neighbors_valid = (
        (edge_cell1 > 0)
        & (edge_cell1 <= nCells)
        & (edge_cell2 > 0)
        & (edge_cell2 <= nCells)
    )
    active_valid = active_connected & edge_neighbors_valid
    if full_solve_compile_path:
        active_valid = active_valid & solve_active_mask[:, None]
    active_c1 = edge_cell1 - 1
    active_c2 = edge_cell2 - 1
    active_c1 = active_c1.masked_fill(~edge_neighbors_valid, 0)
    active_c2 = active_c2.masked_fill(~edge_neighbors_valid, 0)
    active_c1 = _acoustic_step_long_index(active_c1)
    active_c2 = _acoustic_step_long_index(active_c2)
    edge_valid_all = _acoustic_step_static_all_true(edge_valid, allow_cuda_sync=True)
    edge_spec_zone_all_zero = _acoustic_step_static_all_true(
        specZoneMaskEdge[edge_idx] == 0.0,
        allow_cuda_sync=True,
    )
    active_valid_all_by_slot = _acoustic_step_slot_all_true(
        active_valid,
        allow_cuda_sync=True,
    )
    active_edge_sign = edgesOnCell_sign[active_cells, :active_slot_count]
    active_flux_weight = None
    active_inv_area = None
    if dvEdge is not None and invAreaCell is not None:
        active_flux_weight = active_edge_sign * gather_rows(
            dvEdge, active_safe_edge_idx
        )
        active_inv_area = gather_rows(invAreaCell, active_cells)
    active_cell_range = StaticRange(
        "acoustic_step.active_cells", cell_solve_start, cell_solve_end
    )
    edge_cell_relation = StaticPairRelation(
        "acoustic_step.edge_cells",
        first=edge_c1,
        second=edge_c2,
        valid=edge_valid,
        valid_2d=edge_valid_2d,
        valid_all=edge_valid_all,
    )
    active_cell_edge_relation = StaticSlotRelation(
        "acoustic_step.active_cell_edges",
        indices=active_safe_edge_idx,
        valid=active_valid,
        sign=active_edge_sign,
        weights=active_flux_weight,
        valid_all_by_slot=active_valid_all_by_slot,
        valid_weight=active_inv_area,
    )
    active_edge_cell_relation = StaticPairRelation(
        "acoustic_step.active_edge_cells",
        first=active_c1,
        second=active_c2,
        valid=active_valid,
        valid_all=None,
    )
    active_edge_cell_pair = torch.stack((active_c1, active_c2), dim=-1)
    return {
        "edge_valid_2d": edge_valid_2d,
        "edge_valid_all": edge_valid_all,
        "edge_spec_zone_all_zero": edge_spec_zone_all_zero,
        "edge_c1": edge_c1,
        "edge_c2": edge_c2,
        "active_cell_slice": slice(cell_solve_start, cell_solve_end),
        "active_cell_range": active_cell_range,
        "active_cells_are_slice": active_cells_are_slice,
        "active_cells": active_cells,
        "full_solve_compile_path": full_solve_compile_path,
        "solve_active_mask": solve_active_mask,
        "active_safe_edge_idx": active_safe_edge_idx,
        "active_valid": active_valid,
        "active_valid_all_by_slot": active_valid_all_by_slot,
        "active_edge_sign": active_edge_sign,
        "active_flux_weight": active_flux_weight,
        "active_inv_area": active_inv_area,
        "active_c1": active_c1,
        "active_c2": active_c2,
        "edge_cell_relation": edge_cell_relation,
        "active_cell_edge_relation": active_cell_edge_relation,
        "active_edge_cell_relation": active_edge_cell_relation,
        "active_edge_cell_pair": active_edge_cell_pair,
    }


def _acoustic_step_connectivity_payload(**kwargs: object) -> dict[str, Any]:
    """Return cached acoustic-step acoustic topology when topology is static."""
    key = _acoustic_step_connectivity_payload_cache_key(**kwargs)
    build_kwargs = {
        name: value
        for name, value in kwargs.items()
        if name not in _ACOUSTIC_STEP_PAYLOAD_KEY_ONLY_KWARGS
    }
    if key is None:
        return _build_acoustic_step_connectivity_payload(**build_kwargs)  # type: ignore[arg-type]
    cached = _ACOUSTIC_STEP_CONNECTIVITY_CACHE.get(key)
    if cached is not None:
        return cached
    payload = _build_acoustic_step_connectivity_payload(**build_kwargs)  # type: ignore[arg-type]
    if len(_ACOUSTIC_STEP_CONNECTIVITY_CACHE) >= 4:
        _ACOUSTIC_STEP_CONNECTIVITY_CACHE.pop(
            next(iter(_ACOUSTIC_STEP_CONNECTIVITY_CACHE))
        )
    _ACOUSTIC_STEP_CONNECTIVITY_CACHE[key] = payload
    return payload


def _acoustic_step_prebound_edge_cell_pair(
    edge_cell_relation: StaticPairRelation,
) -> torch.Tensor:
    """Reuse the pair tensor owned by the cached acoustic-step connectivity payload."""
    for payload in reversed(tuple(_ACOUSTIC_STEP_CONNECTIVITY_CACHE.values())):
        if payload.get("active_edge_cell_relation") is edge_cell_relation:
            pair = payload.get("active_edge_cell_pair")
            if isinstance(pair, torch.Tensor):
                return pair
    return torch.stack((edge_cell_relation.first, edge_cell_relation.second), dim=-1)


def _sum_acoustic_cell_flux_slots_numeric(
    *,
    dts: float,
    ru_p: torch.Tensor,
    theta_m: torch.Tensor,
    edge_indices: torch.Tensor,
    edge_valid: torch.Tensor,
    edge_valid_all_by_slot: tuple[bool, ...],
    edge_flux_weight: torch.Tensor,
    active_inv_area: torch.Tensor,
    edge_cell_pair: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ordered acoustic-step flux algebra over plan-owned raw topology tensors."""
    n_active = edge_indices.shape[0]
    rs_total: torch.Tensor | None = None
    ts_total: torch.Tensor | None = None
    inv_area = active_inv_area.unsqueeze(1)
    for batch_start, batch_stop in _slot_batches(
        edge_indices.shape[1],
        _slot_gather_batch(ru_p, edge_indices.shape[1]),
    ):
        ru_batch = _gather_slot_batch(ru_p, edge_indices, batch_start, batch_stop)
        metric_batch = edge_flux_weight[:, batch_start:batch_stop]
        theta_pair_batch = _gather_slot_batch(
            theta_m,
            edge_cell_pair,
            batch_start,
            batch_stop,
        )
        for local_slot in range(batch_stop - batch_start):
            slot = batch_start + local_slot
            valid_all = (
                edge_valid_all_by_slot is not None and edge_valid_all_by_slot[slot]
            )
            flux = (
                dts
                * metric_batch[:, local_slot].unsqueeze(1)
                * ru_batch[:, local_slot, :]
                * inv_area
            )
            theta_edge = 0.5 * (
                theta_pair_batch[:, local_slot, 1, :]
                + theta_pair_batch[:, local_slot, 0, :]
            )
            if not valid_all:
                valid = edge_valid[:, slot].unsqueeze(1).to(dtype=ru_p.dtype)
                flux = flux * valid
                theta_edge = theta_edge * valid
            rs_contrib = -flux
            ts_contrib = -(flux * theta_edge)
            rs_total = _accumulate_ordered_slot(rs_total, rs_contrib)
            ts_total = _accumulate_ordered_slot(ts_total, ts_contrib)
    if rs_total is None or ts_total is None:
        shape = (n_active, ru_p.shape[1])
        return ru_p.new_zeros(shape), ru_p.new_zeros(shape)
    return rs_total, ts_total


def _sum_acoustic_cell_flux_slots(
    *,
    dts: float,
    ru_p: torch.Tensor,
    theta_m: torch.Tensor,
    cell_edge_relation: StaticSlotRelation,
    edge_cell_relation: StaticPairRelation,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adapt typed topology once around the ordered numeric flux body."""
    edge_indices = cell_edge_relation.indices
    edge_valid = cell_edge_relation.valid
    if edge_valid is None:
        edge_valid = torch.ones_like(edge_indices, dtype=torch.bool)
    edge_valid_all_by_slot = cell_edge_relation.valid_all_by_slot
    if edge_valid_all_by_slot is None:
        edge_valid_all_by_slot = _acoustic_step_slot_all_true(
            edge_valid,
            allow_cuda_sync=True,
        )
    edge_flux_weight = cell_edge_relation.weights
    active_inv_area = cell_edge_relation.valid_weight
    if edge_flux_weight is None or active_inv_area is None:
        raise ValueError(
            "acoustic-step acoustic cell-flux relation requires metric weights"
        )
    if edge_valid_all_by_slot is None:
        raise ValueError(
            "acoustic-step acoustic cell-flux relation lacks slot-validity metadata"
        )
    return _sum_acoustic_cell_flux_slots_numeric(
        dts=dts,
        ru_p=ru_p,
        theta_m=theta_m,
        edge_indices=edge_indices,
        edge_valid=edge_valid,
        edge_valid_all_by_slot=edge_valid_all_by_slot,
        edge_flux_weight=edge_flux_weight,
        active_inv_area=active_inv_area,
        edge_cell_pair=torch.stack(
            (edge_cell_relation.first, edge_cell_relation.second), dim=-1
        ),
    )


def _sum_acoustic_cell_flux_region(
    *,
    dts: float,
    ru_p: torch.Tensor,
    theta_m: torch.Tensor,
    cell_edge_relation: StaticSlotRelation,
    edge_cell_relation: StaticPairRelation,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate with one ordered implementation for eager, compile, and AD."""
    return _sum_acoustic_cell_flux_slots(
        dts=dts,
        ru_p=ru_p,
        theta_m=theta_m,
        cell_edge_relation=cell_edge_relation,
        edge_cell_relation=edge_cell_relation,
    )


def _acoustic_step_horizontal_momentum_update(
    *,
    small_step: int,
    edge_cell_relation: StaticPairRelation,
    rtheta_pp: torch.Tensor,
    zz: torch.Tensor,
    exner: torch.Tensor,
    rho_pp: torch.Tensor,
    ru_p_edge: torch.Tensor,
    ru_avg_edge: torch.Tensor,
    tend_ru_edge: torch.Tensor,
    inv_dc_edge: torch.Tensor,
    spec_zone_mask_edge: torch.Tensor,
    cqu_edge: torch.Tensor,
    zxu_edge: torch.Tensor,
    dts: float,
    c2: float,
    gravity: float,
    spec_zone_mask_edge_all_zero: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the acoustic-step pressure-gradient ru_p/ruAvg edge patch in one region."""
    edge_valid_2d = edge_cell_relation.valid_2d
    if edge_valid_2d is None:
        edge_valid = edge_cell_relation.valid
        if edge_valid is None:
            edge_valid_2d = torch.ones(
                (ru_p_edge.shape[0], 1),
                dtype=torch.bool,
                device=ru_p_edge.device,
            )
        else:
            edge_valid_2d = edge_valid[:, None]
    edge_valid_all = (
        bool(edge_cell_relation.valid_all)
        if edge_cell_relation.valid_all is not None
        else False
    )
    if small_step != 1:
        rtheta_c1, rtheta_c2 = edge_cell_relation.gather_pair(rtheta_pp)
        zz_c1, zz_c2 = edge_cell_relation.gather_pair(zz)
        exner_c1, exner_c2 = edge_cell_relation.gather_pair(exner)
        rho_c1, rho_c2 = edge_cell_relation.gather_pair(rho_pp)
        pgrad = ((rtheta_c2 - rtheta_c1) * inv_dc_edge.unsqueeze(1)) / (
            0.5 * (zz_c2 + zz_c1)
        )
        pgrad = cqu_edge * 0.5 * c2 * (exner_c1 + exner_c2) * pgrad
        pgrad = pgrad + 0.5 * zxu_edge * gravity * (rho_c1 + rho_c2)

        if spec_zone_mask_edge_all_zero:
            ru_p_updated = ru_p_edge + dts * (tend_ru_edge - pgrad)
        else:
            ru_p_updated = ru_p_edge + dts * (
                tend_ru_edge - (1.0 - spec_zone_mask_edge[:, None]) * pgrad
            )
        if edge_valid_all:
            ru_p_patch = ru_p_updated
        else:
            ru_p_patch = torch.where(edge_valid_2d, ru_p_updated, ru_p_edge)
    else:
        ru_p_init = dts * tend_ru_edge
        if edge_valid_all:
            ru_p_patch = ru_p_init
            ru_avg_patch = ru_p_patch
        else:
            ru_p_patch = torch.where(edge_valid_2d, ru_p_init, ru_p_edge)
            ru_avg_patch = torch.where(edge_valid_2d, ru_p_patch, ru_avg_edge)
        return ru_p_patch, ru_avg_patch

    ru_avg_updated = ru_avg_edge + ru_p_patch
    if edge_valid_all:
        ru_avg_patch = ru_avg_updated
    else:
        ru_avg_patch = torch.where(edge_valid_2d, ru_avg_updated, ru_avg_edge)
    return ru_p_patch, ru_avg_patch


def _acoustic_step_replace_interior_rows(
    rows: torch.Tensor,
    values: torch.Tensor,
    n_vert_levels: int,
    *,
    reuse_rows: bool = False,
) -> torch.Tensor:
    functional = torch.is_grad_enabled() or _compile_active()
    if functional:
        # The functional pack owns its result. Pre-cloning ``rows`` would keep
        # both full tensors live and then copy the clone into the packed output.
        return _replace_all_rows_cols(rows, 1, n_vert_levels, values)
    out = rows if reuse_rows else rows.clone()
    return _replace_all_rows_cols(out, 1, n_vert_levels, values)


def _acoustic_step_zero_row_patch_like(
    field: torch.Tensor, row_start: int, row_end: int
) -> torch.Tensor:
    """Allocate a zero row patch without reading the source rows."""
    return field.new_zeros((row_end - row_start, *field.shape[1:]))


def _acoustic_step_assemble_active_system(
    *,
    nVertLevels: int,
    dts: float,
    resm: float,
    epssm: float,
    cofrz: torch.Tensor,
    rdzw: torch.Tensor,
    rs_flux: torch.Tensor,
    ts_flux: torch.Tensor,
    rho_pp_active: torch.Tensor,
    rtheta_pp_active: torch.Tensor,
    rw_p_active: torch.Tensor,
    ww_avg_active: torch.Tensor,
    tend_rho_active: torch.Tensor,
    tend_rt_active: torch.Tensor,
    tend_rw_active: torch.Tensor,
    coftz_active: torch.Tensor,
    cofwz_active: torch.Tensor,
    cofwr_active: torch.Tensor,
    cofwt_active: torch.Tensor,
    zz_active: torch.Tensor,
    full_solve_compile_path: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble the active acoustic-step factorized-system RHS without solving it."""
    rs_active = (
        rho_pp_active
        + dts * tend_rho_active
        + rs_flux
        - cofrz[None, :] * resm * (rw_p_active[:, 1:] - rw_p_active[:, :nVertLevels])
    )
    ts_active = (
        rtheta_pp_active
        + dts * tend_rt_active
        + ts_flux
        - resm
        * rdzw[None, :]
        * (
            coftz_active[:, 1:] * rw_p_active[:, 1:]
            - coftz_active[:, :nVertLevels] * rw_p_active[:, :nVertLevels]
        )
    )
    rw_rows = rw_p_active
    ww_rows = ww_avg_active
    if nVertLevels > 1:
        ww_active = (
            ww_rows[:, 1:nVertLevels] + 0.5 * (1.0 - epssm) * rw_rows[:, 1:nVertLevels]
        )
        rw_active = (
            rw_rows[:, 1:nVertLevels]
            + dts * tend_rw_active[:, 1:nVertLevels]
            - cofwz_active[:, 1:nVertLevels]
            * (
                (
                    zz_active[:, 1:nVertLevels] * ts_active[:, 1:nVertLevels]
                    - zz_active[:, : nVertLevels - 1] * ts_active[:, : nVertLevels - 1]
                )
                + resm
                * (
                    zz_active[:, 1:nVertLevels] * rtheta_pp_active[:, 1:nVertLevels]
                    - zz_active[:, : nVertLevels - 1]
                    * rtheta_pp_active[:, : nVertLevels - 1]
                )
            )
            - cofwr_active[:, 1:nVertLevels]
            * (
                (rs_active[:, 1:nVertLevels] + rs_active[:, : nVertLevels - 1])
                + resm
                * (
                    rho_pp_active[:, 1:nVertLevels]
                    + rho_pp_active[:, : nVertLevels - 1]
                )
            )
            + cofwt_active[:, 1:nVertLevels]
            * (ts_active[:, 1:nVertLevels] + resm * rtheta_pp_active[:, 1:nVertLevels])
            + cofwt_active[:, : nVertLevels - 1]
            * (
                ts_active[:, : nVertLevels - 1]
                + resm * rtheta_pp_active[:, : nVertLevels - 1]
            )
        )
        reuse_owned_rows = not full_solve_compile_path
        ww_rows = _acoustic_step_replace_interior_rows(
            ww_rows,
            ww_active,
            nVertLevels,
            reuse_rows=reuse_owned_rows,
        )
        rw_rows = _acoustic_step_replace_interior_rows(
            rw_rows,
            rw_active,
            nVertLevels,
            reuse_rows=reuse_owned_rows,
        )
    return rs_active, ts_active, rw_rows, ww_rows


def _acoustic_step_acoustic_solve_region(
    *,
    nVertLevels: int,
    dts: float,
    resm: float,
    epssm: float,
    cofrz: torch.Tensor,
    rdzw: torch.Tensor,
    fzm: torch.Tensor,
    fzp: torch.Tensor,
    rs_flux: torch.Tensor,
    ts_flux: torch.Tensor,
    rho_pp_active: torch.Tensor,
    rtheta_pp_active: torch.Tensor,
    rw_p_active: torch.Tensor,
    ww_avg_active: torch.Tensor,
    tend_rho_active: torch.Tensor,
    tend_rt_active: torch.Tensor,
    tend_rw_active: torch.Tensor,
    coftz_active: torch.Tensor,
    cofwz_active: torch.Tensor,
    cofwr_active: torch.Tensor,
    cofwt_active: torch.Tensor,
    zz_active: torch.Tensor,
    a_tri_active: torch.Tensor,
    alpha_tri_active: torch.Tensor,
    gamma_tri_active: torch.Tensor,
    rho_zz_active: torch.Tensor,
    rw_save_active: torch.Tensor,
    rw_base_active: torch.Tensor,
    dss_active: torch.Tensor,
    w_active: torch.Tensor,
    full_solve_compile_path: bool,
    solve_active_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the acoustic-step active-row acoustic solve around the policy-selected tridiagonal solve."""
    rs_active, ts_active, rw_rows, ww_rows = _acoustic_step_assemble_active_system(
        nVertLevels=nVertLevels,
        dts=dts,
        resm=resm,
        epssm=epssm,
        cofrz=cofrz,
        rdzw=rdzw,
        rs_flux=rs_flux,
        ts_flux=ts_flux,
        rho_pp_active=rho_pp_active,
        rtheta_pp_active=rtheta_pp_active,
        rw_p_active=rw_p_active,
        ww_avg_active=ww_avg_active,
        tend_rho_active=tend_rho_active,
        tend_rt_active=tend_rt_active,
        tend_rw_active=tend_rw_active,
        coftz_active=coftz_active,
        cofwz_active=cofwz_active,
        cofwr_active=cofwr_active,
        cofwt_active=cofwt_active,
        zz_active=zz_active,
        full_solve_compile_path=full_solve_compile_path,
    )

    # Default execution keeps Thomas k-ordering. Scan recurrences are opt-in until
    # the drift-proof/budget chain authorizes them for production.
    rw_rows = TridiagonalSolve.apply(
        a_tri_active, alpha_tri_active, gamma_tri_active, rw_rows
    )

    if nVertLevels > 1:
        zz_interp = (
            fzm[1:nVertLevels][None, :] * zz_active[:, 1:nVertLevels]
            + fzp[1:nVertLevels][None, :] * zz_active[:, : nVertLevels - 1]
        )
        rho_zz_interp = (
            fzm[1:nVertLevels][None, :] * rho_zz_active[:, 1:nVertLevels]
            + fzp[1:nVertLevels][None, :] * rho_zz_active[:, : nVertLevels - 1]
        )
        rw_delta = rw_save_active[:, 1:nVertLevels] - rw_base_active[:, 1:nVertLevels]
        rw_rayleigh = (
            rw_rows[:, 1:nVertLevels]
            + rw_delta
            - dts
            * dss_active[:, 1:nVertLevels]
            * zz_interp
            * rho_zz_interp
            * w_active[:, 1:nVertLevels]
        ) / (1.0 + dts * dss_active[:, 1:nVertLevels]) - rw_delta
        ww_post = ww_rows[:, 1:nVertLevels] + 0.5 * (1.0 + epssm) * rw_rayleigh
        reuse_owned_rows = not full_solve_compile_path
        rw_rows = _acoustic_step_replace_interior_rows(
            rw_rows,
            rw_rayleigh,
            nVertLevels,
            reuse_rows=reuse_owned_rows,
        )
        ww_rows = _acoustic_step_replace_interior_rows(
            ww_rows,
            ww_post,
            nVertLevels,
            reuse_rows=reuse_owned_rows,
        )

    rho_patch = rs_active - cofrz[None, :] * (rw_rows[:, 1:] - rw_rows[:, :nVertLevels])
    rtheta_patch = ts_active - rdzw[None, :] * (
        coftz_active[:, 1:] * rw_rows[:, 1:]
        - coftz_active[:, :nVertLevels] * rw_rows[:, :nVertLevels]
    )
    if full_solve_compile_path:
        active_mask_2d = solve_active_mask[:, None].to(dtype=rho_patch.dtype)
        rho_spec = rho_pp_active + dts * tend_rho_active
        rtheta_spec = rtheta_pp_active + dts * tend_rt_active
        rw_spec = rw_p_active[:, :nVertLevels] + dts * tend_rw_active[:, :nVertLevels]
        ww_spec = ww_avg_active[:, :nVertLevels] + 0.5 * (1.0 + epssm) * rw_spec
        rw_spec_rows = torch.cat((rw_spec, rw_p_active[:, nVertLevels:]), dim=1)
        ww_spec_rows = torch.cat((ww_spec, ww_avg_active[:, nVertLevels:]), dim=1)
        rho_patch = rho_spec + active_mask_2d * (rho_patch - rho_spec)
        rtheta_patch = rtheta_spec + active_mask_2d * (rtheta_patch - rtheta_spec)
        rw_rows = rw_spec_rows + active_mask_2d * (rw_rows - rw_spec_rows)
        ww_rows = ww_spec_rows + active_mask_2d * (ww_rows - ww_spec_rows)
    return rw_rows, ww_rows, rho_patch, rtheta_patch


def _acoustic_step_acoustic_flux_solve_numeric(
    *,
    nVertLevels: int,
    dts: float,
    resm: float,
    epssm: float,
    cofrz: torch.Tensor,
    rdzw: torch.Tensor,
    fzm: torch.Tensor,
    fzp: torch.Tensor,
    edge_indices: torch.Tensor,
    edge_valid: torch.Tensor,
    edge_valid_all_by_slot: tuple[bool, ...],
    edge_flux_weight: torch.Tensor,
    active_inv_area: torch.Tensor,
    edge_cell_pair: torch.Tensor,
    ru_p: torch.Tensor,
    theta_m: torch.Tensor,
    rho_pp_active: torch.Tensor,
    rtheta_pp_active: torch.Tensor,
    rw_p_active: torch.Tensor,
    ww_avg_active: torch.Tensor,
    tend_rho_active: torch.Tensor,
    tend_rt_active: torch.Tensor,
    tend_rw_active: torch.Tensor,
    coftz_active: torch.Tensor,
    cofwz_active: torch.Tensor,
    cofwr_active: torch.Tensor,
    cofwt_active: torch.Tensor,
    zz_active: torch.Tensor,
    a_tri_active: torch.Tensor,
    alpha_tri_active: torch.Tensor,
    gamma_tri_active: torch.Tensor,
    rho_zz_active: torch.Tensor,
    rw_save_active: torch.Tensor,
    rw_base_active: torch.Tensor,
    dss_active: torch.Tensor,
    w_active: torch.Tensor,
    full_solve_compile_path: bool,
    solve_active_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tensor-only acoustic-step flux, RHS, vertical solve, and post-solve update."""
    rs_flux, ts_flux = _sum_acoustic_cell_flux_slots_numeric(
        dts=dts,
        ru_p=ru_p,
        theta_m=theta_m,
        edge_indices=edge_indices,
        edge_valid=edge_valid,
        edge_valid_all_by_slot=edge_valid_all_by_slot,
        edge_flux_weight=edge_flux_weight,
        active_inv_area=active_inv_area,
        edge_cell_pair=edge_cell_pair,
    )
    return _acoustic_step_acoustic_solve_region(
        nVertLevels=nVertLevels,
        dts=dts,
        resm=resm,
        epssm=epssm,
        cofrz=cofrz,
        rdzw=rdzw,
        fzm=fzm,
        fzp=fzp,
        rs_flux=rs_flux,
        ts_flux=ts_flux,
        rho_pp_active=rho_pp_active,
        rtheta_pp_active=rtheta_pp_active,
        rw_p_active=rw_p_active,
        ww_avg_active=ww_avg_active,
        tend_rho_active=tend_rho_active,
        tend_rt_active=tend_rt_active,
        tend_rw_active=tend_rw_active,
        coftz_active=coftz_active,
        cofwz_active=cofwz_active,
        cofwr_active=cofwr_active,
        cofwt_active=cofwt_active,
        zz_active=zz_active,
        a_tri_active=a_tri_active,
        alpha_tri_active=alpha_tri_active,
        gamma_tri_active=gamma_tri_active,
        rho_zz_active=rho_zz_active,
        rw_save_active=rw_save_active,
        rw_base_active=rw_base_active,
        dss_active=dss_active,
        w_active=w_active,
        full_solve_compile_path=full_solve_compile_path,
        solve_active_mask=solve_active_mask,
    )


def _acoustic_step_acoustic_flux_solve_region(
    *,
    nVertLevels: int,
    dts: float,
    resm: float,
    epssm: float,
    cofrz: torch.Tensor,
    rdzw: torch.Tensor,
    fzm: torch.Tensor,
    fzp: torch.Tensor,
    cell_edge_relation: StaticSlotRelation,
    edge_cell_relation: StaticPairRelation,
    ru_p: torch.Tensor,
    theta_m: torch.Tensor,
    rho_pp_active: torch.Tensor,
    rtheta_pp_active: torch.Tensor,
    rw_p_active: torch.Tensor,
    ww_avg_active: torch.Tensor,
    tend_rho_active: torch.Tensor,
    tend_rt_active: torch.Tensor,
    tend_rw_active: torch.Tensor,
    coftz_active: torch.Tensor,
    cofwz_active: torch.Tensor,
    cofwr_active: torch.Tensor,
    cofwt_active: torch.Tensor,
    zz_active: torch.Tensor,
    a_tri_active: torch.Tensor,
    alpha_tri_active: torch.Tensor,
    gamma_tri_active: torch.Tensor,
    rho_zz_active: torch.Tensor,
    rw_save_active: torch.Tensor,
    rw_base_active: torch.Tensor,
    dss_active: torch.Tensor,
    w_active: torch.Tensor,
    full_solve_compile_path: bool,
    solve_active_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compatibility adapter from typed relations to the tensor-only region."""
    edge_indices = cell_edge_relation.indices
    edge_valid = cell_edge_relation.valid
    if edge_valid is None:
        edge_valid = torch.ones_like(edge_indices, dtype=torch.bool)
    edge_valid_all_by_slot = cell_edge_relation.valid_all_by_slot
    if edge_valid_all_by_slot is None:
        edge_valid_all_by_slot = _acoustic_step_slot_all_true(
            edge_valid, allow_cuda_sync=True
        )
    if edge_valid_all_by_slot is None:
        # Conservative metadata-free adapter used by direct/synthetic callers:
        # apply the validity mask for every slot rather than synchronizing the
        # device to discover all-true columns inside a repeated call.
        edge_valid_all_by_slot = (False,) * edge_indices.shape[1]
    edge_flux_weight = cell_edge_relation.weights
    active_inv_area = cell_edge_relation.valid_weight
    if edge_flux_weight is None or active_inv_area is None:
        raise ValueError(
            "acoustic-step acoustic cell-flux relation requires metric weights"
        )
    return _acoustic_step_acoustic_flux_solve_numeric(
        nVertLevels=nVertLevels,
        dts=dts,
        resm=resm,
        epssm=epssm,
        cofrz=cofrz,
        rdzw=rdzw,
        fzm=fzm,
        fzp=fzp,
        edge_indices=edge_indices,
        edge_valid=edge_valid,
        edge_valid_all_by_slot=edge_valid_all_by_slot,
        edge_flux_weight=edge_flux_weight,
        active_inv_area=active_inv_area,
        edge_cell_pair=torch.stack(
            (edge_cell_relation.first, edge_cell_relation.second), dim=-1
        ),
        ru_p=ru_p,
        theta_m=theta_m,
        rho_pp_active=rho_pp_active,
        rtheta_pp_active=rtheta_pp_active,
        rw_p_active=rw_p_active,
        ww_avg_active=ww_avg_active,
        tend_rho_active=tend_rho_active,
        tend_rt_active=tend_rt_active,
        tend_rw_active=tend_rw_active,
        coftz_active=coftz_active,
        cofwz_active=cofwz_active,
        cofwr_active=cofwr_active,
        cofwt_active=cofwt_active,
        zz_active=zz_active,
        a_tri_active=a_tri_active,
        alpha_tri_active=alpha_tri_active,
        gamma_tri_active=gamma_tri_active,
        rho_zz_active=rho_zz_active,
        rw_save_active=rw_save_active,
        rw_base_active=rw_base_active,
        dss_active=dss_active,
        w_active=w_active,
        full_solve_compile_path=full_solve_compile_path,
        solve_active_mask=solve_active_mask,
    )


def _acoustic_step_plan_tensor_fingerprint(tensor: torch.Tensor) -> tuple[object, ...]:
    version_key = _tensor_version_cache_key(tensor)
    if version_key is not None:
        return version_key
    return (
        "tensor_identity",
        id(tensor),
        tuple(tensor.shape),
        str(tensor.dtype),
        str(tensor.device),
    )


def build_acoustic_step_acoustic_solve_plan(
    *,
    nVertLevels: int,
    dts: float,
    resm: float,
    epssm: float,
    cell_edge_relation: StaticSlotRelation,
    edge_cell_relation: StaticPairRelation,
    full_solve_compile_path: bool,
    solve_active_mask: torch.Tensor,
    stage_key: tuple[object, ...] = (),
) -> AcousticSolvePlan:
    """Bind acoustic-step topology once and expose an evolving-tensor-only callable."""
    edge_indices = cell_edge_relation.indices
    edge_valid = cell_edge_relation.valid
    if edge_valid is None:
        edge_valid = torch.ones_like(edge_indices, dtype=torch.bool)
    edge_valid_all_by_slot = cell_edge_relation.valid_all_by_slot
    if edge_valid_all_by_slot is None:
        edge_valid_all_by_slot = _acoustic_step_slot_all_true(
            edge_valid, allow_cuda_sync=True
        )
    edge_flux_weight = cell_edge_relation.weights
    active_inv_area = cell_edge_relation.valid_weight
    if edge_flux_weight is None or active_inv_area is None:
        raise ValueError("acoustic-step acoustic solve plan requires metric weights")
    if edge_valid_all_by_slot is None:
        raise ValueError(
            "acoustic-step acoustic solve plan requires fixed slot-validity metadata"
        )
    if edge_indices.ndim != 2:
        raise ValueError(
            "acoustic-step acoustic solve plan expects a two-dimensional slot relation"
        )
    slot_shape = tuple(edge_indices.shape)
    static_edge_cell_pair = _acoustic_step_prebound_edge_cell_pair(edge_cell_relation)
    for name, tensor in (
        ("edge_valid", edge_valid),
        ("edge_flux_weight", edge_flux_weight),
    ):
        if tuple(tensor.shape) != slot_shape:
            raise ValueError(
                f"acoustic-step acoustic solve plan {name} shape {tuple(tensor.shape)} "
                f"does not match slot shape {slot_shape}"
            )
    if active_inv_area.shape != edge_indices.shape[:1]:
        raise ValueError(
            "acoustic-step acoustic solve plan inv-area rows do not match active cells"
        )
    if solve_active_mask.shape != edge_indices.shape[:1]:
        raise ValueError(
            "acoustic-step acoustic solve plan active mask rows do not match active cells"
        )
    if len(edge_valid_all_by_slot) != edge_indices.shape[1]:
        raise ValueError(
            "acoustic-step acoustic solve plan slot-validity metadata is incomplete"
        )

    static_n_vert_levels = int(nVertLevels)
    static_dts = float(dts)
    static_resm = float(resm)
    static_epssm = float(epssm)
    static_edge_indices = edge_indices
    static_edge_valid = edge_valid
    static_edge_valid_all_by_slot = tuple(
        bool(value) for value in edge_valid_all_by_slot
    )
    static_edge_flux_weight = edge_flux_weight
    static_active_inv_area = active_inv_area
    if tuple(static_edge_cell_pair.shape) != (*slot_shape, 2):
        raise ValueError(
            "acoustic-step acoustic solve plan cell-pair shape does not match slots"
        )
    static_full_solve_compile_path = bool(full_solve_compile_path)
    static_solve_active_mask = solve_active_mask
    static_arguments = MappingProxyType(
        {
            "nVertLevels": static_n_vert_levels,
            "dts": static_dts,
            "resm": static_resm,
            "epssm": static_epssm,
            "edge_indices": static_edge_indices,
            "edge_valid": static_edge_valid,
            "edge_valid_all_by_slot": static_edge_valid_all_by_slot,
            "edge_flux_weight": static_edge_flux_weight,
            "active_inv_area": static_active_inv_area,
            "edge_cell_pair": static_edge_cell_pair,
            "full_solve_compile_path": static_full_solve_compile_path,
            "solve_active_mask": static_solve_active_mask,
        }
    )
    tensor_names = (
        "edge_indices",
        "edge_valid",
        "edge_flux_weight",
        "active_inv_area",
        "edge_cell_pair",
        "solve_active_mask",
    )
    key = (
        "AcousticSolvePlan",
        tuple(stage_key),
        static_n_vert_levels,
        static_dts,
        static_resm,
        static_epssm,
        static_edge_valid_all_by_slot,
        static_full_solve_compile_path,
        tuple(
            (
                name,
                _acoustic_step_plan_tensor_fingerprint(
                    cast(torch.Tensor, static_arguments[name])
                ),
            )
            for name in tensor_names
        ),
    )

    def _acoustic_step_bound_acoustic_solve(
        cofrz: torch.Tensor,
        rdzw: torch.Tensor,
        fzm: torch.Tensor,
        fzp: torch.Tensor,
        ru_p: torch.Tensor,
        theta_m: torch.Tensor,
        rho_pp_active: torch.Tensor,
        rtheta_pp_active: torch.Tensor,
        rw_p_active: torch.Tensor,
        ww_avg_active: torch.Tensor,
        tend_rho_active: torch.Tensor,
        tend_rt_active: torch.Tensor,
        tend_rw_active: torch.Tensor,
        coftz_active: torch.Tensor,
        cofwz_active: torch.Tensor,
        cofwr_active: torch.Tensor,
        cofwt_active: torch.Tensor,
        zz_active: torch.Tensor,
        a_tri_active: torch.Tensor,
        alpha_tri_active: torch.Tensor,
        gamma_tri_active: torch.Tensor,
        rho_zz_active: torch.Tensor,
        rw_save_active: torch.Tensor,
        rw_base_active: torch.Tensor,
        dss_active: torch.Tensor,
        w_active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return _acoustic_step_acoustic_flux_solve_numeric(
            nVertLevels=static_n_vert_levels,
            dts=static_dts,
            resm=static_resm,
            epssm=static_epssm,
            edge_indices=static_edge_indices,
            edge_valid=static_edge_valid,
            edge_valid_all_by_slot=static_edge_valid_all_by_slot,
            edge_flux_weight=static_edge_flux_weight,
            active_inv_area=static_active_inv_area,
            edge_cell_pair=static_edge_cell_pair,
            full_solve_compile_path=static_full_solve_compile_path,
            solve_active_mask=static_solve_active_mask,
            cofrz=cofrz,
            rdzw=rdzw,
            fzm=fzm,
            fzp=fzp,
            ru_p=ru_p,
            theta_m=theta_m,
            rho_pp_active=rho_pp_active,
            rtheta_pp_active=rtheta_pp_active,
            rw_p_active=rw_p_active,
            ww_avg_active=ww_avg_active,
            tend_rho_active=tend_rho_active,
            tend_rt_active=tend_rt_active,
            tend_rw_active=tend_rw_active,
            coftz_active=coftz_active,
            cofwz_active=cofwz_active,
            cofwr_active=cofwr_active,
            cofwt_active=cofwt_active,
            zz_active=zz_active,
            a_tri_active=a_tri_active,
            alpha_tri_active=alpha_tri_active,
            gamma_tri_active=gamma_tri_active,
            rho_zz_active=rho_zz_active,
            rw_save_active=rw_save_active,
            rw_base_active=rw_base_active,
            dss_active=dss_active,
            w_active=w_active,
        )

    _acoustic_step_bound_acoustic_solve.__name__ = "_acoustic_step_bound_acoustic_solve"
    _acoustic_step_bound_acoustic_solve.__qualname__ = (
        "_acoustic_step_bound_acoustic_solve"
    )
    return AcousticSolvePlan(
        key=key,
        static_argument_names=tuple(static_arguments),
        dynamic_argument_names=_ACOUSTIC_STEP_ACOUSTIC_SOLVE_DYNAMIC_NAMES,
        static_arguments=static_arguments,
        numeric_callable=_acoustic_step_bound_acoustic_solve,
    )


def atm_advance_acoustic_step_work(
    # Dimensions
    nCells: int,
    nEdges: int,
    nCellsSolve: int,
    nVertLevels: int,
    maxEdges: int,
    # Loop bounds (1-based Fortran indices)
    cellStart: int,
    cellEnd: int,
    edgeStart: int,
    edgeEnd: int,
    cellSolveStart: int,
    cellSolveEnd: int,
    # Time stepping
    small_step: int,
    dts: float,
    epssm: float,
    # Pre-computed scalar coefficients from checkpoint
    cf1: float,
    cf2: float,
    cf3: float,
    c2: float,
    rcv: float,
    resm: float,
    rdts: float,
    # 1D arrays (nVertLevels,)
    fzm: torch.Tensor,
    fzp: torch.Tensor,
    rdzw: torch.Tensor,
    cofrz: torch.Tensor,
    # 1D arrays (nEdges+1,) or (nCells+1,)
    dcEdge: torch.Tensor,
    invDcEdge: torch.Tensor,
    dvEdge: torch.Tensor,
    invAreaCell: torch.Tensor,
    specZoneMaskCell: torch.Tensor,
    specZoneMaskEdge: torch.Tensor,
    # Connectivity
    nEdgesOnCell: torch.Tensor,  # (nCells+1,) int32
    cellsOnEdge: torch.Tensor,  # (nEdges+1, 2) int32
    edgesOnCell: torch.Tensor,  # (nCells+1, maxEdges) int32
    edgesOnCell_sign: torch.Tensor,  # (nCells+1, maxEdges) float64
    # 2D arrays (nCells+1, nVertLevels) or (nEdges+1, nVertLevels)
    rho_zz: torch.Tensor,
    theta_m: torch.Tensor,
    zz: torch.Tensor,
    exner: torch.Tensor,
    rho_pp: torch.Tensor,
    rtheta_pp: torch.Tensor,
    rtheta_pp_old: torch.Tensor,
    a_tri: torch.Tensor | None,
    alpha_tri: torch.Tensor | None,
    gamma_tri: torch.Tensor | None,
    cofwt: torch.Tensor | None,
    cofwr: torch.Tensor | None,
    cofwz: torch.Tensor | None,
    dss: torch.Tensor,
    cqu: torch.Tensor,
    zxu: torch.Tensor,
    ru: torch.Tensor,
    ru_save: torch.Tensor,
    ru_p: torch.Tensor,
    ruAvg: torch.Tensor,
    tend_ru: torch.Tensor,
    tend_rho: torch.Tensor,
    tend_rt: torch.Tensor,
    # 2D arrays with nVertLevels+1 (nCells+1, nVertLevels+1)
    coftz: torch.Tensor | None,
    zgrid: torch.Tensor,
    w: torch.Tensor,
    rw: torch.Tensor,
    rw_save: torch.Tensor,
    rw_p: torch.Tensor,
    wwAvg: torch.Tensor,
    tend_rw: torch.Tensor,
    workspace=None,  # Optional DycoreWorkspace with pre-allocated acoustic_step tensors
    _compile_region_names: tuple[str, ...] | frozenset[str] | None = None,
    _compile_mode: str = "reduce-overhead",
    _compile_backend: str = "inductor",
    _connectivity_payload: dict[str, Any] | None = None,
    _acoustic_solve_plan: AcousticSolvePlan | None = None,
    _tridiag_scan_policy: str | None = None,
    _tridiag_scan_allowed_shapes: tuple[str, ...] | list[str] | str | None = None,
    _vertical_implicit_coefficients_active_cell_start: int | None = None,
    _vertical_implicit_coefficients_active_cell_end: int | None = None,
    _vertical_implicit_coefficients_cofwt_active: torch.Tensor | None = None,
    _vertical_implicit_coefficients_cofwr_active: torch.Tensor | None = None,
    _vertical_implicit_coefficients_cofwz_active: torch.Tensor | None = None,
    _vertical_implicit_coefficients_coftz_active: torch.Tensor | None = None,
    _vertical_implicit_coefficients_a_tri_active: torch.Tensor | None = None,
    _vertical_implicit_coefficients_alpha_tri_active: torch.Tensor | None = None,
    _vertical_implicit_coefficients_gamma_tri_active: torch.Tensor | None = None,
) -> dict:
    """
    Advance acoustic step for split-explicit time integration.

    This is the core acoustic substep solver implementing forward-backward
    time integration of the acoustic (fast) modes. The algorithm follows
    Klemp et al MWR 2007 with implicit Rayleigh damping from Klemp et al 2008.

    The function performs:
    1. Update horizontal momentum (ru_p) with pressure gradient force
    2. Accumulate ru_p into ruAvg for scalar transport
    3. Save rtheta_pp_old for divergence damping
    4. Vertically implicit solve for rw_p (tridiagonal system)
    5. Apply Rayleigh damping on w (gravity-wave absorbing layer)
    6. Update rho_pp and rtheta_pp with updated rw_p

    Index Convention:
        Fortran uses 1-based indexing: k=1..nVertLevels, iCell=1..nCells
        PyTorch uses 0-based indexing: k=0..nVertLevels-1, iCell=0..nCells-1

        Fortran loop `do k = 2, nVertLevels` maps to Python `for k in range(1, nVertLevels)`

    Args:
        nCells: Number of cells
        nEdges: Number of edges
        nCellsSolve: Number of cells to solve (owned cells)
        nVertLevels: Number of vertical levels
        maxEdges: Maximum edges per cell
        cellStart, cellEnd: Cell loop bounds (1-based)
        edgeStart, edgeEnd: Edge loop bounds (1-based)
        cellSolveStart, cellSolveEnd: Cell solve loop bounds (1-based)
        small_step: Acoustic substep number (1 = first step)
        dts: Small (acoustic) timestep in seconds
        epssm: Off-centering parameter for implicit solve (typically 0.1)
        cf1, cf2, cf3: Weighted averaging coefficients
        c2: cp * rcv (sound speed related)
        rcv: rgas / (cp - rgas)
        resm: (1 - epssm) / (1 + epssm)
        rdts: 1 / dts
        fzm, fzp: Vertical interpolation weights
        rdzw: Inverse dz at w-levels
        cofrz: Coefficient for rho-z coupling
        dcEdge: Distance between cell centers across edge
        invDcEdge: 1 / dcEdge
        dvEdge: Length of edge (dual mesh)
        invAreaCell: 1 / areaCell
        specZoneMaskCell, specZoneMaskEdge: Specified zone masks (regional MPAS)
        nEdgesOnCell: Number of edges on each cell
        cellsOnEdge: Cells adjacent to each edge (1-based indices)
        edgesOnCell: Edges bounding each cell (1-based indices)
        edgesOnCell_sign: Sign of edge relative to cell
        rho_zz: rho * dz/dz_bar (Jacobian-weighted density)
        theta_m: Modified potential temperature
        zz: Jacobian of coordinate transformation
        exner: Exner function
        rho_pp: Density perturbation
        rtheta_pp: rho*theta perturbation
        rtheta_pp_old: Previous rho*theta perturbation (for divergence damping)
        a_tri, alpha_tri, gamma_tri: Tridiagonal matrix coefficients
        cofwt, cofwr, cofwz: Implicit coupling coefficients
        dss: Rayleigh damping coefficient
        cqu: Flux coefficient for u equation
        zxu: Terrain slope at edges
        ru, ru_save: Horizontal momentum and saved value
        ru_p: Horizontal momentum perturbation
        ruAvg: Accumulated ru_p for scalar transport
        tend_ru, tend_rho, tend_rt: Tendencies
        coftz: Theta-z coupling coefficient
        zgrid: Height of grid levels
        w: Vertical velocity
        rw, rw_save: rho*w and saved value
        rw_p: rho*w perturbation
        wwAvg: Accumulated rho*omega for scalar transport
        tend_rw: rho*w tendency

    Returns:
        Dictionary containing updated output arrays:
            ru_p: Updated horizontal momentum perturbation
            ruAvg: Updated accumulated ru_p
            rho_pp: Updated density perturbation
            rtheta_pp: Updated rho*theta perturbation
            rtheta_pp_old: Previous rho*theta perturbation
            rw_p: Updated rho*w perturbation
            wwAvg: Updated accumulated rho*omega
            w: Vertical velocity (unchanged in this version)
            exner: Exner function (unchanged in this version)
    """
    device = require_consistent_device(
        "atm_advance_acoustic_step_work",
        {
            "rho_zz": rho_zz,
            "theta_m": theta_m,
            "zz": zz,
            "exner": exner,
            "ru": ru,
            "rw": rw,
            "tend_ru": tend_ru,
            "tend_rw": tend_rw,
            "tend_rho": tend_rho,
            "tend_rt": tend_rt,
        },
    )
    # =========================================================================
    # IMPORTANT: Use checkpoint-provided scalar values directly
    # The Fortran code computes these at runtime, but we use pre-computed values
    # from the checkpoint to ensure exact match:
    #   rcv = rgas / (cp - rgas)
    #   c2 = cp * rcv
    #   resm = (1.0 - epssm) / (1.0 + epssm)
    #   rdts = 1.0 / dts
    # These are passed in as arguments from the checkpoint.
    # =========================================================================

    # Constant for gravity (from MPAS constants)
    gravity = GRAVITY

    # Clone input arrays to avoid modifying inputs
    ru_p_out = _clone_for_grad(ru_p)
    ruAvg_out = _clone_for_grad(ruAvg)
    rho_pp_out = _clone_for_grad(rho_pp)
    rtheta_pp_out = _clone_for_grad(rtheta_pp)
    rtheta_pp_old_out = _clone_for_grad(rtheta_pp_old)
    rw_p_out = _clone_for_grad(rw_p)
    wwAvg_out = _clone_for_grad(wwAvg)
    # w and exner are read-only in this kernel; avoid unnecessary clones.
    w_out = w
    exner_out = exner

    # Convert Fortran 1-based bounds to Python 0-based
    # Fortran: do iEdge = edgeStart, edgeEnd  (1-based, inclusive)
    # Python: for iEdge in range(edge_start, edge_end)  (0-based, exclusive)
    edge_start = edgeStart - 1
    edge_end = edgeEnd
    cell_start = cellStart - 1
    cell_end = cellEnd
    cell_solve_start = cellSolveStart - 1
    cell_solve_end = cellSolveEnd

    if _connectivity_payload is None:
        conn = _acoustic_step_connectivity_payload(
            device=device,
            nCells=nCells,
            nEdges=nEdges,
            nCellsSolve=nCellsSolve,
            maxEdges=maxEdges,
            edge_start=edge_start,
            edge_end=edge_end,
            cell_solve_start=cell_solve_start,
            cell_solve_end=cell_solve_end,
            cellsOnEdge=cellsOnEdge,
            edgesOnCell=edgesOnCell,
            nEdgesOnCell=nEdgesOnCell,
            edgesOnCell_sign=edgesOnCell_sign,
            specZoneMaskCell=specZoneMaskCell,
            specZoneMaskEdge=specZoneMaskEdge,
            dvEdge=dvEdge,
            invAreaCell=invAreaCell,
        )
    else:
        conn = _connectivity_payload
    edge_cell_relation = conn.get("edge_cell_relation")
    if isinstance(edge_cell_relation, StaticPairRelation):
        edge_valid_2d = (
            edge_cell_relation.valid_2d
            if edge_cell_relation.valid_2d is not None
            else conn["edge_valid_2d"]
        )
        edge_valid_all = (
            bool(edge_cell_relation.valid_all)
            if edge_cell_relation.valid_all is not None
            else bool(conn["edge_valid_all"])
        )
        edge_c1 = edge_cell_relation.first
        edge_c2 = edge_cell_relation.second
    else:
        edge_valid_2d = conn["edge_valid_2d"]
        edge_valid_all = bool(conn["edge_valid_all"])
        edge_c1 = conn["edge_c1"]
        edge_c2 = conn["edge_c2"]
    edge_spec_zone_all_zero = bool(conn.get("edge_spec_zone_all_zero", False))
    active_cells = conn["active_cells"]
    active_cell_range = conn.get("active_cell_range")
    active_cell_slice = (
        active_cell_range.slice
        if isinstance(active_cell_range, StaticRange)
        else conn["active_cell_slice"]
    )
    active_cells_are_slice = bool(conn["active_cells_are_slice"])
    full_solve_compile_path = bool(conn["full_solve_compile_path"])
    solve_active_mask = conn["solve_active_mask"]
    active_cell_edge_relation = conn.get("active_cell_edge_relation")
    if isinstance(active_cell_edge_relation, StaticSlotRelation):
        active_safe_edge_idx = active_cell_edge_relation.indices
        active_valid = (
            active_cell_edge_relation.valid
            if active_cell_edge_relation.valid is not None
            else conn["active_valid"]
        )
        active_valid_all_by_slot = active_cell_edge_relation.valid_all_by_slot
        active_edge_sign = (
            active_cell_edge_relation.sign
            if active_cell_edge_relation.sign is not None
            else conn["active_edge_sign"]
        )
        active_flux_weight = (
            active_cell_edge_relation.weights
            if active_cell_edge_relation.weights is not None
            else conn.get("active_flux_weight")
        )
        active_inv_area = (
            active_cell_edge_relation.valid_weight
            if active_cell_edge_relation.valid_weight is not None
            else conn.get("active_inv_area")
        )
    else:
        active_safe_edge_idx = conn["active_safe_edge_idx"]
        active_valid = conn["active_valid"]
        active_valid_all_by_slot = conn["active_valid_all_by_slot"]
        active_edge_sign = conn["active_edge_sign"]
        active_flux_weight = conn.get("active_flux_weight")
        active_inv_area = conn.get("active_inv_area")
    active_edge_cell_relation = conn.get("active_edge_cell_relation")
    if isinstance(active_edge_cell_relation, StaticPairRelation):
        active_c1 = active_edge_cell_relation.first
        active_c2 = active_edge_cell_relation.second
    else:
        active_c1 = conn["active_c1"]
        active_c2 = conn["active_c2"]
    edge_slice = slice(edge_start, edge_end)

    def _active_rows(field: torch.Tensor) -> torch.Tensor:
        if active_cells_are_slice:
            return field[active_cell_slice]
        return gather_rows(field, active_cells)

    compile_region_names = (
        frozenset(_compile_region_names or ()) & _ACOUSTIC_STEP_REGION_COMPILE_TARGETS
    )
    tridiag_policy_key = normalize_tridiag_scan_policy(_tridiag_scan_policy)
    tridiag_allowed_shape_key = tuple(
        sorted(canonical_tridiag_scan_shape_keys(_tridiag_scan_allowed_shapes))
    )

    def _maybe_compile_acoustic_step_region(fn_name: str, fn, specialization_key=None):
        if fn_name not in compile_region_names:
            return fn
        from mpas_pytorch.dynamics.compile_utils import maybe_compile_kernel

        return maybe_compile_kernel(
            fn_name,
            fn,
            enabled=True,
            mode=_compile_mode,
            backend=_compile_backend,
            specialization_key=specialization_key,
        )

    # =========================================================================
    # SECTION 1: Update horizontal momentum ru_p and accumulate ruAvg

    # =========================================================================

    horizontal_momentum_update = _maybe_compile_acoustic_step_region(
        "acoustic_step_horizontal_momentum_update",
        _acoustic_step_horizontal_momentum_update,
        specialization_key=(
            "small_step",
            small_step,
            "nVertLevels",
            nVertLevels,
            "edge_count",
            edge_end - edge_start,
            "edge_valid_all",
            edge_valid_all,
        ),
    )
    ru_p_patch, ru_avg_patch = horizontal_momentum_update(
        small_step=small_step,
        edge_cell_relation=edge_cell_relation
        if isinstance(edge_cell_relation, StaticPairRelation)
        else StaticPairRelation(
            "acoustic_step.edge_cells.runtime",
            first=edge_c1,
            second=edge_c2,
            valid=edge_valid_2d[:, 0],
            valid_2d=edge_valid_2d,
            valid_all=edge_valid_all,
        ),
        rtheta_pp=rtheta_pp_out,
        zz=zz,
        exner=exner,
        rho_pp=rho_pp_out,
        ru_p_edge=ru_p_out[edge_slice, :],
        ru_avg_edge=ruAvg_out[edge_slice, :],
        tend_ru_edge=tend_ru[edge_slice, :],
        inv_dc_edge=invDcEdge[edge_slice],
        spec_zone_mask_edge=specZoneMaskEdge[edge_slice],
        spec_zone_mask_edge_all_zero=edge_spec_zone_all_zero,
        cqu_edge=cqu[edge_slice, :],
        zxu_edge=zxu[edge_slice, :],
        dts=dts,
        c2=c2,
        gravity=gravity,
    )
    ru_p_out = _replace_row_slice(ru_p_out, edge_start, edge_end, ru_p_patch)
    ruAvg_out = _replace_row_slice(ruAvg_out, edge_start, edge_end, ru_avg_patch)

    # =========================================================================
    # SECTION 2: Save rtheta_pp_old for divergence damping

    # =========================================================================

    cell_slice = slice(cell_start, cell_end)
    if small_step == 1:
        # Initialize to zero on first step
        rtheta_pp_old_out = _replace_row_slice(
            rtheta_pp_old_out,
            cell_start,
            cell_end,
            _acoustic_step_zero_row_patch_like(rtheta_pp_old_out, cell_start, cell_end),
        )
    else:
        # Copy current rtheta_pp
        rtheta_pp_old_out = _replace_row_slice(
            rtheta_pp_old_out,
            cell_start,
            cell_end,
            rtheta_pp_out[cell_slice, :],
        )

    # =========================================================================
    # SECTION 3: Main cell loop - vertically implicit acoustic solve

    # =========================================================================

    # acoustic-step carries the acoustic solve scratch as active-row tensors below; the
    # workspace intentionally has no full-cell ts/rs buffers.

    # first-step initialization on solve cells.
    if small_step == 1:
        rho_pp_out = _replace_row_slice(
            rho_pp_out,
            cell_solve_start,
            cell_solve_end,
            _acoustic_step_zero_row_patch_like(
                rho_pp_out, cell_solve_start, cell_solve_end
            ),
        )
        rtheta_pp_out = _replace_row_slice(
            rtheta_pp_out,
            cell_solve_start,
            cell_solve_end,
            _acoustic_step_zero_row_patch_like(
                rtheta_pp_out, cell_solve_start, cell_solve_end
            ),
        )
        wwAvg_out = _replace_row_slice(
            wwAvg_out,
            cell_solve_start,
            cell_solve_end,
            _acoustic_step_zero_row_patch_like(
                wwAvg_out, cell_solve_start, cell_solve_end
            ),
        )
        rw_p_out = _replace_row_slice(
            rw_p_out,
            cell_solve_start,
            cell_solve_end,
            _acoustic_step_zero_row_patch_like(
                rw_p_out, cell_solve_start, cell_solve_end
            ),
        )

    if active_cells.numel() > 0:
        acoustic_flux_solve_region = None
        if _acoustic_solve_plan is None:
            acoustic_flux_solve_region = _maybe_compile_acoustic_step_region(
                "acoustic_step_acoustic_flux_solve_region",
                _acoustic_step_acoustic_flux_solve_region,
                specialization_key=(
                    "nVertLevels",
                    nVertLevels,
                    "active_cell_count",
                    active_cells.numel(),
                    "slot_count",
                    active_safe_edge_idx.shape[1],
                    "tridiag_scan_policy",
                    tridiag_policy_key,
                    "tridiag_scan_shapes",
                    tridiag_allowed_shape_key,
                ),
            )
        rho_pp_active = _active_rows(rho_pp_out)
        rtheta_pp_active = _active_rows(rtheta_pp_out)
        rw_p_active = _active_rows(rw_p_out)
        ww_avg_active = _active_rows(wwAvg_out)
        tend_rho_active = _active_rows(tend_rho)
        tend_rt_active = _active_rows(tend_rt)
        tend_rw_active = _active_rows(tend_rw)
        cofwt_active = _acoustic_step_select_active_coef_rows(
            cofwt,
            _vertical_implicit_coefficients_cofwt_active,
            active_cell_slice=active_cell_slice,
            active_cells_are_slice=active_cells_are_slice,
            active_cells=active_cells,
            patch_cell_start=_vertical_implicit_coefficients_active_cell_start,
            patch_cell_end=_vertical_implicit_coefficients_active_cell_end,
        )
        cofwr_active = _acoustic_step_select_active_coef_rows(
            cofwr,
            _vertical_implicit_coefficients_cofwr_active,
            active_cell_slice=active_cell_slice,
            active_cells_are_slice=active_cells_are_slice,
            active_cells=active_cells,
            patch_cell_start=_vertical_implicit_coefficients_active_cell_start,
            patch_cell_end=_vertical_implicit_coefficients_active_cell_end,
        )
        cofwz_active = _acoustic_step_select_active_coef_rows(
            cofwz,
            _vertical_implicit_coefficients_cofwz_active,
            active_cell_slice=active_cell_slice,
            active_cells_are_slice=active_cells_are_slice,
            active_cells=active_cells,
            patch_cell_start=_vertical_implicit_coefficients_active_cell_start,
            patch_cell_end=_vertical_implicit_coefficients_active_cell_end,
        )
        coftz_active = _acoustic_step_select_active_coef_rows(
            coftz,
            _vertical_implicit_coefficients_coftz_active,
            active_cell_slice=active_cell_slice,
            active_cells_are_slice=active_cells_are_slice,
            active_cells=active_cells,
            patch_cell_start=_vertical_implicit_coefficients_active_cell_start,
            patch_cell_end=_vertical_implicit_coefficients_active_cell_end,
        )
        a_tri_active = _acoustic_step_select_active_coef_rows(
            a_tri,
            _vertical_implicit_coefficients_a_tri_active,
            active_cell_slice=active_cell_slice,
            active_cells_are_slice=active_cells_are_slice,
            active_cells=active_cells,
            patch_cell_start=_vertical_implicit_coefficients_active_cell_start,
            patch_cell_end=_vertical_implicit_coefficients_active_cell_end,
        )
        alpha_tri_active = _acoustic_step_select_active_coef_rows(
            alpha_tri,
            _vertical_implicit_coefficients_alpha_tri_active,
            active_cell_slice=active_cell_slice,
            active_cells_are_slice=active_cells_are_slice,
            active_cells=active_cells,
            patch_cell_start=_vertical_implicit_coefficients_active_cell_start,
            patch_cell_end=_vertical_implicit_coefficients_active_cell_end,
        )
        gamma_tri_active = _acoustic_step_select_active_coef_rows(
            gamma_tri,
            _vertical_implicit_coefficients_gamma_tri_active,
            active_cell_slice=active_cell_slice,
            active_cells_are_slice=active_cells_are_slice,
            active_cells=active_cells,
            patch_cell_start=_vertical_implicit_coefficients_active_cell_start,
            patch_cell_end=_vertical_implicit_coefficients_active_cell_end,
        )
        zz_active = _active_rows(zz)
        dynamic_solve_arguments = {
            "cofrz": cofrz,
            "rdzw": rdzw,
            "fzm": fzm,
            "fzp": fzp,
            "ru_p": ru_p_out,
            "theta_m": theta_m,
            "rho_pp_active": rho_pp_active,
            "rtheta_pp_active": rtheta_pp_active,
            "rw_p_active": rw_p_active,
            "ww_avg_active": ww_avg_active,
            "tend_rho_active": tend_rho_active,
            "tend_rt_active": tend_rt_active,
            "tend_rw_active": tend_rw_active,
            "coftz_active": coftz_active,
            "cofwz_active": cofwz_active,
            "cofwr_active": cofwr_active,
            "cofwt_active": cofwt_active,
            "zz_active": zz_active,
            "a_tri_active": a_tri_active,
            "alpha_tri_active": alpha_tri_active,
            "gamma_tri_active": gamma_tri_active,
            "rho_zz_active": _active_rows(rho_zz),
            "rw_save_active": _active_rows(rw_save),
            "rw_base_active": _active_rows(rw),
            "dss_active": _active_rows(dss),
            "w_active": _active_rows(w),
        }
        with tridiag_scan_policy(
            tridiag_policy_key,
            allowed_shapes=tridiag_allowed_shape_key,
        ):
            if _acoustic_solve_plan is not None:
                rw_rows, ww_rows, rho_patch, rtheta_patch = (
                    _acoustic_solve_plan.numeric_callable(
                        *_acoustic_solve_plan.dynamic_values(dynamic_solve_arguments)
                    )
                )
            else:
                if acoustic_flux_solve_region is None:
                    raise RuntimeError(
                        "acoustic-step acoustic solve callable was not initialized"
                    )
                rw_rows, ww_rows, rho_patch, rtheta_patch = acoustic_flux_solve_region(
                    nVertLevels=nVertLevels,
                    dts=dts,
                    resm=resm,
                    epssm=epssm,
                    cell_edge_relation=active_cell_edge_relation
                    if isinstance(active_cell_edge_relation, StaticSlotRelation)
                    else StaticSlotRelation(
                        "acoustic_step.active_cell_edges.runtime",
                        indices=active_safe_edge_idx,
                        valid=active_valid,
                        sign=active_edge_sign,
                        weights=active_flux_weight,
                        valid_all_by_slot=active_valid_all_by_slot,
                        valid_weight=active_inv_area,
                    ),
                    edge_cell_relation=active_edge_cell_relation
                    if isinstance(active_edge_cell_relation, StaticPairRelation)
                    else StaticPairRelation(
                        "acoustic_step.active_edge_cells.runtime",
                        first=active_c1,
                        second=active_c2,
                        valid=active_valid,
                    ),
                    full_solve_compile_path=full_solve_compile_path,
                    solve_active_mask=solve_active_mask,
                    **dynamic_solve_arguments,
                )
        if active_cells_are_slice:
            row_start = active_cell_slice.start
            row_end = active_cell_slice.stop
            rw_p_out = _replace_row_slice(rw_p_out, row_start, row_end, rw_rows)
            wwAvg_out = _replace_row_slice(wwAvg_out, row_start, row_end, ww_rows)
            rho_pp_out = _replace_row_slice(rho_pp_out, row_start, row_end, rho_patch)
            rtheta_pp_out = _replace_row_slice(
                rtheta_pp_out, row_start, row_end, rtheta_patch
            )
        else:
            raise RuntimeError(
                "acoustic-step active cell payload must be a contiguous slice"
            )

    # =========================================================================
    # Return updated arrays
    # =========================================================================
    return {
        "ru_p": ru_p_out,
        "ruAvg": ruAvg_out,
        "rho_pp": rho_pp_out,
        "rtheta_pp": rtheta_pp_out,
        "rtheta_pp_old": rtheta_pp_old_out,
        "rw_p": rw_p_out,
        "wwAvg": wwAvg_out,
        "w": w_out,
        "exner": exner_out,
    }


def acoustic_step(
    state: dict,
    mesh: Any,
    dt_acoustic: float | torch.Tensor,
    nAcousticSteps: int,
) -> dict:
    """
    Perform acoustic mode time steps.

    Args:
        state: State dictionary containing all prognostic variables
        mesh: MPAS mesh state (MESHState object or dict of mesh tensors)
        dt_acoustic: Acoustic timestep
        nAcousticSteps: Number of acoustic substeps per large step

    Returns:
        Updated state dictionary
    """
    from mpas_pytorch.dynamics.time_integration import execute_rk_step

    # Convert mesh to dict if it's a MESHState object
    if hasattr(mesh, "__dict__"):
        mesh_dict = dict(mesh.__dict__)
    else:
        mesh_dict = dict(mesh) if mesh is not None else {}
    dts_value = float(dt_acoustic)

    return execute_rk_step(
        rk_step=1,
        state_dict=state,
        mesh_dict=mesh_dict,
        config={"dts": dts_value},
        nAcousticSteps=nAcousticSteps,
    )
