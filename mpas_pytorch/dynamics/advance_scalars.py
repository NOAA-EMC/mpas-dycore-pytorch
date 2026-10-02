"""
Scalar advancement - explicit (explicit-scalar-transport).

Advances scalar (tracer) fields using explicit time integration.
Used for conservative advection with flux-form transport.

This kernel corresponds to Fortran subroutine atm_advance_scalars_work.

Algorithm overview:
1. Compute time weights for density interpolation
2. Compute horizontal fluxes at edges using high-order upwind scheme
3. Accumulate horizontal flux divergence for each cell
4. Compute vertical fluxes using 3rd-order scheme
5. Update scalars with horizontal + vertical flux divergence

Index Convention:
    Fortran uses 1-based indexing. Checkpoint arrays are stored with 0-based
    indexing where Python index 0 corresponds to Fortran index 1.
    Connectivity arrays (advCellsForEdge, edgesOnCell, etc.) store 1-based
    values that must be converted to 0-based when used as indices.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, cast

import torch

from mpas_pytorch.autodiff.soft_limiter import fct_abs
from mpas_pytorch.constants import DTYPE
from mpas_pytorch.dynamics.pattern_c import (
    PatternCOperator,
    _build_pattern_c_connectivity_payload,
    compute_pattern_c_flux_from_static_tensors,
)
from mpas_pytorch.dynamics.static_payloads import (
    StaticRange,
    StaticSlotRelation,
    static_patch_gather,
)
from mpas_pytorch.dynamics.stencil_ops import (
    _accumulate_ordered_slot,
    _active_slot_count,
    _compile_active,
    _gather_slot_batch,
    _slot_batches,
    _slot_gather_batch,
    _static_connectivity_cache_allowed,
    _static_mask_all_true,
    _static_slot_all_true,
    _tensor_version_cache_key,
    gather_rows,
    sum_fortran_order,
)
from mpas_pytorch.utils.checks import require_consistent_device

_EXPLICIT_SCALAR_TRANSPORT_REGION_COMPILE_TARGETS = frozenset(
    {
        "explicit_scalar_transport_horizontal_transport_composition",
        "explicit_scalar_transport_vertical_finish_region",
    }
)
_EXPLICIT_SCALAR_TRANSPORT_CONNECTIVITY_TENSOR_NAMES = (
    "edgesOnCell",
    "nEdgesOnCell",
    "edgesOnCell_sign",
    "nAdvCellsForEdge",
    "advCellsForEdge",
    "adv_coefs",
    "adv_coefs_3rd",
)

_EXPLICIT_SCALAR_TRANSPORT_DYNAMIC_ARGUMENT_NAMES = (
    "scalar_old_cells",
    "scalar_new",
    "scalar_tend_save_cells",
    "rho_zz_old_cells",
    "rho_zz_new_cells",
    "uh_avg",
    "ww_cells",
    "horiz_flux_base",
)


@dataclass(frozen=True)
class ExplicitScalarTransportSectionOps:
    """Plan-owned callables used by the one canonical explicit-scalar-transport composition."""

    horizontal_transport: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    vertical_finish: Callable[..., torch.Tensor]


@dataclass(frozen=True)
class ExplicitScalarTransportNumericPlan:
    """Immutable topology and scalar binding for one explicit-scalar-transport execution variant."""

    key: tuple[object, ...]
    dynamic_argument_names: tuple[str, ...]
    static_arguments: Mapping[str, Any]
    section_ops: ExplicitScalarTransportSectionOps
    numeric_factory: Callable[
        [ExplicitScalarTransportSectionOps], Callable[..., tuple[torch.Tensor, ...]]
    ]
    numeric_callable: Callable[..., tuple[torch.Tensor, ...]]

    def dynamic_values(self, call_arguments: Mapping[str, Any]) -> tuple[object, ...]:
        missing = [
            name for name in self.dynamic_argument_names if name not in call_arguments
        ]
        if missing:
            raise KeyError(
                f"explicit-scalar-transport numeric call is missing dynamic arguments: {missing}"
            )
        return tuple(call_arguments[name] for name in self.dynamic_argument_names)

    def with_section_ops(
        self,
        section_ops: ExplicitScalarTransportSectionOps,
    ) -> "ExplicitScalarTransportNumericPlan":
        return replace(
            self,
            section_ops=section_ops,
            numeric_callable=self.numeric_factory(section_ops),
        )


def _flux4(
    q_im2: torch.Tensor,
    q_im1: torch.Tensor,
    q_i: torch.Tensor,
    q_ip1: torch.Tensor,
    ua: torch.Tensor,
) -> torch.Tensor:
    """
    Fourth-order advection flux.

    MPAS expression: flux4(q_im2, q_im1, q_i, q_ip1, ua) = ua*(7*(q_i + q_im1) - (q_ip1 + q_im2))/12.0
    """
    return ua * (7.0 * (q_i + q_im1) - (q_ip1 + q_im2)) / 12.0


def _flux3(
    q_im2: torch.Tensor,
    q_im1: torch.Tensor,
    q_i: torch.Tensor,
    q_ip1: torch.Tensor,
    ua: torch.Tensor,
    coef3: float,
) -> torch.Tensor:
    """
    Third-order advection flux with upwind correction.

    MPAS expression:
        flux3(q_im2, q_im1, q_i, q_ip1, ua, coef3) =
            flux4(q_im2, q_im1, q_i, q_ip1, ua) +
            coef3*abs(ua)*((q_ip1 - q_im2) - 3*(q_i - q_im1))/12.0
    """
    f4 = _flux4(q_im2, q_im1, q_i, q_ip1, ua)
    upwind_corr = coef3 * fct_abs(ua) * ((q_ip1 - q_im2) - 3.0 * (q_i - q_im1)) / 12.0
    return f4 + upwind_corr


def _sum_horizontal_flux_slots(
    *,
    uh_avg: torch.Tensor,
    horiz_flux: torch.Tensor,
    cell_edges_0based: torch.Tensor,
    edge_sign: torch.Tensor,
    edge_valid: torch.Tensor,
    edge_valid_all_by_slot: tuple[bool, ...] | None = None,
) -> torch.Tensor:
    """Accumulate explicit-scalar-transport horizontal flux divergence without full cell-edge tensors."""
    n_cells = cell_edges_0based.shape[0]
    n_slots = cell_edges_0based.shape[1]
    if (
        not torch.is_grad_enabled()
        and (horiz_flux.device.type == "cuda" or _compile_active())
        and n_slots > 1
    ):
        uh_batch = gather_rows(uh_avg, cell_edges_0based.reshape(-1)).reshape(
            n_cells,
            n_slots,
            uh_avg.shape[1],
        )
        flux_batch = gather_rows(horiz_flux, cell_edges_0based.reshape(-1)).reshape(
            n_cells,
            n_slots,
            horiz_flux.shape[1],
            horiz_flux.shape[2],
        )
        contrib = (
            -edge_sign.view(n_cells, n_slots, 1, 1) * uh_batch.unsqueeze(3) * flux_batch
        )
        valid_all = (
            edge_valid_all_by_slot is not None
            and len(edge_valid_all_by_slot) == n_slots
            and all(edge_valid_all_by_slot)
        )
        if not valid_all:
            contrib = contrib * edge_valid.view(n_cells, n_slots, 1, 1).to(
                dtype=horiz_flux.dtype
            )
        return sum_fortran_order(contrib, dim=1)
    total: torch.Tensor | None = None
    for batch_start, batch_stop in _slot_batches(
        cell_edges_0based.shape[1],
        _slot_gather_batch(horiz_flux, cell_edges_0based.shape[1]),
    ):
        uh_batch = _gather_slot_batch(
            uh_avg, cell_edges_0based, batch_start, batch_stop
        )
        flux_batch = _gather_slot_batch(
            horiz_flux, cell_edges_0based, batch_start, batch_stop
        )
        for local_slot in range(batch_stop - batch_start):
            slot = batch_start + local_slot
            contrib = (
                -edge_sign[:, slot].view(n_cells, 1, 1)
                * uh_batch[:, local_slot, :].unsqueeze(2)
                * flux_batch[:, local_slot, :, :]
            )
            valid_all = (
                edge_valid_all_by_slot is not None and edge_valid_all_by_slot[slot]
            )
            if not valid_all:
                contrib = contrib * edge_valid[:, slot].view(n_cells, 1, 1).to(
                    dtype=horiz_flux.dtype
                )
            total = _accumulate_ordered_slot(total, contrib)
    if total is None:
        return horiz_flux.new_zeros((n_cells, horiz_flux.shape[1], horiz_flux.shape[2]))
    return total


def _explicit_scalar_transport_static_all_true(
    mask: torch.Tensor, *, allow_cuda_sync: bool = False
) -> bool:
    return bool(_static_mask_all_true(mask, allow_cuda_sync=allow_cuda_sync))


def _explicit_scalar_transport_slot_all_true(
    mask: torch.Tensor,
    *,
    allow_cuda_sync: bool = False,
) -> tuple[bool, ...]:
    metadata = _static_slot_all_true(mask, allow_cuda_sync=allow_cuda_sync)
    if metadata is None:
        return tuple(False for _ in range(mask.shape[1]))
    return metadata


def _explicit_scalar_transport_long_index(index: torch.Tensor) -> torch.Tensor:
    return index if index.dtype == torch.int64 else index.to(dtype=torch.int64)


def _explicit_scalar_transport_safe_zero_invalid(
    indices0: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    return _explicit_scalar_transport_long_index(
        torch.where(valid, indices0, torch.zeros_like(indices0))
    )


def _explicit_scalar_transport_zero_row_patch_like(
    field: torch.Tensor, row_start: int, row_end: int
) -> torch.Tensor:
    return field.new_zeros((int(row_end) - int(row_start), *field.shape[1:]))


def _explicit_scalar_transport_density_time_weights(
    *,
    advance_density: int,
    rk_step: int,
    config_time_integration_order: int,
) -> tuple[float, float]:
    """Return the static old/new density weights used by one explicit-scalar-transport RK stage."""
    if advance_density == 0:
        weight_time_new = 1.0
    elif rk_step == 1 and config_time_integration_order == 3:
        weight_time_new = 1.0 / 3.0
    elif (rk_step == 1 and config_time_integration_order == 2) or rk_step == 2:
        weight_time_new = 1.0 / 2.0
    else:
        weight_time_new = 1.0
    return 1.0 - weight_time_new, weight_time_new


def _explicit_scalar_transport_tensor_cache_key(
    tensor: torch.Tensor,
) -> tuple[object, ...] | None:
    return _tensor_version_cache_key(tensor)


def _explicit_scalar_transport_connectivity_payload_cache_key(
    **kwargs: object,
) -> tuple[object, ...] | None:
    """Return the mutation-sensitive cache key for explicit-scalar-transport static topology."""
    tensors = tuple(
        kwargs.get(name)
        for name in _EXPLICIT_SCALAR_TRANSPORT_CONNECTIVITY_TENSOR_NAMES
    )
    if not _static_connectivity_cache_allowed(tensors):
        return None

    tensor_keys: list[tuple[object, ...]] = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            tensor_key = _explicit_scalar_transport_tensor_cache_key(tensor)
            if tensor_key is None:
                return None
            tensor_keys.append(tensor_key)

    return (
        kwargs["device"],
        kwargs["nCells"],
        kwargs["nEdges"],
        kwargs["maxEdges"],
        kwargs["nAdvCellsMax"],
        kwargs["edge_start"],
        kwargs["edge_end"],
        kwargs["cell_solve_start"],
        kwargs["cell_solve_end"],
        *tensor_keys,
    )


def _build_explicit_scalar_transport_connectivity_payload(
    *,
    device: torch.device,
    nCells: int,
    nEdges: int,
    maxEdges: int,
    nAdvCellsMax: int,
    edge_start: int,
    edge_end: int,
    cell_solve_start: int,
    cell_solve_end: int,
    edgesOnCell: torch.Tensor,
    nEdgesOnCell: torch.Tensor,
    edgesOnCell_sign: torch.Tensor,
    nAdvCellsForEdge: torch.Tensor,
    advCellsForEdge: torch.Tensor,
    adv_coefs: torch.Tensor,
    adv_coefs_3rd: torch.Tensor,
) -> dict[str, Any]:
    """Build scalar-independent explicit-scalar-transport topology for one mesh/domain slice."""
    source_topology_key = tuple(
        _explicit_scalar_transport_tensor_cache_key(tensor)
        or (
            "tensor_identity",
            id(tensor),
            tuple(tensor.shape),
            str(tensor.dtype),
            str(tensor.device),
        )
        for tensor in (
            edgesOnCell,
            nEdgesOnCell,
            edgesOnCell_sign,
            nAdvCellsForEdge,
            advCellsForEdge,
            adv_coefs,
            adv_coefs_3rd,
        )
    )
    cell_slice = slice(cell_solve_start, cell_solve_end)
    edge_slice = slice(edge_start, edge_end)
    n_solve_cells = cell_solve_end - cell_solve_start

    n_edges_cell = nEdgesOnCell[cell_slice]
    active_slot_count = _active_slot_count(n_edges_cell, maxEdges, device)
    edge_slots = torch.arange(active_slot_count, device=device).unsqueeze(0)
    edge_valid = edge_slots < n_edges_cell.unsqueeze(1)
    cell_edges_1based = edgesOnCell[cell_slice, :active_slot_count]
    cell_edges_0based = _explicit_scalar_transport_safe_zero_invalid(
        cell_edges_1based - 1, edge_valid
    )
    cell_edges_in_patch = (~edge_valid) | (
        (cell_edges_0based >= edge_start) & (cell_edges_0based < edge_end)
    )
    cell_edges_in_patch_all = _explicit_scalar_transport_static_all_true(
        cell_edges_in_patch,
        allow_cuda_sync=True,
    )
    cell_edges_local = _explicit_scalar_transport_safe_zero_invalid(
        cell_edges_0based - edge_start,
        cell_edges_in_patch & edge_valid,
    )
    edge_sign = edgesOnCell_sign[cell_slice, :active_slot_count].to(dtype=DTYPE)
    edge_valid_all_by_slot = _explicit_scalar_transport_slot_all_true(
        edge_valid, allow_cuda_sync=True
    )
    cell_edge_relation = StaticSlotRelation(
        "explicit_scalar_transport.cell_edges",
        indices=cell_edges_0based,
        valid=edge_valid,
        sign=edge_sign,
        valid_all_by_slot=edge_valid_all_by_slot,
    )
    pattern_c_payload = _build_pattern_c_connectivity_payload(
        scalar_field=adv_coefs,
        n_adv_cells_for_edge=nAdvCellsForEdge,
        adv_cells_for_edge=advCellsForEdge,
        adv_coefs=adv_coefs,
        adv_coefs_3rd=adv_coefs_3rd,
        edge_start=edge_start + 1,
        edge_end=edge_end,
        n_adv_cells_max=nAdvCellsMax,
    )
    pattern_c_operator = PatternCOperator(pattern_c_payload)

    return {
        "cell_slice": cell_slice,
        "cell_range": StaticRange(
            "explicit_scalar_transport.solve_cells", cell_solve_start, cell_solve_end
        ),
        "edge_slice": edge_slice,
        "edge_range": StaticRange(
            "explicit_scalar_transport.edges", edge_start, edge_end
        ),
        "n_solve_cells": n_solve_cells,
        "cell_edges_0based": cell_edges_0based,
        "cell_edges_local": cell_edges_local,
        "cell_edges_in_patch": cell_edges_in_patch,
        "cell_edges_in_patch_all": cell_edges_in_patch_all,
        "edge_valid": edge_valid,
        "edge_sign": edge_sign,
        "edge_valid_all_by_slot": edge_valid_all_by_slot,
        "cell_edge_relation": cell_edge_relation,
        "pattern_c_payload": pattern_c_payload,
        "pattern_c_operator": pattern_c_operator,
        "one": pattern_c_payload["one"],
        "source_topology_key": source_topology_key,
    }


def _explicit_scalar_transport_connectivity_payload(**kwargs: object) -> dict[str, Any]:
    """Build direct-call topology without retaining a second module owner."""
    return _build_explicit_scalar_transport_connectivity_payload(**kwargs)  # type: ignore[arg-type]


def _explicit_scalar_transport_pattern_c_flux_region(
    *,
    scalar_new: torch.Tensor,
    uhAvg: torch.Tensor,
    one: torch.Tensor,
    pattern_c_operator: PatternCOperator,
) -> torch.Tensor:
    """Compute explicit-scalar-transport Pattern-C edge flux for one edge range."""
    _, horiz_flux = pattern_c_operator(
        scalar_field=scalar_new,
        uh_avg=uhAvg,
        one=one,
    )
    return horiz_flux


def _explicit_scalar_transport_horizontal_flux_divergence_region(
    *,
    uhAvg: torch.Tensor,
    horiz_flux_arr: torch.Tensor,
    cell_edge_relation: StaticSlotRelation,
) -> torch.Tensor:
    """Reduce explicit-scalar-transport edge fluxes to cell-centered horizontal divergence."""
    cell_edges_0based = cell_edge_relation.indices
    edge_valid = cell_edge_relation.valid
    edge_sign = cell_edge_relation.sign
    edge_valid_all_by_slot = cell_edge_relation.valid_all_by_slot
    if edge_valid is None:
        edge_valid = torch.ones_like(cell_edges_0based, dtype=torch.bool)
    if edge_sign is None:
        edge_sign = torch.ones_like(cell_edges_0based, dtype=DTYPE)
    return _sum_horizontal_flux_slots(
        uh_avg=uhAvg,
        horiz_flux=horiz_flux_arr,
        cell_edges_0based=cell_edges_0based,
        edge_sign=edge_sign,
        edge_valid=edge_valid,
        edge_valid_all_by_slot=edge_valid_all_by_slot,
    )


def _explicit_scalar_transport_scalar_tend_column_region(
    *,
    scalar_tend_save_cells: torch.Tensor,
    uh_flux_divergence: torch.Tensor,
    inv_area_cells: torch.Tensor,
) -> torch.Tensor:
    """Apply inverse-cell-area scaling to explicit-scalar-transport horizontal flux divergence."""
    return (
        uh_flux_divergence * inv_area_cells.view(inv_area_cells.numel(), 1, 1)
        + scalar_tend_save_cells
    )


def _explicit_scalar_transport_vertical_flux_divergence_region(
    *,
    scalar_new_cells: torch.Tensor,
    ww_cells: torch.Tensor,
    fnm: torch.Tensor,
    fnp: torch.Tensor,
    rdnw: torch.Tensor,
    coef_3rd_order: float,
    nVertLevels: int,
    num_scalars: int,
) -> torch.Tensor:
    """Compatibility wrapper for the split explicit-scalar-transport vertical flux divergence."""
    wdtn_k1, wdtn_last = _explicit_scalar_transport_vertical_boundary_flux_region(
        scalar_new_cells=scalar_new_cells,
        ww_cells=ww_cells,
        fnm=fnm,
        fnp=fnp,
        nVertLevels=nVertLevels,
        num_scalars=num_scalars,
    )
    wdtn_interior = _explicit_scalar_transport_vertical_interior_flux_region(
        scalar_new_cells=scalar_new_cells,
        ww_cells=ww_cells,
        coef_3rd_order=coef_3rd_order,
        nVertLevels=nVertLevels,
        num_scalars=num_scalars,
    )
    return _explicit_scalar_transport_vertical_flux_divergence_from_fluxes_region(
        wdtn_k1=wdtn_k1,
        wdtn_interior=wdtn_interior,
        wdtn_last=wdtn_last,
        rdnw=rdnw,
        nVertLevels=nVertLevels,
    )


def _explicit_scalar_transport_vertical_boundary_flux_region(
    *,
    scalar_new_cells: torch.Tensor,
    ww_cells: torch.Tensor,
    fnm: torch.Tensor,
    fnp: torch.Tensor,
    nVertLevels: int,
    num_scalars: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute explicit-scalar-transport bottom/top vertical interface fluxes.

    The interior third-order flux remains a separate helper because CPU Inductor
    has shown non-bitwise drift for that expression. The boundary expressions
    compile bitwise and are safe to validate independently.
    """
    if nVertLevels < 2:
        zeros = scalar_new_cells.new_zeros((scalar_new_cells.shape[0], num_scalars))
        return zeros, zeros

    wdtn_k1 = ww_cells[:, 1:2] * (
        fnm[1] * scalar_new_cells[:, 1, :] + fnp[1] * scalar_new_cells[:, 0, :]
    )
    if nVertLevels > 2:
        k_last = nVertLevels - 1
        wdtn_last = ww_cells[:, k_last : k_last + 1] * (
            fnm[k_last] * scalar_new_cells[:, k_last, :]
            + fnp[k_last] * scalar_new_cells[:, k_last - 1, :]
        )
    else:
        wdtn_last = scalar_new_cells.new_zeros((scalar_new_cells.shape[0], num_scalars))
    return wdtn_k1, wdtn_last


def _explicit_scalar_transport_vertical_interior_flux_region(
    *,
    scalar_new_cells: torch.Tensor,
    ww_cells: torch.Tensor,
    coef_3rd_order: float,
    nVertLevels: int,
    num_scalars: int,
) -> torch.Tensor:
    """Compute explicit-scalar-transport interior third-order vertical fluxes.

    The helper operates on the interior levels; its caller handles boundary
    levels and assembles the complete vertical flux.
    """
    if nVertLevels <= 3:
        return scalar_new_cells[:, :0, :]
    return _flux3(
        scalar_new_cells[:, 0 : nVertLevels - 3, :],
        scalar_new_cells[:, 1 : nVertLevels - 2, :],
        scalar_new_cells[:, 2 : nVertLevels - 1, :],
        scalar_new_cells[:, 3:nVertLevels, :],
        ww_cells[:, 2 : nVertLevels - 1].unsqueeze(2),
        coef_3rd_order,
    )


def _explicit_scalar_transport_vertical_flux_divergence_from_fluxes_region(
    *,
    wdtn_k1: torch.Tensor,
    wdtn_interior: torch.Tensor,
    wdtn_last: torch.Tensor,
    rdnw: torch.Tensor,
    nVertLevels: int,
) -> torch.Tensor:
    """Assemble explicit-scalar-transport full-level vertical flux divergence from interface fluxes."""
    if nVertLevels < 1:
        return wdtn_k1.new_zeros((wdtn_k1.shape[0], 0, wdtn_k1.shape[1]))

    bands = [rdnw[0] * wdtn_k1.unsqueeze(1)]
    if nVertLevels > 3:
        bands.append(rdnw[1] * (wdtn_interior[:, 0:1, :] - wdtn_k1.unsqueeze(1)))
        if nVertLevels > 4:
            bands.append(
                rdnw[2 : nVertLevels - 2].view(1, -1, 1)
                * (wdtn_interior[:, 1:, :] - wdtn_interior[:, :-1, :])
            )
        bands.append(
            rdnw[nVertLevels - 2] * (wdtn_last.unsqueeze(1) - wdtn_interior[:, -1:, :])
        )
    elif nVertLevels == 3:
        bands.append(rdnw[1] * (wdtn_last.unsqueeze(1) - wdtn_k1.unsqueeze(1)))
    elif nVertLevels == 2:
        bands.append(rdnw[1] * (-wdtn_k1.unsqueeze(1)))

    if nVertLevels > 2:
        bands.append(rdnw[nVertLevels - 1] * (-wdtn_last.unsqueeze(1)))
    return torch.cat(tuple(bands), dim=1)


def _explicit_scalar_transport_final_scalar_update_region(
    *,
    scalar_old_cells: torch.Tensor,
    scalar_tend_column: torch.Tensor,
    vert_flux_divergence: torch.Tensor,
    rho_zz_old_cells: torch.Tensor,
    rho_zz_new_cells: torch.Tensor,
    dt: float,
    weight_time_old: float,
    weight_time_new: float,
) -> torch.Tensor:
    """Apply explicit-scalar-transport density interpolation and final scalar update."""
    rho_zz_interp = (
        weight_time_old * rho_zz_old_cells + weight_time_new * rho_zz_new_cells
    )
    rho_zz_new_inv = 1.0 / rho_zz_interp
    return cast(
        torch.Tensor,
        (
            scalar_old_cells * rho_zz_old_cells.unsqueeze(2)
            + dt * (scalar_tend_column - vert_flux_divergence)
        )
        * rho_zz_new_inv.unsqueeze(2),
    )


def _explicit_scalar_transport_scalar_update_region(
    *,
    scalar_old_cells: torch.Tensor,
    scalar_new_cells: torch.Tensor,
    scalar_tend_save_cells: torch.Tensor,
    rho_zz_old_cells: torch.Tensor,
    rho_zz_new_cells: torch.Tensor,
    uh_flux_divergence: torch.Tensor,
    inv_area_cells: torch.Tensor,
    ww_cells: torch.Tensor,
    fnm: torch.Tensor,
    fnp: torch.Tensor,
    rdnw: torch.Tensor,
    dt: float,
    coef_3rd_order: float,
    weight_time_old: float,
    weight_time_new: float,
    nVertLevels: int,
    num_scalars: int,
) -> torch.Tensor:
    """Compatibility wrapper for the formerly monolithic explicit-scalar-transport scalar update."""
    scalar_tend_column = _explicit_scalar_transport_scalar_tend_column_region(
        scalar_tend_save_cells=scalar_tend_save_cells,
        uh_flux_divergence=uh_flux_divergence,
        inv_area_cells=inv_area_cells,
    )
    vert_flux_divergence = _explicit_scalar_transport_vertical_flux_divergence_region(
        scalar_new_cells=scalar_new_cells,
        ww_cells=ww_cells,
        fnm=fnm,
        fnp=fnp,
        rdnw=rdnw,
        coef_3rd_order=coef_3rd_order,
        nVertLevels=nVertLevels,
        num_scalars=num_scalars,
    )
    return _explicit_scalar_transport_final_scalar_update_region(
        scalar_old_cells=scalar_old_cells,
        scalar_tend_column=scalar_tend_column,
        vert_flux_divergence=vert_flux_divergence,
        rho_zz_old_cells=rho_zz_old_cells,
        rho_zz_new_cells=rho_zz_new_cells,
        dt=dt,
        weight_time_old=weight_time_old,
        weight_time_new=weight_time_new,
    )


def _sum_explicit_scalar_transport_pre_gathered_slots(
    *,
    uh_slots: torch.Tensor,
    flux_slots: torch.Tensor,
    edge_sign: torch.Tensor,
    edge_valid: torch.Tensor,
    edge_valid_all_by_slot: tuple[bool, ...],
) -> torch.Tensor:
    """Reduce pre-gathered explicit-scalar-transport cell-edge fluxes in Fortran slot order."""
    n_cells, n_slots = edge_sign.shape
    if not torch.is_grad_enabled() and (
        _compile_active() or flux_slots.device.type == "cuda"
    ):
        contrib = (
            -edge_sign.view(n_cells, n_slots, 1, 1) * uh_slots.unsqueeze(3) * flux_slots
        )
        if not all(edge_valid_all_by_slot):
            contrib = contrib * edge_valid.view(n_cells, n_slots, 1, 1).to(
                dtype=flux_slots.dtype
            )
        return sum_fortran_order(contrib, dim=1)

    total: torch.Tensor | None = None
    for slot in range(n_slots):
        contrib = (
            -edge_sign[:, slot].view(n_cells, 1, 1)
            * uh_slots[:, slot, :].unsqueeze(2)
            * flux_slots[:, slot, :, :]
        )
        if not edge_valid_all_by_slot[slot]:
            contrib = contrib * edge_valid[:, slot].view(n_cells, 1, 1).to(
                dtype=flux_slots.dtype
            )
        total = _accumulate_ordered_slot(total, contrib)
    if total is None:
        return flux_slots.new_zeros((n_cells, flux_slots.shape[2], flux_slots.shape[3]))
    return total


def _explicit_scalar_transport_horizontal_transport_composition(
    *,
    edge_start: int,
    edge_end: int,
    edge_count: int,
    slot_count: int,
    adv_coef: torch.Tensor,
    adv_coef_3rd: torch.Tensor,
    slot_cells_matrix: torch.Tensor,
    slot_valid_matrix: torch.Tensor,
    slot_valid_matrix_all: bool | None,
    use_full_width_patch_path: bool,
    fallback_rows: torch.Tensor,
    one: torch.Tensor,
    cell_edges_0based: torch.Tensor,
    cell_edges_local: torch.Tensor,
    cell_edges_in_patch: torch.Tensor,
    cell_edges_in_patch_all: bool,
    edge_valid: torch.Tensor,
    edge_sign: torch.Tensor,
    edge_valid_all_by_slot: tuple[bool, ...],
    do_physics: bool,
    inv_area_cells: torch.Tensor,
    scalar_new: torch.Tensor,
    scalar_tend_save_cells: torch.Tensor,
    uh_avg: torch.Tensor,
    horiz_flux_base: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute compact explicit-scalar-transport Pattern-C flux and horizontal tendency together."""
    _, horiz_flux = compute_pattern_c_flux_from_static_tensors(
        scalar_field=scalar_new,
        uh_avg=uh_avg,
        one=one,
        edge_start=edge_start,
        edge_end=edge_end,
        edge_count=edge_count,
        slot_count=slot_count,
        adv_coef=adv_coef,
        adv_coef_3rd=adv_coef_3rd,
        slot_cells_matrix=slot_cells_matrix,
        slot_valid_matrix=slot_valid_matrix,
        slot_valid_matrix_all=slot_valid_matrix_all,
        use_full_width_patch_path=use_full_width_patch_path,
        fallback_rows=fallback_rows,
    )

    uh_slots = gather_rows(uh_avg, cell_edges_0based)
    flux_slots = static_patch_gather(
        horiz_flux_base,
        horiz_flux,
        cell_edges_0based,
        cell_edges_local,
        cell_edges_in_patch,
        cell_edges_in_patch_all,
    )
    horizontal_divergence = _sum_explicit_scalar_transport_pre_gathered_slots(
        uh_slots=uh_slots,
        flux_slots=flux_slots,
        edge_sign=edge_sign,
        edge_valid=edge_valid,
        edge_valid_all_by_slot=edge_valid_all_by_slot,
    )

    scalar_tend_patch = (
        scalar_tend_save_cells
        if do_physics
        else torch.zeros_like(scalar_tend_save_cells)
    )
    scalar_tend_column = _explicit_scalar_transport_scalar_tend_column_region(
        scalar_tend_save_cells=scalar_tend_patch,
        uh_flux_divergence=horizontal_divergence,
        inv_area_cells=inv_area_cells,
    )
    return scalar_tend_column, horiz_flux, scalar_tend_patch


def _explicit_scalar_transport_vertical_finish_region(
    *,
    scalar_old_cells: torch.Tensor,
    scalar_new_cells: torch.Tensor,
    scalar_tend_column: torch.Tensor,
    rho_zz_old_cells: torch.Tensor,
    rho_zz_new_cells: torch.Tensor,
    ww_cells: torch.Tensor,
    wdtn_interior: torch.Tensor,
    fnm: torch.Tensor,
    fnp: torch.Tensor,
    rdnw: torch.Tensor,
    dt: float,
    weight_time_old: float,
    weight_time_new: float,
    nVertLevels: int,
    num_scalars: int,
) -> torch.Tensor:
    """Finish explicit-scalar-transport vertically while the drift-prone interior flux stays eager."""
    wdtn_k1, wdtn_last = _explicit_scalar_transport_vertical_boundary_flux_region(
        scalar_new_cells=scalar_new_cells,
        ww_cells=ww_cells,
        fnm=fnm,
        fnp=fnp,
        nVertLevels=nVertLevels,
        num_scalars=num_scalars,
    )
    vert_flux_divergence = (
        _explicit_scalar_transport_vertical_flux_divergence_from_fluxes_region(
            wdtn_k1=wdtn_k1,
            wdtn_interior=wdtn_interior,
            wdtn_last=wdtn_last,
            rdnw=rdnw,
            nVertLevels=nVertLevels,
        )
    )
    return _explicit_scalar_transport_final_scalar_update_region(
        scalar_old_cells=scalar_old_cells,
        scalar_tend_column=scalar_tend_column,
        vert_flux_divergence=vert_flux_divergence,
        rho_zz_old_cells=rho_zz_old_cells,
        rho_zz_new_cells=rho_zz_new_cells,
        dt=dt,
        weight_time_old=weight_time_old,
        weight_time_new=weight_time_new,
    )


def _explicit_scalar_transport_numeric_composition(
    *,
    _section_ops: ExplicitScalarTransportSectionOps,
    edge_start: int,
    edge_end: int,
    edge_count: int,
    slot_count: int,
    adv_coef: torch.Tensor,
    adv_coef_3rd: torch.Tensor,
    slot_cells_matrix: torch.Tensor,
    slot_valid_matrix: torch.Tensor,
    slot_valid_matrix_all: bool | None,
    use_full_width_patch_path: bool,
    fallback_rows: torch.Tensor,
    one: torch.Tensor,
    cell_edges_0based: torch.Tensor,
    cell_edges_local: torch.Tensor,
    cell_edges_in_patch: torch.Tensor,
    cell_edges_in_patch_all: bool,
    edge_valid: torch.Tensor,
    edge_sign: torch.Tensor,
    edge_valid_all_by_slot: tuple[bool, ...],
    inv_area_cells: torch.Tensor,
    fnm: torch.Tensor,
    fnp: torch.Tensor,
    rdnw: torch.Tensor,
    dt: float,
    coef_3rd_order: float,
    weight_time_old: float,
    weight_time_new: float,
    nVertLevels: int,
    num_scalars: int,
    do_physics: bool,
    cell_start: int,
    cell_end: int,
    scalar_old_cells: torch.Tensor,
    scalar_new: torch.Tensor,
    scalar_tend_save_cells: torch.Tensor,
    rho_zz_old_cells: torch.Tensor,
    rho_zz_new_cells: torch.Tensor,
    uh_avg: torch.Tensor,
    ww_cells: torch.Tensor,
    horiz_flux_base: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute explicit-scalar-transport through one canonical compact composition."""
    scalar_new_cells = scalar_new[cell_start:cell_end]
    scalar_tend_column, horiz_flux, scalar_tend_patch = (
        _section_ops.horizontal_transport(
            scalar_new=scalar_new,
            scalar_tend_save_cells=scalar_tend_save_cells,
            uh_avg=uh_avg,
            horiz_flux_base=horiz_flux_base,
        )
    )
    wdtn_interior = _explicit_scalar_transport_vertical_interior_flux_region(
        scalar_new_cells=scalar_new_cells,
        ww_cells=ww_cells,
        coef_3rd_order=coef_3rd_order,
        nVertLevels=nVertLevels,
        num_scalars=num_scalars,
    )
    scalar_new_patch = _section_ops.vertical_finish(
        scalar_old_cells=scalar_old_cells,
        scalar_new_cells=scalar_new_cells,
        scalar_tend_column=scalar_tend_column,
        rho_zz_old_cells=rho_zz_old_cells,
        rho_zz_new_cells=rho_zz_new_cells,
        ww_cells=ww_cells,
        wdtn_interior=wdtn_interior,
    )
    return scalar_new_patch, scalar_tend_patch, horiz_flux


def _explicit_scalar_transport_plan_tensor_fingerprint(
    tensor: torch.Tensor,
) -> tuple[object, ...]:
    key = _tensor_version_cache_key(tensor)
    if key is not None:
        return key
    return (
        "tensor_identity",
        id(tensor),
        tuple(tensor.shape),
        str(tensor.dtype),
        str(tensor.device),
    )


def build_explicit_scalar_transport_numeric_plan(
    *,
    connectivity_payload: Mapping[str, Any],
    inv_area_cells: torch.Tensor,
    fnm: torch.Tensor,
    fnp: torch.Tensor,
    rdnw: torch.Tensor,
    dt: float,
    coef_3rd_order: float,
    weight_time_old: float,
    weight_time_new: float,
    nVertLevels: int,
    num_scalars: int,
    do_physics: bool,
    stage_key: tuple[object, ...] = (),
) -> ExplicitScalarTransportNumericPlan:
    """Bind explicit-scalar-transport topology and scalar choices into one dynamic-tensor callable."""
    pattern = connectivity_payload["pattern_c_payload"]
    if not isinstance(pattern, Mapping):
        raise TypeError(
            "explicit-scalar-transport numeric plan requires a Pattern-C mapping payload"
        )
    edge_slice = pattern["edge_slice"]
    if not isinstance(edge_slice, slice):
        raise TypeError(
            "explicit-scalar-transport Pattern-C edge_slice must be a slice"
        )
    cell_slice = connectivity_payload["cell_slice"]
    if not isinstance(cell_slice, slice):
        raise TypeError("explicit-scalar-transport cell_slice must be a slice")
    static_arguments = MappingProxyType(
        {
            "edge_start": int(edge_slice.start),
            "edge_end": int(edge_slice.stop),
            "edge_count": int(pattern["edge_count"]),
            "slot_count": int(pattern["slot_count"]),
            "adv_coef": pattern["adv_coef"],
            "adv_coef_3rd": pattern["adv_coef_3rd"],
            "slot_cells_matrix": pattern["slot_cells_matrix"],
            "slot_valid_matrix": pattern["slot_valid_matrix"],
            "slot_valid_matrix_all": pattern.get("slot_valid_matrix_all"),
            "use_full_width_patch_path": bool(pattern["use_full_width_patch_path"]),
            "fallback_rows": pattern["fallback_rows"],
            "one": pattern["one"],
            "cell_edges_0based": connectivity_payload["cell_edges_0based"],
            "cell_edges_local": connectivity_payload["cell_edges_local"],
            "cell_edges_in_patch": connectivity_payload["cell_edges_in_patch"],
            "cell_edges_in_patch_all": bool(
                connectivity_payload["cell_edges_in_patch_all"]
            ),
            "edge_valid": connectivity_payload["edge_valid"],
            "edge_sign": connectivity_payload["edge_sign"],
            "edge_valid_all_by_slot": tuple(
                connectivity_payload["edge_valid_all_by_slot"]
            ),
            "inv_area_cells": inv_area_cells,
            "fnm": fnm,
            "fnp": fnp,
            "rdnw": rdnw,
            "dt": float(dt),
            "coef_3rd_order": float(coef_3rd_order),
            "weight_time_old": float(weight_time_old),
            "weight_time_new": float(weight_time_new),
            "nVertLevels": int(nVertLevels),
            "num_scalars": int(num_scalars),
            "do_physics": bool(do_physics),
            "cell_start": int(cell_slice.start),
            "cell_end": int(cell_slice.stop),
        }
    )
    key = (
        "ExplicitScalarTransportNumericPlan",
        tuple(stage_key),
        connectivity_payload.get("source_topology_key"),
        tuple(
            (name, _explicit_scalar_transport_plan_tensor_fingerprint(value))
            for name, value in static_arguments.items()
            if isinstance(value, torch.Tensor)
        ),
        tuple(
            (name, value)
            for name, value in static_arguments.items()
            if not isinstance(value, torch.Tensor)
        ),
    )

    edge_start_v = int(static_arguments["edge_start"])
    edge_end_v = int(static_arguments["edge_end"])
    edge_count_v = int(static_arguments["edge_count"])
    slot_count_v = int(static_arguments["slot_count"])
    adv_coef_v = static_arguments["adv_coef"]
    adv_coef_3rd_v = static_arguments["adv_coef_3rd"]
    slot_cells_matrix_v = static_arguments["slot_cells_matrix"]
    slot_valid_matrix_v = static_arguments["slot_valid_matrix"]
    slot_valid_matrix_all_v = static_arguments["slot_valid_matrix_all"]
    use_full_width_patch_path_v = bool(static_arguments["use_full_width_patch_path"])
    fallback_rows_v = static_arguments["fallback_rows"]
    one_v = static_arguments["one"]
    cell_edges_0based_v = static_arguments["cell_edges_0based"]
    cell_edges_local_v = static_arguments["cell_edges_local"]
    cell_edges_in_patch_v = static_arguments["cell_edges_in_patch"]
    cell_edges_in_patch_all_v = bool(static_arguments["cell_edges_in_patch_all"])
    edge_valid_v = static_arguments["edge_valid"]
    edge_sign_v = static_arguments["edge_sign"]
    edge_valid_all_by_slot_v = tuple(static_arguments["edge_valid_all_by_slot"])
    inv_area_cells_v = static_arguments["inv_area_cells"]
    fnm_v = static_arguments["fnm"]
    fnp_v = static_arguments["fnp"]
    rdnw_v = static_arguments["rdnw"]
    dt_v = float(static_arguments["dt"])
    coef_3rd_order_v = float(static_arguments["coef_3rd_order"])
    weight_time_old_v = float(static_arguments["weight_time_old"])
    weight_time_new_v = float(static_arguments["weight_time_new"])
    nVertLevels_v = int(static_arguments["nVertLevels"])
    num_scalars_v = int(static_arguments["num_scalars"])
    do_physics_v = bool(static_arguments["do_physics"])
    cell_start_v = int(static_arguments["cell_start"])
    cell_end_v = int(static_arguments["cell_end"])

    def _explicit_scalar_transport_bound_horizontal_transport(
        *,
        scalar_new: torch.Tensor,
        scalar_tend_save_cells: torch.Tensor,
        uh_avg: torch.Tensor,
        horiz_flux_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _explicit_scalar_transport_horizontal_transport_composition(
            scalar_new=scalar_new,
            scalar_tend_save_cells=scalar_tend_save_cells,
            uh_avg=uh_avg,
            horiz_flux_base=horiz_flux_base,
            edge_start=edge_start_v,
            edge_end=edge_end_v,
            edge_count=edge_count_v,
            slot_count=slot_count_v,
            adv_coef=adv_coef_v,
            adv_coef_3rd=adv_coef_3rd_v,
            slot_cells_matrix=slot_cells_matrix_v,
            slot_valid_matrix=slot_valid_matrix_v,
            slot_valid_matrix_all=slot_valid_matrix_all_v,
            use_full_width_patch_path=use_full_width_patch_path_v,
            fallback_rows=fallback_rows_v,
            one=one_v,
            cell_edges_0based=cell_edges_0based_v,
            cell_edges_local=cell_edges_local_v,
            cell_edges_in_patch=cell_edges_in_patch_v,
            cell_edges_in_patch_all=cell_edges_in_patch_all_v,
            edge_valid=edge_valid_v,
            edge_sign=edge_sign_v,
            edge_valid_all_by_slot=edge_valid_all_by_slot_v,
            inv_area_cells=inv_area_cells_v,
            do_physics=do_physics_v,
        )

    def _explicit_scalar_transport_bound_vertical_finish(
        *,
        scalar_old_cells: torch.Tensor,
        scalar_new_cells: torch.Tensor,
        scalar_tend_column: torch.Tensor,
        rho_zz_old_cells: torch.Tensor,
        rho_zz_new_cells: torch.Tensor,
        ww_cells: torch.Tensor,
        wdtn_interior: torch.Tensor,
    ) -> torch.Tensor:
        return _explicit_scalar_transport_vertical_finish_region(
            scalar_old_cells=scalar_old_cells,
            scalar_new_cells=scalar_new_cells,
            scalar_tend_column=scalar_tend_column,
            rho_zz_old_cells=rho_zz_old_cells,
            rho_zz_new_cells=rho_zz_new_cells,
            ww_cells=ww_cells,
            wdtn_interior=wdtn_interior,
            fnm=fnm_v,
            fnp=fnp_v,
            rdnw=rdnw_v,
            dt=dt_v,
            weight_time_old=weight_time_old_v,
            weight_time_new=weight_time_new_v,
            nVertLevels=nVertLevels_v,
            num_scalars=num_scalars_v,
        )

    _explicit_scalar_transport_bound_horizontal_transport.__name__ = (
        "_explicit_scalar_transport_bound_horizontal_transport"
    )
    _explicit_scalar_transport_bound_horizontal_transport.__qualname__ = (
        "_explicit_scalar_transport_bound_horizontal_transport"
    )
    _explicit_scalar_transport_bound_vertical_finish.__name__ = (
        "_explicit_scalar_transport_bound_vertical_finish"
    )
    _explicit_scalar_transport_bound_vertical_finish.__qualname__ = (
        "_explicit_scalar_transport_bound_vertical_finish"
    )
    section_ops = ExplicitScalarTransportSectionOps(
        horizontal_transport=_explicit_scalar_transport_bound_horizontal_transport,
        vertical_finish=_explicit_scalar_transport_bound_vertical_finish,
    )

    def _numeric_factory(
        bound_section_ops: ExplicitScalarTransportSectionOps,
    ) -> Callable[..., tuple[torch.Tensor, ...]]:
        def _explicit_scalar_transport_bound_numeric(
            scalar_old_cells: torch.Tensor,
            scalar_new: torch.Tensor,
            scalar_tend_save_cells: torch.Tensor,
            rho_zz_old_cells: torch.Tensor,
            rho_zz_new_cells: torch.Tensor,
            uh_avg: torch.Tensor,
            ww_cells: torch.Tensor,
            horiz_flux_base: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return _explicit_scalar_transport_numeric_composition(
                _section_ops=bound_section_ops,
                edge_start=edge_start_v,
                edge_end=edge_end_v,
                edge_count=edge_count_v,
                slot_count=slot_count_v,
                adv_coef=adv_coef_v,
                adv_coef_3rd=adv_coef_3rd_v,
                slot_cells_matrix=slot_cells_matrix_v,
                slot_valid_matrix=slot_valid_matrix_v,
                slot_valid_matrix_all=slot_valid_matrix_all_v,
                use_full_width_patch_path=use_full_width_patch_path_v,
                fallback_rows=fallback_rows_v,
                one=one_v,
                cell_edges_0based=cell_edges_0based_v,
                cell_edges_local=cell_edges_local_v,
                cell_edges_in_patch=cell_edges_in_patch_v,
                cell_edges_in_patch_all=cell_edges_in_patch_all_v,
                edge_valid=edge_valid_v,
                edge_sign=edge_sign_v,
                edge_valid_all_by_slot=edge_valid_all_by_slot_v,
                inv_area_cells=inv_area_cells_v,
                fnm=fnm_v,
                fnp=fnp_v,
                rdnw=rdnw_v,
                dt=dt_v,
                coef_3rd_order=coef_3rd_order_v,
                weight_time_old=weight_time_old_v,
                weight_time_new=weight_time_new_v,
                nVertLevels=nVertLevels_v,
                num_scalars=num_scalars_v,
                do_physics=do_physics_v,
                cell_start=cell_start_v,
                cell_end=cell_end_v,
                scalar_old_cells=scalar_old_cells,
                scalar_new=scalar_new,
                scalar_tend_save_cells=scalar_tend_save_cells,
                rho_zz_old_cells=rho_zz_old_cells,
                rho_zz_new_cells=rho_zz_new_cells,
                uh_avg=uh_avg,
                ww_cells=ww_cells,
                horiz_flux_base=horiz_flux_base,
            )

        _explicit_scalar_transport_bound_numeric.__name__ = (
            "_explicit_scalar_transport_bound_numeric"
        )
        _explicit_scalar_transport_bound_numeric.__qualname__ = (
            "_explicit_scalar_transport_bound_numeric"
        )
        return _explicit_scalar_transport_bound_numeric

    return ExplicitScalarTransportNumericPlan(
        key=key,
        dynamic_argument_names=_EXPLICIT_SCALAR_TRANSPORT_DYNAMIC_ARGUMENT_NAMES,
        static_arguments=static_arguments,
        section_ops=section_ops,
        numeric_factory=_numeric_factory,
        numeric_callable=_numeric_factory(section_ops),
    )


def compile_explicit_scalar_transport_numeric_plan(
    plan: ExplicitScalarTransportNumericPlan,
    *,
    compile_region_names: tuple[str, ...] | frozenset[str],
    compile_kernel: Callable[..., Callable[..., Any]],
    mode: str,
    backend: str,
) -> ExplicitScalarTransportNumericPlan:
    """Compile only strict explicit-scalar-transport section boundaries and rebuild the plan."""
    names = (
        frozenset(compile_region_names)
        & _EXPLICIT_SCALAR_TRANSPORT_REGION_COMPILE_TARGETS
    )
    if not names:
        return plan
    section_ops = ExplicitScalarTransportSectionOps(
        horizontal_transport=compile_kernel(
            "explicit_scalar_transport_horizontal_transport_composition",
            plan.section_ops.horizontal_transport,
            enabled="explicit_scalar_transport_horizontal_transport_composition"
            in names,
            mode=mode,
            backend=backend,
            specialization_key=(*plan.key, "horizontal_transport"),
        ),
        vertical_finish=compile_kernel(
            "explicit_scalar_transport_vertical_finish_region",
            plan.section_ops.vertical_finish,
            enabled="explicit_scalar_transport_vertical_finish_region" in names,
            mode=mode,
            backend=backend,
            specialization_key=(*plan.key, "vertical_finish"),
        ),
    )
    return plan.with_section_ops(section_ops)


def _explicit_scalar_transport_pack_public_rows(
    base: torch.Tensor,
    row_start: int,
    row_end: int,
    values: torch.Tensor,
) -> torch.Tensor:
    """Materialize one owned padded public result from a compact row patch."""
    if row_start == 0 and row_end == base.shape[0]:
        return values
    return torch.cat((base[:row_start], values, base[row_end:]), dim=0)


def atm_advance_scalars_work(
    nCells: int,
    nEdges: int,
    nVertLevels: int,
    num_scalars: int,
    maxEdges: int,
    nAdvCellsMax: int,
    dt: float,
    coef_3rd_order: float,
    edgeStart: int,
    edgeEnd: int,
    cellSolveStart: int,
    cellSolveEnd: int,
    rk_step: int,
    config_time_integration_order: int,
    advance_density: int,
    scalar_old: torch.Tensor,  # (nCells+1, nVertLevels, num_scalars)
    scalar_new: torch.Tensor,  # (nCells+1, nVertLevels, num_scalars) - input/output
    scalar_tend_save: torch.Tensor,  # (nCells+1, nVertLevels, num_scalars) - input/output
    rho_zz_old: torch.Tensor,  # (nCells+1, nVertLevels)
    rho_zz_new: torch.Tensor,  # (nCells+1, nVertLevels)
    uhAvg: torch.Tensor,  # (nEdges+1, nVertLevels)
    wwAvg: torch.Tensor,  # (nCells+1, nVertLevels+1)
    dvEdge: torch.Tensor,  # (nEdges+1,)
    invAreaCell: torch.Tensor,  # (nCells+1,)
    fnm: torch.Tensor,  # (nVertLevels,)
    fnp: torch.Tensor,  # (nVertLevels,)
    rdnw: torch.Tensor,  # (nVertLevels,)
    cellsOnEdge: torch.Tensor,  # (nEdges+1, 2) - contains 1-based cell indices
    edgesOnCell: torch.Tensor,  # (nCells+1, maxEdges) - contains 1-based edge indices
    nEdgesOnCell: torch.Tensor,  # (nCells+1,)
    edgesOnCell_sign: torch.Tensor,  # (nCells+1, maxEdges)
    bdyMaskCell: torch.Tensor,  # (nCells+1,)
    bdyMaskEdge: torch.Tensor,  # (nEdges+1,)
    nAdvCellsForEdge: torch.Tensor,  # (nEdges+1,)
    advCellsForEdge: torch.Tensor,  # (nEdges+1, nAdvCellsMax) - contains 1-based cell indices
    adv_coefs: torch.Tensor,  # (nEdges+1, nAdvCellsMax)
    adv_coefs_3rd: torch.Tensor,  # (nEdges+1, nAdvCellsMax)
    horiz_flux_arr: torch.Tensor,  # (nEdges+1, nVertLevels, num_scalars) - input/output
    do_physics: bool = False,
    _compile_region_names: tuple[str, ...] | frozenset[str] | None = None,
    _compile_mode: str = "reduce-overhead",
    _compile_backend: str = "inductor",
    _connectivity_payload: dict[str, Any] | None = None,
    _numeric_plan: ExplicitScalarTransportNumericPlan | None = None,
) -> dict:
    """
    Advance scalar fields using explicit time stepping with flux-form advection.

    This is a Runge-Kutta step: fluxes are computed from scalar_new values,
    and the update starts from scalar_old.

    Args:
        nCells: Number of cells in mesh
        nEdges: Number of edges in mesh
        nVertLevels: Number of vertical levels
        num_scalars: Number of scalar fields to advance
        maxEdges: Maximum edges per cell
        nAdvCellsMax: Maximum advection cells per edge
        dt: Timestep (seconds)
        coef_3rd_order: Third-order upwind coefficient (typically 0.25)
        edgeStart: First edge to process (1-indexed Fortran convention)
        edgeEnd: Last edge to process (1-indexed Fortran convention)
        cellSolveStart: First cell to update (1-indexed Fortran convention)
        cellSolveEnd: Last cell to update (1-indexed Fortran convention)
        rk_step: Current Runge-Kutta substep (1, 2, or 3)
        config_time_integration_order: Time integration order (2 or 3)
        advance_density: Whether density is being advanced (1=true, 0=false)
        scalar_old: Scalar values at beginning of RK step
        scalar_new: Scalar values to compute fluxes from (updated on output)
        scalar_tend_save: Saved scalar tendencies
        rho_zz_old: Old dry density
        rho_zz_new: New dry density
        uhAvg: Time-averaged horizontal mass flux at edges
        wwAvg: Time-averaged vertical mass flux
        dvEdge: Edge lengths
        invAreaCell: Inverse cell areas
        fnm, fnp: Vertical interpolation weights (negative/positive)
        rdnw: Inverse vertical layer thickness
        cellsOnEdge: Cells adjacent to each edge (contains 1-based indices)
        edgesOnCell: Edges bounding each cell (contains 1-based indices)
        nEdgesOnCell: Number of edges for each cell
        edgesOnCell_sign: Sign of edge normal relative to cell
        bdyMaskCell: Cell boundary mask
        bdyMaskEdge: Edge boundary mask
        nAdvCellsForEdge: Number of advection cells per edge
        advCellsForEdge: Advection cell indices per edge (contains 1-based indices)
        adv_coefs: Advection coefficients
        adv_coefs_3rd: Third-order advection coefficients
        horiz_flux_arr: Horizontal flux array (updated on output)

    Returns:
        Dictionary containing:
            - scalar_new: Updated scalar values
            - scalar_tend_save: Scalar tendencies
            - horiz_flux_arr: Horizontal fluxes at edges
    """
    device = require_consistent_device(
        "atm_advance_scalars_work",
        scalar_old,
        scalar_new,
        scalar_tend_save,
        rho_zz_old,
        rho_zz_new,
        uhAvg,
        wwAvg,
        dvEdge,
        invAreaCell,
        fnm,
        fnp,
        rdnw,
        cellsOnEdge,
        edgesOnCell,
        nEdgesOnCell,
        edgesOnCell_sign,
        bdyMaskCell,
        bdyMaskEdge,
        nAdvCellsForEdge,
        advCellsForEdge,
        adv_coefs,
        adv_coefs_3rd,
        horiz_flux_arr,
    )
    compile_region_names = (
        frozenset(_compile_region_names or ())
        & _EXPLICIT_SCALAR_TRANSPORT_REGION_COMPILE_TARGETS
    )
    cell_start = cellSolveStart - 1
    cell_end = cellSolveEnd
    weight_time_old, weight_time_new = _explicit_scalar_transport_density_time_weights(
        advance_density=advance_density,
        rk_step=rk_step,
        config_time_integration_order=config_time_integration_order,
    )

    edge_start = edgeStart - 1
    edge_end = edgeEnd
    if _connectivity_payload is None:
        conn = _explicit_scalar_transport_connectivity_payload(
            device=device,
            nCells=nCells,
            nEdges=nEdges,
            maxEdges=maxEdges,
            nAdvCellsMax=nAdvCellsMax,
            edge_start=edge_start,
            edge_end=edge_end,
            cell_solve_start=cell_start,
            cell_solve_end=cell_end,
            edgesOnCell=edgesOnCell,
            nEdgesOnCell=nEdgesOnCell,
            edgesOnCell_sign=edgesOnCell_sign,
            nAdvCellsForEdge=nAdvCellsForEdge,
            advCellsForEdge=advCellsForEdge,
            adv_coefs=adv_coefs,
            adv_coefs_3rd=adv_coefs_3rd,
        )
    else:
        conn = _connectivity_payload
    cell_range = conn.get("cell_range")
    cell_slice = (
        cell_range.slice if isinstance(cell_range, StaticRange) else conn["cell_slice"]
    )
    edge_range = conn.get("edge_range")
    edge_slice = (
        edge_range.slice if isinstance(edge_range, StaticRange) else conn["edge_slice"]
    )
    if _numeric_plan is None:
        numeric_plan = build_explicit_scalar_transport_numeric_plan(
            connectivity_payload=conn,
            inv_area_cells=invAreaCell[cell_slice],
            fnm=fnm,
            fnp=fnp,
            rdnw=rdnw,
            dt=dt,
            coef_3rd_order=coef_3rd_order,
            weight_time_old=weight_time_old,
            weight_time_new=weight_time_new,
            nVertLevels=nVertLevels,
            num_scalars=num_scalars,
            do_physics=do_physics,
            stage_key=(
                rk_step,
                config_time_integration_order,
                int(advance_density),
                cell_start,
                cell_end,
                edge_start,
                edge_end,
            ),
        )
    else:
        numeric_plan = _numeric_plan

    if compile_region_names:
        from mpas_pytorch.dynamics.compile_utils import maybe_compile_kernel

        numeric_plan = compile_explicit_scalar_transport_numeric_plan(
            numeric_plan,
            compile_region_names=compile_region_names,
            compile_kernel=maybe_compile_kernel,
            mode=_compile_mode,
            backend=_compile_backend,
        )

    scalar_new_patch, scalar_tend_patch, horiz_flux = numeric_plan.numeric_callable(
        scalar_old[cell_slice, :, :],
        scalar_new,
        scalar_tend_save[cell_slice, :, :],
        rho_zz_old[cell_slice, :],
        rho_zz_new[cell_slice, :],
        uhAvg,
        wwAvg[cell_slice, :],
        horiz_flux_arr,
    )
    edge_start_idx = 0 if edge_slice.start is None else edge_slice.start
    edge_stop_idx = nEdges if edge_slice.stop is None else edge_slice.stop
    scalar_new_out = _explicit_scalar_transport_pack_public_rows(
        scalar_new, cell_start, cell_end, scalar_new_patch
    )
    scalar_tend_save_out = _explicit_scalar_transport_pack_public_rows(
        scalar_tend_save, cell_start, cell_end, scalar_tend_patch
    )
    horiz_flux_arr_out = _explicit_scalar_transport_pack_public_rows(
        horiz_flux_arr, edge_start_idx, edge_stop_idx, horiz_flux
    )

    return {
        "scalar_new": scalar_new_out,
        "scalar_tend_save": scalar_tend_save_out,
        "horiz_flux_arr": horiz_flux_arr_out,
    }
