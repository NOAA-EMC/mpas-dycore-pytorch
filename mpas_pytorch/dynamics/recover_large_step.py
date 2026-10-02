"""
Large timestep recovery (large-step-recovery).

Recovers the full state variables from perturbations after acoustic substeps.
This is the outer loop of split-explicit time integration.


Subroutine: atm_recover_large_step_variables_work
"""

import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

import torch
import torch.nn.functional as F

from mpas_pytorch.autodiff import is_grad_mode
from mpas_pytorch.autodiff.soft_limiter import fct_sign_pm
from mpas_pytorch.constants import DTYPE, PhysicalConstants
from mpas_pytorch.dynamics.static_payloads import (
    StaticPairRelation,
    StaticRange,
    StaticSlotRelation,
)
from mpas_pytorch.dynamics.stencil_ops import (
    _accumulate_ordered_slot,
    _active_slot_count,
    _clone_for_grad,
    _replace_row_slice,
    _replace_row_slice_cols,
    _static_connectivity_cache_allowed,
    _static_mask_all_true,
    _static_slot_all_true,
    _tensor_version_cache_key,
    gather_rows,
)
from mpas_pytorch.mesh import MESHState
from mpas_pytorch.utils.checks import require_consistent_device


def _mpas_debug_enabled() -> bool:
    """Return True when expensive runtime validations are enabled."""
    return os.environ.get("MPAS_DEBUG", "0") == "1"


def _large_step_recovery_static_all_true(
    mask: torch.Tensor, *, allow_cuda_sync: bool = False
) -> bool:
    return bool(_static_mask_all_true(mask, allow_cuda_sync=allow_cuda_sync))


def _large_step_recovery_slot_all_true(
    mask: torch.Tensor,
    *,
    allow_cuda_sync: bool = False,
) -> tuple[bool, ...]:
    metadata = _static_slot_all_true(mask, allow_cuda_sync=allow_cuda_sync)
    if metadata is None:
        return tuple(False for _ in range(mask.shape[1]))
    return metadata


def _large_step_recovery_long_index(index: torch.Tensor) -> torch.Tensor:
    return index if index.dtype == torch.int64 else index.to(dtype=torch.int64)


def _large_step_recovery_tensor_has_forward_ad(tensor: object) -> bool:
    """Return True when ``tensor`` carries a forward-AD tangent."""
    if not isinstance(tensor, torch.Tensor):
        return False
    try:
        _primal, tangent = torch.autograd.forward_ad.unpack_dual(tensor)
    except RuntimeError:
        return False
    return tangent is not None


_LARGE_STEP_RECOVERY_REGION_COMPILE_TARGETS = frozenset(
    {
        "large_step_recovery_recovery_region",
    }
)
_LARGE_STEP_RECOVERY_RECOVERY_DYNAMIC_NAMES = (
    "rho_p_save_cells",
    "rho_pp_cells",
    "rho_base_cells",
    "rtheta_p_save_cells",
    "rtheta_pp_cells",
    "rtheta_base_cells",
    "rt_diabatic_tend_cells",
    "exner_base_cells",
    "zz_cells",
    "ww_avg_cells",
    "rw_save_cells",
    "rw_p_cells",
    "ru_avg_edges",
    "ru_save_edges",
    "ru_p_edges",
    "zb_cell_solve",
    "zb3_cell_solve",
)


@dataclass(frozen=True)
class LargeStepRecoveryPlan:
    """Immutable static binding for the compact large-step-recovery recovery region."""

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
                f"large-step-recovery numeric call is missing dynamic arguments: {missing}"
            )
        return tuple(call_arguments[name] for name in self.dynamic_argument_names)

    def with_numeric_callable(
        self,
        numeric_callable: Callable[..., tuple[torch.Tensor, ...]],
    ) -> "LargeStepRecoveryPlan":
        return replace(self, numeric_callable=numeric_callable)


_LARGE_STEP_RECOVERY_CONNECTIVITY_TENSOR_NAMES = (
    "cellsOnEdge",
    "edgesOnCell",
    "nEdgesOnCell",
    "edgesOnCell_sign",
    "bdyMaskCell",
)


def _large_step_recovery_tensor_cache_key(
    tensor: torch.Tensor,
) -> tuple[object, ...] | None:
    return _tensor_version_cache_key(tensor)


def _large_step_recovery_connectivity_payload_cache_key(
    **kwargs: object,
) -> tuple[object, ...] | None:
    """Return the mutation-sensitive cache key for a large-step-recovery static payload."""
    tensors = tuple(
        kwargs.get(name) for name in _LARGE_STEP_RECOVERY_CONNECTIVITY_TENSOR_NAMES
    )
    if not _static_connectivity_cache_allowed(tensors):
        return None

    tensor_keys: list[tuple[object, ...]] = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            tensor_key = _large_step_recovery_tensor_cache_key(tensor)
            if tensor_key is None:
                return None
            tensor_keys.append(tensor_key)

    return (
        kwargs["device"],
        kwargs["nCells"],
        kwargs["nEdges"],
        kwargs["maxEdges"],
        kwargs["nRelaxZone"],
        kwargs["cell_start"],
        kwargs["cell_end"],
        kwargs["edge_start"],
        kwargs["edge_end"],
        *tensor_keys,
    )


def _build_large_step_recovery_connectivity_payload(
    *,
    device: torch.device,
    nCells: int,
    nEdges: int,
    maxEdges: int,
    nRelaxZone: int,
    cell_start: int,
    cell_end: int,
    edge_start: int,
    edge_end: int,
    cellsOnEdge: torch.Tensor,
    edgesOnCell: torch.Tensor,
    nEdgesOnCell: torch.Tensor,
    edgesOnCell_sign: torch.Tensor,
    bdyMaskCell: torch.Tensor,
) -> dict[str, Any]:
    """Build reusable large-step-recovery edge-pair and cell-edge recovery topology."""
    cell_slice = slice(cell_start, cell_end)
    edge_slice = slice(edge_start, edge_end)

    cell1_1based = cellsOnEdge[edge_slice, 0]
    cell2_1based = cellsOnEdge[edge_slice, 1]
    edge_valid = (
        (cell1_1based > 0)
        & (cell1_1based <= nCells)
        & (cell2_1based > 0)
        & (cell2_1based <= nCells)
    )
    c1 = _large_step_recovery_long_index(
        torch.where(edge_valid, cell1_1based - 1, torch.zeros_like(cell1_1based))
    )
    c2 = _large_step_recovery_long_index(
        torch.where(edge_valid, cell2_1based - 1, torch.zeros_like(cell2_1based))
    )
    c1_in_cell_patch = (c1 >= cell_start) & (c1 < cell_end)
    c2_in_cell_patch = (c2 >= cell_start) & (c2 < cell_end)
    c1_local = _large_step_recovery_long_index(
        (c1 - cell_start).clamp(min=0, max=max(cell_end - cell_start - 1, 0))
    )
    c2_local = _large_step_recovery_long_index(
        (c2 - cell_start).clamp(min=0, max=max(cell_end - cell_start - 1, 0))
    )

    active_mask = bdyMaskCell[cell_slice] <= nRelaxZone
    active_slot_count = _active_slot_count(nEdgesOnCell[cell_slice], maxEdges, device)
    edge_slots = torch.arange(active_slot_count, dtype=torch.int64, device=device)
    cell_edges_1based = edgesOnCell[cell_slice, :active_slot_count]
    cell_edges0 = cell_edges_1based - 1
    slot_valid = edge_slots[None, :] < nEdgesOnCell[cell_slice][:, None]
    edge_valid_2d = (cell_edges0 >= 0) & (cell_edges0 < nEdges)
    valid = slot_valid & edge_valid_2d & active_mask[:, None]
    safe_edge = _large_step_recovery_long_index(
        cell_edges0.clamp(min=0, max=max(nEdges - 1, 0))
    )
    edge_in_edge_patch = (safe_edge >= edge_start) & (safe_edge < edge_end)
    safe_edge_local = _large_step_recovery_long_index(
        (safe_edge - edge_start).clamp(min=0, max=max(edge_end - edge_start - 1, 0))
    )
    edge_valid_all = _large_step_recovery_static_all_true(
        edge_valid, allow_cuda_sync=True
    )
    active_mask_all = _large_step_recovery_static_all_true(
        active_mask, allow_cuda_sync=True
    )
    valid_all_by_slot = _large_step_recovery_slot_all_true(valid, allow_cuda_sync=True)
    edge_sign = edgesOnCell_sign[cell_slice, :active_slot_count]
    signed_valid_edge_weight = edge_sign * valid.to(dtype=DTYPE)
    cell_range = StaticRange("large_step_recovery.cells", cell_start, cell_end)
    edge_range = StaticRange("large_step_recovery.edges", edge_start, edge_end)
    edge_pair_cells = torch.stack((c1, c2), dim=1)
    edge_pair_relation = StaticPairRelation(
        "large_step_recovery.edge_cells",
        first=c1,
        second=c2,
        valid=edge_valid,
        valid_all=edge_valid_all,
    )
    cell_edge_relation = StaticSlotRelation(
        "large_step_recovery.cell_edges",
        indices=safe_edge,
        valid=valid,
        sign=edge_sign,
        weights=signed_valid_edge_weight,
        valid_all_by_slot=valid_all_by_slot,
    )

    return {
        "cell_slice": cell_slice,
        "cell_range": cell_range,
        "edge_slice": edge_slice,
        "edge_range": edge_range,
        "edge_valid": edge_valid,
        "edge_valid_all": edge_valid_all,
        "c1": c1,
        "c2": c2,
        "c1_local": c1_local,
        "c2_local": c2_local,
        "c1_in_cell_patch": c1_in_cell_patch,
        "c2_in_cell_patch": c2_in_cell_patch,
        "c1_in_cell_patch_all": _large_step_recovery_static_all_true(
            c1_in_cell_patch, allow_cuda_sync=True
        ),
        "c2_in_cell_patch_all": _large_step_recovery_static_all_true(
            c2_in_cell_patch, allow_cuda_sync=True
        ),
        "edge_pair_cells": edge_pair_cells,
        "edge_pair_relation": edge_pair_relation,
        "active_mask": active_mask,
        "active_mask_all": active_mask_all,
        "safe_edge": safe_edge,
        "safe_edge_local": safe_edge_local,
        "edge_in_edge_patch": edge_in_edge_patch,
        "edge_in_edge_patch_all": _large_step_recovery_static_all_true(
            edge_in_edge_patch, allow_cuda_sync=True
        ),
        "valid": valid,
        "valid_all_by_slot": valid_all_by_slot,
        "edge_sign": edge_sign,
        "signed_valid_edge_weight": signed_valid_edge_weight,
        "cell_edge_relation": cell_edge_relation,
        "active_slot_count": active_slot_count,
    }


def _large_step_recovery_connectivity_payload(**kwargs: object) -> dict[str, Any]:
    """Build large-step-recovery topology for direct callers; the runner owns warm reuse."""
    return _build_large_step_recovery_connectivity_payload(**kwargs)  # type: ignore[arg-type]


def _large_step_recovery_gather_compact_patch(
    patch: torch.Tensor,
    local_indices: torch.Tensor,
    in_patch: torch.Tensor,
    *,
    all_in_patch: bool,
) -> torch.Tensor:
    """Gather compact rows with explicit zero fill outside the owned patch."""
    gathered = gather_rows(patch, local_indices)
    if all_in_patch:
        return gathered
    mask = in_patch.reshape(
        *in_patch.shape,
        *([1] * (gathered.ndim - in_patch.ndim)),
    )
    return gathered.masked_fill(~mask, 0)


def _sum_w_recovery_compact_slots(
    *,
    ru_patch: torch.Tensor,
    safe_edge_local: torch.Tensor,
    edge_in_edge_patch: torch.Tensor,
    edge_in_edge_patch_all: bool,
    valid: torch.Tensor,
    edge_sign: torch.Tensor,
    zb_cell_solve: torch.Tensor,
    zb3_cell_solve: torch.Tensor,
    fzm: torch.Tensor,
    fzp: torch.Tensor,
    cf1: float,
    cf2: float,
    cf3: float,
    nVertLevels: int,
    valid_all_by_slot: tuple[bool, ...] | None,
    signed_valid_edge_weight: torch.Tensor | None,
    interior_initial: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply ordered large-step-recovery flux terms directly from the compact edge patch.

    When ``interior_initial`` is supplied, interior edge contributions are
    added to that partially recovered vertical velocity in upstream statement
    order.  Returning a separately reduced sum and adding it afterward changes
    the binary64 association before the density normalization.
    """
    n_cells, n_slots = safe_edge_local.shape
    surface_total: torch.Tensor | None = None
    interior_total = interior_initial
    for slot in range(n_slots):
        ru_slot = _large_step_recovery_gather_compact_patch(
            ru_patch,
            safe_edge_local[:, slot],
            edge_in_edge_patch[:, slot],
            all_in_patch=edge_in_edge_patch_all,
        )
        valid_all = valid_all_by_slot is not None and valid_all_by_slot[slot]
        flux_surface = cf1 * ru_slot[:, 0] + cf2 * ru_slot[:, 1] + cf3 * ru_slot[:, 2]
        coef_surface = (
            zb_cell_solve[:, slot, 0]
            + fct_sign_pm(flux_surface) * zb3_cell_solve[:, slot, 0]
        )
        if signed_valid_edge_weight is not None:
            surface_contrib = (
                signed_valid_edge_weight[:, slot] * coef_surface * flux_surface
            )
        else:
            surface_contrib = edge_sign[:, slot] * coef_surface * flux_surface
            if not valid_all:
                surface_contrib = surface_contrib * valid[:, slot].to(
                    dtype=ru_patch.dtype
                )
        surface_total = _accumulate_ordered_slot(surface_total, surface_contrib)

        if nVertLevels > 1:
            flux_interior = (
                fzm[1:nVertLevels][None, :] * ru_slot[:, 1:nVertLevels]
                + fzp[1:nVertLevels][None, :] * ru_slot[:, : nVertLevels - 1]
            )
            coef_interior = (
                zb_cell_solve[:, slot, 1:nVertLevels]
                + fct_sign_pm(flux_interior) * zb3_cell_solve[:, slot, 1:nVertLevels]
            )
            if signed_valid_edge_weight is not None:
                interior_contrib = (
                    signed_valid_edge_weight[:, slot].unsqueeze(1)
                    * coef_interior
                    * flux_interior
                )
            else:
                interior_contrib = (
                    edge_sign[:, slot].unsqueeze(1) * coef_interior * flux_interior
                )
                if not valid_all:
                    interior_contrib = interior_contrib * valid[:, slot].view(
                        n_cells, 1
                    ).to(dtype=ru_patch.dtype)
            interior_total = _accumulate_ordered_slot(interior_total, interior_contrib)

    if surface_total is None:
        surface_total = ru_patch.new_zeros(n_cells)
    return surface_total, interior_total


def _large_step_recovery_density_theta_recovery_region(
    *,
    rho_p_save_cells: torch.Tensor,
    rho_pp_cells: torch.Tensor,
    rho_base_cells: torch.Tensor,
    rtheta_p_save_cells: torch.Tensor,
    rtheta_pp_cells: torch.Tensor,
    rtheta_base_cells: torch.Tensor,
    rt_diabatic_tend_cells: torch.Tensor,
    exner_base_cells: torch.Tensor,
    zz_cells: torch.Tensor,
    dt: float,
    rk_step: int,
    rgas: float,
    cp: float,
    p0: float,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Recover cell density/theta patches for the active large-step-recovery cell range."""
    rho_p_cell = rho_p_save_cells + rho_pp_cells
    rho_zz_cell = rho_p_cell + rho_base_cells

    if rk_step == 3:
        rtheta_p_cell = (
            rtheta_p_save_cells
            + rtheta_pp_cells
            - dt * rho_zz_cell * rt_diabatic_tend_cells
        )
    else:
        rtheta_p_cell = rtheta_p_save_cells + rtheta_pp_cells

    theta_cell = (rtheta_p_cell + rtheta_base_cells) / rho_zz_cell

    rcv = rgas / (cp - rgas)
    if rk_step == 3:
        exner_cell = (
            zz_cells * (rgas / p0) * (rtheta_p_cell + rtheta_base_cells)
        ) ** rcv
        pressure_cell = (
            zz_cells
            * rgas
            * (
                exner_cell * rtheta_p_cell
                + rtheta_base_cells * (exner_cell - exner_base_cells)
            )
        )
    else:
        exner_cell = torch.zeros_like(rtheta_p_cell)
        pressure_cell = torch.zeros_like(rtheta_p_cell)

    return rho_p_cell, rho_zz_cell, rtheta_p_cell, theta_cell, exner_cell, pressure_cell


def _large_step_recovery_density_theta_recovery_no_thermo_region(
    *,
    rho_p_save_cells: torch.Tensor,
    rho_pp_cells: torch.Tensor,
    rho_base_cells: torch.Tensor,
    rtheta_p_save_cells: torch.Tensor,
    rtheta_pp_cells: torch.Tensor,
    rtheta_base_cells: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recover RK1/RK2 cell fields without allocating unused thermo patches."""
    rho_p_cell = rho_p_save_cells + rho_pp_cells
    rho_zz_cell = rho_p_cell + rho_base_cells
    rtheta_p_cell = rtheta_p_save_cells + rtheta_pp_cells
    theta_cell = (rtheta_p_cell + rtheta_base_cells) / rho_zz_cell
    return rho_p_cell, rho_zz_cell, rtheta_p_cell, theta_cell


def _large_step_recovery_edge_momentum_recovery_region(
    *,
    ru_save_edges: torch.Tensor,
    ru_avg_edges: torch.Tensor,
    ru_p_edges: torch.Tensor,
    rho_zz: torch.Tensor,
    c1: torch.Tensor,
    c2: torch.Tensor,
    edge_valid: torch.Tensor,
    edge_valid_all: bool,
    invNs: float,
    rho_zz_cell1: torch.Tensor | None = None,
    rho_zz_cell2: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recover edge momentum/velocity patches for the active large-step-recovery edge range."""
    ru_avg_new = ru_save_edges + (ru_avg_edges * invNs)
    ru_new = ru_save_edges + ru_p_edges
    if rho_zz_cell1 is None or rho_zz_cell2 is None:
        rho_zz_cell1 = gather_rows(rho_zz, c1)
        rho_zz_cell2 = gather_rows(rho_zz, c2)
    rho_sum = rho_zz_cell1 + rho_zz_cell2
    u_new = 2.0 * ru_new / rho_sum
    if edge_valid_all:
        return ru_avg_new, ru_new, u_new
    return (
        torch.where(edge_valid[:, None], ru_avg_new, ru_avg_edges),
        torch.where(edge_valid[:, None], ru_new, torch.zeros_like(ru_new)),
        torch.where(edge_valid[:, None], u_new, torch.zeros_like(u_new)),
    )


def _large_step_recovery_recovery_numeric(
    *,
    nVertLevels: int,
    rk_step: int,
    dt: float,
    rgas: float,
    cp: float,
    p0: float,
    invNs: float,
    cf1: float,
    cf2: float,
    cf3: float,
    fzm: torch.Tensor,
    fzp: torch.Tensor,
    c1_local: torch.Tensor,
    c2_local: torch.Tensor,
    c1_in_cell_patch: torch.Tensor,
    c2_in_cell_patch: torch.Tensor,
    c1_in_cell_patch_all: bool,
    c2_in_cell_patch_all: bool,
    edge_valid: torch.Tensor,
    edge_valid_all: bool,
    safe_edge_local: torch.Tensor,
    edge_in_edge_patch: torch.Tensor,
    edge_in_edge_patch_all: bool,
    valid: torch.Tensor,
    edge_sign: torch.Tensor,
    active_mask: torch.Tensor,
    active_mask_all: bool,
    valid_all_by_slot: tuple[bool, ...] | None,
    signed_valid_edge_weight: torch.Tensor | None,
    rho_p_save_cells: torch.Tensor,
    rho_pp_cells: torch.Tensor,
    rho_base_cells: torch.Tensor,
    rtheta_p_save_cells: torch.Tensor,
    rtheta_pp_cells: torch.Tensor,
    rtheta_base_cells: torch.Tensor,
    rt_diabatic_tend_cells: torch.Tensor,
    exner_base_cells: torch.Tensor,
    zz_cells: torch.Tensor,
    ww_avg_cells: torch.Tensor,
    rw_save_cells: torch.Tensor,
    rw_p_cells: torch.Tensor,
    ru_avg_edges: torch.Tensor,
    ru_save_edges: torch.Tensor,
    ru_p_edges: torch.Tensor,
    zb_cell_solve: torch.Tensor,
    zb3_cell_solve: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Recover all compact large-step-recovery output patches in one tensor-only region."""
    if rk_step == 3:
        (
            rho_p_cell,
            rho_zz_cell,
            rtheta_p_cell,
            theta_cell,
            exner_cell,
            pressure_cell,
        ) = _large_step_recovery_density_theta_recovery_region(
            rho_p_save_cells=rho_p_save_cells,
            rho_pp_cells=rho_pp_cells,
            rho_base_cells=rho_base_cells,
            rtheta_p_save_cells=rtheta_p_save_cells,
            rtheta_pp_cells=rtheta_pp_cells,
            rtheta_base_cells=rtheta_base_cells,
            rt_diabatic_tend_cells=rt_diabatic_tend_cells,
            exner_base_cells=exner_base_cells,
            zz_cells=zz_cells,
            dt=dt,
            rk_step=rk_step,
            rgas=rgas,
            cp=cp,
            p0=p0,
        )
    else:
        rho_p_cell, rho_zz_cell, rtheta_p_cell, theta_cell = (
            _large_step_recovery_density_theta_recovery_no_thermo_region(
                rho_p_save_cells=rho_p_save_cells,
                rho_pp_cells=rho_pp_cells,
                rho_base_cells=rho_base_cells,
                rtheta_p_save_cells=rtheta_p_save_cells,
                rtheta_pp_cells=rtheta_pp_cells,
                rtheta_base_cells=rtheta_base_cells,
            )
        )
        exner_cell = rtheta_p_cell
        pressure_cell = rtheta_p_cell

    rho_zz_cell1 = _large_step_recovery_gather_compact_patch(
        rho_zz_cell,
        c1_local,
        c1_in_cell_patch,
        all_in_patch=c1_in_cell_patch_all,
    )
    rho_zz_cell2 = _large_step_recovery_gather_compact_patch(
        rho_zz_cell,
        c2_local,
        c2_in_cell_patch,
        all_in_patch=c2_in_cell_patch_all,
    )
    ru_avg_patch, ru_patch, u_patch = (
        _large_step_recovery_edge_momentum_recovery_region(
            ru_save_edges=ru_save_edges,
            ru_avg_edges=ru_avg_edges,
            ru_p_edges=ru_p_edges,
            rho_zz=rho_zz_cell,
            c1=c1_local,
            c2=c2_local,
            edge_valid=edge_valid,
            edge_valid_all=edge_valid_all,
            invNs=invNs,
            rho_zz_cell1=rho_zz_cell1,
            rho_zz_cell2=rho_zz_cell2,
        )
    )

    if nVertLevels > 1:
        ww_avg_mid = rw_save_cells[:, 1:nVertLevels] + (
            ww_avg_cells[:, 1:nVertLevels] * invNs
        )
        rw_mid = rw_save_cells[:, 1:nVertLevels] + rw_p_cells[:, 1:nVertLevels]
        zz_interp = (
            fzm[1:nVertLevels][None, :] * zz_cells[:, 1:nVertLevels]
            + fzp[1:nVertLevels][None, :] * zz_cells[:, : nVertLevels - 1]
        )
        w_mid = rw_mid / zz_interp
    else:
        ww_avg_mid = rw_save_cells[:, 1:1]
        rw_mid = rw_save_cells[:, 1:1]
        w_mid = rw_save_cells[:, 1:1]

    surface_flux_sum, interior_flux_sum = _sum_w_recovery_compact_slots(
        ru_patch=ru_patch,
        safe_edge_local=safe_edge_local,
        edge_in_edge_patch=edge_in_edge_patch,
        edge_in_edge_patch_all=edge_in_edge_patch_all,
        valid=valid,
        edge_sign=edge_sign,
        zb_cell_solve=zb_cell_solve,
        zb3_cell_solve=zb3_cell_solve,
        fzm=fzm,
        fzp=fzp,
        cf1=cf1,
        cf2=cf2,
        cf3=cf3,
        nVertLevels=nVertLevels,
        valid_all_by_slot=valid_all_by_slot,
        signed_valid_edge_weight=signed_valid_edge_weight,
        interior_initial=w_mid,
    )
    denom_surface = (
        cf1 * rho_zz_cell[:, 0] + cf2 * rho_zz_cell[:, 1] + cf3 * rho_zz_cell[:, 2]
    )
    w_surface_norm = surface_flux_sum / denom_surface
    w_surface_final = (
        w_surface_norm
        if active_mask_all
        else torch.where(active_mask, w_surface_norm, torch.zeros_like(w_surface_norm))
    )

    if nVertLevels > 1:
        if interior_flux_sum is None:
            raise RuntimeError(
                "large-step-recovery interior recovery did not return active levels"
            )
        w_interior_with_flux = interior_flux_sum
        denom = (
            fzm[1:nVertLevels][None, :] * rho_zz_cell[:, 1:nVertLevels]
            + fzp[1:nVertLevels][None, :] * rho_zz_cell[:, : nVertLevels - 1]
        )
        w_interior_norm = w_interior_with_flux / denom
        w_interior_final = (
            w_interior_norm
            if active_mask_all
            else torch.where(active_mask[:, None], w_interior_norm, w_mid)
        )
    else:
        w_interior_final = w_mid

    return (
        rho_p_cell,
        rho_zz_cell,
        rtheta_p_cell,
        theta_cell,
        exner_cell,
        pressure_cell,
        ww_avg_mid,
        rw_mid,
        w_surface_final,
        w_interior_final,
        ru_avg_patch,
        ru_patch,
        u_patch,
    )


def _large_step_recovery_plan_tensor_fingerprint(
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


def large_step_recovery_recovery_plan_owner_key(
    *,
    connectivity_identity: object,
    nVertLevels: int,
    ns: int,
    rk_step: int,
    dt: float,
    cf1: float,
    cf2: float,
    cf3: float,
    fzm: torch.Tensor,
    fzp: torch.Tensor,
    stage_key: tuple[object, ...] = (),
) -> tuple[object, ...]:
    """Return the exact large-step-recovery runner key without constructing a numeric closure."""
    if ns <= 0:
        raise ValueError(f"ns must be > 0, got {ns}")
    return (
        "LargeStepRecoveryPlan.v2",
        tuple(stage_key),
        connectivity_identity,
        int(nVertLevels),
        int(rk_step),
        float(dt),
        1.0 / float(ns),
        float(cf1),
        float(cf2),
        float(cf3),
        float(PhysicalConstants.RD),
        float(PhysicalConstants.CP),
        1.0e05,
        _large_step_recovery_plan_tensor_fingerprint(fzm),
        _large_step_recovery_plan_tensor_fingerprint(fzp),
    )


def _large_step_recovery_connectivity_plan_fingerprint(
    connectivity_payload: Mapping[str, Any],
) -> tuple[object, ...]:
    """Fingerprint direct-call topology when no runner cache key is available."""
    names = (
        "c1_local",
        "c2_local",
        "c1_in_cell_patch",
        "c2_in_cell_patch",
        "edge_valid",
        "safe_edge_local",
        "edge_in_edge_patch",
        "valid",
        "edge_sign",
        "active_mask",
        "signed_valid_edge_weight",
    )
    return tuple(
        (name, _large_step_recovery_plan_tensor_fingerprint(value))
        for name in names
        if isinstance((value := connectivity_payload.get(name)), torch.Tensor)
    )


def build_large_step_recovery_recovery_plan(
    *,
    connectivity_payload: Mapping[str, Any],
    nVertLevels: int,
    ns: int,
    rk_step: int,
    dt: float,
    cf1: float,
    cf2: float,
    cf3: float,
    fzm: torch.Tensor,
    fzp: torch.Tensor,
    stage_key: tuple[object, ...] = (),
    connectivity_key: tuple[object, ...] | None = None,
) -> LargeStepRecoveryPlan:
    """Bind large-step-recovery topology/scalars and expose a compact tensor-only callable."""
    if ns <= 0:
        raise ValueError(f"ns must be > 0, got {ns}")
    static_arguments = MappingProxyType(
        {
            "nVertLevels": int(nVertLevels),
            "rk_step": int(rk_step),
            "dt": float(dt),
            "rgas": float(PhysicalConstants.RD),
            "cp": float(PhysicalConstants.CP),
            "p0": 1.0e05,
            "invNs": 1.0 / float(ns),
            "cf1": float(cf1),
            "cf2": float(cf2),
            "cf3": float(cf3),
            "fzm": fzm,
            "fzp": fzp,
            "c1_local": connectivity_payload["c1_local"],
            "c2_local": connectivity_payload["c2_local"],
            "c1_in_cell_patch": connectivity_payload["c1_in_cell_patch"],
            "c2_in_cell_patch": connectivity_payload["c2_in_cell_patch"],
            "c1_in_cell_patch_all": bool(connectivity_payload["c1_in_cell_patch_all"]),
            "c2_in_cell_patch_all": bool(connectivity_payload["c2_in_cell_patch_all"]),
            "edge_valid": connectivity_payload["edge_valid"],
            "edge_valid_all": bool(connectivity_payload["edge_valid_all"]),
            "safe_edge_local": connectivity_payload["safe_edge_local"],
            "edge_in_edge_patch": connectivity_payload["edge_in_edge_patch"],
            "edge_in_edge_patch_all": bool(
                connectivity_payload["edge_in_edge_patch_all"]
            ),
            "valid": connectivity_payload["valid"],
            "edge_sign": connectivity_payload["edge_sign"],
            "active_mask": connectivity_payload["active_mask"],
            "active_mask_all": bool(connectivity_payload["active_mask_all"]),
            "valid_all_by_slot": tuple(connectivity_payload["valid_all_by_slot"]),
            "signed_valid_edge_weight": connectivity_payload[
                "signed_valid_edge_weight"
            ],
        }
    )
    connectivity_identity: object = (
        connectivity_key
        if connectivity_key is not None
        else _large_step_recovery_connectivity_plan_fingerprint(connectivity_payload)
    )
    key = large_step_recovery_recovery_plan_owner_key(
        connectivity_identity=connectivity_identity,
        nVertLevels=nVertLevels,
        ns=ns,
        rk_step=rk_step,
        dt=dt,
        cf1=cf1,
        cf2=cf2,
        cf3=cf3,
        fzm=fzm,
        fzp=fzp,
        stage_key=stage_key,
    )

    def _large_step_recovery_bound_recovery(
        rho_p_save_cells: torch.Tensor,
        rho_pp_cells: torch.Tensor,
        rho_base_cells: torch.Tensor,
        rtheta_p_save_cells: torch.Tensor,
        rtheta_pp_cells: torch.Tensor,
        rtheta_base_cells: torch.Tensor,
        rt_diabatic_tend_cells: torch.Tensor,
        exner_base_cells: torch.Tensor,
        zz_cells: torch.Tensor,
        ww_avg_cells: torch.Tensor,
        rw_save_cells: torch.Tensor,
        rw_p_cells: torch.Tensor,
        ru_avg_edges: torch.Tensor,
        ru_save_edges: torch.Tensor,
        ru_p_edges: torch.Tensor,
        zb_cell_solve: torch.Tensor,
        zb3_cell_solve: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        return _large_step_recovery_recovery_numeric(
            **static_arguments,
            rho_p_save_cells=rho_p_save_cells,
            rho_pp_cells=rho_pp_cells,
            rho_base_cells=rho_base_cells,
            rtheta_p_save_cells=rtheta_p_save_cells,
            rtheta_pp_cells=rtheta_pp_cells,
            rtheta_base_cells=rtheta_base_cells,
            rt_diabatic_tend_cells=rt_diabatic_tend_cells,
            exner_base_cells=exner_base_cells,
            zz_cells=zz_cells,
            ww_avg_cells=ww_avg_cells,
            rw_save_cells=rw_save_cells,
            rw_p_cells=rw_p_cells,
            ru_avg_edges=ru_avg_edges,
            ru_save_edges=ru_save_edges,
            ru_p_edges=ru_p_edges,
            zb_cell_solve=zb_cell_solve,
            zb3_cell_solve=zb3_cell_solve,
        )

    _large_step_recovery_bound_recovery.__name__ = "_large_step_recovery_bound_recovery"
    _large_step_recovery_bound_recovery.__qualname__ = (
        "_large_step_recovery_bound_recovery"
    )
    return LargeStepRecoveryPlan(
        key=key,
        dynamic_argument_names=_LARGE_STEP_RECOVERY_RECOVERY_DYNAMIC_NAMES,
        static_arguments=static_arguments,
        numeric_callable=_large_step_recovery_bound_recovery,
    )


def _large_step_recovery_pack_zero_rows(
    base: torch.Tensor | None,
    patch: torch.Tensor,
    row_start: int,
    row_end: int,
    total_rows: int,
) -> torch.Tensor:
    """Pack a zero-outside active patch with one full-field materialization."""
    if base is not None:
        return _replace_row_slice(base, row_start, row_end, patch)
    if row_start == 0 and row_end == total_rows:
        return patch
    pad = [0, 0] * (patch.ndim - 1) + [row_start, total_rows - row_end]
    return F.pad(patch, tuple(pad), mode="constant", value=0.0)


def _large_step_recovery_pack_recovery_region(
    *,
    rho_p: torch.Tensor | None,
    rho_zz: torch.Tensor | None,
    rtheta_p: torch.Tensor | None,
    theta_m: torch.Tensor | None,
    rw: torch.Tensor | None,
    w: torch.Tensor | None,
    ru: torch.Tensor | None,
    u: torch.Tensor | None,
    wwAvg_out: torch.Tensor,
    ruAvg_out: torch.Tensor,
    exner: torch.Tensor,
    pressure_p: torch.Tensor,
    rho_p_cell: torch.Tensor,
    rho_zz_cell: torch.Tensor,
    rtheta_p_cell: torch.Tensor,
    theta_cell: torch.Tensor,
    exner_cell: torch.Tensor,
    pressure_cell: torch.Tensor,
    wwAvg_mid: torch.Tensor,
    rw_mid: torch.Tensor,
    w_surface_final: torch.Tensor,
    w_interior_final: torch.Tensor,
    ru_avg_patch: torch.Tensor,
    ru_patch: torch.Tensor,
    u_patch: torch.Tensor,
    cell_start: int,
    cell_end: int,
    edge_start: int,
    edge_end: int,
    nCells: int,
    nVertLevels: int,
    rk_step: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Pack active large-step-recovery patches back into padded MPAS-shaped tensors."""
    rho_p = _large_step_recovery_pack_zero_rows(
        rho_p, rho_p_cell, cell_start, cell_end, nCells + 1
    )
    if rho_zz is None:
        rho_zz = _large_step_recovery_pack_zero_rows(
            None, rho_zz_cell, cell_start, cell_end, nCells + 1
        )
        rho_zz = _replace_row_slice(
            rho_zz,
            nCells,
            nCells + 1,
            torch.ones((1, rho_zz.shape[1]), dtype=rho_zz.dtype, device=rho_zz.device),
        )
    else:
        rho_zz = _replace_row_slice(
            rho_zz,
            nCells,
            nCells + 1,
            torch.ones((1, rho_zz.shape[1]), dtype=rho_zz.dtype, device=rho_zz.device),
        )
        rho_zz = _replace_row_slice(rho_zz, cell_start, cell_end, rho_zz_cell)
    rtheta_p = _large_step_recovery_pack_zero_rows(
        rtheta_p, rtheta_p_cell, cell_start, cell_end, nCells + 1
    )
    theta_m = _large_step_recovery_pack_zero_rows(
        theta_m, theta_cell, cell_start, cell_end, nCells + 1
    )

    if rw is None:
        if not torch.is_grad_enabled():
            rw = rw_mid.new_zeros((nCells + 1, nVertLevels + 1))
            if nVertLevels > 1:
                rw[cell_start:cell_end, 1:nVertLevels] = rw_mid
        elif nVertLevels > 1:
            rw_active = F.pad(
                rw_mid,
                (1, 1),
                mode="constant",
                value=0.0,
            )
            rw = _large_step_recovery_pack_zero_rows(
                rw, rw_active, cell_start, cell_end, nCells + 1
            )
        else:
            rw_active = rw_mid.new_zeros((cell_end - cell_start, nVertLevels + 1))
            rw = _large_step_recovery_pack_zero_rows(
                rw, rw_active, cell_start, cell_end, nCells + 1
            )
    elif nVertLevels > 1:
        rw = _replace_row_slice_cols(rw, cell_start, cell_end, 1, nVertLevels, rw_mid)

    if w is None:
        if not torch.is_grad_enabled():
            w = w_surface_final.new_zeros((nCells + 1, nVertLevels + 1))
            w[cell_start:cell_end, 0] = w_surface_final
            if nVertLevels > 1:
                w[cell_start:cell_end, 1:nVertLevels] = w_interior_final
        else:
            w_parts = [w_surface_final[:, None]]
            if nVertLevels > 1:
                w_parts.append(w_interior_final)
            w_active = F.pad(
                w_parts[0] if len(w_parts) == 1 else torch.cat(tuple(w_parts), dim=1),
                (0, 1),
                mode="constant",
                value=0.0,
            )
            w = _large_step_recovery_pack_zero_rows(
                w, w_active, cell_start, cell_end, nCells + 1
            )
    else:
        if nVertLevels > 1:
            w = _replace_row_slice_cols(
                w, cell_start, cell_end, 1, nVertLevels, w_interior_final
            )
        w = _replace_row_slice_cols(
            w, cell_start, cell_end, 0, 1, w_surface_final[:, None]
        )

    if nVertLevels > 1:
        wwAvg_out = _replace_row_slice_cols(
            wwAvg_out, cell_start, cell_end, 1, nVertLevels, wwAvg_mid
        )

    ruAvg_out = _replace_row_slice(ruAvg_out, edge_start, edge_end, ru_avg_patch)
    ru = _large_step_recovery_pack_zero_rows(
        ru, ru_patch, edge_start, edge_end, ruAvg_out.shape[0]
    )
    u = _large_step_recovery_pack_zero_rows(
        u, u_patch, edge_start, edge_end, ruAvg_out.shape[0]
    )

    if rk_step == 3:
        exner = _replace_row_slice(exner, cell_start, cell_end, exner_cell)
        pressure_p = _replace_row_slice(pressure_p, cell_start, cell_end, pressure_cell)

    return (
        rho_p,
        rho_zz,
        rtheta_p,
        theta_m,
        rw,
        w,
        ru,
        u,
        wwAvg_out,
        ruAvg_out,
        exner,
        pressure_p,
    )


def atm_recover_large_step_variables_work(
    # Dimensions
    nCells: int,
    nEdges: int,
    nCellsSolve: int,
    nEdgesSolve: int,
    nVertLevels: int,
    maxEdges: int,
    # Time stepping
    ns: int,
    rk_step: int,
    dt: float,
    # Loop bounds (1-based Fortran indices)
    cellStart: int,
    cellEnd: int,
    vertexStart: int,
    vertexEnd: int,
    edgeStart: int,
    edgeEnd: int,
    cellSolveStart: int,
    cellSolveEnd: int,
    vertexSolveStart: int,
    vertexSolveEnd: int,
    edgeSolveStart: int,
    edgeSolveEnd: int,
    # Regional MPAS boundary parameters
    nRelaxZone: int,
    # Pre-computed coefficients
    cf1: float,
    cf2: float,
    cf3: float,
    # 1D arrays (nVertLevels,)
    fzm: torch.Tensor,
    fzp: torch.Tensor,
    # Connectivity (1-based indices)
    cellsOnEdge: torch.Tensor,  # (nEdges+1, 2) int32
    edgesOnCell: torch.Tensor,  # (nCells+1, maxEdges) int32
    nEdgesOnCell: torch.Tensor,  # (nCells+1,) int32
    edgesOnCell_sign: torch.Tensor,  # (nCells+1, maxEdges) float64
    bdyMaskCell: torch.Tensor,  # (nCells+1,) int32
    # Cell-based 2D arrays (nCells+1, nVertLevels)
    rho_p_save: torch.Tensor,
    rho_pp: torch.Tensor,
    rho_base: torch.Tensor,
    rtheta_p_save: torch.Tensor,
    rtheta_pp: torch.Tensor,
    rtheta_base: torch.Tensor,
    rt_diabatic_tend: torch.Tensor,
    exner_base: torch.Tensor,
    zz: torch.Tensor,
    pressure_b: torch.Tensor,
    # Cell-based 2D arrays (nCells+1, nVertLevels+1)
    wwAvg: torch.Tensor,
    rw_save: torch.Tensor,
    rw_p: torch.Tensor,
    # Edge-based 2D arrays (nEdges+1, nVertLevels)
    ruAvg: torch.Tensor,
    ru_save: torch.Tensor,
    ru_p: torch.Tensor,
    # 3D terrain coefficient arrays
    zb_cell: torch.Tensor,  # (nCells+1, maxEdges, nVertLevels+1)
    zb3_cell: torch.Tensor,  # (nCells+1, maxEdges, nVertLevels+1)
    zb: torch.Tensor,  # (nEdges+1, 2, nVertLevels+1)
    zb3: torch.Tensor,  # (nEdges+1, 2, nVertLevels+1)
    # Scalars (nCells+1, nVertLevels, num_scalars)
    scalars: torch.Tensor,
    _connectivity_payload: dict[str, Any] | None = None,
    _compile_region_names: tuple[str, ...] | frozenset[str] | None = None,
    _compile_mode: str = "reduce-overhead",
    _compile_backend: str = "inductor",
    _recovery_plan: LargeStepRecoveryPlan | None = None,
) -> dict:
    """
    Recover full state variables from perturbation variables after acoustic substeps.

    This function is called at the end of each RK step after the acoustic substeps
    have been completed. It reconstructs the full state variables (rho, theta, u, w)
    from the perturbation variables computed during the acoustic substeps.

    The algorithm:
    1. Compute new density: rho_p = rho_p_save + rho_pp, rho_zz = rho_p + rho_base
    2. Compute rw and w at interior levels using time-averaged values
    3. For rk_step == 3: include diabatic tendencies and compute thermodynamic vars
    4. Recover ruAvg, ru, and compute u from ru and rho_zz
    5. Finish recovering w from omega using flux divergence

    Index Convention:
        Fortran uses 1-based indexing: k=1..nVertLevels, iCell=1..nCells
        PyTorch uses 0-based indexing: k=0..nVertLevels-1, iCell=0..nCells-1

        Fortran loop `do k = 2, nVertLevels` maps to Python `for k in range(1, nVertLevels)`

    Args:
        nCells: Number of cells
        nEdges: Number of edges
        nCellsSolve: Number of owned cells to solve
        nEdgesSolve: Number of owned edges to solve
        nVertLevels: Number of vertical levels
        maxEdges: Maximum edges per cell
        ns: Number of acoustic substeps
        rk_step: RK substep number (1, 2, or 3)
        dt: Large timestep in seconds
        cellStart..edgeSolveEnd: Loop bounds (1-based Fortran indices)
        nRelaxZone: Number of relaxation zone cells for regional MPAS
        cf1, cf2, cf3: Vertical interpolation coefficients for surface
        fzm, fzp: Vertical interpolation weights
        cellsOnEdge: Cells adjacent to each edge (1-based indices)
        edgesOnCell: Edges bounding each cell (1-based indices)
        nEdgesOnCell: Number of edges on each cell
        edgesOnCell_sign: Sign of edge relative to cell
        bdyMaskCell: Boundary mask for regional MPAS (0 = interior)
        rho_p_save: Saved density perturbation from start of acoustic loop
        rho_pp: Accumulated density perturbation from acoustic substeps
        rho_base: Base state density
        rtheta_p_save: Saved rho*theta perturbation
        rtheta_pp: Accumulated rho*theta perturbation
        rtheta_base: Base state rho*theta
        rt_diabatic_tend: Diabatic tendency for theta
        exner_base: Base state Exner function
        zz: Jacobian of coordinate transformation
        pressure_b: Base state pressure
        wwAvg: Accumulated rho*omega from acoustic substeps
        rw_save: Saved rho*w from start of acoustic loop
        rw_p: rho*w perturbation from acoustic substeps
        ruAvg: Accumulated ru from acoustic substeps
        ru_save: Saved ru from start of acoustic loop
        ru_p: ru perturbation from acoustic substeps
        zb_cell, zb3_cell: Terrain coefficients for w recovery (cell-based)
        zb, zb3: Terrain coefficients for w recovery (edge-based)
        scalars: Scalar tracer fields

    Returns:
        Dictionary containing:
            w: Vertical velocity (nCells+1, nVertLevels+1)
            rw: rho*w (nCells+1, nVertLevels+1)
            wwAvg: Time-averaged rho*omega (nCells+1, nVertLevels+1)
            rho_p: Density perturbation (nCells+1, nVertLevels)
            rho_zz: Jacobian-weighted density (nCells+1, nVertLevels)
            rtheta_p: rho*theta perturbation (nCells+1, nVertLevels)
            theta_m: Modified potential temperature (nCells+1, nVertLevels)
            u: Edge-normal velocity (nEdges+1, nVertLevels)
            ru: rho*u (nEdges+1, nVertLevels)
            ruAvg: Time-averaged ru (nEdges+1, nVertLevels)
            exner: Exner function (nCells+1, nVertLevels) [only for rk_step==3]
            pressure_p: Pressure perturbation (nCells+1, nVertLevels) [only for rk_step==3]
    """
    require_consistent_device(
        "atm_recover_large_step_variables_work",
        fzm,
        fzp,
        cellsOnEdge,
        edgesOnCell,
        nEdgesOnCell,
        edgesOnCell_sign,
        bdyMaskCell,
        rho_p_save,
        rho_pp,
        rho_base,
        rtheta_p_save,
        rtheta_pp,
        rtheta_base,
        rt_diabatic_tend,
        exner_base,
        zz,
        pressure_b,
        wwAvg,
        rw_save,
        rw_p,
        ruAvg,
        ru_save,
        ru_p,
        zb_cell,
        zb3_cell,
        zb,
        zb3,
        scalars,
    )

    def _require_finite_nonzero(value, label: str) -> None:
        if not _mpas_debug_enabled():
            return
        if isinstance(value, torch.Tensor):
            if not torch.isfinite(value).all():
                raise ValueError(f"{label} is not finite")
            if torch.any(value == 0.0):
                raise ValueError(f"{label} is zero (division by zero risk)")
            return
        value_f = float(value)
        if not math.isfinite(value_f):
            raise ValueError(f"{label} is not finite")
        if value_f == 0.0:
            raise ValueError(f"{label} is zero (division by zero risk)")

    # Physical constants
    float(PhysicalConstants.RD)  # 287.0
    float(PhysicalConstants.CP)  # 1004.5

    # Convert Fortran 1-based bounds to Python 0-based
    cell_start = cellStart - 1
    cell_end = cellEnd
    edge_start = edgeStart - 1
    edge_end = edgeEnd
    if _connectivity_payload is None:
        conn = _large_step_recovery_connectivity_payload(
            device=rho_pp.device,
            nCells=nCells,
            nEdges=nEdges,
            maxEdges=maxEdges,
            nRelaxZone=nRelaxZone,
            cell_start=cell_start,
            cell_end=cell_end,
            edge_start=edge_start,
            edge_end=edge_end,
            cellsOnEdge=cellsOnEdge,
            edgesOnCell=edgesOnCell,
            nEdgesOnCell=nEdgesOnCell,
            edgesOnCell_sign=edgesOnCell_sign,
            bdyMaskCell=bdyMaskCell,
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

    edge_pair_relation = conn.get("edge_pair_relation")
    if isinstance(edge_pair_relation, StaticPairRelation):
        (
            edge_pair_relation.valid
            if edge_pair_relation.valid is not None
            else conn["edge_valid"]
        )
        (
            bool(edge_pair_relation.valid_all)
            if edge_pair_relation.valid_all is not None
            else bool(conn["edge_valid_all"])
        )
        c1 = edge_pair_relation.first
        c2 = edge_pair_relation.second
    else:
        conn["edge_valid"]
        bool(conn["edge_valid_all"])
        c1 = conn["c1"]
        c2 = conn["c2"]
    edge_pair_cells = conn.get("edge_pair_cells")
    if not isinstance(edge_pair_cells, torch.Tensor):
        edge_pair_cells = torch.stack((c1, c2), dim=1)

    cell_edge_relation = conn.get("cell_edge_relation")
    if isinstance(cell_edge_relation, StaticSlotRelation):
        (
            cell_edge_relation.valid
            if cell_edge_relation.valid is not None
            else conn["valid"]
        )
        (
            cell_edge_relation.sign
            if cell_edge_relation.sign is not None
            else conn["edge_sign"]
        )
        (
            cell_edge_relation.weights
            if cell_edge_relation.weights is not None
            else conn.get("signed_valid_edge_weight")
        )
    else:
        conn["valid"]
        conn["safe_edge"]
        conn["edge_sign"]
        conn.get("signed_valid_edge_weight")
        conn["valid_all_by_slot"]

    requested_compile_region_names = (
        frozenset(_compile_region_names or ())
        & _LARGE_STEP_RECOVERY_REGION_COMPILE_TARGETS
    )
    autodiff_tensors = (
        rho_p_save,
        rho_pp,
        rtheta_p_save,
        rtheta_pp,
        rw_save,
        rw_p,
        wwAvg,
        ru_save,
        ru_p,
        ruAvg,
    )
    autodiff_active = bool(
        is_grad_mode()
        or any(getattr(tensor, "requires_grad", False) for tensor in autodiff_tensors)
        or any(
            _large_step_recovery_tensor_has_forward_ad(tensor)
            for tensor in autodiff_tensors
        )
    )
    compile_region_names = (
        frozenset() if autodiff_active else requested_compile_region_names
    )

    def _maybe_compile_large_step_recovery_region(
        fn_name: str, fn, specialization_key=None
    ):
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

    if _recovery_plan is None:
        _recovery_plan = build_large_step_recovery_recovery_plan(
            connectivity_payload=conn,
            nVertLevels=nVertLevels,
            ns=ns,
            rk_step=rk_step,
            dt=dt,
            cf1=cf1,
            cf2=cf2,
            cf3=cf3,
            fzm=fzm,
            fzp=fzp,
            stage_key=("direct_call", "rk_step", int(rk_step)),
        )
        if "large_step_recovery_recovery_region" in compile_region_names:
            _recovery_plan = _recovery_plan.with_numeric_callable(
                _maybe_compile_large_step_recovery_region(
                    "large_step_recovery_recovery_region",
                    _recovery_plan.numeric_callable,
                    specialization_key=_recovery_plan.key,
                )
            )

    if _recovery_plan is not None:
        numeric_arguments = {
            "rho_p_save_cells": rho_p_save[cell_slice, :],
            "rho_pp_cells": rho_pp[cell_slice, :],
            "rho_base_cells": rho_base[cell_slice, :],
            "rtheta_p_save_cells": rtheta_p_save[cell_slice, :],
            "rtheta_pp_cells": rtheta_pp[cell_slice, :],
            "rtheta_base_cells": rtheta_base[cell_slice, :],
            "rt_diabatic_tend_cells": rt_diabatic_tend[cell_slice, :],
            "exner_base_cells": exner_base[cell_slice, :],
            "zz_cells": zz[cell_slice, :],
            "ww_avg_cells": wwAvg[cell_slice, :],
            "rw_save_cells": rw_save[cell_slice, :],
            "rw_p_cells": rw_p[cell_slice, :],
            "ru_avg_edges": ruAvg[edge_slice, :],
            "ru_save_edges": ru_save[edge_slice, :],
            "ru_p_edges": ru_p[edge_slice, :],
            "zb_cell_solve": zb_cell[
                cell_start:cell_end, : int(conn["active_slot_count"]), :
            ],
            "zb3_cell_solve": zb3_cell[
                cell_start:cell_end, : int(conn["active_slot_count"]), :
            ],
        }
        (
            rho_p_cell,
            rho_zz_cell,
            rtheta_p_cell,
            theta_cell,
            exner_cell,
            pressure_cell,
            wwAvg_mid,
            rw_mid,
            w_surface_final,
            w_interior_final,
            ru_avg_patch,
            ru_patch,
            u_patch,
        ) = _recovery_plan.numeric_callable(
            *_recovery_plan.dynamic_values(numeric_arguments)
        )

        wwAvg_out = _clone_for_grad(wwAvg)
        ruAvg_out = _clone_for_grad(ruAvg)
        # These outputs are zero outside their active patches. Let the packer
        # construct each padded result once instead of allocating a full zero
        # tensor and then rewriting its active rows.
        rho_p = None
        rho_zz = None
        rtheta_p = None
        theta_m = None
        rw = None
        w = None
        ru = None
        u = None
        exner = torch.zeros_like(rtheta_pp) if rk_step == 3 else rtheta_pp
        pressure_p = torch.zeros_like(rtheta_pp) if rk_step == 3 else rtheta_pp
        (
            rho_p,
            rho_zz,
            rtheta_p,
            theta_m,
            rw,
            w,
            ru,
            u,
            wwAvg_out,
            ruAvg_out,
            exner,
            pressure_p,
        ) = _large_step_recovery_pack_recovery_region(
            rho_p=rho_p,
            rho_zz=rho_zz,
            rtheta_p=rtheta_p,
            theta_m=theta_m,
            rw=rw,
            w=w,
            ru=ru,
            u=u,
            wwAvg_out=wwAvg_out,
            ruAvg_out=ruAvg_out,
            exner=exner,
            pressure_p=pressure_p,
            rho_p_cell=rho_p_cell,
            rho_zz_cell=rho_zz_cell,
            rtheta_p_cell=rtheta_p_cell,
            theta_cell=theta_cell,
            exner_cell=exner_cell,
            pressure_cell=pressure_cell,
            wwAvg_mid=wwAvg_mid,
            rw_mid=rw_mid,
            w_surface_final=w_surface_final,
            w_interior_final=w_interior_final,
            ru_avg_patch=ru_avg_patch,
            ru_patch=ru_patch,
            u_patch=u_patch,
            cell_start=cell_start,
            cell_end=cell_end,
            edge_start=edge_start,
            edge_end=edge_end,
            nCells=nCells,
            nVertLevels=nVertLevels,
            rk_step=rk_step,
        )
        result = {
            "w": w,
            "rw": rw,
            "wwAvg": wwAvg_out,
            "rho_p": rho_p,
            "rho_zz": rho_zz,
            "rtheta_p": rtheta_p,
            "theta_m": theta_m,
            "u": u,
            "ru": ru,
            "ruAvg": ruAvg_out,
        }
        if rk_step == 3:
            result["exner"] = exner
            result["pressure_p"] = pressure_p
        return result


def recover_large_step(
    state: dict,
    state_old: dict,
    mesh: MESHState,
    dt: torch.Tensor,
) -> dict:
    """
    High-level wrapper for atm_recover_large_step_variables_work - INTENTIONALLY NOT IMPLEMENTED.

    This wrapper is a placeholder for a potential future API simplification.
    The low-level atm_recover_large_step_variables_work() function is used directly
    in execute_rk_step() to maintain explicit control over all parameters and
    avoid the complexity of automatic parameter extraction from state dicts.

    Design Decision:
        The explicit parameter passing pattern in atm_recover_large_step_variables_work()
        provides better type safety, clearer documentation, and easier debugging
        compared to a wrapper that would need to extract ~30+ parameters from
        nested dictionaries. This is intentional, not a TODO.

    Usage:
        Use atm_recover_large_step_variables_work() directly with explicit parameters.
        See time_integration.py execute_rk_step() for the expected call pattern.

    Raises:
        NotImplementedError: Always - use atm_recover_large_step_variables_work() instead.
    """
    raise NotImplementedError(
        "recover_large_step() wrapper is intentionally not implemented. "
        "Use atm_recover_large_step_variables_work() directly with explicit parameters. "
        "See execute_rk_step() for the expected call pattern."
    )
