"""
Small timestep perturbation (small-step-perturbation).

Handles perturbation variables and small timestep momentum updates
in the split-explicit time integration scheme.


Lines: 2427-2508
"""

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch

from mpas_pytorch.autodiff import is_grad_mode
from mpas_pytorch.autodiff.soft_limiter import fct_sign_pm
from mpas_pytorch.dynamics.static_payloads import StaticRange, StaticSlotRelation
from mpas_pytorch.dynamics.stencil_ops import (
    _active_slot_count,
    _replace_row_slice_cols,
    _static_connectivity_cache_allowed,
    _static_mask_all_true,
    _static_slot_all_true,
    _tensor_version_cache_key,
    gather_rows,
)
from mpas_pytorch.utils.checks import require_consistent_device

_SMALL_STEP_PERTURBATION_REGION_COMPILE_TARGETS = frozenset(
    {
        "small_step_perturbation_smlstep_pert_w_tendency_region",
    }
)
_SMALL_STEP_PERTURBATION_DYNAMIC_NAMES = (
    "w_tend_cells",
    "u_tend",
    "zz_cells",
    "zb_cell_solve",
    "zb3_cell_solve",
)


@dataclass(frozen=True)
class SmallStepPerturbationNumericPlan:
    """Immutable static binding for the compact small-step-perturbation tendency region."""

    key: tuple[object, ...]
    dynamic_argument_names: tuple[str, ...]
    static_arguments: Mapping[str, Any]
    numeric_callable: Callable[..., torch.Tensor]

    def dynamic_values(self, call_arguments: Mapping[str, Any]) -> tuple[object, ...]:
        missing = [
            name for name in self.dynamic_argument_names if name not in call_arguments
        ]
        if missing:
            raise KeyError(
                f"small-step-perturbation numeric call is missing dynamic arguments: {missing}"
            )
        return tuple(call_arguments[name] for name in self.dynamic_argument_names)


_SMALL_STEP_PERTURBATION_CONNECTIVITY_TENSOR_NAMES = (
    "nEdgesOnCell",
    "edgesOnCell",
    "edgesOnCell_sign",
    "bdyMaskCell",
)


def _small_step_perturbation_static_all_true(
    mask: torch.Tensor, *, allow_cuda_sync: bool = False
) -> bool:
    return bool(_static_mask_all_true(mask, allow_cuda_sync=allow_cuda_sync))


def _small_step_perturbation_slot_all_true(
    mask: torch.Tensor,
    *,
    allow_cuda_sync: bool = False,
) -> tuple[bool, ...]:
    metadata = _static_slot_all_true(mask, allow_cuda_sync=allow_cuda_sync)
    if metadata is None:
        return tuple(False for _ in range(mask.shape[1]))
    return metadata


def _small_step_perturbation_tensor_cache_key(
    tensor: torch.Tensor,
) -> tuple[object, ...] | None:
    return _tensor_version_cache_key(tensor)


def _small_step_perturbation_tensor_has_forward_ad(tensor: object) -> bool:
    if not isinstance(tensor, torch.Tensor):
        return False
    try:
        _primal, tangent = torch.autograd.forward_ad.unpack_dual(tensor)
    except Exception:  # noqa: BLE001
        return False
    return tangent is not None


def _small_step_perturbation_connectivity_payload_cache_key(
    **kwargs: object,
) -> tuple[object, ...] | None:
    """Return the mutation-sensitive cache key for small-step-perturbation static connectivity."""
    tensors = tuple(
        kwargs.get(name) for name in _SMALL_STEP_PERTURBATION_CONNECTIVITY_TENSOR_NAMES
    )
    if not _static_connectivity_cache_allowed(tensors):
        return None

    tensor_keys: list[tuple[object, ...]] = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            tensor_key = _small_step_perturbation_tensor_cache_key(tensor)
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
        *tensor_keys,
    )


def _build_small_step_perturbation_connectivity_payload(
    *,
    device: torch.device,
    nCells: int,
    nEdges: int,
    maxEdges: int,
    nRelaxZone: int,
    cell_start: int,
    cell_end: int,
    nEdgesOnCell: torch.Tensor,
    edgesOnCell: torch.Tensor,
    edgesOnCell_sign: torch.Tensor,
    bdyMaskCell: torch.Tensor,
) -> dict[str, Any]:
    """Build reusable small-step-perturbation cell-edge perturbation topology."""
    cell_slice = slice(cell_start, cell_end)
    solve_cells = torch.arange(cell_start, cell_end, dtype=torch.int64, device=device)
    active_mask = bdyMaskCell[solve_cells] <= nRelaxZone
    active_slot_count = _active_slot_count(nEdgesOnCell[solve_cells], maxEdges, device)
    edge_slots = torch.arange(active_slot_count, dtype=torch.int64, device=device)
    edge_1based = edgesOnCell[solve_cells, :active_slot_count]
    edge0 = edge_1based - 1
    slot_valid = edge_slots[None, :] < nEdgesOnCell[solve_cells][:, None]
    edge_valid = (edge0 >= 0) & (edge0 < nEdges)
    valid = slot_valid & edge_valid & active_mask[:, None]
    safe_edge = edge0.clamp(min=0, max=max(nEdges - 1, 0)).to(dtype=torch.int64)
    edge_sign = edgesOnCell_sign[solve_cells, :active_slot_count]
    cell_range = StaticRange("small_step_perturbation.cells", cell_start, cell_end)
    cell_edge_relation = StaticSlotRelation(
        "small_step_perturbation.cell_edges",
        indices=safe_edge,
        valid=valid,
        sign=edge_sign,
        valid_all_by_slot=_small_step_perturbation_slot_all_true(
            valid, allow_cuda_sync=True
        ),
    )
    return {
        "cell_slice": cell_slice,
        "cell_range": cell_range,
        "solve_cells": solve_cells,
        "active_mask": active_mask,
        "active_mask_all": _small_step_perturbation_static_all_true(
            active_mask, allow_cuda_sync=True
        ),
        "safe_edge": safe_edge,
        "valid": valid,
        "edge_sign": edge_sign,
        "valid_all_by_slot": cell_edge_relation.valid_all_by_slot,
        "active_slot_count": active_slot_count,
        "cell_edge_relation": cell_edge_relation,
    }


def _small_step_perturbation_connectivity_payload(**kwargs: object) -> dict[str, Any]:
    """Build small-step-perturbation topology for direct callers; the runner owns warm reuse."""
    return _build_small_step_perturbation_connectivity_payload(**kwargs)  # type: ignore[arg-type]


def _small_step_perturbation_smlstep_pert_w_tendency_region(
    *,
    w_tend_cells: torch.Tensor,
    u_tend: torch.Tensor,
    zz_cells: torch.Tensor,
    safe_edge: torch.Tensor,
    valid: torch.Tensor,
    edge_sign: torch.Tensor,
    active_mask: torch.Tensor,
    active_mask_all: bool,
    zb_cell_solve: torch.Tensor,
    zb3_cell_solve: torch.Tensor,
    fzm: torch.Tensor,
    fzp: torch.Tensor,
    nVertLevels: int,
) -> torch.Tensor:
    """Return updated small-step-perturbation active-cell interior w-tendency rows."""
    if nVertLevels <= 1:
        return w_tend_cells[:, 1:1]

    current_block = w_tend_cells[:, 1:nVertLevels]
    # Match the upstream statement association exactly: each edge correction is
    # subtracted from the live tendency before the next edge is visited.  A
    # pre-summed delta is mathematically equivalent but changes binary64 results
    # before the acoustic solve.
    w_after_flux = current_block
    for slot in range(safe_edge.shape[1]):
        # Keep one slot's temporary arrays live while preserving edge order.
        u_slot = gather_rows(u_tend[:, :nVertLevels], safe_edge[:, slot])
        u_k = u_slot[:, 1:nVertLevels]
        u_interp = (
            fzm[1:nVertLevels][None, :] * u_k
            + fzp[1:nVertLevels][None, :] * u_slot[:, : nVertLevels - 1]
        )
        flux = edge_sign[:, slot, None] * u_interp
        coef = (
            zb_cell_solve[:, slot, 1:nVertLevels]
            + fct_sign_pm(u_k) * zb3_cell_solve[:, slot, 1:nVertLevels]
        )
        corrected = w_after_flux - coef * flux
        w_after_flux = torch.where(valid[:, slot, None], corrected, w_after_flux)
    if not active_mask_all:
        w_after_flux = torch.where(active_mask[:, None], w_after_flux, current_block)

    zz_interp = (
        fzm[1:nVertLevels][None, :] * zz_cells[:, 1:nVertLevels]
        + fzp[1:nVertLevels][None, :] * zz_cells[:, : nVertLevels - 1]
    )
    w_scaled = zz_interp * w_after_flux
    if active_mask_all:
        return w_scaled
    return torch.where(active_mask[:, None], w_scaled, current_block)


def _small_step_perturbation_plan_tensor_fingerprint(
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


def small_step_perturbation_numeric_plan_owner_key(
    *,
    connectivity_identity: object,
    nVertLevels: int,
    fzm: torch.Tensor,
    fzp: torch.Tensor,
    stage_key: tuple[object, ...] = (),
) -> tuple[object, ...]:
    """Return the exact runner-owned small-step-perturbation specialization key."""
    return (
        "SmallStepPerturbationNumericPlan.v1",
        tuple(stage_key),
        connectivity_identity,
        int(nVertLevels),
        _small_step_perturbation_plan_tensor_fingerprint(fzm),
        _small_step_perturbation_plan_tensor_fingerprint(fzp),
    )


def _small_step_perturbation_connectivity_plan_fingerprint(
    connectivity_payload: Mapping[str, Any],
) -> tuple[object, ...]:
    names = ("safe_edge", "valid", "edge_sign", "active_mask")
    return tuple(
        (name, _small_step_perturbation_plan_tensor_fingerprint(value))
        for name in names
        if isinstance((value := connectivity_payload.get(name)), torch.Tensor)
    )


def build_small_step_perturbation_numeric_plan(
    *,
    connectivity_payload: Mapping[str, Any],
    nVertLevels: int,
    fzm: torch.Tensor,
    fzp: torch.Tensor,
    stage_key: tuple[object, ...] = (),
    connectivity_key: tuple[object, ...] | None = None,
) -> SmallStepPerturbationNumericPlan:
    """Bind small-step-perturbation topology and vertical constants around one numeric callable."""
    relation = connectivity_payload.get("cell_edge_relation")
    if isinstance(relation, StaticSlotRelation):
        safe_edge = relation.indices
        valid = relation.valid
        edge_sign = relation.sign
    else:
        safe_edge = connectivity_payload["safe_edge"]
        valid = connectivity_payload["valid"]
        edge_sign = connectivity_payload["edge_sign"]
    if not all(
        isinstance(value, torch.Tensor) for value in (safe_edge, valid, edge_sign)
    ):
        raise TypeError(
            "small-step-perturbation numeric plan requires tensor-valued cell-edge topology"
        )

    static_arguments = MappingProxyType(
        {
            "safe_edge": safe_edge,
            "valid": valid,
            "edge_sign": edge_sign,
            "active_mask": connectivity_payload["active_mask"],
            "active_mask_all": bool(connectivity_payload["active_mask_all"]),
            "fzm": fzm,
            "fzp": fzp,
            "nVertLevels": int(nVertLevels),
        }
    )
    identity: object = (
        connectivity_key
        if connectivity_key is not None
        else _small_step_perturbation_connectivity_plan_fingerprint(
            connectivity_payload
        )
    )
    key = small_step_perturbation_numeric_plan_owner_key(
        connectivity_identity=identity,
        nVertLevels=nVertLevels,
        fzm=fzm,
        fzp=fzp,
        stage_key=stage_key,
    )

    def _small_step_perturbation_bound_numeric(
        w_tend_cells: torch.Tensor,
        u_tend: torch.Tensor,
        zz_cells: torch.Tensor,
        zb_cell_solve: torch.Tensor,
        zb3_cell_solve: torch.Tensor,
    ) -> torch.Tensor:
        return _small_step_perturbation_smlstep_pert_w_tendency_region(
            **static_arguments,
            w_tend_cells=w_tend_cells,
            u_tend=u_tend,
            zz_cells=zz_cells,
            zb_cell_solve=zb_cell_solve,
            zb3_cell_solve=zb3_cell_solve,
        )

    return SmallStepPerturbationNumericPlan(
        key=key,
        dynamic_argument_names=_SMALL_STEP_PERTURBATION_DYNAMIC_NAMES,
        static_arguments=static_arguments,
        numeric_callable=_small_step_perturbation_bound_numeric,
    )


def atm_set_smlstep_pert_variables_work(
    nCells: int,
    nEdges: int,
    nVertLevels: int,
    maxEdges: int,
    cellSolveStart: int,
    cellSolveEnd: int,
    nRelaxZone: int,
    nEdgesOnCell: torch.Tensor,  # (nCells+1,) int32
    cellsOnEdge: torch.Tensor,  # (nEdges+1, 2) int32
    edgesOnCell: torch.Tensor,  # (nCells+1, maxEdges) int32
    edgesOnCell_sign: torch.Tensor,  # (nCells+1, maxEdges) float64
    bdyMaskCell: torch.Tensor,  # (nCells+1,) int32
    fzm: torch.Tensor,  # (nVertLevels,) float64
    fzp: torch.Tensor,  # (nVertLevels,) float64
    zb: torch.Tensor,  # (nEdges+1, 2, nVertLevels+1) float64
    zb3: torch.Tensor,  # (nEdges+1, 2, nVertLevels+1) float64
    zb_cell: torch.Tensor,  # (nCells+1, maxEdges, nVertLevels+1) float64
    zb3_cell: torch.Tensor,  # (nCells+1, maxEdges, nVertLevels+1) float64
    zz: torch.Tensor,  # (nCells+1, nVertLevels) float64
    w_tend: torch.Tensor,  # (nCells+1, nVertLevels+1) float64
    u_tend: torch.Tensor,  # (nEdges+1, nVertLevels) float64
    _connectivity_payload: dict[str, Any] | None = None,
    _compile_region_names: tuple[str, ...] | frozenset[str] | None = None,
    _compile_mode: str = "reduce-overhead",
    _compile_backend: str = "inductor",
    _numeric_plan: SmallStepPerturbationNumericPlan | None = None,
    _take_w_tend_ownership: bool = False,
) -> dict:
    """
    Set perturbation variables for acoustic substep.

    Converts w tendency to omega tendency for the split-explicit acoustic solver.
    This uses the same flux divergence as the theta equation (Klemp et al MWR 2003).

    Args:
        nCells: Number of cells
        nEdges: Number of edges
        nVertLevels: Number of vertical levels
        maxEdges: Maximum edges per cell
        cellSolveStart: Start cell index for solve loop (1-based Fortran index)
        cellSolveEnd: End cell index for solve loop (1-based Fortran index)
        nRelaxZone: Relaxation zone width for regional MPAS
        nEdgesOnCell: Number of edges on each cell (nCells+1,)
        cellsOnEdge: Cells adjacent to each edge (nEdges+1, 2)
        edgesOnCell: Edges bounding each cell (nCells+1, maxEdges)
        edgesOnCell_sign: Sign of edge relative to cell (+1 or -1)
        bdyMaskCell: Boundary mask for regional MPAS
        fzm: Vertical interpolation weight (minus)
        fzp: Vertical interpolation weight (plus)
        zb: Edge-based vertical metric coefficient
        zb3: Edge-based vertical metric coefficient (3rd order)
        zb_cell: Cell-based vertical metric coefficient
        zb3_cell: Cell-based vertical metric coefficient (3rd order)
        zz: Jacobian of coordinate transformation
        w_tend: Vertical velocity tendency (input/output)
        u_tend: Horizontal velocity tendency

    Returns:
        Dictionary containing updated w_tend
    """
    device = require_consistent_device(
        "atm_set_smlstep_pert_variables_work",
        nEdgesOnCell,
        cellsOnEdge,
        edgesOnCell,
        edgesOnCell_sign,
        bdyMaskCell,
        fzm,
        fzp,
        zb,
        zb3,
        zb_cell,
        zb3_cell,
        zz,
        w_tend,
        u_tend,
    )
    # Convert one-based MPAS indices to zero-based Python indices.
    # Upstream bounds: cellSolveStart=1, cellSolveEnd=nCells
    # Python: range(0, nCells)
    cell_start = cellSolveStart - 1  # 0
    cell_end = cellSolveEnd  # nCells (exclusive in Python)

    if _connectivity_payload is None:
        conn = _small_step_perturbation_connectivity_payload(
            device=device,
            nCells=nCells,
            nEdges=nEdges,
            maxEdges=maxEdges,
            nRelaxZone=nRelaxZone,
            cell_start=cell_start,
            cell_end=cell_end,
            nEdgesOnCell=nEdgesOnCell,
            edgesOnCell=edgesOnCell,
            edgesOnCell_sign=edgesOnCell_sign,
            bdyMaskCell=bdyMaskCell,
        )
    else:
        conn = _connectivity_payload

    cell_range = conn.get("cell_range")
    cell_slice = (
        cell_range.slice if isinstance(cell_range, StaticRange) else conn["cell_slice"]
    )
    active_slot_count = int(conn["active_slot_count"])

    requested_compile_region_names = (
        frozenset(_compile_region_names or ())
        & _SMALL_STEP_PERTURBATION_REGION_COMPILE_TARGETS
    )
    autodiff_tensors = (w_tend, u_tend, zz)
    autodiff_active = bool(
        is_grad_mode()
        or any(getattr(tensor, "requires_grad", False) for tensor in autodiff_tensors)
        or any(
            _small_step_perturbation_tensor_has_forward_ad(tensor)
            for tensor in autodiff_tensors
        )
    )
    compile_region_names = (
        frozenset() if autodiff_active else requested_compile_region_names
    )

    def _maybe_compile_small_step_perturbation_region(
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

    # The timestep runner may transfer a freshly produced dynamics-tendency tendency into
    # small-step-perturbation during inference. Public calls and every AD-active call retain the
    # non-mutating API contract.
    w_tend_out = (
        w_tend if _take_w_tend_ownership and not autodiff_active else w_tend.clone()
    )

    if nVertLevels > 1:
        if _numeric_plan is None:
            _numeric_plan = build_small_step_perturbation_numeric_plan(
                connectivity_payload=conn,
                nVertLevels=nVertLevels,
                fzm=fzm,
                fzp=fzp,
                stage_key=("direct_call",),
            )
        numeric_callable = _numeric_plan.numeric_callable
        if (
            "small_step_perturbation_smlstep_pert_w_tendency_region"
            in compile_region_names
        ):
            numeric_callable = _maybe_compile_small_step_perturbation_region(
                "small_step_perturbation_smlstep_pert_w_tendency_region",
                numeric_callable,
                specialization_key=_numeric_plan.key,
            )
        dynamic_arguments = {
            "w_tend_cells": w_tend_out[cell_slice, :],
            "u_tend": u_tend,
            "zz_cells": zz[cell_slice, :],
            "zb_cell_solve": zb_cell[cell_start:cell_end, :active_slot_count, :],
            "zb3_cell_solve": zb3_cell[cell_start:cell_end, :active_slot_count, :],
        }
        updated_block = numeric_callable(
            *_numeric_plan.dynamic_values(dynamic_arguments)
        )
        w_tend_out = _replace_row_slice_cols(
            w_tend_out,
            cellSolveStart - 1,
            cellSolveEnd,
            1,
            nVertLevels,
            updated_block,
        )

    return {"w_tend": w_tend_out}


def smlstep_pert(
    state: dict,
    mesh: Any,
    dt: torch.Tensor,
    *,
    nCells: int | None = None,
    nEdges: int | None = None,
    nVertLevels: int | None = None,
    maxEdges: int | None = None,
) -> dict:
    """
    Set perturbation variables for acoustic substep.

    High-level wrapper that delegates to atm_set_smlstep_pert_variables_work.

    Args:
        state: State dictionary containing all prognostic variables
        mesh: MPAS mesh state
        dt: Small timestep (unused but kept for interface consistency)

    Returns:
        Updated state dictionary with modified w_tend
    """
    # Handle DycoreState vs dict input for state
    if hasattr(state, "__dict__"):
        state_dict = dict(state.__dict__)
    else:
        state_dict = dict(state)

    # Handle MESHState vs dict input for mesh
    if hasattr(mesh, "__dict__"):
        mesh_dict = dict(mesh.__dict__)
    else:
        mesh_dict = dict(mesh)

    device = require_consistent_device("smlstep_pert inputs", state_dict, mesh_dict)

    def _coerce_int_param(value, name: str, default: int | None = None) -> int:
        if value is None:
            if default is None:
                raise ValueError("Missing required scalar dimension")
            return int(default)
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(
                    f"Expected scalar tensor, got shape {tuple(value.shape)}"
                )
            if value.device.type != "cpu":
                raise ValueError(
                    f"{name} must be provided as a Python int when tensor is on {value.device}. "
                    "Pass explicit integer args from caller to avoid host synchronization."
                )
            return int(value)
        return int(value)

    # Extract dimensions (prefer Python int parameters from caller).
    nCells = _coerce_int_param(
        nCells if nCells is not None else state_dict.get("nCells"), "nCells"
    )
    nEdges = _coerce_int_param(
        nEdges if nEdges is not None else state_dict.get("nEdges"),
        "nEdges",
        default=nCells * 3,
    )
    nVertLevels = _coerce_int_param(
        nVertLevels if nVertLevels is not None else state_dict.get("nVertLevels"),
        "nVertLevels",
    )
    maxEdges = _coerce_int_param(
        maxEdges if maxEdges is not None else mesh_dict.get("maxEdges"),
        "maxEdges",
        default=10,
    )

    # Require Fortran-style 1-based+padded connectivity arrays.
    if "cellsOnEdge" in mesh_dict:
        coe = mesh_dict["cellsOnEdge"]
        invalid_shape = coe.shape[0] == nEdges
        has_zero_based_sentinel = False
        if os.environ.get("MPAS_DEBUG", "0") == "1" and isinstance(coe, torch.Tensor):
            has_zero_based_sentinel = bool(torch.any(coe == -1))
        if invalid_shape or has_zero_based_sentinel:
            raise ValueError(
                "smlstep_pert requires Fortran 1-based+padded mesh connectivity arrays. "
                "Use raw Fortran-style mesh data instead of MESHState."
            )

    # Extract loop bounds
    cellSolveStart = _coerce_int_param(
        state_dict.get("cellSolveStart"), "cellSolveStart", default=1
    )
    cellSolveEnd = _coerce_int_param(
        state_dict.get("cellSolveEnd"), "cellSolveEnd", default=nCells
    )
    nRelaxZone = _coerce_int_param(
        state_dict.get("nRelaxZone"), "nRelaxZone", default=0
    )

    # Call work function
    result = atm_set_smlstep_pert_variables_work(
        nCells=nCells,
        nEdges=nEdges,
        nVertLevels=nVertLevels,
        maxEdges=maxEdges,
        cellSolveStart=cellSolveStart,
        cellSolveEnd=cellSolveEnd,
        nRelaxZone=nRelaxZone,
        nEdgesOnCell=mesh_dict["nEdgesOnCell"],
        cellsOnEdge=mesh_dict["cellsOnEdge"],
        edgesOnCell=mesh_dict["edgesOnCell"],
        edgesOnCell_sign=mesh_dict["edgesOnCell_sign"],
        bdyMaskCell=mesh_dict.get(
            "bdyMaskCell", torch.zeros(nCells + 1, dtype=torch.int32, device=device)
        ),
        fzm=state_dict["fzm"],
        fzp=state_dict["fzp"],
        zb=state_dict["zb"],
        zb3=state_dict["zb3"],
        zb_cell=state_dict["zb_cell"],
        zb3_cell=state_dict["zb3_cell"],
        zz=state_dict["zz"],
        w_tend=state_dict["w_tend"],
        u_tend=state_dict["u_tend"],
    )

    # Merge result into state_dict
    output = dict(state_dict)
    output.update(result)

    return output
