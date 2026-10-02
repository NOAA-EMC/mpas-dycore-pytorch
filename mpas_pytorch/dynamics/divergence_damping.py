"""
3D Divergence Damping for acoustic noise control.

This kernel applies divergence damping to the horizontal momentum (ru_p)
to suppress acoustic noise that can develop during split-explicit time integration.

The damping is based on the time derivative of rtheta_pp (perturbed potential
temperature) which represents horizontal divergence in the acoustic modes.



"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch

from mpas_pytorch.autodiff import is_grad_mode
from mpas_pytorch.dynamics.stencil_ops import (
    _compile_active,
    _replace_row_slice,
    _static_connectivity_cache_allowed,
    _tensor_version_cache_key,
    gather_rows,
)
from mpas_pytorch.utils.checks import require_consistent_device

_DIVERGENCE_DAMPING_REGION_COMPILE_TARGETS = frozenset(
    {
        "dynamics_tendency_divergence_damping_region",
    }
)
_DIVERGENCE_DYNAMIC_NAMES = (
    "ru_p_edges",
    "theta_m",
    "rtheta_pp",
    "rtheta_pp_old",
)


@dataclass(frozen=True)
class DivergenceDampingPlan:
    """Immutable static binding for the divergence-damping numeric region."""

    key: tuple[object, ...]
    dynamic_argument_names: tuple[str, ...]
    static_arguments: Mapping[str, torch.Tensor]
    numeric_callable: Callable[..., torch.Tensor]

    def dynamic_values(self, call_arguments: Mapping[str, Any]) -> tuple[object, ...]:
        missing = [
            name for name in self.dynamic_argument_names if name not in call_arguments
        ]
        if missing:
            raise KeyError(
                f"divergence-damping numeric call is missing arguments: {missing}"
            )
        return tuple(call_arguments[name] for name in self.dynamic_argument_names)


_DIVERGENCE_CONNECTIVITY_TENSOR_NAMES = (
    "cellsOnEdge",
    "specZoneMaskEdge",
)
_DIVERGENCE_PAYLOAD_KEY_ONLY_KWARGS = frozenset({"specZoneMaskEdgeKnownAllZero"})


def _divergence_tensor_cache_key(tensor: torch.Tensor) -> tuple[object, ...] | None:
    return _tensor_version_cache_key(tensor)


def _divergence_spec_zone_mask_cache_key(
    tensor: torch.Tensor,
    *,
    known_all_zero: bool = False,
) -> tuple[object, ...] | None:
    if known_all_zero:
        return (
            "specZoneMaskEdge",
            "all_zero",
            str(tensor.device),
            str(tensor.dtype),
            tuple(tensor.shape),
        )
    return _divergence_tensor_cache_key(tensor)


def _divergence_connectivity_payload_cache_key(
    **kwargs: object,
) -> tuple[object, ...] | None:
    """Return the mutation-sensitive cache key for divergence damping topology."""
    tensors = tuple(kwargs.get(name) for name in _DIVERGENCE_CONNECTIVITY_TENSOR_NAMES)
    if not _static_connectivity_cache_allowed(tensors):
        return None

    tensor_keys: list[tuple[object, ...]] = []
    for name, tensor in zip(_DIVERGENCE_CONNECTIVITY_TENSOR_NAMES, tensors):
        if isinstance(tensor, torch.Tensor):
            if name == "specZoneMaskEdge":
                tensor_key = _divergence_spec_zone_mask_cache_key(
                    tensor,
                    known_all_zero=bool(
                        kwargs.get("specZoneMaskEdgeKnownAllZero", False)
                    ),
                )
            else:
                tensor_key = _divergence_tensor_cache_key(tensor)
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


def _build_divergence_connectivity_payload(
    *,
    device: torch.device,
    nCells: int,
    nEdges: int,
    nCellsSolve: int,
    edge_start: int,
    edge_end: int,
    cellsOnEdge: torch.Tensor,
    specZoneMaskEdge: torch.Tensor,
) -> dict[str, Any]:
    """Build static edge topology for 3-D divergence damping."""
    edge_start = max(int(edge_start), 0)
    edge_end = min(int(edge_end), int(nEdges))
    edge_ids = torch.arange(nEdges, dtype=torch.int64, device=device)
    edge_in_range = (edge_ids >= edge_start) & (edge_ids < edge_end)

    cells_on_edge = cellsOnEdge[:nEdges, :2]
    cell1_1based = cells_on_edge[:, 0]
    cell2_1based = cells_on_edge[:, 1]
    valid_cells = (
        (cell1_1based >= 1)
        & (cell1_1based <= nCells)
        & (cell2_1based >= 1)
        & (cell2_1based <= nCells)
    )
    owned_edge = (cell1_1based <= nCellsSolve) | (cell2_1based <= nCellsSolve)
    active_edge = edge_in_range & valid_cells & owned_edge

    return {
        "cell1": torch.clamp(cell1_1based - 1, min=0, max=max(0, nCells - 1)).to(
            dtype=torch.int64
        ),
        "cell2": torch.clamp(cell2_1based - 1, min=0, max=max(0, nCells - 1)).to(
            dtype=torch.int64
        ),
        "active_edge": active_edge,
        "edge_base_scale": 1.0 - specZoneMaskEdge[:nEdges],
    }


def _divergence_connectivity_payload(**kwargs: object) -> dict[str, Any]:
    """Build topology for direct callers; the runner owns warm reuse."""
    build_kwargs = {
        name: value
        for name, value in kwargs.items()
        if name not in _DIVERGENCE_PAYLOAD_KEY_ONLY_KWARGS
    }
    return _build_divergence_connectivity_payload(**build_kwargs)  # type: ignore[arg-type]


def _divergence_tensor_has_forward_ad(tensor: object) -> bool:
    if not isinstance(tensor, torch.Tensor):
        return False
    try:
        _primal, tangent = torch.autograd.forward_ad.unpack_dual(tensor)
    except Exception:  # noqa: BLE001
        return False
    return tangent is not None


def _divergence_damping_region(
    *,
    ru_p_edges: torch.Tensor,
    theta_m: torch.Tensor,
    rtheta_pp: torch.Tensor,
    rtheta_pp_old: torch.Tensor,
    cell1: torch.Tensor,
    cell2: torch.Tensor,
    active_edge: torch.Tensor,
    edge_scale: torch.Tensor,
) -> torch.Tensor:
    """Return active-edge ru_p patch with scaled 3-D divergence damping applied."""
    # Subtract once on cell rows before gathering the identical edge values.
    div_cells = -(rtheta_pp - rtheta_pp_old)
    div_cell1 = gather_rows(div_cells, cell1)
    div_cell2 = gather_rows(div_cells, cell2)
    theta_sum = gather_rows(theta_m, cell1) + gather_rows(theta_m, cell2)
    nonzero_theta = torch.abs(theta_sum) >= 1.0e-30
    damp = edge_scale[:, None] * (div_cell2 - div_cell1) / theta_sum
    damp = torch.where(
        active_edge[:, None] & nonzero_theta, damp, torch.zeros_like(damp)
    )
    return ru_p_edges + damp


def divergence_damping_plan_owner_key(
    *,
    connectivity_identity: object,
    nVertLevels: int,
    dts: float,
    config_smdiv: float,
    config_len_disp: float,
    stage_key: tuple[object, ...] = (),
) -> tuple[object, ...]:
    """Return the exact runner-owned divergence-damping specialization key."""
    if dts <= 0.0:
        raise ValueError(f"dts must be > 0 for an enabled damping plan, got {dts}")
    return (
        "DivergenceDampingPlan.v1",
        tuple(stage_key),
        connectivity_identity,
        int(nVertLevels),
        float(dts),
        float(config_smdiv),
        float(config_len_disp),
    )


def _divergence_connectivity_plan_fingerprint(
    connectivity_payload: Mapping[str, torch.Tensor],
) -> tuple[object, ...]:
    return tuple(
        (name, _divergence_tensor_cache_key(connectivity_payload[name]))
        for name in ("cell1", "cell2", "active_edge", "edge_base_scale")
    )


def build_divergence_damping_plan(
    *,
    connectivity_payload: Mapping[str, torch.Tensor],
    nVertLevels: int,
    dts: float,
    config_smdiv: float,
    config_len_disp: float,
    stage_key: tuple[object, ...] = (),
    connectivity_key: tuple[object, ...] | None = None,
) -> DivergenceDampingPlan:
    """Bind topology and the damping coefficient around one numeric callable."""
    key = divergence_damping_plan_owner_key(
        connectivity_identity=(
            connectivity_key
            if connectivity_key is not None
            else _divergence_connectivity_plan_fingerprint(connectivity_payload)
        ),
        nVertLevels=nVertLevels,
        dts=dts,
        config_smdiv=config_smdiv,
        config_len_disp=config_len_disp,
        stage_key=stage_key,
    )
    coefficient = 2.0 * float(config_smdiv) * float(config_len_disp) / float(dts)
    static_arguments = MappingProxyType(
        {
            "cell1": connectivity_payload["cell1"],
            "cell2": connectivity_payload["cell2"],
            "active_edge": connectivity_payload["active_edge"],
            "edge_scale": coefficient * connectivity_payload["edge_base_scale"],
        }
    )

    def _bound_divergence_damping(
        ru_p_edges: torch.Tensor,
        theta_m: torch.Tensor,
        rtheta_pp: torch.Tensor,
        rtheta_pp_old: torch.Tensor,
    ) -> torch.Tensor:
        return _divergence_damping_region(
            **static_arguments,
            ru_p_edges=ru_p_edges,
            theta_m=theta_m,
            rtheta_pp=rtheta_pp,
            rtheta_pp_old=rtheta_pp_old,
        )

    return DivergenceDampingPlan(
        key=key,
        dynamic_argument_names=_DIVERGENCE_DYNAMIC_NAMES,
        static_arguments=static_arguments,
        numeric_callable=_bound_divergence_damping,
    )


def atm_divergence_damping_3d(
    # Dimensions
    nCells: int,
    nEdges: int,
    nCellsSolve: int,
    nVertLevels: int,
    # Loop bounds (1-based Fortran indices)
    edgeStart: int,
    edgeEnd: int,
    # Damping parameters
    dts: float,
    config_smdiv: float,
    config_len_disp: float,
    # Mesh geometry
    specZoneMaskEdge: torch.Tensor,  # (nEdges+1,) float64
    # Connectivity
    cellsOnEdge: torch.Tensor,  # (nEdges+1, 2) int
    # State variables
    theta_m: torch.Tensor,  # (nCells+1, nVertLevels) float64
    rtheta_pp: torch.Tensor,  # (nCells+1, nVertLevels) float64
    rtheta_pp_old: torch.Tensor,  # (nCells+1, nVertLevels) float64
    ru_p: torch.Tensor,  # (nEdges+1, nVertLevels) float64 - INPUT
    _compile_region_names: tuple[str, ...] | frozenset[str] | None = None,
    _compile_mode: str = "reduce-overhead",
    _compile_backend: str = "inductor",
    _connectivity_payload: Mapping[str, torch.Tensor] | None = None,
    _allow_inplace_update: bool = False,
    _numeric_plan: DivergenceDampingPlan | None = None,
) -> dict:
    """
    Apply 3D divergence damping to control acoustic noise.

    This implements "scaled 3d divergence damping" from the Fortran code.
    The damping modifies ru_p based on the divergence gradient computed from
    the time derivative of rtheta_pp between neighboring cells.

    Algorithm:
        1. Compute divergence at each cell as the time derivative of rtheta_pp:
           divCell = -(rtheta_pp - rtheta_pp_old)
        2. For each edge, compute the damping term:
           damp = coef_divdamp * (divCell2 - divCell1) * (1 - specZoneMaskEdge) / (theta_m1 + theta_m2)
        3. Update ru_p: ru_p += damp

    Args:
        nCells: Number of cells in mesh
        nEdges: Number of edges in mesh
        nCellsSolve: Number of cells to solve (owned cells)
        nVertLevels: Number of vertical levels
        edgeStart: First edge index to process (1-based)
        edgeEnd: Last edge index to process (1-based)
        dts: Acoustic timestep size in seconds
        config_smdiv: Divergence damping coefficient (typically 0.1)
        config_len_disp: Dispersion length scale for damping
        specZoneMaskEdge: Spectral zone mask for edges (0=interior, 1=boundary)
        cellsOnEdge: Cell indices on each side of edge (nEdges+1, 2)
        theta_m: Modified potential temperature (nCells+1, nVertLevels)
        rtheta_pp: Perturbed rho*theta at current time (nCells+1, nVertLevels)
        rtheta_pp_old: Perturbed rho*theta at previous time (nCells+1, nVertLevels)
        ru_p: Perturbed rho*u momentum (nEdges+1, nVertLevels)

    Returns:
        Dictionary with:
            'ru_p': Updated ru_p tensor with divergence damping applied

    Notes:
        - Uses "scaled" divergence damping (not "unscaled" variant)
        - Only applies to interior edges (specZoneMaskEdge == 0)
        - Only processes edges where at least one cell is owned (cell <= nCellsSolve)
        - Damping coefficient: 2 * smdiv * len_disp / dts
    """
    require_consistent_device(
        "atm_divergence_damping_3d",
        {
            "specZoneMaskEdge": specZoneMaskEdge,
            "cellsOnEdge": cellsOnEdge,
            "theta_m": theta_m,
            "rtheta_pp": rtheta_pp,
            "rtheta_pp_old": rtheta_pp_old,
            "ru_p": ru_p,
        },
    )

    requested_compile_region_names = (
        frozenset(_compile_region_names or ())
        & _DIVERGENCE_DAMPING_REGION_COMPILE_TARGETS
    )
    autodiff_tensors = (theta_m, rtheta_pp, rtheta_pp_old, ru_p)
    autodiff_active = bool(
        is_grad_mode()
        or any(getattr(tensor, "requires_grad", False) for tensor in autodiff_tensors)
        or any(_divergence_tensor_has_forward_ad(tensor) for tensor in autodiff_tensors)
    )
    compile_region_names = (
        frozenset() if autodiff_active else requested_compile_region_names
    )

    def _maybe_compile_divergence_region(fn_name: str, fn, specialization_key=None):
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

    (
        bool(_allow_inplace_update)
        and not torch.is_grad_enabled()
        and not _compile_active()
    )
    # Functional execution reads ``ru_p`` and materializes exactly once when
    # the active patch is packed below. A defensive pre-clone followed by the
    # functional pack writes the full edge field twice without adding alias
    # safety. The in-place branch remains explicit and runner-owned.
    ru_p_out = ru_p

    # Compute damping coefficient
    # Preserve the upstream reciprocal-then-multiply association.
    #          coef_divdamp = 2.0 * smdiv * config_len_disp * rdts
    if dts > 0.0:
        rdts = 1.0 / dts
    else:
        # No damping if timestep is zero or negative
        return {"ru_p": ru_p_out}

    coef_divdamp = 2.0 * config_smdiv * config_len_disp * rdts

    # Skip if no damping
    if coef_divdamp == 0.0:
        return {"ru_p": ru_p_out}

    edge_start = max(edgeStart - 1, 0)
    edge_end = min(edgeEnd, nEdges)
    conn: Mapping[str, Any]
    if _connectivity_payload is None:
        conn = _divergence_connectivity_payload(
            device=ru_p_out.device,
            nCells=nCells,
            nEdges=nEdges,
            nCellsSolve=nCellsSolve,
            edge_start=edge_start,
            edge_end=edge_end,
            cellsOnEdge=cellsOnEdge,
            specZoneMaskEdge=specZoneMaskEdge,
        )
    else:
        conn = _connectivity_payload

    if _numeric_plan is None:
        _numeric_plan = build_divergence_damping_plan(
            connectivity_payload=conn,
            nVertLevels=nVertLevels,
            dts=dts,
            config_smdiv=config_smdiv,
            config_len_disp=config_len_disp,
            stage_key=("direct_call",),
        )
    numeric_callable = _numeric_plan.numeric_callable
    if "dynamics_tendency_divergence_damping_region" in compile_region_names:
        numeric_callable = _maybe_compile_divergence_region(
            "dynamics_tendency_divergence_damping_region",
            numeric_callable,
            specialization_key=_numeric_plan.key,
        )
    dynamic_arguments = {
        "ru_p_edges": ru_p_out[:nEdges, :],
        "theta_m": theta_m,
        "rtheta_pp": rtheta_pp,
        "rtheta_pp_old": rtheta_pp_old,
    }
    edge_patch = numeric_callable(*_numeric_plan.dynamic_values(dynamic_arguments))
    if not torch.is_grad_enabled() and not _compile_active():
        ru_p_out[:nEdges, :] = edge_patch
    else:
        ru_p_out = _replace_row_slice(ru_p_out, 0, nEdges, edge_patch)

    return {"ru_p": ru_p_out}
