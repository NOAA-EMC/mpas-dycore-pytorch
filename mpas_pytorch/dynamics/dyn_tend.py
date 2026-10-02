"""
Dynamical tendency computation (dynamics-tendency).

Computes the RHS of the momentum and continuity equations for explicit stepping.
This is one of the most complex kernels, involving all momentum advection,
Coriolis, pressure gradient, and vertical mixing terms.


Subroutine: atm_compute_dyn_tend_work

GPU invariants for this kernel:
- Workspace scratch tensors are reset in place (`.zero_()`) under inference-only
  execution. If autograd is ever required, switch these paths to fresh allocations.
- Scatter-style accumulations rely on MPAS mesh topology producing duplicate-free
  index sets within each stencil gather (for example `edge_ids` / `solve_cells`).
  This keeps updates race-free without atomics.

Scratch reuse note: inference-only scratch reuse; not for autograd paths.
"""

import inspect
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from time import perf_counter
from types import MappingProxyType
from typing import Any, cast

import torch
import torch.nn.functional as F

from mpas_pytorch.autodiff import is_grad_mode
from mpas_pytorch.autodiff.soft_limiter import (
    fct_abs,
    fct_max,
    fct_min,
    fct_sign_pm,
    fct_sqrt_pos,
)
from mpas_pytorch.constants import DTYPE, GRAVITY, OMEGA, PRANDTL
from mpas_pytorch.dynamics.static_payloads import (
    StaticPairRelation,
    StaticSlotRelation,
    build_ordered_output_transpose,
)
from mpas_pytorch.dynamics.stencil_ops import (
    _accumulate_ordered_slot,
    _active_slot_count,
    _compile_active,
    _gather_slot_batch,
    _replace_row_slice,
    _replace_row_slice_cols,
    _slot_batches,
    _slot_gather_batch,
    _static_connectivity_cache_allowed,
    _static_mask_all_true,
    _static_slot_all_true,
    _tensor_carries_autodiff,
    _tensor_version_cache_key,
    cell_divergence_gather,
    gather_rows,
)
from mpas_pytorch.dynamics.stencil_ops import (
    sum_fortran_order as _sum_fortran_order,
)
from mpas_pytorch.mesh import MESHState
from mpas_pytorch.profiling import profile_section
from mpas_pytorch.utils.checks import require_consistent_device

_DYNAMICS_TENDENCY_WHOLE_STAGE_COMPILE_TARGET = "dynamics_tendency_dyn_tend_whole_stage"
_DYNAMICS_TENDENCY_REGION_COMPILE_TARGETS = frozenset(
    {
        _DYNAMICS_TENDENCY_WHOLE_STAGE_COMPILE_TARGET,
        "dynamics_tendency_normal_velocity_tendency_normal_velocity_section",
        "dynamics_tendency_normal_velocity_mixing_u_mixing_section",
        "dynamics_tendency_normal_velocity_finalization_u_final_section",
        "dynamics_tendency_normal_velocity_mixing_normal_velocity_finalization_u_tendency_region",
        "dynamics_tendency_vertical_velocity_horizontal_advection_w_hadv_section",
        "dynamics_tendency_vertical_velocity_horizontal_mixing_w_horizontal_mix_section",
        "dynamics_tendency_vertical_velocity_horizontal_mixing_vertical_velocity_vertical_transport_w_tendency_region",
        "dynamics_tendency_vertical_velocity_vertical_transport_w_vertical_section",
        "dynamics_tendency_theta_horizontal_advection_theta_hadv_section",
        "dynamics_tendency_theta_horizontal_advection_theta_finalization_theta_tendency_region",
        "dynamics_tendency_theta_horizontal_mixing_theta_horizontal_mix_section",
        "dynamics_tendency_theta_vertical_advection_diabatic_theta_vertical_section",
        "dynamics_tendency_theta_vertical_mixing_theta_finalization_theta_final_section",
    }
)

# Evolving fields cross the whole-stage compiled boundary. Dimensions,
# connectivity, metrics, namelist choices, and orchestration controls are bound
# once by DynamicsTendencyNumericPlan and never appear as runtime compiler inputs.
_DYNAMICS_TENDENCY_DYNAMIC_ARGUMENT_NAMES = (
    "u",
    "v",
    "w",
    "ru",
    "rw",
    "ru_save",
    "rw_save",
    "theta_m",
    "theta_m_save",
    "rho_zz",
    "rho_edge",
    "divergence",
    "vorticity",
    "ke",
    "pv_edge",
    "pp",
    "rr_save",
    "rb",
    "zgrid",
    "zz",
    "zxu",
    "cqw",
    "cqu",
    "rt_diabatic_tend",
    "ur_cell",
    "vr_cell",
    "qtot",
    "kdiff_in",
    "tend_u_euler_in",
    "tend_w_euler_in",
    "tend_theta_euler_in",
    "tend_rho_in",
    "tend_ru_physics",
    "tend_rtheta_physics",
    "tend_rho_physics",
)

# Retain these legacy/public arguments at the API boundary, but do not expose
# them as whole-stage compiler inputs.  The canonical numeric body does not read
# them.  Tests verify this from the AST so a future equation change cannot
# silently consume the plan-owned ``None`` placeholders.
_DYNAMICS_TENDENCY_UNUSED_SOURCE_ARGUMENT_NAMES = (
    "qv_init",
    "exner",
    "rr",
    "pressure_b",
    "scalars",
    "tend_w_pgf",
    "tend_w_buoy",
)

# These public arguments are consumed when the runner constructs the immutable
# connectivity payload. The compiled numeric callable must not receive them a
# second time: doing so creates duplicate-input guards without adding data. Their
# mutation-sensitive fingerprints remain part of DynamicsTendencyNumericPlan.key.
_DYNAMICS_TENDENCY_PAYLOAD_OWNED_SOURCE_ARGUMENT_NAMES = (
    "cellsOnEdge",
    "verticesOnEdge",
    "edgesOnCell",
    "edgesOnEdge",
    "cellsOnCell",
    "edgesOnVertex",
    "nEdgesOnCell",
    "nEdgesOnEdge",
    "advCellsForEdge",
    "nAdvCellsForEdge",
    "adv_coefs",
    "adv_coefs_3rd",
    "weightsOnEdge",
    "deriv_two",
    "edgesOnCell_sign",
    "edgesOnVertex_sign",
)
_DYNAMICS_TENDENCY_COMPACT_OUTPUT_NAMES = (
    "tend_u_solve_patch",
    "tend_rho_cell_patch",
    "tend_theta_solve_patch",
    "tend_w_solve_inner_patch",
    "tend_u_euler_edge_patch",
    "tend_w_euler_cell_patch",
    "tend_theta_euler_cell_patch",
    "h_divergence_cell_patch",
    "kdiff_cell_patch",
    "rthdynten_solve_patch",
)
_DYNAMICS_TENDENCY_SLOT_GATHER_BATCH = 4
_DYNAMICS_TENDENCY_EDGE_NEIGHBOR_FULL_SLOT_COMPILE_LIMIT = 10
_DYNAMICS_TENDENCY_AD_COMPILE_MODES = frozenset(
    {"eager", "native_autograd", "compiled_adjoint", "whole_region_aot"}
)


@dataclass(frozen=True)
class DynamicsTendencyADCompilePlan:
    """Runner-owned execution policy for the three ordered dynamics-tendency adjoints."""

    mode: str
    device_type: str
    compile_mode: str
    compile_backend: str
    normal_velocity_tendency_edge_neighbor_pv_backward: Callable[..., Any]
    normal_velocity_tendency_vertical_backward: Callable[..., Any]
    edge_advective_cell_tendency_backward: Callable[..., Any]

    @property
    def compiled_adjoints(self) -> bool:
        return self.mode == "compiled_adjoint"

    @property
    def ordered_adjoints(self) -> bool:
        """Whether reverse AD uses the explicit Fortran-ordered transposes."""
        return self.mode != "native_autograd"

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "device_type": self.device_type,
            "compile_mode": self.compile_mode,
            "compile_backend": self.compile_backend,
            "compiled_adjoints": self.compiled_adjoints,
            "ordered_adjoints": self.ordered_adjoints,
        }


_DYNAMICS_TENDENCY_AD_COMPILE_PLAN: ContextVar[DynamicsTendencyADCompilePlan | None] = (
    ContextVar(
        "_DYNAMICS_TENDENCY_AD_COMPILE_PLAN",
        default=None,
    )
)


@contextmanager
def use_dynamics_tendency_ad_compile_plan(
    plan: DynamicsTendencyADCompilePlan,
) -> Iterator[None]:
    """Install one immutable dynamics-tendency AD plan for a differentiable forward."""
    token = _DYNAMICS_TENDENCY_AD_COMPILE_PLAN.set(plan)
    try:
        yield
    finally:
        _DYNAMICS_TENDENCY_AD_COMPILE_PLAN.reset(token)


def _planned_dynamics_tendency_adjoint(
    attribute: str,
    eager_callable: Callable[..., Any],
) -> Callable[..., Any]:
    """Capture the selected adjoint callable in an autograd context."""
    plan = _DYNAMICS_TENDENCY_AD_COMPILE_PLAN.get()
    if plan is None:
        return eager_callable
    selected = getattr(plan, attribute, None)
    if not callable(selected):
        raise RuntimeError(  # noqa: TRY004
            f"dynamics-tendency AD compile plan has no callable {attribute!r}"
        )
    return cast(Callable[..., Any], selected)


def resolve_dynamics_tendency_ad_compile_mode(
    config: Mapping[str, Any],
    *,
    device: torch.device | str,
) -> str:
    """Resolve one explicit runner-owned dynamics-tendency AD plan and reject mixed modes."""
    from mpas_pytorch.dynamics.compile_utils import (
        compile_enabled,
        compile_target_kernels,
    )

    requested = str(config.get("config_dynamics_tendency_ad_compile_plan", "auto"))
    requested = requested.strip().lower().replace("-", "_")
    if requested != "auto" and requested not in _DYNAMICS_TENDENCY_AD_COMPILE_MODES:
        allowed = ", ".join(["auto", *sorted(_DYNAMICS_TENDENCY_AD_COMPILE_MODES)])
        raise ValueError(
            f"unknown dynamics-tendency AD compile mode {requested!r}; allowed: {allowed}"
        )
    compile_requested = compile_enabled(dict(config))
    targets = compile_target_kernels(dict(config)) if compile_requested else ()
    whole_region_selected = (
        compile_requested
        and targets is not None
        and _DYNAMICS_TENDENCY_WHOLE_STAGE_COMPILE_TARGET in targets
    )
    device_type = torch.device(device).type
    # Compile changes the ordered adjoint's floating-point accumulation on CPU.
    # Keep the implicit policy exact; compiled modes require an explicit runner
    # request until their AD drift proof is independently authorized.
    effective = (
        ("whole_region_aot" if whole_region_selected else "eager")
        if requested == "auto"
        else requested
    )
    if whole_region_selected != (effective == "whole_region_aot"):
        raise ValueError(
            "dynamics-tendency whole-region AOTAutograd mode must exactly match the "
            "dynamics_tendency_dyn_tend_whole_stage compile target"
        )
    if effective == "compiled_adjoint" and device_type != "cpu":
        raise ValueError(
            "compiled dynamics-tendency ordered adjoints are CPU-qualified only; "
            f"device {device_type!r} remains fail-closed"
        )
    return effective


class _NoProfileSection:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def _dynamics_tendency_slot_batches(
    slot_count: int, batch_size: int = _DYNAMICS_TENDENCY_SLOT_GATHER_BATCH
) -> tuple[tuple[int, int], ...]:
    width = max(1, min(int(batch_size), max(int(slot_count), 1)))
    bounds: list[tuple[int, int]] = []
    for start in range(0, int(slot_count), width):
        bounds.append((start, min(start + width, int(slot_count))))
    return tuple(bounds)


def _dynamics_tendency_slot_gather_batch(
    source: torch.Tensor, slot_count: int | None = None
) -> int:
    if source.device.type == "cuda" or _compile_active():
        if slot_count is not None:
            slots = max(int(slot_count), 1)
            if slots <= 8:
                return slots
        return _DYNAMICS_TENDENCY_SLOT_GATHER_BATCH
    return 1


def _dynamics_tendency_long_index(index: torch.Tensor) -> torch.Tensor:
    return index if index.dtype == torch.int64 else index.to(dtype=torch.int64)


def _dynamics_tendency_gather_slot_batch(
    field: torch.Tensor,
    indices: torch.Tensor,
    start: int,
    stop: int,
) -> torch.Tensor:
    if stop - start == 1:
        return gather_rows(field, indices[:, start]).unsqueeze(1)
    return gather_rows(field, indices[:, start:stop])


def _dynamics_tendency_gather_rows_patch_or_base(
    field: torch.Tensor,
    indices: torch.Tensor,
    patch: torch.Tensor | None,
    patch_row_start: int,
    patch_row_end: int,
) -> torch.Tensor:
    """Gather rows from an active patch when possible, falling back to base rows.

    This avoids materializing a full mesh-sized scratch tensor solely so later
    topology-indexed stencils can read active rows.
    """
    if patch is None:
        return gather_rows(field, indices)
    patch_row_start = int(patch_row_start)
    patch_row_end = int(patch_row_end)
    if patch_row_end <= patch_row_start or patch.shape[0] == 0:
        return gather_rows(field, indices)

    rel = indices - patch_row_start
    in_patch = (indices >= patch_row_start) & (indices < patch_row_end)
    if not _compile_active() and field.device.type == "cpu":
        all_true = _static_mask_all_true(in_patch)
        if all_true is True:
            return gather_rows(patch, rel)

    base_values = gather_rows(field, indices)
    safe_rel = torch.clamp(rel, min=0, max=patch.shape[0] - 1)
    patch_values = gather_rows(patch, safe_rel)
    mask = in_patch
    while mask.ndim < patch_values.ndim:
        mask = mask.unsqueeze(-1)
    return torch.where(mask, patch_values, base_values)


def _dynamics_tendency_replace_patch_cols(
    patch: torch.Tensor,
    col_start: int,
    col_end: int,
    values: torch.Tensor,
) -> torch.Tensor:
    """Functionally replace a fixed column range in an active dynamics-tendency patch."""
    col_start = int(col_start)
    col_end = int(col_end)
    if col_start < 0 or col_end < col_start or col_end > patch.shape[1]:
        raise ValueError(
            f"invalid active patch column slice [{col_start}:{col_end}] "
            f"for dim-1 size {patch.shape[1]}"
        )
    if (col_end - col_start) != values.shape[1]:
        raise ValueError(
            f"active patch column slice length {col_end - col_start} "
            f"does not match values columns {values.shape[1]}"
        )
    if col_start == 0 and col_end == patch.shape[1]:
        return values
    if col_start == col_end:
        return patch
    out = patch.clone()
    out[:, col_start:col_end, ...] = values
    return out


def _dynamics_tendency_replace_patch_rows(
    patch: torch.Tensor,
    row_start: int,
    row_end: int,
    values: torch.Tensor,
) -> torch.Tensor:
    """Functionally replace a fixed row range in an active dynamics-tendency patch."""
    row_start = int(row_start)
    row_end = int(row_end)
    if row_start < 0 or row_end < row_start or row_end > patch.shape[0]:
        raise ValueError(
            f"invalid active patch row slice [{row_start}:{row_end}] "
            f"for dim-0 size {patch.shape[0]}"
        )
    if (row_end - row_start) != values.shape[0]:
        raise ValueError(
            f"active patch row slice length {row_end - row_start} "
            f"does not match values rows {values.shape[0]}"
        )
    if row_start == 0 and row_end == patch.shape[0]:
        return values
    if row_start == row_end:
        return patch
    out = patch.clone()
    out[row_start:row_end, ...] = values
    return out


def _dynamics_tendency_replace_patch_rows_cols(
    patch: torch.Tensor,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    values: torch.Tensor,
) -> torch.Tensor:
    """Functionally replace a fixed row/column rectangle in a dynamics-tendency patch."""
    row_start = int(row_start)
    row_end = int(row_end)
    col_start = int(col_start)
    col_end = int(col_end)
    if row_start < 0 or row_end < row_start or row_end > patch.shape[0]:
        raise ValueError(
            f"invalid active patch row slice [{row_start}:{row_end}] "
            f"for dim-0 size {patch.shape[0]}"
        )
    if col_start < 0 or col_end < col_start or col_end > patch.shape[1]:
        raise ValueError(
            f"invalid active patch column slice [{col_start}:{col_end}] "
            f"for dim-1 size {patch.shape[1]}"
        )
    if (row_end - row_start) != values.shape[0]:
        raise ValueError(
            f"active patch row slice length {row_end - row_start} "
            f"does not match values rows {values.shape[0]}"
        )
    if (col_end - col_start) != values.shape[1]:
        raise ValueError(
            f"active patch column slice length {col_end - col_start} "
            f"does not match values columns {values.shape[1]}"
        )
    if row_start == 0 and row_end == patch.shape[0]:
        return _dynamics_tendency_replace_patch_cols(patch, col_start, col_end, values)
    if col_start == 0 and col_end == patch.shape[1]:
        return _dynamics_tendency_replace_patch_rows(patch, row_start, row_end, values)
    if row_start == row_end or col_start == col_end:
        return patch
    out = patch.clone()
    out[row_start:row_end, col_start:col_end, ...] = values
    return out


_DYNAMICS_TENDENCY_PAIR_RELATION_FIELDS = (
    "first",
    "second",
    "valid",
    "valid_2d",
    "valid_all",
    "first_local",
    "second_local",
    "first_local_valid",
    "second_local_valid",
    "first_local_all",
    "second_local_all",
    "active",
    "active_3d",
    "active_all",
    "first_transpose_destinations",
    "first_transpose_valid",
    "second_transpose_destinations",
    "second_transpose_valid",
)
_DYNAMICS_TENDENCY_SLOT_RELATION_FIELDS = (
    "indices",
    "valid",
    "sign",
    "weights",
    "coefficients",
    "coefficients_3rd",
    "valid_all_by_slot",
    "valid_weight",
    "local_indices",
    "local_valid",
    "local_all",
    "transpose_destinations",
    "transpose_valid",
)


def _dynamics_tendency_numeric_relation(value: Any) -> Any:
    """Flatten a static relation into tensor/scalar values for Dynamo."""
    if isinstance(value, StaticPairRelation):
        return tuple(
            getattr(value, field) for field in _DYNAMICS_TENDENCY_PAIR_RELATION_FIELDS
        )
    if isinstance(value, StaticSlotRelation):
        return tuple(
            getattr(value, field) for field in _DYNAMICS_TENDENCY_SLOT_RELATION_FIELDS
        )
    return value


def _dynamics_tendency_pair_field(relation: Any, index: int, name: str) -> Any:
    if isinstance(relation, tuple):
        return relation[index]
    return getattr(relation, name)


def _dynamics_tendency_slot_field(relation: Any, index: int, name: str) -> Any:
    if isinstance(relation, tuple):
        return relation[index]
    return getattr(relation, name)


def _dynamics_tendency_relation_slot_slice(
    value: torch.Tensor | None,
    slot_start: int | None,
    slot_stop: int | None,
) -> torch.Tensor | None:
    if value is None or slot_start is None or slot_stop is None or value.ndim < 2:
        return value
    return value[:, int(slot_start) : int(slot_stop)]


def _dynamics_tendency_pair_gather_side_patch_or_base(
    field: torch.Tensor,
    patch: torch.Tensor,
    indices: torch.Tensor,
    local_indices: torch.Tensor | None,
    local_valid: torch.Tensor | None,
    local_all: bool | None,
    patch_row_start: int,
    patch_row_end: int,
) -> torch.Tensor:
    if local_indices is None or local_valid is None:
        rel = indices - patch_row_start
        local_valid = (indices >= patch_row_start) & (indices < patch_row_end)
        safe_local = torch.clamp(rel, min=0, max=patch.shape[0] - 1)
    else:
        safe_local = local_indices
    if local_all is True:
        return gather_rows(patch, safe_local)
    patch_values = gather_rows(patch, safe_local)
    # Whole-dynamics-tendency compact mode represents an otherwise-zero public base with one
    # row.  Mixed local/nonlocal relations must use zero fill for rows outside
    # the active patch instead of materializing or indexing the padded field.
    if field.shape[0] == 1 and patch_row_end - patch_row_start > field.shape[0]:
        base_values = torch.zeros_like(patch_values)
    else:
        base_values = gather_rows(field, indices)
    mask = local_valid
    while mask.ndim < patch_values.ndim:
        mask = mask.unsqueeze(-1)
    return torch.where(mask, patch_values, base_values)


def _dynamics_tendency_pair_gather_patch_or_base(
    relation: Any,
    field: torch.Tensor,
    patch: torch.Tensor | None,
    *,
    patch_row_start: int,
    patch_row_end: int,
    slot_start: int | None = None,
    slot_stop: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    first = _dynamics_tendency_relation_slot_slice(
        _dynamics_tendency_pair_field(relation, 0, "first"), slot_start, slot_stop
    )
    second = _dynamics_tendency_relation_slot_slice(
        _dynamics_tendency_pair_field(relation, 1, "second"), slot_start, slot_stop
    )
    assert first is not None and second is not None
    if patch is None or patch_row_end <= patch_row_start or patch.shape[0] == 0:
        return gather_rows(field, first), gather_rows(field, second)
    first_local = _dynamics_tendency_relation_slot_slice(
        _dynamics_tendency_pair_field(relation, 5, "first_local"), slot_start, slot_stop
    )
    second_local = _dynamics_tendency_relation_slot_slice(
        _dynamics_tendency_pair_field(relation, 6, "second_local"),
        slot_start,
        slot_stop,
    )
    first_local_valid = _dynamics_tendency_relation_slot_slice(
        _dynamics_tendency_pair_field(relation, 7, "first_local_valid"),
        slot_start,
        slot_stop,
    )
    second_local_valid = _dynamics_tendency_relation_slot_slice(
        _dynamics_tendency_pair_field(relation, 8, "second_local_valid"),
        slot_start,
        slot_stop,
    )
    return (
        _dynamics_tendency_pair_gather_side_patch_or_base(
            field,
            patch,
            first,
            first_local,
            first_local_valid,
            _dynamics_tendency_pair_field(relation, 9, "first_local_all"),
            int(patch_row_start),
            int(patch_row_end),
        ),
        _dynamics_tendency_pair_gather_side_patch_or_base(
            field,
            patch,
            second,
            second_local,
            second_local_valid,
            _dynamics_tendency_pair_field(relation, 10, "second_local_all"),
            int(patch_row_start),
            int(patch_row_end),
        ),
    )


def _dynamics_tendency_pair_gather_slot_patch_or_base(
    relation: Any,
    field: torch.Tensor,
    patch: torch.Tensor | None,
    *,
    patch_row_start: int,
    patch_row_end: int,
    slot: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather one relation slot without creating a width-one batch axis."""
    first = _dynamics_tendency_pair_field(relation, 0, "first")[:, slot]
    second = _dynamics_tendency_pair_field(relation, 1, "second")[:, slot]
    if patch is None or patch_row_end <= patch_row_start or patch.shape[0] == 0:
        return gather_rows(field, first), gather_rows(field, second)
    first_local = _dynamics_tendency_pair_field(relation, 5, "first_local")
    second_local = _dynamics_tendency_pair_field(relation, 6, "second_local")
    first_local_valid = _dynamics_tendency_pair_field(relation, 7, "first_local_valid")
    second_local_valid = _dynamics_tendency_pair_field(
        relation, 8, "second_local_valid"
    )
    return (
        _dynamics_tendency_pair_gather_side_patch_or_base(
            field,
            patch,
            first,
            first_local[:, slot] if first_local is not None else None,
            first_local_valid[:, slot] if first_local_valid is not None else None,
            _dynamics_tendency_pair_field(relation, 9, "first_local_all"),
            int(patch_row_start),
            int(patch_row_end),
        ),
        _dynamics_tendency_pair_gather_side_patch_or_base(
            field,
            patch,
            second,
            second_local[:, slot] if second_local is not None else None,
            second_local_valid[:, slot] if second_local_valid is not None else None,
            _dynamics_tendency_pair_field(relation, 10, "second_local_all"),
            int(patch_row_start),
            int(patch_row_end),
        ),
    )


def _dynamics_tendency_slot_gather_patch_or_base(
    relation: Any,
    field: torch.Tensor,
    patch: torch.Tensor | None,
    *,
    patch_row_start: int,
    patch_row_end: int,
    slot_start: int | None = None,
    slot_stop: int | None = None,
) -> torch.Tensor:
    indices = _dynamics_tendency_relation_slot_slice(
        _dynamics_tendency_slot_field(relation, 0, "indices"), slot_start, slot_stop
    )
    assert indices is not None
    if patch is None:
        return gather_rows(field, indices)
    local_indices = _dynamics_tendency_relation_slot_slice(
        _dynamics_tendency_slot_field(relation, 8, "local_indices"),
        slot_start,
        slot_stop,
    )
    local_valid = _dynamics_tendency_relation_slot_slice(
        _dynamics_tendency_slot_field(relation, 9, "local_valid"), slot_start, slot_stop
    )
    return _dynamics_tendency_pair_gather_side_patch_or_base(
        field,
        patch,
        indices,
        local_indices,
        local_valid,
        _dynamics_tendency_slot_field(relation, 10, "local_all"),
        int(patch_row_start),
        int(patch_row_end),
    )


def _dynamics_tendency_gather_pair_patch_or_base_cols(
    relation: Any,
    field: torch.Tensor,
    patch: torch.Tensor | None,
    *,
    patch_row_start: int,
    patch_row_end: int,
    col_start: int,
    col_stop: int,
    patch_col_start: int = 0,
    slot_start: int | None = None,
    slot_stop: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather pair-relation columns from a compact active patch or base field.

    This is equivalent to materializing ``field[patch_row_start:patch_row_end,
    patch_col_start:patch_col_start + patch.shape[1]] = patch`` and then
    gathering ``col_start:col_stop``.  It keeps compact column patches compact
    through dynamics-tendency reducers instead of padding them to public field width.
    """
    col_start = int(col_start)
    col_stop = int(col_stop)
    if col_start < 0 or col_stop < col_start or col_stop > field.shape[1]:
        raise ValueError(
            f"invalid gather column slice [{col_start}:{col_stop}] "
            f"for dim-1 size {field.shape[1]}"
        )
    field_window = field[:, col_start:col_stop, ...]
    first = _dynamics_tendency_relation_slot_slice(
        _dynamics_tendency_pair_field(relation, 0, "first"), slot_start, slot_stop
    )
    second = _dynamics_tendency_relation_slot_slice(
        _dynamics_tendency_pair_field(relation, 1, "second"), slot_start, slot_stop
    )
    assert first is not None and second is not None
    if patch is None:
        return gather_rows(field_window, first), gather_rows(field_window, second)

    patch_row_start = int(patch_row_start)
    patch_row_end = int(patch_row_end)
    if patch_row_end <= patch_row_start or patch.shape[0] == 0:
        return gather_rows(field_window, first), gather_rows(field_window, second)
    patch_col_start = int(patch_col_start)
    patch_col_stop = patch_col_start + int(patch.shape[1])
    if col_start < patch_col_start or col_stop > patch_col_stop:
        raise ValueError(
            f"compact patch columns [{patch_col_start}:{patch_col_stop}] "
            f"do not cover requested gather columns [{col_start}:{col_stop}]"
        )
    patch_window = patch[
        :, col_start - patch_col_start : col_stop - patch_col_start, ...
    ]

    first_local = _dynamics_tendency_relation_slot_slice(
        _dynamics_tendency_pair_field(relation, 5, "first_local"), slot_start, slot_stop
    )
    second_local = _dynamics_tendency_relation_slot_slice(
        _dynamics_tendency_pair_field(relation, 6, "second_local"),
        slot_start,
        slot_stop,
    )
    first_local_valid = _dynamics_tendency_relation_slot_slice(
        _dynamics_tendency_pair_field(relation, 7, "first_local_valid"),
        slot_start,
        slot_stop,
    )
    second_local_valid = _dynamics_tendency_relation_slot_slice(
        _dynamics_tendency_pair_field(relation, 8, "second_local_valid"),
        slot_start,
        slot_stop,
    )
    return (
        _dynamics_tendency_pair_gather_side_patch_or_base(
            field_window,
            patch_window,
            first,
            first_local,
            first_local_valid,
            _dynamics_tendency_pair_field(relation, 9, "first_local_all"),
            patch_row_start,
            patch_row_end,
        ),
        _dynamics_tendency_pair_gather_side_patch_or_base(
            field_window,
            patch_window,
            second,
            second_local,
            second_local_valid,
            _dynamics_tendency_pair_field(relation, 10, "second_local_all"),
            patch_row_start,
            patch_row_end,
        ),
    )


def _dynamics_tendency_can_use_no_ad_full_slot_reduce(
    reference: torch.Tensor, *tensors: torch.Tensor | None
) -> bool:
    """Return True for full-slot row-owned reductions in inference/compile paths."""
    if torch.is_grad_enabled():
        return False
    if reference.device.type != "cuda" and not _compile_active():
        return False
    if _tensor_carries_autodiff(reference):
        return False
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor) and _tensor_carries_autodiff(tensor):
            return False
    return True


def flux4(
    q_im2: torch.Tensor,
    q_im1: torch.Tensor,
    q_i: torch.Tensor,
    q_ip1: torch.Tensor,
    ua: torch.Tensor,
) -> torch.Tensor:
    """
    4th-order horizontal flux.
    upstream MPAS statement-function evaluation.
    """
    return ua * (7.0 * (q_i + q_im1) - (q_ip1 + q_im2)) / 12.0


def flux3(
    q_im2: torch.Tensor,
    q_im1: torch.Tensor,
    q_i: torch.Tensor,
    q_ip1: torch.Tensor,
    ua: torch.Tensor,
    coef3: float,
) -> torch.Tensor:
    """
    3rd-order horizontal flux with upwind bias.
    upstream MPAS statement-function evaluation.
    """
    return _flux3_from_abs(
        q_im2,
        q_im1,
        q_i,
        q_ip1,
        ua,
        fct_abs(ua),
        coef3,
    )


def _flux3_from_abs(
    q_im2: torch.Tensor,
    q_im1: torch.Tensor,
    q_i: torch.Tensor,
    q_ip1: torch.Tensor,
    ua: torch.Tensor,
    abs_ua: torch.Tensor,
    coef3: float,
) -> torch.Tensor:
    """Evaluate ``flux3`` with an explicitly supplied absolute-value primal."""
    return (
        flux4(q_im2, q_im1, q_i, q_ip1, ua)
        + coef3 * abs_ua * ((q_ip1 - q_im2) - 3.0 * (q_i - q_im1)) / 12.0
    )


def _full_level_vertical_flux_delta(
    interface_flux: torch.Tensor,
    n_levels: int,
) -> torch.Tensor:
    """Match padded[:, 1:n+1] - padded[:, :n] with zero boundary fluxes."""
    expected_interfaces = max(n_levels - 1, 0)
    if interface_flux.shape[1] != expected_interfaces:
        raise ValueError(
            "interface_flux has wrong vertical width: "
            f"got {interface_flux.shape[1]}, expected {expected_interfaces}"
        )
    if n_levels == 0:
        return interface_flux.new_zeros((interface_flux.shape[0], 0))
    if expected_interfaces == 0:
        return interface_flux.new_zeros((interface_flux.shape[0], n_levels))
    out = interface_flux.new_empty((interface_flux.shape[0], n_levels))
    out[:, 0] = interface_flux[:, 0]
    if n_levels > 2:
        out[:, 1:-1] = interface_flux[:, 1:] - interface_flux[:, :-1]
    out[:, -1] = -interface_flux[:, -1]
    return out


def _interior_level_vertical_flux_delta(
    interface_flux: torch.Tensor,
    n_levels: int,
) -> torch.Tensor:
    """Match padded[:, 2:n+1] - padded[:, 1:n] with zero top flux."""
    expected_interfaces = max(n_levels - 1, 0)
    if interface_flux.shape[1] != expected_interfaces:
        raise ValueError(
            "interface_flux has wrong vertical width: "
            f"got {interface_flux.shape[1]}, expected {expected_interfaces}"
        )
    if expected_interfaces == 0:
        return interface_flux.new_zeros((interface_flux.shape[0], 0))
    out = interface_flux.new_empty((interface_flux.shape[0], expected_interfaces))
    if expected_interfaces > 1:
        out[:, :-1] = interface_flux[:, 1:] - interface_flux[:, :-1]
    out[:, -1] = -interface_flux[:, -1]
    return out


def _full_level_vertical_flux_delta_from_parts(
    first_flux: torch.Tensor,
    middle_flux: torch.Tensor | None,
    last_flux: torch.Tensor | None,
    n_levels: int,
) -> torch.Tensor:
    """Build full-level vertical flux deltas without a materialized flux array."""
    if n_levels == 0:
        return first_flux.new_zeros((first_flux.shape[0], 0))
    if n_levels == 1:
        return first_flux.new_zeros((first_flux.shape[0], 1))
    if n_levels == 2:
        return torch.cat((first_flux.unsqueeze(1), (-first_flux).unsqueeze(1)), dim=1)
    if last_flux is None:
        raise ValueError("last_flux is required when n_levels > 2")
    if n_levels == 3:
        return torch.cat(
            (
                first_flux.unsqueeze(1),
                (last_flux - first_flux).unsqueeze(1),
                (-last_flux).unsqueeze(1),
            ),
            dim=1,
        )
    else:
        if middle_flux is None or middle_flux.shape[1] != n_levels - 3:
            raise ValueError(
                "middle_flux has wrong vertical width: "
                f"got {None if middle_flux is None else middle_flux.shape[1]}, "
                f"expected {n_levels - 3}"
            )
        segments = [
            first_flux.unsqueeze(1),
            middle_flux[:, :1] - first_flux.unsqueeze(1),
        ]
        if n_levels > 4:
            segments.append(middle_flux[:, 1:] - middle_flux[:, :-1])
        segments.extend(
            (
                last_flux.unsqueeze(1) - middle_flux[:, -1:],
                (-last_flux).unsqueeze(1),
            )
        )
        return torch.cat(tuple(segments), dim=1)


def _interior_level_vertical_flux_delta_from_parts(
    first_flux: torch.Tensor,
    middle_flux: torch.Tensor | None,
    last_flux: torch.Tensor | None,
    n_levels: int,
) -> torch.Tensor:
    """Build interior-level vertical flux deltas without a materialized flux array."""
    if n_levels <= 1:
        return first_flux.new_zeros((first_flux.shape[0], 0))
    if n_levels == 2:
        return (-first_flux).unsqueeze(1)
    if last_flux is None:
        raise ValueError("last_flux is required when n_levels > 2")
    if n_levels == 3:
        return torch.cat(
            ((last_flux - first_flux).unsqueeze(1), (-last_flux).unsqueeze(1)),
            dim=1,
        )
    else:
        if middle_flux is None or middle_flux.shape[1] != n_levels - 3:
            raise ValueError(
                "middle_flux has wrong vertical width: "
                f"got {None if middle_flux is None else middle_flux.shape[1]}, "
                f"expected {n_levels - 3}"
            )
        segments = [middle_flux[:, :1] - first_flux.unsqueeze(1)]
        if n_levels > 4:
            segments.append(middle_flux[:, 1:] - middle_flux[:, :-1])
        segments.extend(
            (
                last_flux.unsqueeze(1) - middle_flux[:, -1:],
                (-last_flux).unsqueeze(1),
            )
        )
        return torch.cat(tuple(segments), dim=1)


def _dynamics_tendency_tensor_has_forward_tangent(tensor: torch.Tensor) -> bool:
    try:
        _primal, tangent = torch.autograd.forward_ad.unpack_dual(tensor)
    except RuntimeError:
        return False
    return tangent is not None


def _dynamics_tendency_use_ordered_reverse_vjp(*tensors: torch.Tensor) -> bool:
    """Use measured output-owned transposes only for ordinary reverse AD."""
    # Compiled forward graphs never use the reverse-only custom VJPs.  Return
    # before consulting runner-owned Python policy so Dynamo sees tensor math
    # rather than a ContextVar access at every ordered stencil site.
    if _compile_active():
        return False
    plan = _DYNAMICS_TENDENCY_AD_COMPILE_PLAN.get()
    return (
        (plan is None or plan.ordered_adjoints)
        and is_grad_mode()
        and torch.is_grad_enabled()
        and any(tensor.requires_grad for tensor in tensors)
        and not any(
            _dynamics_tendency_tensor_has_forward_tangent(tensor) for tensor in tensors
        )
    )


def _density_tendency_density_section_region(
    *,
    ru: torch.Tensor,
    rw_cells: torch.Tensor,
    rb_cells: torch.Tensor,
    rr_save_cells: torch.Tensor,
    qtot_cells: torch.Tensor,
    dv_edge: torch.Tensor,
    inv_area_cell: torch.Tensor,
    rdzw: torch.Tensor,
    cell_edge_relation: StaticSlotRelation | tuple[Any, ...],
    gravity: float,
    include_density_update: bool,
    tend_rho_physics_cells: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Compute the complete active-cell dynamics-tendency density section."""
    safe_cell_edges = _dynamics_tendency_slot_field(cell_edge_relation, 0, "indices")
    cell_edge_valid = _dynamics_tendency_slot_field(cell_edge_relation, 1, "valid")
    cell_edge_sign = _dynamics_tendency_slot_field(cell_edge_relation, 2, "sign")
    valid_all_by_slot = _dynamics_tendency_slot_field(
        cell_edge_relation, 6, "valid_all_by_slot"
    )
    if not all(
        isinstance(value, torch.Tensor)
        for value in (safe_cell_edges, cell_edge_valid, cell_edge_sign)
    ):
        raise ValueError(
            "dynamics-tendency density-tendency requires indexed, signed cell-edge topology"
        )

    h_divergence = cell_divergence_gather(
        edge_field=ru,
        edge_metric=dv_edge,
        cell_edges_idx=safe_cell_edges,
        cell_edges_sign=cell_edge_sign,
        cell_edges_valid=cell_edge_valid,
        inv_area_cell=inv_area_cell,
        cell_edges_valid_all_by_slot=valid_all_by_slot,
    )
    if not include_density_update:
        return h_divergence, None, None

    n_vert_levels = h_divergence.shape[1]
    tend_rho = -h_divergence - rdzw[None, :n_vert_levels] * (
        rw_cells[:, 1 : n_vert_levels + 1] - rw_cells[:, :n_vert_levels]
    )
    if tend_rho_physics_cells is not None:
        tend_rho = tend_rho + tend_rho_physics_cells
    dpdz = -gravity * (rb_cells * qtot_cells + rr_save_cells * (1.0 + qtot_cells))
    return h_divergence, tend_rho, dpdz


def _normal_velocity_tendency_edge_neighbor_pv_ordered_numeric(
    u: torch.Tensor,
    pv_edge: torch.Tensor,
    pv_self: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    total: torch.Tensor | None = None
    reuse_private = not torch.is_grad_enabled() and not _compile_active()
    batch_size = _slot_gather_batch(u, indices.shape[1]) if _compile_active() else 1
    for batch_start, batch_stop in _slot_batches(indices.shape[1], batch_size):
        pv_batch = _gather_slot_batch(pv_edge, indices, batch_start, batch_stop)
        u_batch = _gather_slot_batch(u, indices, batch_start, batch_stop)
        for local_slot in range(batch_stop - batch_start):
            slot = batch_start + local_slot
            if reuse_private:
                # The gathers own fresh storage. Reuse it as OpenACC-style private
                # scratch while retaining the original operation and slot order.
                work_pv = pv_batch[:, local_slot, :]
                torch.add(pv_self, work_pv, out=work_pv)
                work_pv.mul_(0.5)
                contribution = u_batch[:, local_slot, :]
                torch.mul(weights[:, slot].unsqueeze(1), contribution, out=contribution)
                contribution.mul_(work_pv)
            else:
                work_pv = 0.5 * (pv_self + pv_batch[:, local_slot, :])
                contribution = (
                    weights[:, slot].unsqueeze(1) * u_batch[:, local_slot, :] * work_pv
                )
            total = _accumulate_ordered_slot(total, contribution)
    if total is None:
        return u.new_zeros((indices.shape[0], u.shape[1]))
    return total


def _normal_velocity_tendency_edge_neighbor_pv_ordered_backward(
    u: torch.Tensor,
    pv_edge: torch.Tensor,
    pv_self: torch.Tensor,
    grad_output: torch.Tensor,
    indices: torch.Tensor,
    valid: torch.Tensor,
    weights: torch.Tensor,
    transpose_destinations: torch.Tensor,
    transpose_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source_rows = pv_edge.shape[0]
    if source_rows > u.shape[0] or source_rows > transpose_destinations.shape[0]:
        raise ValueError(
            "dynamics-tendency normal-velocity-tendency ordered VJP source-row mismatch: "
            f"pv_edge={source_rows}, u={u.shape[0]}, "
            f"transpose={transpose_destinations.shape[0]}"
        )
    source_u = u[:source_rows]
    source_destinations = transpose_destinations[:source_rows]
    source_valid = transpose_valid[:source_rows]
    grad_u: torch.Tensor | None = None
    grad_pv: torch.Tensor | None = None
    for slot in range(indices.shape[1] - 1, -1, -1):
        slot_grad_u = torch.zeros_like(source_u)
        slot_grad_pv = torch.zeros_like(pv_edge)
        for duplicate in range(source_destinations.shape[2]):
            destination = source_destinations[:, slot, duplicate]
            active = source_valid[:, slot, duplicate].unsqueeze(1)
            gathered_grad = gather_rows(grad_output, destination)
            gathered_self = gather_rows(pv_self, destination)
            work_pv = 0.5 * (gathered_self + pv_edge)
            weight = gather_rows(weights[:, slot], destination).unsqueeze(1)
            u_part = (gathered_grad * work_pv) * weight
            pv_part = (gathered_grad * (weight * source_u)) * 0.5
            slot_grad_u.add_(torch.where(active, u_part, torch.zeros_like(u_part)))
            slot_grad_pv.add_(torch.where(active, pv_part, torch.zeros_like(pv_part)))
        grad_u = slot_grad_u if grad_u is None else grad_u + slot_grad_u
        grad_pv = slot_grad_pv if grad_pv is None else grad_pv + slot_grad_pv
    if grad_u is None or grad_pv is None:
        grad_u = torch.zeros_like(source_u)
        grad_pv = torch.zeros_like(pv_edge)
    if source_rows < u.shape[0]:
        grad_u = torch.cat((grad_u, torch.zeros_like(u[source_rows:])), dim=0)

    grad_self: torch.Tensor | None = None
    for slot in range(indices.shape[1] - 1, -1, -1):
        neighbor_u = gather_rows(u, indices[:, slot])
        masked_grad = torch.where(
            valid[:, slot].unsqueeze(1),
            grad_output,
            torch.zeros_like(grad_output),
        )
        contribution = (
            masked_grad * (weights[:, slot].unsqueeze(1) * neighbor_u)
        ) * 0.5
        grad_self = contribution if grad_self is None else grad_self + contribution
    if grad_self is None:
        grad_self = torch.zeros_like(pv_self)
    return grad_u, grad_pv, grad_self


class _NormalVelocityTendencyEdgeNeighborPVFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        u: torch.Tensor,
        pv_edge: torch.Tensor,
        pv_self: torch.Tensor,
        indices: torch.Tensor,
        valid: torch.Tensor,
        weights: torch.Tensor,
        transpose_destinations: torch.Tensor,
        transpose_valid: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(
            u,
            pv_edge,
            pv_self,
            indices,
            valid,
            weights,
            transpose_destinations,
            transpose_valid,
        )
        ctx.backward_callable = _planned_dynamics_tendency_adjoint(
            "normal_velocity_tendency_edge_neighbor_pv_backward",
            _normal_velocity_tendency_edge_neighbor_pv_ordered_backward,
        )
        return _normal_velocity_tendency_edge_neighbor_pv_ordered_numeric(
            u, pv_edge, pv_self, indices, weights
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        saved = ctx.saved_tensors
        grad_u, grad_pv, grad_self = ctx.backward_callable(
            saved[0],
            saved[1],
            saved[2],
            grad_output,
            saved[3],
            saved[4],
            saved[5],
            saved[6],
            saved[7],
        )
        return grad_u, grad_pv, grad_self, None, None, None, None, None


def _edge_neighbor_pv_reduction(
    *,
    edge_neighbor_relation: StaticSlotRelation,
    u: torch.Tensor,
    pv_edge: torch.Tensor,
    pv_self: torch.Tensor,
) -> torch.Tensor:
    """Reduce the dynamics-tendency PV edge-neighbor stencil without a full 3D work tensor."""
    eoe_safe = _dynamics_tendency_slot_field(edge_neighbor_relation, 0, "indices")
    eoe_valid = _dynamics_tendency_slot_field(edge_neighbor_relation, 1, "valid")
    edge_weights = _dynamics_tendency_slot_field(edge_neighbor_relation, 3, "weights")
    if eoe_valid is None:
        eoe_valid = torch.ones_like(eoe_safe, dtype=torch.bool)
    if edge_weights is None:
        raise ValueError(
            "dynamics-tendency edge-neighbor PV reduction requires relation weights"
        )
    n_edges = eoe_safe.shape[0]
    n_levels = u.shape[1]
    if eoe_safe.shape[1] == 0:
        return u.new_zeros((n_edges, n_levels))

    transpose_destinations = _dynamics_tendency_slot_field(
        edge_neighbor_relation, 11, "transpose_destinations"
    )
    transpose_valid = _dynamics_tendency_slot_field(
        edge_neighbor_relation, 12, "transpose_valid"
    )
    if (
        transpose_destinations is not None
        and transpose_valid is not None
        and _dynamics_tendency_use_ordered_reverse_vjp(u, pv_edge, pv_self)
    ):
        return cast(
            torch.Tensor,
            _NormalVelocityTendencyEdgeNeighborPVFunction.apply(
                u,
                pv_edge,
                pv_self,
                eoe_safe,
                eoe_valid,
                edge_weights,
                transpose_destinations,
                transpose_valid,
            ),
        )

    edge_weights = edge_weights[:, : eoe_safe.shape[1]]
    if (
        eoe_safe.shape[1] > 1
        and eoe_safe.shape[1]
        <= _DYNAMICS_TENDENCY_EDGE_NEIGHBOR_FULL_SLOT_COMPILE_LIMIT
        and _dynamics_tendency_can_use_no_ad_full_slot_reduce(
            u,
            eoe_safe,
            eoe_valid,
            edge_weights,
            pv_edge,
            pv_self,
        )
    ):
        # Keep the upstream slot chain explicit inside one output-owned region.
        # Inductor fuses the gathers and pointwise terms without constructing
        # the full (edge, slot, level) pv/u/contribution tensors; eager CUDA
        # likewise avoids those large global intermediates.
        return _normal_velocity_tendency_edge_neighbor_pv_ordered_numeric(
            u,
            pv_edge,
            pv_self,
            eoe_safe,
            edge_weights,
        )

    q: torch.Tensor | None = None
    batch_size = _dynamics_tendency_slot_gather_batch(u, eoe_safe.shape[1])
    if batch_size == 1:
        reuse_private = not torch.is_grad_enabled() and not any(
            _tensor_carries_autodiff(tensor)
            for tensor in (u, pv_edge, pv_self, edge_weights)
        )
        for slot in range(eoe_safe.shape[1]):
            neighbor_edges = eoe_safe[:, slot]
            if reuse_private:
                workpv = gather_rows(pv_edge, neighbor_edges)
                torch.add(pv_self, workpv, out=workpv)
                workpv.mul_(0.5)
                contrib = gather_rows(u, neighbor_edges)
                torch.mul(edge_weights[:, slot].unsqueeze(1), contrib, out=contrib)
                contrib.mul_(workpv)
            else:
                workpv = 0.5 * (pv_self + gather_rows(pv_edge, neighbor_edges))
                contrib = (
                    edge_weights[:, slot].unsqueeze(1)
                    * gather_rows(u, neighbor_edges)
                    * workpv
                )
            q = _accumulate_ordered_slot(q, contrib)
        if q is None:
            return u.new_zeros((n_edges, n_levels))
        return q
    if batch_size == eoe_safe.shape[1] and eoe_safe.shape[1] > 1:
        pv_batch = _dynamics_tendency_gather_slot_batch(
            pv_edge, eoe_safe, 0, eoe_safe.shape[1]
        )
        u_batch = _dynamics_tendency_gather_slot_batch(
            u, eoe_safe, 0, eoe_safe.shape[1]
        )
        workpv_batch = 0.5 * (pv_self.unsqueeze(1) + pv_batch)
        contrib_batch = edge_weights.unsqueeze(2) * u_batch * workpv_batch
        return _sum_fortran_order(contrib_batch, dim=1)

    for batch_start, batch_stop in _dynamics_tendency_slot_batches(
        eoe_safe.shape[1], batch_size
    ):
        pv_batch = _dynamics_tendency_gather_slot_batch(
            pv_edge, eoe_safe, batch_start, batch_stop
        )
        u_batch = _dynamics_tendency_gather_slot_batch(
            u, eoe_safe, batch_start, batch_stop
        )
        for local_slot in range(batch_stop - batch_start):
            slot = batch_start + local_slot
            workpv = 0.5 * (pv_self + pv_batch[:, local_slot, :])
            contrib = (
                edge_weights[:, slot].unsqueeze(1) * u_batch[:, local_slot, :] * workpv
            )
            q = _accumulate_ordered_slot(q, contrib)
    if q is None:
        return u.new_zeros((n_edges, n_levels))
    return q


def _normal_velocity_tendency_pair_average_backward(
    grad_output: torch.Tensor,
    first_destinations: torch.Tensor,
    first_valid: torch.Tensor,
    second_destinations: torch.Tensor,
    second_valid: torch.Tensor,
    n_source_rows: int,
    n_source_levels: int,
    n_vert_levels: int,
) -> torch.Tensor:
    def reduce_side(
        destinations: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        total: torch.Tensor | None = None
        for duplicate in range(destinations.shape[1]):
            contribution = 0.5 * gather_rows(grad_output, destinations[:, duplicate])
            contribution = torch.where(
                valid[:, duplicate].unsqueeze(1),
                contribution,
                torch.zeros_like(contribution),
            )
            total = contribution if total is None else total + contribution
        if total is None:
            return grad_output.new_zeros(
                (int(n_source_rows), max(int(n_vert_levels) - 1, 0))
            )
        return total

    middle = reduce_side(first_destinations, first_valid) + reduce_side(
        second_destinations, second_valid
    )
    before = middle.new_zeros((int(n_source_rows), 1))
    after = middle.new_zeros(
        (int(n_source_rows), int(n_source_levels) - int(n_vert_levels))
    )
    return torch.cat((before, middle, after), dim=1)


class _NormalVelocityTendencyPairAverageFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        rw: torch.Tensor,
        first: torch.Tensor,
        second: torch.Tensor,
        first_destinations: torch.Tensor,
        first_valid: torch.Tensor,
        second_destinations: torch.Tensor,
        second_valid: torch.Tensor,
        n_vert_levels: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(
            first_destinations,
            first_valid,
            second_destinations,
            second_valid,
        )
        ctx.n_source_rows = int(rw.shape[0])
        ctx.n_source_levels = int(rw.shape[1])
        ctx.n_vert_levels = int(n_vert_levels)
        return 0.5 * (rw[first, 1:n_vert_levels] + rw[second, 1:n_vert_levels])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        first_destinations, first_valid, second_destinations, second_valid = (
            ctx.saved_tensors
        )
        grad_rw = _normal_velocity_tendency_pair_average_backward(
            grad_output,
            first_destinations,
            first_valid,
            second_destinations,
            second_valid,
            ctx.n_source_rows,
            ctx.n_source_levels,
            ctx.n_vert_levels,
        )
        return grad_rw, None, None, None, None, None, None, None


def _normal_velocity_tendency_vertical_tendency_from_edge_rw(
    u_edge: torch.Tensor,
    edge_rw: torch.Tensor,
    rdzw: torch.Tensor,
    fzm: torch.Tensor,
    fzp: torch.Tensor,
    solve_edge_mask: torch.Tensor,
    solve_edge_mask_all: bool,
    n_vert_levels: int,
    middle_abs_edge_rw: torch.Tensor | None = None,
) -> torch.Tensor:
    n_edges = u_edge.shape[0]
    if n_vert_levels > 1:
        wduz_k1 = edge_rw[:, 0] * (fzm[1] * u_edge[:, 1] + fzp[1] * u_edge[:, 0])

        wduz_mid = None
        if n_vert_levels > 3:
            middle_rw = edge_rw[:, 1 : n_vert_levels - 2]
            if middle_abs_edge_rw is None:
                middle_abs_edge_rw = fct_abs(middle_rw)
            wduz_mid = _flux3_from_abs(
                u_edge[:, 0 : n_vert_levels - 3],
                u_edge[:, 1 : n_vert_levels - 2],
                u_edge[:, 2 : n_vert_levels - 1],
                u_edge[:, 3:n_vert_levels],
                middle_rw,
                middle_abs_edge_rw,
                1.0,
            )

        wduz_klast = None
        if n_vert_levels > 2:
            wduz_klast = edge_rw[:, n_vert_levels - 2] * (
                fzm[n_vert_levels - 1] * u_edge[:, n_vert_levels - 1]
                + fzp[n_vert_levels - 1] * u_edge[:, n_vert_levels - 2]
            )

        wduz_delta = _full_level_vertical_flux_delta_from_parts(
            wduz_k1,
            wduz_mid,
            wduz_klast,
            n_vert_levels,
        )
    else:
        wduz_delta = u_edge.new_zeros((n_edges, n_vert_levels))

    tend_u_patch = -rdzw[:n_vert_levels].unsqueeze(0) * wduz_delta
    if not solve_edge_mask_all:
        tend_u_patch = tend_u_patch * solve_edge_mask
    return tend_u_patch


def _normal_velocity_tendency_vertical_hard_scalar(
    u_edge: torch.Tensor,
    edge_rw: torch.Tensor,
    grad_output: torch.Tensor,
    rdzw: torch.Tensor,
    fzm: torch.Tensor,
    fzp: torch.Tensor,
    solve_edge_mask: torch.Tensor,
    solve_edge_mask_all: bool,
    n_vert_levels: int,
) -> torch.Tensor:
    middle_rw = edge_rw[:, 1 : n_vert_levels - 2]
    hard_abs = torch.abs(middle_rw)
    output = _normal_velocity_tendency_vertical_tendency_from_edge_rw(
        u_edge,
        edge_rw,
        rdzw,
        fzm,
        fzp,
        solve_edge_mask,
        solve_edge_mask_all,
        n_vert_levels,
        hard_abs,
    )
    return (output * grad_output).sum()


_normal_velocity_tendency_vertical_hard_grad = torch.func.grad(
    _normal_velocity_tendency_vertical_hard_scalar,
    argnums=(0, 1),
)


def _normal_velocity_tendency_vertical_ordered_backward(
    u_edge: torch.Tensor,
    edge_rw: torch.Tensor,
    grad_output: torch.Tensor,
    rdzw: torch.Tensor,
    fzm: torch.Tensor,
    fzp: torch.Tensor,
    solve_edge_mask: torch.Tensor,
    solve_edge_mask_all: bool,
    n_vert_levels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the exact hard-primal vertical-transport transpose."""
    return cast(
        tuple[torch.Tensor, torch.Tensor],
        _normal_velocity_tendency_vertical_hard_grad(
            u_edge,
            edge_rw,
            grad_output,
            rdzw,
            fzm,
            fzp,
            solve_edge_mask,
            solve_edge_mask_all,
            n_vert_levels,
        ),
    )


class _NormalVelocityTendencyVerticalTendencyFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        u_edge: torch.Tensor,
        edge_rw: torch.Tensor,
        rdzw: torch.Tensor,
        fzm: torch.Tensor,
        fzp: torch.Tensor,
        solve_edge_mask: torch.Tensor,
        solve_edge_mask_all: bool,
        n_vert_levels: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(
            u_edge,
            edge_rw,
            rdzw,
            fzm,
            fzp,
            solve_edge_mask,
        )
        ctx.solve_edge_mask_all = bool(solve_edge_mask_all)
        ctx.n_vert_levels = int(n_vert_levels)
        ctx.backward_callable = _planned_dynamics_tendency_adjoint(
            "normal_velocity_tendency_vertical_backward",
            _normal_velocity_tendency_vertical_ordered_backward,
        )
        return _normal_velocity_tendency_vertical_tendency_from_edge_rw(
            u_edge,
            edge_rw,
            rdzw,
            fzm,
            fzp,
            solve_edge_mask,
            bool(solve_edge_mask_all),
            int(n_vert_levels),
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        u_edge, edge_rw, rdzw, fzm, fzp, solve_edge_mask = ctx.saved_tensors
        grad_u, grad_rw = ctx.backward_callable(
            u_edge,
            edge_rw,
            grad_output,
            rdzw,
            fzm,
            fzp,
            solve_edge_mask,
            ctx.solve_edge_mask_all,
            ctx.n_vert_levels,
        )
        return grad_u, grad_rw, None, None, None, None, None, None


def _normal_velocity_tendency_vertical_tendency_region(
    *,
    edge_cell_relation: StaticPairRelation,
    u_edge: torch.Tensor,
    rw: torch.Tensor,
    rdzw: torch.Tensor,
    fzm: torch.Tensor,
    fzp: torch.Tensor,
    solve_edge_mask: torch.Tensor,
    solve_edge_mask_all: bool,
    n_vert_levels: int,
) -> torch.Tensor:
    """Compute the normal-velocity-tendency vertical transport tendency without the rest of normal-velocity-tendency."""
    edge_c1 = _dynamics_tendency_pair_field(edge_cell_relation, 0, "first")
    edge_c2 = _dynamics_tendency_pair_field(edge_cell_relation, 1, "second")
    first_destinations = _dynamics_tendency_pair_field(
        edge_cell_relation, 14, "first_transpose_destinations"
    )
    first_valid = _dynamics_tendency_pair_field(
        edge_cell_relation, 15, "first_transpose_valid"
    )
    second_destinations = _dynamics_tendency_pair_field(
        edge_cell_relation, 16, "second_transpose_destinations"
    )
    second_valid = _dynamics_tendency_pair_field(
        edge_cell_relation, 17, "second_transpose_valid"
    )
    use_ordered_vjp = (
        first_destinations is not None
        and first_valid is not None
        and second_destinations is not None
        and second_valid is not None
        and _dynamics_tendency_use_ordered_reverse_vjp(u_edge, rw)
    )
    if use_ordered_vjp:
        edge_rw = _NormalVelocityTendencyPairAverageFunction.apply(
            rw,
            edge_c1,
            edge_c2,
            first_destinations,
            first_valid,
            second_destinations,
            second_valid,
            n_vert_levels,
        )
        return cast(
            torch.Tensor,
            _NormalVelocityTendencyVerticalTendencyFunction.apply(
                u_edge,
                edge_rw,
                rdzw,
                fzm,
                fzp,
                solve_edge_mask,
                solve_edge_mask_all,
                n_vert_levels,
            ),
        )

    rw_levels = rw[:, 1:n_vert_levels]
    edge_rw = 0.5 * (
        rw_levels.index_select(0, edge_c1) + rw_levels.index_select(0, edge_c2)
    )
    return _normal_velocity_tendency_vertical_tendency_from_edge_rw(
        u_edge,
        edge_rw,
        rdzw,
        fzm,
        fzp,
        solve_edge_mask,
        solve_edge_mask_all,
        n_vert_levels,
    )


def _normal_velocity_tendency_base_coupling_region(
    *,
    edge_cell_relation: StaticPairRelation,
    u_edge: torch.Tensor,
    ke: torch.Tensor,
    h_divergence: torch.Tensor,
    edge_neighbor_pv_patch: torch.Tensor,
    rho_edge_solve: torch.Tensor,
    inv_dc_edge_solve: torch.Tensor,
    solve_edge_mask: torch.Tensor,
    solve_edge_mask_all: bool,
    h_divergence_cell1: torch.Tensor | None = None,
    h_divergence_cell2: torch.Tensor | None = None,
    h_divergence_cell_patch: torch.Tensor | None = None,
    cell_patch_row_start: int = 0,
    cell_patch_row_end: int = 0,
) -> torch.Tensor:
    """Compute the nonvertical normal-velocity-tendency PV/KE/divergence coupling."""
    edge_c1 = _dynamics_tendency_pair_field(edge_cell_relation, 0, "first")
    edge_c2 = _dynamics_tendency_pair_field(edge_cell_relation, 1, "second")
    reuse_private = (
        not torch.is_grad_enabled()
        and not _compile_active()
        and not any(
            _tensor_carries_autodiff(tensor)
            for tensor in (
                u_edge,
                ke,
                h_divergence,
                edge_neighbor_pv_patch,
                rho_edge_solve,
            )
        )
    )
    if reuse_private:
        ke_delta = ke.index_select(0, edge_c2)
        ke_cell1 = ke.index_select(0, edge_c1)
        torch.sub(ke_delta, ke_cell1, out=ke_delta)
        del ke_cell1
    else:
        ke_delta = ke.index_select(0, edge_c2) - ke.index_select(0, edge_c1)
    gathered_hdiv = False
    if h_divergence_cell1 is None and h_divergence_cell2 is None:
        if h_divergence_cell_patch is None:
            h_divergence_cell1, h_divergence_cell2 = (
                _dynamics_tendency_pair_gather_patch_or_base(
                    edge_cell_relation,
                    h_divergence,
                    None,
                    patch_row_start=0,
                    patch_row_end=0,
                )
            )
        else:
            h_divergence_cell1, h_divergence_cell2 = (
                _dynamics_tendency_pair_gather_patch_or_base(
                    edge_cell_relation,
                    h_divergence,
                    h_divergence_cell_patch,
                    patch_row_start=cell_patch_row_start,
                    patch_row_end=cell_patch_row_end,
                )
            )
        gathered_hdiv = True
    else:
        if h_divergence_cell1 is None:
            h_divergence_cell1 = h_divergence.index_select(0, edge_c1)
        if h_divergence_cell2 is None:
            h_divergence_cell2 = h_divergence.index_select(0, edge_c2)
    if reuse_private and gathered_hdiv:
        torch.add(h_divergence_cell1, h_divergence_cell2, out=h_divergence_cell1)
        hdiv_sum = h_divergence_cell1
    else:
        hdiv_sum = h_divergence_cell1 + h_divergence_cell2
    if reuse_private:
        ke_delta.mul_(inv_dc_edge_solve.unsqueeze(1))
        torch.sub(edge_neighbor_pv_patch, ke_delta, out=ke_delta)
        ke_delta.mul_(rho_edge_solve)
        u_hdiv = u_edge * 0.5
        u_hdiv.mul_(hdiv_sum)
        ke_delta.sub_(u_hdiv)
        u_base_patch = ke_delta
    else:
        u_base_patch = (
            rho_edge_solve
            * (edge_neighbor_pv_patch - ke_delta * inv_dc_edge_solve.unsqueeze(1))
            - u_edge * 0.5 * hdiv_sum
        )
    if not solve_edge_mask_all:
        if reuse_private:
            u_base_patch.mul_(solve_edge_mask)
        else:
            u_base_patch = solve_edge_mask * u_base_patch
    return u_base_patch


def _normal_velocity_tendency_h_divergence_proof_tensors(
    *,
    ru: torch.Tensor,
    dv_edge: torch.Tensor,
    cell_edge_relation: StaticSlotRelation,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return unnormalized normal-velocity-tendency divergence sums, absolute term sums, and counts."""
    cell_edges = _dynamics_tendency_slot_field(cell_edge_relation, 0, "indices")
    cell_sign = _dynamics_tendency_slot_field(cell_edge_relation, 2, "sign")
    cell_valid = _dynamics_tendency_slot_field(cell_edge_relation, 1, "valid")
    if cell_sign is None:
        raise ValueError(
            "dynamics-tendency normal-velocity-tendency proof divergence requires cell-edge signs"
        )
    n_cells = cell_edges.shape[0]
    n_levels = ru.shape[1]
    total = ru.new_zeros((n_cells, n_levels))
    abs_total = ru.new_zeros((n_cells, n_levels))
    count = ru.new_zeros((n_cells, n_levels))
    terms = ru.new_zeros((n_cells, cell_edges.shape[1], n_levels))
    for slot in range(cell_edges.shape[1]):
        edge_idx = cell_edges[:, slot]
        weight = cell_sign[:, slot] * dv_edge[edge_idx]
        valid = None if cell_valid is None else cell_valid[:, slot]
        if valid is not None:
            weight = torch.where(valid, weight, torch.zeros_like(weight))
        contrib = weight.unsqueeze(1) * gather_rows(ru, edge_idx)
        terms[:, slot, :] = contrib
        total = total + contrib
        abs_total = abs_total + torch.abs(contrib)
        if valid is None:
            count = count + torch.ones_like(total)
        else:
            count = count + valid.to(dtype=ru.dtype).unsqueeze(1)
    return total, abs_total, count, terms


def _normal_velocity_tendency_vertical_flux_proof_tensor(
    *,
    edge_cell_relation: StaticPairRelation,
    u_edge: torch.Tensor,
    rw: torch.Tensor,
    fzm: torch.Tensor,
    fzp: torch.Tensor,
    n_vert_levels: int,
) -> torch.Tensor:
    """Reconstruct the normal-velocity-tendency Fortran ``wduz`` scratch array for debug proof."""
    n_edges = u_edge.shape[0]
    wduz = u_edge.new_zeros((n_edges, n_vert_levels + 1))
    if n_vert_levels <= 1:
        return wduz
    edge_c1 = _dynamics_tendency_pair_field(edge_cell_relation, 0, "first")
    edge_c2 = _dynamics_tendency_pair_field(edge_cell_relation, 1, "second")
    edge_c1_rw = rw[edge_c1, 1:n_vert_levels]
    edge_c2_rw = rw[edge_c2, 1:n_vert_levels]
    wduz[:, 1] = (
        0.5
        * (edge_c1_rw[:, 0] + edge_c2_rw[:, 0])
        * (fzm[1] * u_edge[:, 1] + fzp[1] * u_edge[:, 0])
    )
    if n_vert_levels > 3:
        wduz[:, 2 : n_vert_levels - 1] = flux3(
            u_edge[:, 0 : n_vert_levels - 3],
            u_edge[:, 1 : n_vert_levels - 2],
            u_edge[:, 2 : n_vert_levels - 1],
            u_edge[:, 3:n_vert_levels],
            0.5
            * (
                edge_c1_rw[:, 1 : n_vert_levels - 2]
                + edge_c2_rw[:, 1 : n_vert_levels - 2]
            ),
            1.0,
        )
    if n_vert_levels > 2:
        wduz[:, n_vert_levels - 1] = (
            0.5
            * (edge_c1_rw[:, n_vert_levels - 2] + edge_c2_rw[:, n_vert_levels - 2])
            * (
                fzm[n_vert_levels - 1] * u_edge[:, n_vert_levels - 1]
                + fzp[n_vert_levels - 1] * u_edge[:, n_vert_levels - 2]
            )
        )
    return wduz


def _normal_velocity_tendency_vertical_flux_abs_term_sum_proof_tensor(
    *,
    edge_cell_relation: StaticPairRelation,
    u_edge: torch.Tensor,
    rw: torch.Tensor,
    fzm: torch.Tensor,
    fzp: torch.Tensor,
    n_vert_levels: int,
) -> torch.Tensor:
    """Source-derived absolute scale for the normal-velocity-tendency Fortran ``wduz`` scratch array."""
    n_edges = u_edge.shape[0]
    wduz_abs = u_edge.new_zeros((n_edges, n_vert_levels + 1))
    if n_vert_levels <= 1:
        return wduz_abs
    edge_c1 = _dynamics_tendency_pair_field(edge_cell_relation, 0, "first")
    edge_c2 = _dynamics_tendency_pair_field(edge_cell_relation, 1, "second")
    edge_c1_rw = rw[edge_c1, 1:n_vert_levels]
    edge_c2_rw = rw[edge_c2, 1:n_vert_levels]
    wduz_abs[:, 1] = 0.5 * (
        torch.abs(edge_c1_rw[:, 0] * fzm[1] * u_edge[:, 1])
        + torch.abs(edge_c1_rw[:, 0] * fzp[1] * u_edge[:, 0])
        + torch.abs(edge_c2_rw[:, 0] * fzm[1] * u_edge[:, 1])
        + torch.abs(edge_c2_rw[:, 0] * fzp[1] * u_edge[:, 0])
    )
    if n_vert_levels > 3:
        wduz_abs[:, 2 : n_vert_levels - 1] = _flux3_abs_term_sum(
            u_edge[:, 0 : n_vert_levels - 3],
            u_edge[:, 1 : n_vert_levels - 2],
            u_edge[:, 2 : n_vert_levels - 1],
            u_edge[:, 3:n_vert_levels],
            0.5
            * (
                edge_c1_rw[:, 1 : n_vert_levels - 2]
                + edge_c2_rw[:, 1 : n_vert_levels - 2]
            ),
            1.0,
        )
    if n_vert_levels > 2:
        wduz_abs[:, n_vert_levels - 1] = 0.5 * (
            torch.abs(
                edge_c1_rw[:, n_vert_levels - 2]
                * fzm[n_vert_levels - 1]
                * u_edge[:, n_vert_levels - 1]
            )
            + torch.abs(
                edge_c1_rw[:, n_vert_levels - 2]
                * fzp[n_vert_levels - 1]
                * u_edge[:, n_vert_levels - 2]
            )
            + torch.abs(
                edge_c2_rw[:, n_vert_levels - 2]
                * fzm[n_vert_levels - 1]
                * u_edge[:, n_vert_levels - 1]
            )
            + torch.abs(
                edge_c2_rw[:, n_vert_levels - 2]
                * fzp[n_vert_levels - 1]
                * u_edge[:, n_vert_levels - 2]
            )
        )
    return wduz_abs


def _normal_velocity_tendency_edge_neighbor_pv_proof_tensors(
    *,
    edge_neighbor_relation: StaticSlotRelation,
    u: torch.Tensor,
    pv_edge: torch.Tensor,
    pv_self: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normal-velocity-tendency PV reduction and absolute term sums in Fortran slot order."""
    eoe_safe = _dynamics_tendency_slot_field(edge_neighbor_relation, 0, "indices")
    eoe_valid = _dynamics_tendency_slot_field(edge_neighbor_relation, 1, "valid")
    edge_weights = _dynamics_tendency_slot_field(edge_neighbor_relation, 3, "weights")
    if edge_weights is None:
        raise ValueError(
            "dynamics-tendency normal-velocity-tendency proof PV requires edge-neighbor weights"
        )
    n_edges = eoe_safe.shape[0]
    n_levels = u.shape[1]
    q = u.new_zeros((n_edges, n_levels))
    abs_total = u.new_zeros((n_edges, n_levels))
    for slot in range(eoe_safe.shape[1]):
        neighbor_edges = eoe_safe[:, slot]
        workpv = 0.5 * (pv_self + gather_rows(pv_edge, neighbor_edges))
        contrib = (
            edge_weights[:, slot].unsqueeze(1) * gather_rows(u, neighbor_edges) * workpv
        )
        if eoe_valid is not None:
            valid = eoe_valid[:, slot].unsqueeze(1)
            contrib = torch.where(valid, contrib, torch.zeros_like(contrib))
        q = q + contrib
        abs_total = abs_total + torch.abs(contrib)
    return q, abs_total


def _normal_velocity_tendency_pressure_proof_tensor(
    *,
    edge_cell_relation: StaticPairRelation,
    cqu_solve: torch.Tensor,
    inv_dc_edge_solve: torch.Tensor,
    zxu_solve: torch.Tensor,
    zz: torch.Tensor,
    pp: torch.Tensor,
    dpdz: torch.Tensor,
    dpdz_cell_patch: torch.Tensor | None,
    cell_patch_row_start: int,
    cell_patch_row_end: int,
) -> torch.Tensor:
    """Return the normal-velocity-tendency horizontal pressure-gradient contribution."""
    edge_c1 = _dynamics_tendency_pair_field(edge_cell_relation, 0, "first")
    edge_c2 = _dynamics_tendency_pair_field(edge_cell_relation, 1, "second")
    zz_pair_sum = zz[edge_c1, :] + zz[edge_c2, :]
    pp_delta = pp[edge_c2, :] - pp[edge_c1, :]
    dpdz_c1, dpdz_c2 = _dynamics_tendency_pair_gather_patch_or_base(
        edge_cell_relation,
        dpdz,
        dpdz_cell_patch,
        patch_row_start=cell_patch_row_start,
        patch_row_end=cell_patch_row_end,
    )
    return -cqu_solve * (
        pp_delta * inv_dc_edge_solve.unsqueeze(1) / (0.5 * zz_pair_sum)
        - 0.5 * zxu_solve * (dpdz_c1 + dpdz_c2)
    )


def _normal_velocity_tendency_pressure_abs_term_sum_proof_tensor(
    *,
    edge_cell_relation: StaticPairRelation,
    cqu_solve: torch.Tensor,
    inv_dc_edge_solve: torch.Tensor,
    zxu_solve: torch.Tensor,
    zz: torch.Tensor,
    pp: torch.Tensor,
    dpdz: torch.Tensor,
    dpdz_cell_patch: torch.Tensor | None,
    cell_patch_row_start: int,
    cell_patch_row_end: int,
) -> torch.Tensor:
    """Source-derived absolute scale for the normal-velocity-tendency horizontal pressure term."""
    edge_c1 = _dynamics_tendency_pair_field(edge_cell_relation, 0, "first")
    edge_c2 = _dynamics_tendency_pair_field(edge_cell_relation, 1, "second")
    zz_pair_sum = zz[edge_c1, :] + zz[edge_c2, :]
    denom = torch.clamp(torch.abs(0.5 * zz_pair_sum), min=torch.finfo(zz.dtype).tiny)
    dc_scale = torch.abs(inv_dc_edge_solve).unsqueeze(1) / denom
    dpdz_c1, dpdz_c2 = _dynamics_tendency_pair_gather_patch_or_base(
        edge_cell_relation,
        dpdz,
        dpdz_cell_patch,
        patch_row_start=cell_patch_row_start,
        patch_row_end=cell_patch_row_end,
    )
    return torch.abs(cqu_solve) * (
        dc_scale * (torch.abs(pp[edge_c2, :]) + torch.abs(pp[edge_c1, :]))
        + 0.5 * torch.abs(zxu_solve) * (torch.abs(dpdz_c1) + torch.abs(dpdz_c2))
    )


def _normal_velocity_tendency_proof_tensors(
    *,
    edge_cell_relation: StaticPairRelation,
    edge_neighbor_relation: StaticSlotRelation,
    u: torch.Tensor,
    u_edge: torch.Tensor,
    rw: torch.Tensor,
    ke: torch.Tensor,
    h_divergence: torch.Tensor,
    h_divergence_cell_patch: torch.Tensor,
    h_divergence_abs_term_sum_cell_patch: torch.Tensor,
    pv_edge: torch.Tensor,
    pv_edge_solve: torch.Tensor,
    rho_edge_solve: torch.Tensor,
    inv_dc_edge_solve: torch.Tensor,
    rdzw: torch.Tensor,
    fzm: torch.Tensor,
    fzp: torch.Tensor,
    n_vert_levels: int,
    include_pressure_gradient: bool,
    zz: torch.Tensor,
    pp: torch.Tensor,
    dpdz: torch.Tensor,
    dpdz_cell_patch: torch.Tensor | None,
    cell_patch_row_start: int,
    cell_patch_row_end: int,
    inv_area_cell: torch.Tensor,
    cqu_solve: torch.Tensor,
    zxu_solve: torch.Tensor,
) -> dict[str, Any]:
    """Build dynamics-tendency normal-velocity-tendency proof operands/intermediates for debug comparisons only."""
    wduz = _normal_velocity_tendency_vertical_flux_proof_tensor(
        edge_cell_relation=edge_cell_relation,
        u_edge=u_edge,
        rw=rw,
        fzm=fzm,
        fzp=fzp,
        n_vert_levels=n_vert_levels,
    )
    wduz_abs_term_sum = (
        _normal_velocity_tendency_vertical_flux_abs_term_sum_proof_tensor(
            edge_cell_relation=edge_cell_relation,
            u_edge=u_edge,
            rw=rw,
            fzm=fzm,
            fzp=fzp,
            n_vert_levels=n_vert_levels,
        )
    )
    vertical_tendency = -rdzw[:n_vert_levels].unsqueeze(0) * (
        wduz[:, 1 : n_vert_levels + 1] - wduz[:, :n_vert_levels]
    )
    vertical_tendency_abs_term_sum = torch.abs(rdzw[:n_vert_levels]).unsqueeze(0) * (
        wduz_abs_term_sum[:, 1 : n_vert_levels + 1]
        + wduz_abs_term_sum[:, :n_vert_levels]
    )
    q, q_abs_term_sum = _normal_velocity_tendency_edge_neighbor_pv_proof_tensors(
        edge_neighbor_relation=edge_neighbor_relation,
        u=u,
        pv_edge=pv_edge,
        pv_self=pv_edge_solve,
    )
    edge_c1 = _dynamics_tendency_pair_field(edge_cell_relation, 0, "first")
    edge_c2 = _dynamics_tendency_pair_field(edge_cell_relation, 1, "second")
    ke_delta = ke[edge_c2, :] - ke[edge_c1, :]
    hdiv_c1, hdiv_c2 = _dynamics_tendency_pair_gather_patch_or_base(
        edge_cell_relation,
        h_divergence,
        h_divergence_cell_patch,
        patch_row_start=cell_patch_row_start,
        patch_row_end=cell_patch_row_end,
    )
    patch_rows = torch.arange(
        cell_patch_row_start,
        cell_patch_row_end,
        device=h_divergence_abs_term_sum_cell_patch.device,
    )
    hdiv_abs_norm_patch = h_divergence_abs_term_sum_cell_patch * torch.abs(
        inv_area_cell[patch_rows]
    ).unsqueeze(1)
    hdiv_abs_c1, hdiv_abs_c2 = _dynamics_tendency_pair_gather_patch_or_base(
        edge_cell_relation,
        torch.abs(h_divergence),
        hdiv_abs_norm_patch,
        patch_row_start=cell_patch_row_start,
        patch_row_end=cell_patch_row_end,
    )
    base_coupling = rho_edge_solve * (
        q - ke_delta * inv_dc_edge_solve.unsqueeze(1)
    ) - u_edge * 0.5 * (hdiv_c1 + hdiv_c2)
    base_coupling_abs_term_sum = torch.abs(rho_edge_solve) * (
        q_abs_term_sum + torch.abs(ke_delta * inv_dc_edge_solve.unsqueeze(1))
    ) + 0.5 * torch.abs(u_edge) * (hdiv_abs_c1 + hdiv_abs_c2)
    pressure = u_edge.new_zeros(u_edge.shape)
    pressure_abs_term_sum = u_edge.new_zeros(u_edge.shape)
    if include_pressure_gradient:
        pressure = _normal_velocity_tendency_pressure_proof_tensor(
            edge_cell_relation=edge_cell_relation,
            cqu_solve=cqu_solve,
            inv_dc_edge_solve=inv_dc_edge_solve,
            zxu_solve=zxu_solve,
            zz=zz,
            pp=pp,
            dpdz=dpdz,
            dpdz_cell_patch=dpdz_cell_patch,
            cell_patch_row_start=cell_patch_row_start,
            cell_patch_row_end=cell_patch_row_end,
        )
        pressure_abs_term_sum = (
            _normal_velocity_tendency_pressure_abs_term_sum_proof_tensor(
                edge_cell_relation=edge_cell_relation,
                cqu_solve=cqu_solve,
                inv_dc_edge_solve=inv_dc_edge_solve,
                zxu_solve=zxu_solve,
                zz=zz,
                pp=pp,
                dpdz=dpdz,
                dpdz_cell_patch=dpdz_cell_patch,
                cell_patch_row_start=cell_patch_row_start,
                cell_patch_row_end=cell_patch_row_end,
            )
        )
    return {
        "normal_velocity_tendency_wduz": wduz,
        "normal_velocity_tendency_wduz_abs_term_sum": wduz_abs_term_sum,
        "normal_velocity_tendency_q": q,
        "normal_velocity_tendency_q_abs_term_sum": q_abs_term_sum,
        "normal_velocity_tendency_vertical_tendency": vertical_tendency,
        "normal_velocity_tendency_vertical_tendency_abs_term_sum": vertical_tendency_abs_term_sum,
        "normal_velocity_tendency_base_coupling": base_coupling,
        "normal_velocity_tendency_base_coupling_abs_term_sum": base_coupling_abs_term_sum,
        "normal_velocity_tendency_pressure_gradient": pressure,
        "normal_velocity_tendency_pressure_gradient_abs_term_sum": pressure_abs_term_sum,
        "normal_velocity_tendency_tend_u_abs_term_sum": (
            vertical_tendency_abs_term_sum + base_coupling_abs_term_sum
        ),
        "normal_velocity_tendency_tend_u_euler_abs_term_sum": pressure_abs_term_sum,
    }


def _normal_velocity_tendency_pressure_curvature_region(
    *,
    edge_cell_relation: StaticPairRelation,
    base_tendency_patch: torch.Tensor,
    u_edge: torch.Tensor,
    rho_edge_solve: torch.Tensor,
    inv_dc_edge_solve: torch.Tensor,
    solve_edge_mask: torch.Tensor,
    solve_edge_mask_all: bool,
    tend_u_euler_edge: torch.Tensor,
    include_pressure_gradient: bool,
    zz: torch.Tensor,
    pp: torch.Tensor,
    dpdz: torch.Tensor,
    cqu_solve: torch.Tensor,
    zxu_solve: torch.Tensor,
    include_curvature: bool,
    w: torch.Tensor,
    angle_edge_solve: torch.Tensor,
    lat_edge_solve: torch.Tensor,
    omega: float,
    inv_r_earth: float,
    dpdz_cell1: torch.Tensor | None = None,
    dpdz_cell2: torch.Tensor | None = None,
    dpdz_cell_patch: torch.Tensor | None = None,
    cell_patch_row_start: int = 0,
    cell_patch_row_end: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the normal-velocity-tendency pressure-gradient/Euler and optional curvature tail."""
    edge_c1 = _dynamics_tendency_pair_field(edge_cell_relation, 0, "first")
    edge_c2 = _dynamics_tendency_pair_field(edge_cell_relation, 1, "second")
    reuse_private = (
        not torch.is_grad_enabled()
        and not _compile_active()
        and not any(
            _tensor_carries_autodiff(tensor)
            for tensor in (zz, pp, dpdz, cqu_solve, zxu_solve)
        )
    )
    if include_pressure_gradient:
        if reuse_private:
            zz_pair_sum = zz.index_select(0, edge_c2)
            zz_cell1 = zz.index_select(0, edge_c1)
            torch.add(zz_pair_sum, zz_cell1, out=zz_pair_sum)
            del zz_cell1
            pp_delta = pp.index_select(0, edge_c2)
            pp_cell1 = pp.index_select(0, edge_c1)
            torch.sub(pp_delta, pp_cell1, out=pp_delta)
            del pp_cell1
        else:
            zz_pair_sum = zz.index_select(0, edge_c2) + zz.index_select(0, edge_c1)
            pp_delta = pp.index_select(0, edge_c2) - pp.index_select(0, edge_c1)
        gathered_dpdz = False
        if dpdz_cell1 is None and dpdz_cell2 is None:
            if dpdz_cell_patch is None:
                dpdz_cell1, dpdz_cell2 = _dynamics_tendency_pair_gather_patch_or_base(
                    edge_cell_relation,
                    dpdz,
                    None,
                    patch_row_start=0,
                    patch_row_end=0,
                )
            else:
                dpdz_cell1, dpdz_cell2 = _dynamics_tendency_pair_gather_patch_or_base(
                    edge_cell_relation,
                    dpdz,
                    dpdz_cell_patch,
                    patch_row_start=cell_patch_row_start,
                    patch_row_end=cell_patch_row_end,
                )
            gathered_dpdz = True
        else:
            if dpdz_cell1 is None:
                dpdz_cell1 = dpdz.index_select(0, edge_c1)
            if dpdz_cell2 is None:
                dpdz_cell2 = dpdz.index_select(0, edge_c2)
        if reuse_private and gathered_dpdz:
            torch.add(dpdz_cell1, dpdz_cell2, out=dpdz_cell1)
            dpdz_pair_sum = dpdz_cell1
        else:
            dpdz_pair_sum = dpdz_cell1 + dpdz_cell2
        if reuse_private:
            zz_pair_sum.mul_(0.5)
            pp_delta.mul_(inv_dc_edge_solve.unsqueeze(1))
            pp_delta.div_(zz_pair_sum)
            zxu_term = 0.5 * zxu_solve
            zxu_term.mul_(dpdz_pair_sum)
            pp_delta.sub_(zxu_term)
            neg_cqu = -cqu_solve
            torch.mul(neg_cqu, pp_delta, out=pp_delta)
            pgf = pp_delta
        else:
            pgf = -cqu_solve * (
                pp_delta * inv_dc_edge_solve.unsqueeze(1) / (0.5 * zz_pair_sum)
                - 0.5 * zxu_solve * dpdz_pair_sum
            )
        del zz_pair_sum, pp_delta, dpdz_pair_sum
        if solve_edge_mask_all:
            tend_u_euler_patch = pgf
        else:
            tend_u_euler_patch = (
                tend_u_euler_edge * (1.0 - solve_edge_mask) + pgf * solve_edge_mask
            )
    else:
        tend_u_euler_patch = tend_u_euler_edge

    tend_u_patch = base_tendency_patch
    if include_curvature:
        tend_u_patch = _apply_u_curvature_tendency(
            tend_u_patch=tend_u_patch,
            u_edge=u_edge,
            w=w,
            edge_c1=edge_c1,
            edge_c2=edge_c2,
            rho_edge_solve=rho_edge_solve,
            angle_edge_solve=angle_edge_solve,
            lat_edge_solve=lat_edge_solve,
            omega=omega,
            inv_r_earth=inv_r_earth,
            solve_edge_mask=solve_edge_mask,
            solve_edge_mask_all=solve_edge_mask_all,
        )
    return tend_u_patch, tend_u_euler_patch


def _normal_velocity_tendency_coupling_pressure_section_region(
    *,
    edge_cell_relation: StaticPairRelation,
    vertical_tendency_patch: torch.Tensor,
    edge_neighbor_pv_patch: torch.Tensor,
    u_edge: torch.Tensor,
    ke: torch.Tensor,
    h_divergence: torch.Tensor,
    rho_edge_solve: torch.Tensor,
    inv_dc_edge_solve: torch.Tensor,
    solve_edge_mask: torch.Tensor,
    solve_edge_mask_all: bool,
    tend_u_euler_edge: torch.Tensor,
    include_pressure_gradient: bool,
    zz: torch.Tensor,
    pp: torch.Tensor,
    dpdz: torch.Tensor,
    cqu_solve: torch.Tensor,
    zxu_solve: torch.Tensor,
    include_curvature: bool,
    w: torch.Tensor,
    angle_edge_solve: torch.Tensor,
    lat_edge_solve: torch.Tensor,
    omega: float,
    inv_r_earth: float,
    h_divergence_cell1: torch.Tensor | None = None,
    h_divergence_cell2: torch.Tensor | None = None,
    h_divergence_cell_patch: torch.Tensor | None = None,
    dpdz_cell1: torch.Tensor | None = None,
    dpdz_cell2: torch.Tensor | None = None,
    dpdz_cell_patch: torch.Tensor | None = None,
    cell_patch_row_start: int = 0,
    cell_patch_row_end: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse the strict normal-velocity-tendency nonvertical coupling and pressure/curvature tail.

    The drift-prone normal-velocity-tendency vertical transport and the ordered PV neighbor reducer stay
    as explicit upstream inputs.  This region therefore shortens the live
    ``normal_velocity_tendency_coupling_patch`` handoff while preserving the existing arithmetic order
    of each sub-expression.
    """
    coupling_patch = _normal_velocity_tendency_base_coupling_region(
        edge_cell_relation=edge_cell_relation,
        u_edge=u_edge,
        ke=ke,
        h_divergence=h_divergence,
        h_divergence_cell1=h_divergence_cell1,
        h_divergence_cell2=h_divergence_cell2,
        h_divergence_cell_patch=h_divergence_cell_patch,
        edge_neighbor_pv_patch=edge_neighbor_pv_patch,
        rho_edge_solve=rho_edge_solve,
        inv_dc_edge_solve=inv_dc_edge_solve,
        solve_edge_mask=solve_edge_mask,
        solve_edge_mask_all=solve_edge_mask_all,
        cell_patch_row_start=cell_patch_row_start,
        cell_patch_row_end=cell_patch_row_end,
    )
    base_tendency_patch = vertical_tendency_patch + coupling_patch
    return _normal_velocity_tendency_pressure_curvature_region(
        edge_cell_relation=edge_cell_relation,
        base_tendency_patch=base_tendency_patch,
        u_edge=u_edge,
        rho_edge_solve=rho_edge_solve,
        inv_dc_edge_solve=inv_dc_edge_solve,
        solve_edge_mask=solve_edge_mask,
        solve_edge_mask_all=solve_edge_mask_all,
        tend_u_euler_edge=tend_u_euler_edge,
        include_pressure_gradient=include_pressure_gradient,
        zz=zz,
        pp=pp,
        dpdz=dpdz,
        cqu_solve=cqu_solve,
        zxu_solve=zxu_solve,
        include_curvature=include_curvature,
        w=w,
        angle_edge_solve=angle_edge_solve,
        lat_edge_solve=lat_edge_solve,
        omega=omega,
        inv_r_earth=inv_r_earth,
        dpdz_cell1=dpdz_cell1,
        dpdz_cell2=dpdz_cell2,
        dpdz_cell_patch=dpdz_cell_patch,
        cell_patch_row_start=cell_patch_row_start,
        cell_patch_row_end=cell_patch_row_end,
    )


def _normal_velocity_tendency_base_normal_velocity_tendency(
    *,
    edge_cell_relation: StaticPairRelation,
    edge_neighbor_relation: StaticSlotRelation,
    u: torch.Tensor,
    u_edge: torch.Tensor,
    rw: torch.Tensor,
    ke: torch.Tensor,
    h_divergence: torch.Tensor,
    pv_edge: torch.Tensor,
    pv_edge_solve: torch.Tensor,
    rho_edge_solve: torch.Tensor,
    inv_dc_edge_solve: torch.Tensor,
    rdzw: torch.Tensor,
    fzm: torch.Tensor,
    fzp: torch.Tensor,
    solve_edge_mask: torch.Tensor,
    solve_edge_mask_all: bool,
    n_vert_levels: int,
    h_divergence_cell1: torch.Tensor | None = None,
    h_divergence_cell2: torch.Tensor | None = None,
    h_divergence_cell_patch: torch.Tensor | None = None,
    cell_patch_row_start: int = 0,
    cell_patch_row_end: int = 0,
    vertical_tendency_patch: torch.Tensor | None = None,
    edge_neighbor_pv_patch: torch.Tensor | None = None,
    edge_neighbor_pv_reducer=None,
) -> torch.Tensor:
    """Compute dynamics-tendency normal-velocity-tendency base normal-velocity tendency in one PyTorch region."""
    edge_weights = _dynamics_tendency_slot_field(edge_neighbor_relation, 3, "weights")
    if edge_weights is None:
        raise ValueError(
            "dynamics-tendency normal-velocity-tendency base tendency requires edge-neighbor weights"
        )
    if vertical_tendency_patch is None:
        tend_u_patch = _normal_velocity_tendency_vertical_tendency_region(
            edge_cell_relation=edge_cell_relation,
            u_edge=u_edge,
            rw=rw,
            rdzw=rdzw,
            fzm=fzm,
            fzp=fzp,
            solve_edge_mask=solve_edge_mask,
            solve_edge_mask_all=solve_edge_mask_all,
            n_vert_levels=n_vert_levels,
        )
    else:
        tend_u_patch = vertical_tendency_patch

    if edge_neighbor_pv_patch is None:
        if edge_neighbor_pv_reducer is None:
            edge_neighbor_pv_reducer = _edge_neighbor_pv_reduction
        q = edge_neighbor_pv_reducer(
            edge_neighbor_relation=edge_neighbor_relation,
            u=u,
            pv_edge=pv_edge,
            pv_self=pv_edge_solve,
        )
    else:
        q = edge_neighbor_pv_patch

    u_base_patch = _normal_velocity_tendency_base_coupling_region(
        edge_cell_relation=edge_cell_relation,
        u_edge=u_edge,
        ke=ke,
        h_divergence=h_divergence,
        h_divergence_cell1=h_divergence_cell1,
        h_divergence_cell2=h_divergence_cell2,
        h_divergence_cell_patch=h_divergence_cell_patch,
        cell_patch_row_start=cell_patch_row_start,
        cell_patch_row_end=cell_patch_row_end,
        edge_neighbor_pv_patch=q,
        rho_edge_solve=rho_edge_solve,
        inv_dc_edge_solve=inv_dc_edge_solve,
        solve_edge_mask=solve_edge_mask,
        solve_edge_mask_all=solve_edge_mask_all,
    )
    del q
    return tend_u_patch + u_base_patch


def _apply_u_curvature_tendency(
    *,
    tend_u_patch: torch.Tensor,
    u_edge: torch.Tensor,
    w: torch.Tensor,
    edge_c1: torch.Tensor,
    edge_c2: torch.Tensor,
    rho_edge_solve: torch.Tensor,
    angle_edge_solve: torch.Tensor,
    lat_edge_solve: torch.Tensor,
    omega: float,
    inv_r_earth: float,
    solve_edge_mask: torch.Tensor,
    solve_edge_mask_all: bool,
) -> torch.Tensor:
    """Apply dynamics-tendency spherical-curvature terms without dropping prior tendencies."""
    n_vert_levels = tend_u_patch.shape[1]
    w_avg = 0.25 * (
        w[edge_c1, 0:n_vert_levels]
        + w[edge_c1, 1 : n_vert_levels + 1]
        + w[edge_c2, 0:n_vert_levels]
        + w[edge_c2, 1 : n_vert_levels + 1]
    )
    curvature_patch = (
        2.0
        * omega
        * torch.cos(angle_edge_solve).unsqueeze(1)
        * torch.cos(lat_edge_solve).unsqueeze(1)
        * rho_edge_solve
        * w_avg
        + u_edge * w_avg * rho_edge_solve * inv_r_earth
    )
    if not solve_edge_mask_all:
        curvature_patch = solve_edge_mask * curvature_patch
    return tend_u_patch - curvature_patch


def _normal_velocity_mixing_u_del2_diffusion_region(
    *,
    edge_cell_relation: StaticPairRelation,
    edge_vertex_relation: StaticPairRelation,
    divergence: torch.Tensor,
    vorticity: torch.Tensor,
    kdiff: torch.Tensor,
    kdiff_patch: torch.Tensor | None,
    patch_row_start: int,
    patch_row_end: int,
    rho_edge_patch: torch.Tensor,
    tend_u_euler_patch: torch.Tensor,
    inv_dc_edge_patch: torch.Tensor,
    min_inv_dv_dc_edge_patch: torch.Tensor,
    mesh_scaling_del2_patch: torch.Tensor,
    edge_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute dynamics-tendency normal-velocity-mixing del2(u) and its second-order Euler update."""
    div_c1, div_c2 = _dynamics_tendency_pair_gather_patch_or_base(
        edge_cell_relation,
        divergence,
        None,
        patch_row_start=0,
        patch_row_end=0,
    )
    vort_v1, vort_v2 = _dynamics_tendency_pair_gather_patch_or_base(
        edge_vertex_relation,
        vorticity,
        None,
        patch_row_start=0,
        patch_row_end=0,
    )
    u_diffusion = (
        (div_c2 - div_c1) * inv_dc_edge_patch.unsqueeze(1)
        - (vort_v2 - vort_v1) * min_inv_dv_dc_edge_patch.unsqueeze(1)
    ) * edge_valid.unsqueeze(1).to(dtype=divergence.dtype)
    kdiffu_c1, kdiffu_c2 = _dynamics_tendency_pair_gather_patch_or_base(
        edge_cell_relation,
        kdiff,
        kdiff_patch,
        patch_row_start=patch_row_start,
        patch_row_end=patch_row_end,
    )
    kdiffu = 0.5 * (kdiffu_c1 + kdiffu_c2)
    updated_tend_u_euler_patch = (
        tend_u_euler_patch
        + rho_edge_patch * kdiffu * u_diffusion * mesh_scaling_del2_patch.unsqueeze(1)
    )
    return u_diffusion, updated_tend_u_euler_patch


def _normal_velocity_mixing_u_del4_diffusion_region(
    *,
    vertex_edge_relation: StaticSlotRelation,
    cell_edge_relation: StaticSlotRelation,
    solve_edge_pair_relation: StaticPairRelation,
    solve_edge_vertex_relation: StaticPairRelation,
    delsq_u: torch.Tensor,
    delsq_u_edge_patch: torch.Tensor,
    delsq_divergence: torch.Tensor,
    delsq_vorticity: torch.Tensor,
    tend_u_euler_edge_patch: torch.Tensor,
    vertex_edge_sign: torch.Tensor,
    cell_div_edge_sign: torch.Tensor,
    r_dc_solve: torch.Tensor,
    r_dv_solve: torch.Tensor,
    rho_edge_solve_patch: torch.Tensor | None,
    solve_edge_valid: torch.Tensor,
    edge_patch_row_start: int,
    edge_patch_row_end: int,
    cell_patch_row_start: int,
    cell_patch_row_end: int,
    vertex_patch_row_start: int,
    vertex_patch_row_end: int,
    edge_solve_rel_start: int,
    edge_solve_rel_end: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute dynamics-tendency normal-velocity-mixing del4(u) and update Euler u tendency in one region."""
    if rho_edge_solve_patch is None:
        raise ValueError(
            "dynamics-tendency normal-velocity-mixing del4 requires solve-edge density"
        )
    if (
        torch.is_grad_enabled()
        and not _compile_active()
        and delsq_u.shape[0] >= int(edge_patch_row_end)
        and delsq_divergence.shape[0] >= int(cell_patch_row_end)
        and delsq_vorticity.shape[0] >= int(vertex_patch_row_end)
    ):
        materialized_delsq_u = delsq_u.clone()
        materialized_delsq_u[
            int(edge_patch_row_start) : int(edge_patch_row_end),
            :,
        ] = delsq_u_edge_patch
        vertex_flux = (
            vertex_edge_sign.unsqueeze(2)
            * materialized_delsq_u[
                _dynamics_tendency_slot_field(vertex_edge_relation, 0, "indices"), :
            ]
        )
        cell_flux = (
            cell_div_edge_sign.unsqueeze(2)
            * materialized_delsq_u[
                _dynamics_tendency_slot_field(cell_edge_relation, 0, "indices"), :
            ]
        )
        delsq_vorticity_vertex_patch = _sum_fortran_order(vertex_flux, dim=1)
        delsq_divergence_cell_patch = _sum_fortran_order(cell_flux, dim=1)
        materialized_divergence = delsq_divergence.clone()
        materialized_divergence[
            int(cell_patch_row_start) : int(cell_patch_row_end),
            :,
        ] = delsq_divergence_cell_patch
        materialized_vorticity = delsq_vorticity.clone()
        materialized_vorticity[
            int(vertex_patch_row_start) : int(vertex_patch_row_end),
            :,
        ] = delsq_vorticity_vertex_patch
        div_cell1 = materialized_divergence[
            _dynamics_tendency_pair_field(solve_edge_pair_relation, 0, "first"), :
        ]
        div_cell2 = materialized_divergence[
            _dynamics_tendency_pair_field(solve_edge_pair_relation, 1, "second"), :
        ]
        vort_vertex1 = materialized_vorticity[
            _dynamics_tendency_pair_field(solve_edge_vertex_relation, 0, "first"), :
        ]
        vort_vertex2 = materialized_vorticity[
            _dynamics_tendency_pair_field(solve_edge_vertex_relation, 1, "second"), :
        ]
        u_diffusion = (
            rho_edge_solve_patch
            * (
                (div_cell2 - div_cell1) * r_dc_solve.unsqueeze(1)
                - (vort_vertex2 - vort_vertex1) * r_dv_solve.unsqueeze(1)
            )
            * solve_edge_valid.unsqueeze(1).to(dtype=rho_edge_solve_patch.dtype)
        )
        updated_tend_u_euler_edge_patch = tend_u_euler_edge_patch.clone()
        updated_tend_u_euler_edge_patch[
            int(edge_solve_rel_start) : int(edge_solve_rel_end),
            :,
        ] = (
            updated_tend_u_euler_edge_patch[
                int(edge_solve_rel_start) : int(edge_solve_rel_end),
                :,
            ]
            - u_diffusion
        )
        return (
            delsq_divergence_cell_patch,
            delsq_vorticity_vertex_patch,
            updated_tend_u_euler_edge_patch,
        )

    delsq_vorticity_vertex_patch = _sum_weighted_index_slots(
        field=delsq_u,
        weights=vertex_edge_sign,
        index_relation=vertex_edge_relation,
        field_patch=delsq_u_edge_patch,
        patch_row_start=edge_patch_row_start,
        patch_row_end=edge_patch_row_end,
    )
    delsq_divergence_cell_patch = _sum_weighted_index_slots(
        field=delsq_u,
        weights=cell_div_edge_sign,
        index_relation=cell_edge_relation,
        field_patch=delsq_u_edge_patch,
        patch_row_start=edge_patch_row_start,
        patch_row_end=edge_patch_row_end,
    )

    div_cell1, div_cell2 = _dynamics_tendency_pair_gather_patch_or_base(
        solve_edge_pair_relation,
        delsq_divergence,
        delsq_divergence_cell_patch,
        patch_row_start=cell_patch_row_start,
        patch_row_end=cell_patch_row_end,
    )
    vort_vertex1, vort_vertex2 = _dynamics_tendency_pair_gather_patch_or_base(
        solve_edge_vertex_relation,
        delsq_vorticity,
        delsq_vorticity_vertex_patch,
        patch_row_start=vertex_patch_row_start,
        patch_row_end=vertex_patch_row_end,
    )
    u_diffusion = (
        rho_edge_solve_patch
        * (
            (div_cell2 - div_cell1) * r_dc_solve.unsqueeze(1)
            - (vort_vertex2 - vort_vertex1) * r_dv_solve.unsqueeze(1)
        )
        * solve_edge_valid.unsqueeze(1).to(dtype=rho_edge_solve_patch.dtype)
    )
    tend_u_del4_patch = (
        tend_u_euler_edge_patch[edge_solve_rel_start:edge_solve_rel_end, :]
        - u_diffusion
    )
    if _compile_active():
        if (
            edge_solve_rel_start == 0
            and edge_solve_rel_end == tend_u_euler_edge_patch.shape[0]
        ):
            updated_tend_u_euler_edge_patch = tend_u_del4_patch
        else:
            updated_tend_u_euler_edge_patch = tend_u_euler_edge_patch.clone()
            updated_tend_u_euler_edge_patch[
                edge_solve_rel_start:edge_solve_rel_end, :
            ] = tend_u_del4_patch
    else:
        updated_tend_u_euler_edge_patch = _replace_row_slice(
            tend_u_euler_edge_patch,
            edge_solve_rel_start,
            edge_solve_rel_end,
            tend_u_del4_patch,
        )
    return (
        delsq_divergence_cell_patch,
        delsq_vorticity_vertex_patch,
        updated_tend_u_euler_edge_patch,
    )


def _normal_velocity_mixing_u_vertical_mix_update_region(
    *,
    u_mix_solve: torch.Tensor,
    rho_edge_solve: torch.Tensor,
    tend_u_euler_solve_inner: torch.Tensor,
    zgrid_cell1: torch.Tensor | None,
    zgrid_cell2: torch.Tensor | None,
    edge_valid: torch.Tensor,
    v_mom_eddy_visc2: float,
    n_vert_levels: int,
) -> torch.Tensor:
    """Apply dynamics-tendency normal-velocity-mixing vertical u-mixing on active solve-edge columns."""
    if n_vert_levels <= 2:
        return tend_u_euler_solve_inner[:, 0:0]
    if zgrid_cell1 is None or zgrid_cell2 is None:
        raise ValueError(
            "dynamics-tendency normal-velocity-mixing vertical mixing requires both edge-cell heights"
        )

    z1 = 0.5 * (
        zgrid_cell1[:, 0 : n_vert_levels - 2] + zgrid_cell2[:, 0 : n_vert_levels - 2]
    )
    z2 = 0.5 * (
        zgrid_cell1[:, 1 : n_vert_levels - 1] + zgrid_cell2[:, 1 : n_vert_levels - 1]
    )
    z3 = 0.5 * (zgrid_cell1[:, 2:n_vert_levels] + zgrid_cell2[:, 2:n_vert_levels])
    z4 = 0.5 * (
        zgrid_cell1[:, 3 : n_vert_levels + 1] + zgrid_cell2[:, 3 : n_vert_levels + 1]
    )

    zm = 0.5 * (z1 + z2)
    z0 = 0.5 * (z2 + z3)
    zp = 0.5 * (z3 + z4)
    vert_mix = (
        rho_edge_solve[:, 1 : n_vert_levels - 1]
        * v_mom_eddy_visc2
        * (
            (u_mix_solve[:, 2:n_vert_levels] - u_mix_solve[:, 1 : n_vert_levels - 1])
            / (zp - z0)
            - (
                u_mix_solve[:, 1 : n_vert_levels - 1]
                - u_mix_solve[:, 0 : n_vert_levels - 2]
            )
            / (z0 - zm)
        )
        / (0.5 * (zp - zm))
    ) * edge_valid.unsqueeze(1).to(dtype=u_mix_solve.dtype)
    return tend_u_euler_solve_inner + vert_mix


def _replace_active_rows_cols_numeric(
    patch: torch.Tensor,
    values: torch.Tensor,
    *,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
) -> torch.Tensor:
    """Functionally replace one static rectangle without indexed mutation."""
    if row_start == row_end or col_start == col_end:
        return patch
    middle = torch.cat(
        (
            patch[row_start:row_end, :col_start],
            values,
            patch[row_start:row_end, col_end:],
        ),
        dim=1,
    )
    if row_start == 0 and row_end == patch.shape[0]:
        return middle
    return torch.cat((patch[:row_start], middle, patch[row_end:]), dim=0)


def _normal_velocity_mixing_u_mixing_section_region(
    *,
    edge_cell_relation: StaticPairRelation,
    edge_vertex_relation: StaticPairRelation,
    divergence: torch.Tensor,
    vorticity: torch.Tensor,
    kdiff: torch.Tensor,
    kdiff_patch: torch.Tensor | None,
    patch_row_start: int,
    patch_row_end: int,
    rho_edge_patch: torch.Tensor,
    tend_u_euler_edge_patch: torch.Tensor,
    inv_dc_edge_patch: torch.Tensor,
    min_inv_dv_dc_edge_patch: torch.Tensor,
    mesh_scaling_del2_patch: torch.Tensor,
    edge_valid: torch.Tensor,
    include_del4: bool,
    vertex_edge_relation: StaticSlotRelation | None,
    cell_edge_relation: StaticSlotRelation | None,
    solve_edge_pair_relation: StaticPairRelation | None,
    solve_edge_vertex_relation: StaticPairRelation | None,
    delsq_u: torch.Tensor,
    delsq_divergence: torch.Tensor,
    delsq_vorticity: torch.Tensor,
    vertex_edge_sign: torch.Tensor | None,
    cell_div_edge_sign: torch.Tensor | None,
    r_dc_solve: torch.Tensor | None,
    r_dv_solve: torch.Tensor | None,
    rho_edge_solve_patch: torch.Tensor | None,
    solve_edge_valid: torch.Tensor,
    edge_patch_row_start: int,
    edge_patch_row_end: int,
    cell_patch_row_start: int,
    cell_patch_row_end: int,
    vertex_patch_row_start: int,
    vertex_patch_row_end: int,
    edge_solve_rel_start: int,
    edge_solve_rel_end: int,
    include_vertical_mix: bool,
    u_mix_solve: torch.Tensor | None,
    zgrid_cell1: torch.Tensor | None,
    zgrid_cell2: torch.Tensor | None,
    v_mom_eddy_visc2: float,
    n_vert_levels: int,
) -> torch.Tensor:
    """Execute normal-velocity-mixing and return only the tendency consumed by the next section."""
    delsq_u_edge_patch, updated_tend_u_euler = (
        _normal_velocity_mixing_u_del2_diffusion_region(
            edge_cell_relation=edge_cell_relation,
            edge_vertex_relation=edge_vertex_relation,
            divergence=divergence,
            vorticity=vorticity,
            kdiff=kdiff,
            kdiff_patch=kdiff_patch,
            patch_row_start=patch_row_start,
            patch_row_end=patch_row_end,
            rho_edge_patch=rho_edge_patch,
            tend_u_euler_patch=tend_u_euler_edge_patch,
            inv_dc_edge_patch=inv_dc_edge_patch,
            min_inv_dv_dc_edge_patch=min_inv_dv_dc_edge_patch,
            mesh_scaling_del2_patch=mesh_scaling_del2_patch,
            edge_valid=edge_valid,
        )
    )
    if include_del4:
        if (
            vertex_edge_relation is None
            or cell_edge_relation is None
            or solve_edge_pair_relation is None
            or solve_edge_vertex_relation is None
            or vertex_edge_sign is None
            or cell_div_edge_sign is None
            or r_dc_solve is None
            or r_dv_solve is None
        ):
            raise ValueError(
                "dynamics-tendency normal-velocity-mixing del4 requires complete static topology"
            )
        (
            _delsq_divergence_cell_patch,
            _delsq_vorticity_vertex_patch,
            updated_tend_u_euler,
        ) = _normal_velocity_mixing_u_del4_diffusion_region(
            vertex_edge_relation=vertex_edge_relation,
            cell_edge_relation=cell_edge_relation,
            solve_edge_pair_relation=solve_edge_pair_relation,
            solve_edge_vertex_relation=solve_edge_vertex_relation,
            delsq_u=delsq_u,
            delsq_u_edge_patch=delsq_u_edge_patch,
            delsq_divergence=delsq_divergence,
            delsq_vorticity=delsq_vorticity,
            tend_u_euler_edge_patch=updated_tend_u_euler,
            vertex_edge_sign=vertex_edge_sign,
            cell_div_edge_sign=cell_div_edge_sign,
            r_dc_solve=r_dc_solve,
            r_dv_solve=r_dv_solve,
            rho_edge_solve_patch=rho_edge_solve_patch,
            solve_edge_valid=solve_edge_valid,
            edge_patch_row_start=edge_patch_row_start,
            edge_patch_row_end=edge_patch_row_end,
            cell_patch_row_start=cell_patch_row_start,
            cell_patch_row_end=cell_patch_row_end,
            vertex_patch_row_start=vertex_patch_row_start,
            vertex_patch_row_end=vertex_patch_row_end,
            edge_solve_rel_start=edge_solve_rel_start,
            edge_solve_rel_end=edge_solve_rel_end,
        )
    if include_vertical_mix:
        if u_mix_solve is None or rho_edge_solve_patch is None:
            raise ValueError(
                "dynamics-tendency normal-velocity-mixing vertical mixing requires velocity and density"
            )
        mix_patch = _normal_velocity_mixing_u_vertical_mix_update_region(
            u_mix_solve=u_mix_solve,
            rho_edge_solve=rho_edge_solve_patch,
            tend_u_euler_solve_inner=updated_tend_u_euler[
                edge_solve_rel_start:edge_solve_rel_end,
                1 : n_vert_levels - 1,
            ],
            zgrid_cell1=zgrid_cell1,
            zgrid_cell2=zgrid_cell2,
            edge_valid=solve_edge_valid,
            v_mom_eddy_visc2=v_mom_eddy_visc2,
            n_vert_levels=n_vert_levels,
        )
        updated_tend_u_euler = _replace_active_rows_cols_numeric(
            updated_tend_u_euler,
            mix_patch,
            row_start=edge_solve_rel_start,
            row_end=edge_solve_rel_end,
            col_start=1,
            col_end=n_vert_levels - 1,
        )
    return updated_tend_u_euler


def _normal_velocity_finalization_u_rayleigh_damp_region(
    *,
    tend_u_solve_levels: torch.Tensor,
    rho_edge_solve_levels: torch.Tensor,
    u_solve_levels: torch.Tensor,
    rayleigh_damp_levels: torch.Tensor,
) -> torch.Tensor:
    """Apply dynamics-tendency normal-velocity-finalization Rayleigh damping on the configured top-level band."""
    return (
        tend_u_solve_levels
        - rho_edge_solve_levels * u_solve_levels * rayleigh_damp_levels.unsqueeze(0)
    )


def _normal_velocity_finalization_u_final_section_region(
    *,
    tend_u_solve: torch.Tensor,
    tend_u_euler_solve: torch.Tensor,
    rho_edge_solve: torch.Tensor | None,
    u_solve: torch.Tensor | None,
    tend_ru_physics_solve: torch.Tensor | None,
    apply_rayleigh: bool,
    rayleigh_coef_inverse: float,
    rayleigh_damp_level_count: int,
    n_vert_levels: int,
) -> torch.Tensor:
    """Apply optional normal-velocity-finalization damping and complete the normal-momentum tendency."""
    if apply_rayleigh:
        if rho_edge_solve is None or u_solve is None:
            raise ValueError(
                "dynamics-tendency normal-velocity-finalization Rayleigh damping requires density and velocity"
            )
        level_start = n_vert_levels - rayleigh_damp_level_count
        rayleigh_levels = torch.arange(
            level_start,
            n_vert_levels,
            dtype=tend_u_solve.dtype,
            device=tend_u_solve.device,
        )
        rayleigh_damp = (
            (rayleigh_levels + 1.0) - float(level_start)
        ) * rayleigh_coef_inverse
        damped_levels = _normal_velocity_finalization_u_rayleigh_damp_region(
            tend_u_solve_levels=tend_u_solve[:, level_start:n_vert_levels],
            rho_edge_solve_levels=rho_edge_solve[:, level_start:n_vert_levels],
            u_solve_levels=u_solve[:, level_start:n_vert_levels],
            rayleigh_damp_levels=rayleigh_damp,
        )
        tend_u_solve = torch.cat(
            (tend_u_solve[:, :level_start], damped_levels),
            dim=1,
        )
    tend_u_solve = tend_u_solve + tend_u_euler_solve
    if tend_ru_physics_solve is not None:
        tend_u_solve = tend_u_solve + tend_ru_physics_solve
    return tend_u_solve


def _normal_velocity_mixing_normal_velocity_finalization_u_tendency_region(
    *,
    edge_cell_relation: StaticPairRelation,
    edge_vertex_relation: StaticPairRelation,
    divergence: torch.Tensor,
    vorticity: torch.Tensor,
    kdiff: torch.Tensor,
    kdiff_patch: torch.Tensor | None,
    patch_row_start: int,
    patch_row_end: int,
    rho_edge_patch: torch.Tensor,
    tend_u_euler_edge_patch: torch.Tensor,
    inv_dc_edge_patch: torch.Tensor,
    min_inv_dv_dc_edge_patch: torch.Tensor,
    mesh_scaling_del2_patch: torch.Tensor,
    edge_valid: torch.Tensor,
    include_del4: bool,
    vertex_edge_relation: StaticSlotRelation | None,
    cell_edge_relation: StaticSlotRelation | None,
    solve_edge_pair_relation: StaticPairRelation | None,
    solve_edge_vertex_relation: StaticPairRelation | None,
    delsq_u: torch.Tensor,
    delsq_divergence: torch.Tensor,
    delsq_vorticity: torch.Tensor,
    vertex_edge_sign: torch.Tensor | None,
    cell_div_edge_sign: torch.Tensor | None,
    r_dc_solve: torch.Tensor | None,
    r_dv_solve: torch.Tensor | None,
    rho_edge_solve_patch: torch.Tensor | None,
    solve_edge_valid: torch.Tensor,
    edge_patch_row_start: int,
    edge_patch_row_end: int,
    cell_patch_row_start: int,
    cell_patch_row_end: int,
    vertex_patch_row_start: int,
    vertex_patch_row_end: int,
    edge_solve_rel_start: int,
    edge_solve_rel_end: int,
    include_vertical_mix: bool,
    u_mix_solve: torch.Tensor | None,
    zgrid_cell1: torch.Tensor | None,
    zgrid_cell2: torch.Tensor | None,
    v_mom_eddy_visc2: float,
    tend_u_solve: torch.Tensor,
    u_solve_rayleigh: torch.Tensor | None,
    tend_ru_physics_solve: torch.Tensor | None,
    apply_rayleigh: bool,
    rayleigh_coef_inverse: float,
    rayleigh_damp_level_count: int,
    n_vert_levels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose normal-velocity-mixing and normal-velocity-finalization while materializing the public Euler tendency once."""
    updated_tend_u_euler = _normal_velocity_mixing_u_mixing_section_region(
        edge_cell_relation=edge_cell_relation,
        edge_vertex_relation=edge_vertex_relation,
        divergence=divergence,
        vorticity=vorticity,
        kdiff=kdiff,
        kdiff_patch=kdiff_patch,
        patch_row_start=patch_row_start,
        patch_row_end=patch_row_end,
        rho_edge_patch=rho_edge_patch,
        tend_u_euler_edge_patch=tend_u_euler_edge_patch,
        inv_dc_edge_patch=inv_dc_edge_patch,
        min_inv_dv_dc_edge_patch=min_inv_dv_dc_edge_patch,
        mesh_scaling_del2_patch=mesh_scaling_del2_patch,
        edge_valid=edge_valid,
        include_del4=include_del4,
        vertex_edge_relation=vertex_edge_relation,
        cell_edge_relation=cell_edge_relation,
        solve_edge_pair_relation=solve_edge_pair_relation,
        solve_edge_vertex_relation=solve_edge_vertex_relation,
        delsq_u=delsq_u,
        delsq_divergence=delsq_divergence,
        delsq_vorticity=delsq_vorticity,
        vertex_edge_sign=vertex_edge_sign,
        cell_div_edge_sign=cell_div_edge_sign,
        r_dc_solve=r_dc_solve,
        r_dv_solve=r_dv_solve,
        rho_edge_solve_patch=rho_edge_solve_patch,
        solve_edge_valid=solve_edge_valid,
        edge_patch_row_start=edge_patch_row_start,
        edge_patch_row_end=edge_patch_row_end,
        cell_patch_row_start=cell_patch_row_start,
        cell_patch_row_end=cell_patch_row_end,
        vertex_patch_row_start=vertex_patch_row_start,
        vertex_patch_row_end=vertex_patch_row_end,
        edge_solve_rel_start=edge_solve_rel_start,
        edge_solve_rel_end=edge_solve_rel_end,
        include_vertical_mix=include_vertical_mix,
        u_mix_solve=u_mix_solve,
        zgrid_cell1=zgrid_cell1,
        zgrid_cell2=zgrid_cell2,
        v_mom_eddy_visc2=v_mom_eddy_visc2,
        n_vert_levels=n_vert_levels,
    )
    final_tend_u = _normal_velocity_finalization_u_final_section_region(
        tend_u_solve=tend_u_solve,
        tend_u_euler_solve=updated_tend_u_euler[
            edge_solve_rel_start:edge_solve_rel_end, :
        ],
        rho_edge_solve=rho_edge_solve_patch if apply_rayleigh else None,
        u_solve=u_solve_rayleigh,
        tend_ru_physics_solve=tend_ru_physics_solve,
        apply_rayleigh=apply_rayleigh,
        rayleigh_coef_inverse=rayleigh_coef_inverse,
        rayleigh_damp_level_count=rayleigh_damp_level_count,
        n_vert_levels=n_vert_levels,
    )
    return updated_tend_u_euler, final_tend_u


def _normal_velocity_tendency_normal_velocity_section_region(
    *,
    edge_cell_relation: StaticPairRelation,
    edge_neighbor_relation: StaticSlotRelation,
    u: torch.Tensor,
    u_edge: torch.Tensor,
    rw: torch.Tensor,
    ke: torch.Tensor,
    h_divergence: torch.Tensor,
    pv_edge: torch.Tensor,
    pv_edge_solve: torch.Tensor,
    rho_edge_solve: torch.Tensor,
    inv_dc_edge_solve: torch.Tensor,
    rdzw: torch.Tensor,
    fzm: torch.Tensor,
    fzp: torch.Tensor,
    solve_edge_mask: torch.Tensor,
    solve_edge_mask_all: bool,
    n_vert_levels: int,
    tend_u_euler_edge: torch.Tensor,
    include_pressure_gradient: bool,
    zz: torch.Tensor,
    pp: torch.Tensor,
    dpdz: torch.Tensor,
    cqu_solve: torch.Tensor,
    zxu_solve: torch.Tensor,
    include_curvature: bool,
    w: torch.Tensor,
    angle_edge_solve: torch.Tensor,
    lat_edge_solve: torch.Tensor,
    omega: float,
    inv_r_earth: float,
    h_divergence_cell1: torch.Tensor | None = None,
    h_divergence_cell2: torch.Tensor | None = None,
    h_divergence_cell_patch: torch.Tensor | None = None,
    dpdz_cell1: torch.Tensor | None = None,
    dpdz_cell2: torch.Tensor | None = None,
    dpdz_cell_patch: torch.Tensor | None = None,
    cell_patch_row_start: int = 0,
    cell_patch_row_end: int = 0,
    base_tendency_patch: torch.Tensor | None = None,
    vertical_tendency_patch: torch.Tensor | None = None,
    edge_neighbor_pv_patch: torch.Tensor | None = None,
    edge_neighbor_pv_reducer=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the full dynamics-tendency normal-velocity-tendency normal-velocity section as one PyTorch region."""
    if _dynamics_tendency_slot_field(edge_neighbor_relation, 3, "weights") is None:
        raise ValueError(
            "dynamics-tendency normal-velocity-tendency normal-velocity section requires edge-neighbor weights"
        )
    if base_tendency_patch is None:
        base_tendency_patch = _normal_velocity_tendency_base_normal_velocity_tendency(
            edge_cell_relation=edge_cell_relation,
            edge_neighbor_relation=edge_neighbor_relation,
            u=u,
            u_edge=u_edge,
            rw=rw,
            ke=ke,
            h_divergence=h_divergence,
            pv_edge=pv_edge,
            pv_edge_solve=pv_edge_solve,
            rho_edge_solve=rho_edge_solve,
            inv_dc_edge_solve=inv_dc_edge_solve,
            rdzw=rdzw,
            fzm=fzm,
            fzp=fzp,
            solve_edge_mask=solve_edge_mask,
            solve_edge_mask_all=solve_edge_mask_all,
            n_vert_levels=n_vert_levels,
            h_divergence_cell1=h_divergence_cell1,
            h_divergence_cell2=h_divergence_cell2,
            h_divergence_cell_patch=h_divergence_cell_patch,
            cell_patch_row_start=cell_patch_row_start,
            cell_patch_row_end=cell_patch_row_end,
            vertical_tendency_patch=vertical_tendency_patch,
            edge_neighbor_pv_patch=edge_neighbor_pv_patch,
            edge_neighbor_pv_reducer=edge_neighbor_pv_reducer,
        )
    return _normal_velocity_tendency_pressure_curvature_region(
        edge_cell_relation=edge_cell_relation,
        base_tendency_patch=base_tendency_patch,
        u_edge=u_edge,
        rho_edge_solve=rho_edge_solve,
        inv_dc_edge_solve=inv_dc_edge_solve,
        solve_edge_mask=solve_edge_mask,
        solve_edge_mask_all=solve_edge_mask_all,
        tend_u_euler_edge=tend_u_euler_edge,
        include_pressure_gradient=include_pressure_gradient,
        zz=zz,
        pp=pp,
        dpdz=dpdz,
        dpdz_cell1=dpdz_cell1,
        dpdz_cell2=dpdz_cell2,
        dpdz_cell_patch=dpdz_cell_patch,
        cell_patch_row_start=cell_patch_row_start,
        cell_patch_row_end=cell_patch_row_end,
        cqu_solve=cqu_solve,
        zxu_solve=zxu_solve,
        include_curvature=include_curvature,
        w=w,
        angle_edge_solve=angle_edge_solve,
        lat_edge_solve=lat_edge_solve,
        omega=omega,
        inv_r_earth=inv_r_earth,
    )


def _normal_velocity_tendency_normal_velocity_owned_section_region(
    *,
    edge_cell_relation: StaticPairRelation,
    edge_neighbor_relation: StaticSlotRelation,
    edge_solve_start: int,
    edge_solve_end: int,
    tend_u_euler_source_row_start: int,
    u: torch.Tensor,
    rw: torch.Tensor,
    ke: torch.Tensor,
    h_divergence: torch.Tensor,
    pv_edge: torch.Tensor,
    rho_edge: torch.Tensor,
    inv_dc_edge: torch.Tensor,
    rdzw: torch.Tensor,
    fzm: torch.Tensor,
    fzp: torch.Tensor,
    solve_edge_mask: torch.Tensor,
    solve_edge_mask_all: bool,
    n_vert_levels: int,
    tend_u_euler_source: torch.Tensor,
    include_pressure_gradient: bool,
    zz: torch.Tensor,
    pp: torch.Tensor,
    dpdz: torch.Tensor,
    cqu: torch.Tensor,
    zxu: torch.Tensor,
    include_curvature: bool,
    w: torch.Tensor,
    angle_edge: torch.Tensor,
    lat_edge: torch.Tensor,
    omega: float,
    inv_r_earth: float,
    h_divergence_cell_patch: torch.Tensor | None = None,
    dpdz_cell_patch: torch.Tensor | None = None,
    cell_patch_row_start: int = 0,
    cell_patch_row_end: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run normal-velocity-tendency from owning inputs, forming every solve view inside the region."""
    solve_slice = slice(int(edge_solve_start), int(edge_solve_end))
    euler_start = int(edge_solve_start) - int(tend_u_euler_source_row_start)
    euler_end = int(edge_solve_end) - int(tend_u_euler_source_row_start)
    return _normal_velocity_tendency_normal_velocity_section_region(
        edge_cell_relation=edge_cell_relation,
        edge_neighbor_relation=edge_neighbor_relation,
        u=u,
        u_edge=u[solve_slice, :],
        rw=rw,
        ke=ke,
        h_divergence=h_divergence,
        h_divergence_cell_patch=h_divergence_cell_patch,
        pv_edge=pv_edge,
        pv_edge_solve=pv_edge[solve_slice, :],
        rho_edge_solve=rho_edge[solve_slice, :],
        inv_dc_edge_solve=inv_dc_edge[solve_slice],
        rdzw=rdzw,
        fzm=fzm,
        fzp=fzp,
        solve_edge_mask=solve_edge_mask,
        solve_edge_mask_all=solve_edge_mask_all,
        n_vert_levels=n_vert_levels,
        tend_u_euler_edge=tend_u_euler_source[euler_start:euler_end, :],
        include_pressure_gradient=include_pressure_gradient,
        zz=zz,
        pp=pp,
        dpdz=dpdz,
        dpdz_cell_patch=dpdz_cell_patch,
        cell_patch_row_start=cell_patch_row_start,
        cell_patch_row_end=cell_patch_row_end,
        cqu_solve=cqu[solve_slice, :],
        zxu_solve=zxu[solve_slice, :],
        include_curvature=include_curvature,
        w=w,
        angle_edge_solve=angle_edge[solve_slice],
        lat_edge_solve=lat_edge[solve_slice],
        omega=omega,
        inv_r_earth=inv_r_earth,
    )


def _sum_smagorinsky_deformation_slots(
    *,
    cell_edge_relation: StaticSlotRelation,
    defc_a_cell: torch.Tensor,
    defc_b_cell: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate Smagorinsky deformation in the Fortran statement order."""
    edge_ids = _dynamics_tendency_slot_field(cell_edge_relation, 0, "indices")
    active = _dynamics_tendency_slot_field(cell_edge_relation, 1, "valid")
    active_all_by_slot = _dynamics_tendency_slot_field(
        cell_edge_relation, 6, "valid_all_by_slot"
    )
    if active is None:
        active = torch.ones_like(edge_ids, dtype=torch.bool)
    n_cells = edge_ids.shape[0]
    d_diag = torch.zeros((n_cells, u.shape[1]), dtype=u.dtype, device=u.device)
    d_off_diag = torch.zeros_like(d_diag)
    gather_batch = _dynamics_tendency_slot_gather_batch(u, edge_ids.shape[1])
    if gather_batch == 1:
        for slot in range(edge_ids.shape[1]):
            u_slot = gather_rows(u, edge_ids[:, slot])
            v_slot = gather_rows(v, edge_ids[:, slot])
            defc_a_slot = defc_a_cell[:, slot].view(n_cells, 1)
            defc_b_slot = defc_b_cell[:, slot].view(n_cells, 1)
            next_diag = (d_diag + defc_a_slot * u_slot) - defc_b_slot * v_slot
            next_off_diag = (d_off_diag + defc_b_slot * u_slot) + defc_a_slot * v_slot
            valid_all = active_all_by_slot is not None and active_all_by_slot[slot]
            if not valid_all:
                valid = active[:, slot].view(n_cells, 1)
                next_diag = torch.where(valid, next_diag, d_diag)
                next_off_diag = torch.where(valid, next_off_diag, d_off_diag)
            d_diag = next_diag
            d_off_diag = next_off_diag
        return d_diag, d_off_diag
    for batch_start, batch_stop in _dynamics_tendency_slot_batches(
        edge_ids.shape[1], gather_batch
    ):
        u_batch = _dynamics_tendency_gather_slot_batch(
            u, edge_ids, batch_start, batch_stop
        )
        v_batch = _dynamics_tendency_gather_slot_batch(
            v, edge_ids, batch_start, batch_stop
        )
        for local_slot in range(batch_stop - batch_start):
            slot = batch_start + local_slot
            u_slot = u_batch[:, local_slot, :]
            v_slot = v_batch[:, local_slot, :]
            defc_a_slot = defc_a_cell[:, slot].view(n_cells, 1)
            defc_b_slot = defc_b_cell[:, slot].view(n_cells, 1)
            # Upstream evaluates these assignments from left to right:
            #   d_diag     = d_diag     + a*u - b*v
            #   d_off_diag = d_off_diag + b*u + a*v
            # Preserve the per-edge operation order: regrouping the products
            # changes binary64 results across a large mesh.
            next_diag = (d_diag + defc_a_slot * u_slot) - defc_b_slot * v_slot
            next_off_diag = (d_off_diag + defc_b_slot * u_slot) + defc_a_slot * v_slot
            valid_all = active_all_by_slot is not None and active_all_by_slot[slot]
            if not valid_all:
                valid = active[:, slot].view(n_cells, 1)
                next_diag = torch.where(valid, next_diag, d_diag)
                next_off_diag = torch.where(valid, next_off_diag, d_off_diag)
            d_diag = next_diag
            d_off_diag = next_off_diag
    return d_diag, d_off_diag


def _mixing_initialization_mixing_initialization_region(
    *,
    rk_step: int,
    edge_count: int,
    n_vert_levels: int,
    config_horiz_mixing: str,
    cell_edge_relation: StaticSlotRelation,
    defc_a_cell: torch.Tensor,
    defc_b_cell: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    c_s: float,
    config_len_disp: float,
    inv_dt: float,
    config_h_theta_eddy_visc2: float,
    config_visc4_2dsmag: float,
    config_h_mom_eddy_visc4: float,
    config_h_theta_eddy_visc4: float,
    config_mpas_cam_coef: float,
    config_number_cam_damping_levels: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, float, float]:
    """Initialize dynamics-tendency mixing fields while preserving the Fortran operation order."""
    horiz_mixing = str(config_horiz_mixing).strip()
    if horiz_mixing not in {"2d_smagorinsky", "2d_fixed"}:
        raise ValueError(
            "Unsupported config_horiz_mixing="
            f"{config_horiz_mixing!r}. Supported modes are '2d_smagorinsky' "
            "and '2d_fixed'."
        )

    if horiz_mixing == "2d_smagorinsky":
        h_mom_eddy_visc4 = config_visc4_2dsmag * config_len_disp**3
        h_theta_eddy_visc4 = h_mom_eddy_visc4
    else:
        h_mom_eddy_visc4 = config_h_mom_eddy_visc4
        h_theta_eddy_visc4 = config_h_theta_eddy_visc4

    if rk_step != 1:
        return None, None, h_mom_eddy_visc4, h_theta_eddy_visc4

    tend_u_euler_edge_patch = torch.zeros(
        (int(edge_count), n_vert_levels),
        dtype=DTYPE,
        device=u.device,
    )
    if horiz_mixing == "2d_smagorinsky":
        d_diag, d_off_diag = _sum_smagorinsky_deformation_slots(
            cell_edge_relation=cell_edge_relation,
            defc_a_cell=defc_a_cell,
            defc_b_cell=defc_b_cell,
            u=u,
            v=v,
        )
        deformation = fct_sqrt_pos(d_diag * d_diag + d_off_diag * d_off_diag)
        kdiff_cap = (0.01 * config_len_disp**2) * inv_dt
        kdiff_cell_patch = fct_min(
            ((c_s * config_len_disp) ** 2) * deformation,
            torch.full_like(deformation, kdiff_cap),
        )
    else:
        kdiff_cell_patch = torch.full(
            (defc_a_cell.shape[0], n_vert_levels),
            config_h_theta_eddy_visc2,
            dtype=DTYPE,
            device=u.device,
        )

    if config_mpas_cam_coef > 0.0 and config_number_cam_damping_levels > 0:
        k_start = n_vert_levels - config_number_cam_damping_levels
        k_idx = torch.arange(
            k_start,
            n_vert_levels,
            device=u.device,
            dtype=torch.int64,
        )
        visc2cam_base = 4.0 * 2.0833 * config_len_disp * config_mpas_cam_coef
        cam_factor = 1.0 - (n_vert_levels - (k_idx + 1)).to(DTYPE) / float(
            config_number_cam_damping_levels
        )
        visc2cam_levels = visc2cam_base * cam_factor
        kdiff_cam_patch = fct_max(
            kdiff_cell_patch[:, k_start:n_vert_levels],
            visc2cam_levels.unsqueeze(0),
        )
        kdiff_cell_patch = _replace_row_slice_cols(
            kdiff_cell_patch,
            0,
            kdiff_cell_patch.shape[0],
            k_start,
            n_vert_levels,
            kdiff_cam_patch,
        )

    return (
        tend_u_euler_edge_patch,
        kdiff_cell_patch,
        h_mom_eddy_visc4,
        h_theta_eddy_visc4,
    )


def _sum_local_cell_edge_flux_slots(
    *,
    edge_flux: torch.Tensor,
    local_edge_relation: StaticSlotRelation,
) -> torch.Tensor:
    """Accumulate local edge fluxes by cell without a full cell-edge gather."""
    local_edges = _dynamics_tendency_slot_field(local_edge_relation, 0, "indices")
    edge_sign = _dynamics_tendency_slot_field(local_edge_relation, 2, "sign")
    edge_valid_weight = _dynamics_tendency_slot_field(
        local_edge_relation, 7, "valid_weight"
    )
    edge_valid_all_by_slot = _dynamics_tendency_slot_field(
        local_edge_relation, 6, "valid_all_by_slot"
    )
    if local_edges is None or edge_sign is None or edge_valid_weight is None:
        raise ValueError(
            "dynamics-tendency local cell-edge flux requires a static relation"
        )
    n_cells = local_edges.shape[0]
    total: torch.Tensor | None = None
    for batch_start, batch_stop in _dynamics_tendency_slot_batches(
        local_edges.shape[1],
        _dynamics_tendency_slot_gather_batch(edge_flux, local_edges.shape[1]),
    ):
        if batch_start == 0 and batch_stop == local_edges.shape[1] and batch_stop > 1:
            flux_batch = _dynamics_tendency_gather_slot_batch(
                edge_flux, local_edges, batch_start, batch_stop
            )
            contrib_batch = edge_sign.unsqueeze(2) * flux_batch
            if edge_valid_all_by_slot is None or not all(edge_valid_all_by_slot):
                contrib_batch = contrib_batch * edge_valid_weight.unsqueeze(2)
            return _sum_fortran_order(contrib_batch, dim=1)
        flux_batch = _dynamics_tendency_gather_slot_batch(
            edge_flux, local_edges, batch_start, batch_stop
        )
        for local_slot in range(batch_stop - batch_start):
            slot = batch_start + local_slot
            contrib = edge_sign[:, slot].unsqueeze(1) * flux_batch[:, local_slot, :]
            valid_all = (
                edge_valid_all_by_slot is not None and edge_valid_all_by_slot[slot]
            )
            if not valid_all:
                contrib = contrib * edge_valid_weight[:, slot].unsqueeze(1)
            total = _accumulate_ordered_slot(total, contrib)
    if total is None:
        return edge_flux.new_zeros((n_cells, edge_flux.shape[1]))
    return total


def _edge_advective_cell_tendency(
    *,
    field: torch.Tensor,
    field_start: int,
    field_stop: int,
    ru_edge: torch.Tensor,
    edge_adv_cell_relation: StaticSlotRelation | tuple[Any, ...],
    local_edge_relation: StaticSlotRelation | tuple[Any, ...],
    sign_ru: torch.Tensor,
) -> torch.Tensor:
    """Accumulate dynamics-tendency edge-advective fluxes and reduce them back to cells."""
    safe_adv_cells = _dynamics_tendency_slot_field(edge_adv_cell_relation, 0, "indices")
    adv_valid = _dynamics_tendency_slot_field(edge_adv_cell_relation, 1, "valid")
    adv_coef = _dynamics_tendency_slot_field(edge_adv_cell_relation, 4, "coefficients")
    adv_coef_3rd = _dynamics_tendency_slot_field(
        edge_adv_cell_relation, 5, "coefficients_3rd"
    )
    local_indices = _dynamics_tendency_slot_field(local_edge_relation, 0, "indices")
    local_valid = _dynamics_tendency_slot_field(local_edge_relation, 1, "valid")
    local_sign = _dynamics_tendency_slot_field(local_edge_relation, 2, "sign")
    # These tuples are runner-owned topology facts. Keeping them visible while
    # tracing lets Inductor omit masks for uniformly valid slots instead of
    # rediscovering the same fact over every edge/cell/level element.
    adv_valid_all_by_slot = _dynamics_tendency_slot_field(
        edge_adv_cell_relation,
        6,
        "valid_all_by_slot",
    )
    local_valid_all_by_slot = _dynamics_tendency_slot_field(
        local_edge_relation,
        6,
        "valid_all_by_slot",
    )
    adv_destinations = _dynamics_tendency_slot_field(
        edge_adv_cell_relation, 11, "transpose_destinations"
    )
    adv_transpose_valid = _dynamics_tendency_slot_field(
        edge_adv_cell_relation, 12, "transpose_valid"
    )
    local_destinations = _dynamics_tendency_slot_field(
        local_edge_relation, 11, "transpose_destinations"
    )
    local_transpose_valid = _dynamics_tendency_slot_field(
        local_edge_relation, 12, "transpose_valid"
    )
    if safe_adv_cells is None or adv_coef is None or adv_coef_3rd is None:
        raise ValueError(
            "dynamics-tendency edge advection requires an advection-cell relation"
        )
    if local_indices is None or local_sign is None:
        raise ValueError(
            "dynamics-tendency edge advection requires a local edge relation"
        )
    if adv_valid is None:
        adv_valid = torch.ones_like(safe_adv_cells, dtype=torch.bool)
    if local_valid is None:
        local_valid = torch.ones_like(local_indices, dtype=torch.bool)

    field_levels = field[:, field_start:field_stop]
    ordered_reverse = _dynamics_tendency_use_ordered_reverse_vjp(field_levels, ru_edge)
    if ordered_reverse:
        if any(
            value is None
            for value in (
                adv_destinations,
                adv_transpose_valid,
                local_destinations,
                local_transpose_valid,
            )
        ):
            raise ValueError(
                "dynamics-tendency vertical-velocity-horizontal-advection/theta-horizontal-advection differentiable execution requires plan-owned ordered "
                "transpose metadata for both advection and local-edge relations"
            )
        return cast(
            torch.Tensor,
            _EdgeAdvectiveCellTendencyFunction.apply(
                field_levels,
                ru_edge,
                safe_adv_cells,
                adv_valid,
                adv_coef,
                adv_coef_3rd,
                local_indices,
                local_valid,
                local_sign,
                adv_destinations,
                adv_transpose_valid,
                local_destinations,
                local_transpose_valid,
                adv_valid_all_by_slot,
                local_valid_all_by_slot,
            ),
        )
    return _edge_advective_cell_tendency_ordered_from_sign_numeric(
        field_levels,
        ru_edge,
        sign_ru,
        safe_adv_cells,
        adv_valid,
        adv_coef,
        adv_coef_3rd,
        local_indices,
        local_valid,
        local_sign,
        adv_valid_all_by_slot,
        local_valid_all_by_slot,
    )


def _edge_advective_flux_ordered_numeric(
    field_levels: torch.Tensor,
    sign_ru: torch.Tensor,
    adv_indices: torch.Tensor,
    adv_valid: torch.Tensor,
    adv_coef: torch.Tensor,
    adv_coef_3rd: torch.Tensor,
    adv_valid_all_by_slot: tuple[bool, ...] | None = None,
) -> torch.Tensor:
    """Build one edge-owned flux patch with the exact upstream slot association."""
    flux_sum: torch.Tensor | None = None
    reuse_private = not torch.is_grad_enabled() and not _compile_active()
    batch_size = (
        _slot_gather_batch(field_levels, adv_indices.shape[1])
        if _compile_active()
        else 1
    )
    for batch_start, batch_stop in _slot_batches(adv_indices.shape[1], batch_size):
        field_batch = _gather_slot_batch(
            field_levels, adv_indices, batch_start, batch_stop
        )
        for local_slot in range(batch_stop - batch_start):
            slot = batch_start + local_slot
            field_slot = field_batch[:, local_slot, :]
            if reuse_private:
                weight = sign_ru * adv_coef_3rd[:, slot].unsqueeze(1)
                torch.add(adv_coef[:, slot].unsqueeze(1), weight, out=weight)
                torch.mul(weight, field_slot, out=field_slot)
                contribution = field_slot
            else:
                weight = adv_coef[:, slot].unsqueeze(1) + sign_ru * adv_coef_3rd[
                    :, slot
                ].unsqueeze(1)
                contribution = weight * field_slot
            valid_all = (
                adv_valid_all_by_slot is not None and adv_valid_all_by_slot[slot]
            )
            if not valid_all:
                if reuse_private:
                    contribution.masked_fill_(~adv_valid[:, slot].unsqueeze(1), 0.0)
                else:
                    contribution = contribution.masked_fill(
                        ~adv_valid[:, slot].unsqueeze(1),
                        0.0,
                    )
            flux_sum = _accumulate_ordered_slot(flux_sum, contribution)
    if flux_sum is None:
        return field_levels.new_zeros(sign_ru.shape)
    return flux_sum


def _edge_advective_cell_tendency_ordered_from_sign_numeric(
    field_levels: torch.Tensor,
    ru_edge: torch.Tensor,
    sign_ru: torch.Tensor,
    adv_indices: torch.Tensor,
    adv_valid: torch.Tensor,
    adv_coef: torch.Tensor,
    adv_coef_3rd: torch.Tensor,
    local_indices: torch.Tensor,
    local_valid: torch.Tensor,
    local_sign: torch.Tensor,
    adv_valid_all_by_slot: tuple[bool, ...] | None = None,
    local_valid_all_by_slot: tuple[bool, ...] | None = None,
) -> torch.Tensor:
    """Exact output-owned vertical-velocity-horizontal-advection/theta-horizontal-advection primal for eager, Inductor, VJP, and JVP."""
    flux_sum = _edge_advective_flux_ordered_numeric(
        field_levels,
        sign_ru,
        adv_indices,
        adv_valid,
        adv_coef,
        adv_coef_3rd,
        adv_valid_all_by_slot,
    )

    total: torch.Tensor | None = None
    reuse_private = not torch.is_grad_enabled() and not _compile_active()
    # The edge product is invariant across every adjacent-cell slot. Form it
    # once so compiled execution gathers one edge field per slot instead of
    # independently gathering ru_edge and flux_sum. Eager inference reuses its
    # private flux buffer; differentiable/compiled execution remains functional.
    if reuse_private:
        torch.mul(ru_edge, flux_sum, out=flux_sum)
        edge_product = flux_sum
    else:
        edge_product = ru_edge * flux_sum
    batch_size = (
        _slot_gather_batch(edge_product, local_indices.shape[1])
        if _compile_active()
        else 1
    )
    for batch_start, batch_stop in _slot_batches(local_indices.shape[1], batch_size):
        edge_flux_batch = _gather_slot_batch(
            edge_product, local_indices, batch_start, batch_stop
        )
        for local_slot in range(batch_stop - batch_start):
            slot = batch_start + local_slot
            edge_flux = edge_flux_batch[:, local_slot, :]
            if reuse_private:
                edge_flux.mul_(-local_sign[:, slot].unsqueeze(1))
                contribution = edge_flux
            else:
                contribution = -local_sign[:, slot].unsqueeze(1) * edge_flux
            valid_all = (
                local_valid_all_by_slot is not None and local_valid_all_by_slot[slot]
            )
            if not valid_all:
                if reuse_private:
                    contribution.masked_fill_(~local_valid[:, slot].unsqueeze(1), 0.0)
                else:
                    contribution = contribution.masked_fill(
                        ~local_valid[:, slot].unsqueeze(1),
                        0.0,
                    )
            total = _accumulate_ordered_slot(total, contribution)
    if total is None:
        return field_levels.new_zeros((local_indices.shape[0], field_levels.shape[1]))
    return total


def _edge_advective_cell_tendency_ordered_numeric(
    field_levels: torch.Tensor,
    ru_edge: torch.Tensor,
    adv_indices: torch.Tensor,
    adv_valid: torch.Tensor,
    adv_coef: torch.Tensor,
    adv_coef_3rd: torch.Tensor,
    local_indices: torch.Tensor,
    local_valid: torch.Tensor,
    local_sign: torch.Tensor,
    adv_valid_all_by_slot: tuple[bool, ...] | None = None,
    local_valid_all_by_slot: tuple[bool, ...] | None = None,
) -> torch.Tensor:
    """Derive the smooth upwind sign and invoke the shared ordered primal."""
    return _edge_advective_cell_tendency_ordered_from_sign_numeric(
        field_levels,
        ru_edge,
        fct_sign_pm(ru_edge),
        adv_indices,
        adv_valid,
        adv_coef,
        adv_coef_3rd,
        local_indices,
        local_valid,
        local_sign,
        adv_valid_all_by_slot,
        local_valid_all_by_slot,
    )


def _edge_advective_cell_tendency_ordered_backward(
    field_levels: torch.Tensor,
    ru_edge: torch.Tensor,
    grad_output: torch.Tensor,
    adv_indices: torch.Tensor,
    adv_valid: torch.Tensor,
    adv_coef: torch.Tensor,
    adv_coef_3rd: torch.Tensor,
    local_indices: torch.Tensor,
    local_valid: torch.Tensor,
    local_sign: torch.Tensor,
    adv_destinations: torch.Tensor,
    adv_transpose_valid: torch.Tensor,
    local_destinations: torch.Tensor,
    local_transpose_valid: torch.Tensor,
    adv_valid_all_by_slot: tuple[bool, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Output-owned transpose for the ordered vertical-velocity-horizontal-advection/theta-horizontal-advection edge-advection stencil."""
    sign_ru = torch.where(
        ru_edge >= 0.0, torch.ones_like(ru_edge), -torch.ones_like(ru_edge)
    )
    flux_sum: torch.Tensor | None = None
    for slot in range(adv_indices.shape[1]):
        field_slot = gather_rows(field_levels, adv_indices[:, slot])
        weight = adv_coef[:, slot].unsqueeze(1) + sign_ru * adv_coef_3rd[
            :, slot
        ].unsqueeze(1)
        contribution = weight * field_slot
        valid_all = adv_valid_all_by_slot is not None and adv_valid_all_by_slot[slot]
        if not valid_all:
            contribution = contribution.masked_fill(
                ~adv_valid[:, slot].unsqueeze(1), 0.0
            )
        flux_sum = _accumulate_ordered_slot(flux_sum, contribution)
    if flux_sum is None:
        flux_sum = field_levels.new_zeros(ru_edge.shape)

    grad_flux: torch.Tensor | None = None
    grad_ru_direct: torch.Tensor | None = None
    for slot in range(local_indices.shape[1] - 1, -1, -1):
        slot_grad_flux = torch.zeros_like(ru_edge)
        slot_grad_ru = torch.zeros_like(ru_edge)
        for duplicate in range(local_destinations.shape[2]):
            destination = local_destinations[:, slot, duplicate]
            active = local_transpose_valid[:, slot, duplicate].unsqueeze(1)
            gathered_grad = gather_rows(grad_output, destination)
            gathered_sign = gather_rows(local_sign[:, slot], destination).unsqueeze(1)
            grad_edge_flux = (-gathered_grad) * gathered_sign
            flux_part = grad_edge_flux * ru_edge
            ru_part = grad_edge_flux * flux_sum
            slot_grad_flux = slot_grad_flux + flux_part.masked_fill(~active, 0.0)
            slot_grad_ru = slot_grad_ru + ru_part.masked_fill(~active, 0.0)
        grad_flux = _accumulate_ordered_slot(grad_flux, slot_grad_flux)
        grad_ru_direct = _accumulate_ordered_slot(grad_ru_direct, slot_grad_ru)
    if grad_flux is None or grad_ru_direct is None:
        grad_flux = torch.zeros_like(ru_edge)
        grad_ru_direct = torch.zeros_like(ru_edge)

    grad_field: torch.Tensor | None = None
    for slot in range(adv_indices.shape[1] - 1, -1, -1):
        slot_grad_field = torch.zeros_like(field_levels)
        for duplicate in range(adv_destinations.shape[2]):
            destination = adv_destinations[:, slot, duplicate]
            active = adv_transpose_valid[:, slot, duplicate].unsqueeze(1)
            gathered_grad = gather_rows(grad_flux, destination)
            gathered_weight = gather_rows(adv_coef[:, slot], destination).unsqueeze(
                1
            ) + gather_rows(sign_ru * adv_coef_3rd[:, slot].unsqueeze(1), destination)
            contribution = gathered_grad * gathered_weight
            slot_grad_field = slot_grad_field + contribution.masked_fill(~active, 0.0)
        grad_field = _accumulate_ordered_slot(grad_field, slot_grad_field)

    if grad_field is None:
        grad_field = torch.zeros_like(field_levels)

    # ``sign(ru_edge)`` is locally constant away from its switch at zero.  The
    # release preflight rejects directions that cross that switch, so the exact
    # smooth-region contribution through the sign selector is zero.
    grad_ru = grad_ru_direct
    return grad_field, grad_ru


class _EdgeAdvectiveCellTendencyFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        field_levels: torch.Tensor,
        ru_edge: torch.Tensor,
        adv_indices: torch.Tensor,
        adv_valid: torch.Tensor,
        adv_coef: torch.Tensor,
        adv_coef_3rd: torch.Tensor,
        local_indices: torch.Tensor,
        local_valid: torch.Tensor,
        local_sign: torch.Tensor,
        adv_destinations: torch.Tensor,
        adv_transpose_valid: torch.Tensor,
        local_destinations: torch.Tensor,
        local_transpose_valid: torch.Tensor,
        adv_valid_all_by_slot: tuple[bool, ...] | None,
        local_valid_all_by_slot: tuple[bool, ...] | None,
    ) -> torch.Tensor:
        ctx.save_for_backward(
            field_levels,
            ru_edge,
            adv_indices,
            adv_valid,
            adv_coef,
            adv_coef_3rd,
            local_indices,
            local_valid,
            local_sign,
            adv_destinations,
            adv_transpose_valid,
            local_destinations,
            local_transpose_valid,
        )
        ctx.adv_valid_all_by_slot = adv_valid_all_by_slot
        ctx.backward_callable = _planned_dynamics_tendency_adjoint(
            "edge_advective_cell_tendency_backward",
            _edge_advective_cell_tendency_ordered_backward,
        )
        return _edge_advective_cell_tendency_ordered_numeric(
            field_levels,
            ru_edge,
            adv_indices,
            adv_valid,
            adv_coef,
            adv_coef_3rd,
            local_indices,
            local_valid,
            local_sign,
            adv_valid_all_by_slot,
            local_valid_all_by_slot,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        saved = ctx.saved_tensors
        grad_field, grad_ru = ctx.backward_callable(
            saved[0],
            saved[1],
            grad_output,
            *saved[2:],
            ctx.adv_valid_all_by_slot,
        )
        return grad_field, grad_ru, *(None for _ in range(13))


def build_dynamics_tendency_ad_compile_plan(
    *,
    mode: str,
    device: torch.device | str,
    compile_mode: str = "default",
    compile_backend: str = "inductor",
) -> DynamicsTendencyADCompilePlan:
    """Build the immutable eager/compiled-adjoint/AOTAutograd dynamics-tendency policy.

    Compiled adjoints are an explicit CPU experiment, not the automatic policy.
    Whole-region AOTAutograd is selected by the outer dynamics-tendency compile target and
    does not compile these custom transposes separately.
    """
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized not in _DYNAMICS_TENDENCY_AD_COMPILE_MODES:
        allowed = ", ".join(sorted(_DYNAMICS_TENDENCY_AD_COMPILE_MODES))
        raise ValueError(
            f"unknown dynamics-tendency AD compile mode {mode!r}; allowed: {allowed}"
        )
    device_type = torch.device(device).type
    if normalized == "compiled_adjoint" and device_type != "cpu":
        raise ValueError(
            "compiled dynamics-tendency ordered adjoints are CPU-qualified only; "
            f"device {device_type!r} remains fail-closed"
        )

    callables: dict[str, Callable[..., Any]] = {
        "normal_velocity_tendency_edge_neighbor_pv_backward": _normal_velocity_tendency_edge_neighbor_pv_ordered_backward,
        "normal_velocity_tendency_vertical_backward": _normal_velocity_tendency_vertical_ordered_backward,
        "edge_advective_cell_tendency_backward": (
            _edge_advective_cell_tendency_ordered_backward
        ),
    }
    if normalized == "compiled_adjoint":
        from mpas_pytorch.dynamics.compile_utils import maybe_compile_kernel

        for attribute, fn in tuple(callables.items()):
            callables[attribute] = maybe_compile_kernel(
                f"dynamics_tendency_adjoint_{attribute}",
                fn,
                enabled=True,
                mode=str(compile_mode),
                backend=str(compile_backend),
                specialization_key=(
                    "dynamics_tendency_ad_compile_plan",
                    attribute,
                    device_type,
                ),
                allow_reverse_autodiff=True,
            )

    return DynamicsTendencyADCompilePlan(
        mode=normalized,
        device_type=device_type,
        compile_mode=str(compile_mode),
        compile_backend=str(compile_backend),
        normal_velocity_tendency_edge_neighbor_pv_backward=callables[
            "normal_velocity_tendency_edge_neighbor_pv_backward"
        ],
        normal_velocity_tendency_vertical_backward=callables[
            "normal_velocity_tendency_vertical_backward"
        ],
        edge_advective_cell_tendency_backward=callables[
            "edge_advective_cell_tendency_backward"
        ],
    )


def _cell_owned_edge_advective_cell_tendency(
    *,
    field: torch.Tensor,
    field_start: int,
    field_stop: int,
    ru_edge: torch.Tensor,
    edge_adv_cell_relation: StaticSlotRelation | tuple[Any, ...],
    local_edge_relation: StaticSlotRelation | tuple[Any, ...],
    sign_ru: torch.Tensor,
) -> torch.Tensor:
    """A/B-only cell-owned form of the vertical-velocity-horizontal-advection/theta-horizontal-advection advective cell tendency.

    This deliberately trades redundant per-cell edge-flux work for avoiding the
    global edge-owned ``edge_flux_sum`` tensor. It is not the default path; it is
    wired only for controlled CUDA A/B runs.
    """
    safe_adv_cells = _dynamics_tendency_slot_field(edge_adv_cell_relation, 0, "indices")
    adv_coef = _dynamics_tendency_slot_field(edge_adv_cell_relation, 4, "coefficients")
    adv_coef_3rd = _dynamics_tendency_slot_field(
        edge_adv_cell_relation, 5, "coefficients_3rd"
    )
    local_edges = _dynamics_tendency_slot_field(local_edge_relation, 0, "indices")
    edge_sign = _dynamics_tendency_slot_field(local_edge_relation, 2, "sign")
    edge_valid_weight = _dynamics_tendency_slot_field(
        local_edge_relation, 7, "valid_weight"
    )
    edge_valid_all_by_slot = _dynamics_tendency_slot_field(
        local_edge_relation, 6, "valid_all_by_slot"
    )
    if safe_adv_cells is None or adv_coef is None or adv_coef_3rd is None:
        raise ValueError(
            "dynamics-tendency cell-owned advection requires an advection-cell relation"
        )
    if local_edges is None or edge_sign is None or edge_valid_weight is None:
        raise ValueError(
            "dynamics-tendency cell-owned advection requires a local edge relation"
        )
    n_cells = local_edges.shape[0]
    n_levels = field_stop - field_start
    field_levels = field[:, field_start:field_stop]
    total: torch.Tensor | None = None
    for edge_slot in range(local_edges.shape[1]):
        edge_idx = local_edges[:, edge_slot]
        ru_slot = gather_rows(ru_edge, edge_idx)
        sign_slot = gather_rows(sign_ru, edge_idx)
        adv_cells_slot = gather_rows(safe_adv_cells, edge_idx)
        adv_coef_slot = gather_rows(adv_coef, edge_idx)
        adv_coef_3rd_slot = gather_rows(adv_coef_3rd, edge_idx)
        flux_sum: torch.Tensor | None = None
        for adv_slot in range(safe_adv_cells.shape[1]):
            field_slot = gather_rows(field_levels, adv_cells_slot[:, adv_slot])
            weight_slot = adv_coef_slot[:, adv_slot].unsqueeze(
                1
            ) + sign_slot * adv_coef_3rd_slot[:, adv_slot].unsqueeze(1)
            flux_sum = _accumulate_ordered_slot(flux_sum, weight_slot * field_slot)
        if flux_sum is None:
            flux_sum = field.new_zeros((n_cells, n_levels))
        contrib = edge_sign[:, edge_slot].unsqueeze(1) * ru_slot * flux_sum
        valid_all = (
            edge_valid_all_by_slot is not None and edge_valid_all_by_slot[edge_slot]
        )
        if not valid_all:
            contrib = contrib * edge_valid_weight[:, edge_slot].unsqueeze(1)
        total = _accumulate_ordered_slot(total, contrib)
    if total is None:
        return field.new_zeros((n_cells, n_levels))
    return -total


def _edge_advective_cell_tendency_proof_tensors(
    *,
    field: torch.Tensor,
    field_start: int,
    field_stop: int,
    ru_edge: torch.Tensor,
    edge_adv_cell_relation: StaticSlotRelation | tuple[Any, ...],
    local_edge_relation: StaticSlotRelation | tuple[Any, ...],
    sign_ru: torch.Tensor,
    full_level_count: int,
    ru_edge_abs_term_sum: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return local vertical-velocity-horizontal-advection/theta-horizontal-advection advection components in cell, slot, level order."""
    safe_adv_cells = _dynamics_tendency_slot_field(edge_adv_cell_relation, 0, "indices")
    adv_coef = _dynamics_tendency_slot_field(edge_adv_cell_relation, 4, "coefficients")
    adv_coef_3rd = _dynamics_tendency_slot_field(
        edge_adv_cell_relation, 5, "coefficients_3rd"
    )
    adv_valid = _dynamics_tendency_slot_field(edge_adv_cell_relation, 1, "valid")
    local_edges = _dynamics_tendency_slot_field(local_edge_relation, 0, "indices")
    edge_sign = _dynamics_tendency_slot_field(local_edge_relation, 2, "sign")
    local_valid = _dynamics_tendency_slot_field(local_edge_relation, 1, "valid")
    if safe_adv_cells is None or adv_coef is None or adv_coef_3rd is None:
        raise ValueError(
            "dynamics-tendency proof advection requires an advection-cell relation"
        )
    if local_edges is None or edge_sign is None:
        raise ValueError(
            "dynamics-tendency proof advection requires a local edge relation"
        )

    n_edges = safe_adv_cells.shape[0]
    n_levels = field_stop - field_start
    field_levels = field[:, field_start:field_stop]
    flux_sum = field.new_zeros((n_edges, n_levels))
    flux_abs_sum = field.new_zeros((n_edges, n_levels))
    for slot in range(safe_adv_cells.shape[1]):
        idx_slot = safe_adv_cells[:, slot]
        weight_slot = adv_coef[:, slot].unsqueeze(1) + sign_ru * adv_coef_3rd[
            :, slot
        ].unsqueeze(1)
        contrib = weight_slot * gather_rows(field_levels, idx_slot)
        if adv_valid is not None:
            contrib = torch.where(
                adv_valid[:, slot].unsqueeze(1),
                contrib,
                torch.zeros_like(contrib),
            )
        flux_sum = flux_sum + contrib
        flux_abs_sum = flux_abs_sum + torch.abs(contrib)

    flux_slots = gather_rows(flux_sum, local_edges)
    flux_abs_slots = gather_rows(flux_abs_sum, local_edges)
    ru_slots = gather_rows(ru_edge, local_edges)
    ru_abs_source = (
        torch.abs(ru_edge) if ru_edge_abs_term_sum is None else ru_edge_abs_term_sum
    )
    ru_abs_slots = gather_rows(ru_abs_source, local_edges)
    if local_valid is not None:
        valid = local_valid.unsqueeze(2)
        flux_slots = torch.where(valid, flux_slots, torch.zeros_like(flux_slots))
        flux_abs_slots = torch.where(
            valid, flux_abs_slots, torch.zeros_like(flux_abs_slots)
        )
        ru_slots = torch.where(valid, ru_slots, torch.zeros_like(ru_slots))
        ru_abs_slots = torch.where(valid, ru_abs_slots, torch.zeros_like(ru_abs_slots))

    n_cells, n_slots = local_edges.shape
    ru_full = field.new_zeros((n_cells, n_slots, full_level_count))
    ru_abs_full = field.new_zeros((n_cells, n_slots, full_level_count))
    flux_full = field.new_zeros((n_cells, n_slots, full_level_count))
    flux_abs_full = field.new_zeros((n_cells, n_slots, full_level_count))
    ru_full[:, :, field_start:field_stop] = ru_slots
    ru_abs_full[:, :, field_start:field_stop] = ru_abs_slots
    flux_full[:, :, field_start:field_stop] = flux_slots
    flux_abs_full[:, :, field_start:field_stop] = flux_abs_slots
    terms = -edge_sign.unsqueeze(2) * ru_slots * flux_slots
    term_abs = torch.abs(terms)
    if local_valid is not None:
        valid = local_valid.unsqueeze(2)
        terms = torch.where(valid, terms, torch.zeros_like(terms))
        term_abs = torch.where(valid, term_abs, torch.zeros_like(term_abs))
    terms_full = field.new_zeros((n_cells, n_slots, full_level_count))
    terms_full[:, :, field_start:field_stop] = terms
    terms_abs_source_full = field.new_zeros((n_cells, n_slots, full_level_count))
    terms_abs_source_full[:, :, field_start:field_stop] = ru_abs_slots * flux_abs_slots
    term_abs_full = field.new_zeros((n_cells, full_level_count))
    term_abs_full[:, field_start:field_stop] = _sum_fortran_order(
        ru_abs_slots * flux_abs_slots,
        dim=1,
    )
    return (
        ru_full,
        ru_abs_full,
        flux_full,
        flux_abs_full,
        term_abs_full,
        terms_full,
        terms_abs_source_full,
    )


def _vertical_velocity_horizontal_advection_w_advective_cell_tendency_region(
    *,
    w: torch.Tensor,
    ru_solve: torch.Tensor,
    fzm: torch.Tensor,
    fzp: torch.Tensor,
    n_vert_levels: int,
    edge_adv_cell_relation: StaticSlotRelation,
    local_edge_relation: StaticSlotRelation,
    cell_owned_vertical_velocity_horizontal_advection_theta_horizontal_advection_ab: bool = False,
) -> torch.Tensor:
    """Compute dynamics-tendency vertical-velocity-horizontal-advection horizontal w-advection as one PyTorch regional helper."""
    ru_edge = (
        fzm[1:n_vert_levels].view(1, -1) * ru_solve[:, 1:n_vert_levels]
        + fzp[1:n_vert_levels].view(1, -1) * ru_solve[:, 0 : n_vert_levels - 1]
    )
    sign_ru = fct_sign_pm(ru_edge)
    if cell_owned_vertical_velocity_horizontal_advection_theta_horizontal_advection_ab:
        return _cell_owned_edge_advective_cell_tendency(
            field=w,
            field_start=1,
            field_stop=n_vert_levels,
            ru_edge=ru_edge,
            edge_adv_cell_relation=edge_adv_cell_relation,
            local_edge_relation=local_edge_relation,
            sign_ru=sign_ru,
        )
    return _edge_advective_cell_tendency(
        field=w,
        field_start=1,
        field_stop=n_vert_levels,
        ru_edge=ru_edge,
        edge_adv_cell_relation=edge_adv_cell_relation,
        local_edge_relation=local_edge_relation,
        sign_ru=sign_ru,
    )


def _vertical_velocity_horizontal_advection_w_hadv_section_region(
    *,
    w: torch.Tensor,
    ru_solve: torch.Tensor,
    fzm: torch.Tensor,
    fzp: torch.Tensor,
    n_vert_levels: int,
    edge_adv_cell_relation: StaticSlotRelation,
    local_edge_relation: StaticSlotRelation,
    include_curvature: bool,
    ur_cell_solve: torch.Tensor,
    vr_cell_solve: torch.Tensor,
    rho_zz_solve: torch.Tensor,
    lat_cell_solve: torch.Tensor,
    omega: float,
    r_earth: float,
    cell_owned_vertical_velocity_horizontal_advection_theta_horizontal_advection_ab: bool = False,
) -> torch.Tensor:
    """Compute dynamics-tendency vertical-velocity-horizontal-advection horizontal w-advection and optional curvature as one region."""
    tend_w_patch = _vertical_velocity_horizontal_advection_w_advective_cell_tendency_region(
        w=w,
        ru_solve=ru_solve,
        fzm=fzm,
        fzp=fzp,
        n_vert_levels=n_vert_levels,
        edge_adv_cell_relation=edge_adv_cell_relation,
        local_edge_relation=local_edge_relation,
        cell_owned_vertical_velocity_horizontal_advection_theta_horizontal_advection_ab=cell_owned_vertical_velocity_horizontal_advection_theta_horizontal_advection_ab,
    )
    if include_curvature:
        fzm_inner = fzm[1:n_vert_levels].unsqueeze(0)
        fzp_inner = fzp[1:n_vert_levels].unsqueeze(0)
        ur_interp = (
            fzm_inner * ur_cell_solve[:, 1:n_vert_levels]
            + fzp_inner * ur_cell_solve[:, 0 : n_vert_levels - 1]
        )
        vr_interp = (
            fzm_inner * vr_cell_solve[:, 1:n_vert_levels]
            + fzp_inner * vr_cell_solve[:, 0 : n_vert_levels - 1]
        )
        rho_interp = (
            rho_zz_solve[:, 1:n_vert_levels] * fzm_inner
            + rho_zz_solve[:, 0 : n_vert_levels - 1] * fzp_inner
        )
        curvature_patch = rho_interp * (ur_interp**2 + vr_interp**2) / r_earth
        tend_w_patch = tend_w_patch + curvature_patch
        curvature_patch = (
            2.0
            * omega
            * torch.cos(lat_cell_solve).unsqueeze(1)
            * ur_interp
            * rho_interp
        )
        del fzm_inner, fzp_inner, ur_interp, vr_interp, rho_interp
        tend_w_patch = tend_w_patch + curvature_patch
        del curvature_patch
    return tend_w_patch


def _vertical_velocity_horizontal_advection_w_hadv_owned_section_region(
    *,
    w: torch.Tensor,
    ru: torch.Tensor,
    edge_solve_start: int,
    edge_solve_end: int,
    fzm: torch.Tensor,
    fzp: torch.Tensor,
    n_vert_levels: int,
    edge_adv_cell_relation: StaticSlotRelation,
    local_edge_relation: StaticSlotRelation,
    include_curvature: bool,
    ur_cell_solve: torch.Tensor,
    vr_cell_solve: torch.Tensor,
    rho_zz_solve: torch.Tensor,
    lat_cell_solve: torch.Tensor,
    omega: float,
    r_earth: float,
    cell_owned_vertical_velocity_horizontal_advection_theta_horizontal_advection_ab: bool = False,
) -> torch.Tensor:
    """Run vertical-velocity-horizontal-advection from owning fields so no aliased solve view crosses compilation."""
    edge_slice = slice(int(edge_solve_start), int(edge_solve_end))
    return _vertical_velocity_horizontal_advection_w_hadv_section_region(
        w=w,
        ru_solve=ru[edge_slice, :],
        fzm=fzm,
        fzp=fzp,
        n_vert_levels=n_vert_levels,
        edge_adv_cell_relation=edge_adv_cell_relation,
        local_edge_relation=local_edge_relation,
        include_curvature=include_curvature,
        ur_cell_solve=ur_cell_solve,
        vr_cell_solve=vr_cell_solve,
        rho_zz_solve=rho_zz_solve,
        lat_cell_solve=lat_cell_solve,
        omega=omega,
        r_earth=r_earth,
        cell_owned_vertical_velocity_horizontal_advection_theta_horizontal_advection_ab=cell_owned_vertical_velocity_horizontal_advection_theta_horizontal_advection_ab,
    )


def _theta_horizontal_advection_theta_advective_cell_tendency_region(
    *,
    theta_m: torch.Tensor,
    ru_solve: torch.Tensor,
    n_vert_levels: int,
    edge_adv_cell_relation: StaticSlotRelation | tuple[Any, ...],
    local_edge_relation: StaticSlotRelation | tuple[Any, ...],
    cell_owned_vertical_velocity_horizontal_advection_theta_horizontal_advection_ab: bool = False,
) -> torch.Tensor:
    """Compute dynamics-tendency theta-horizontal-advection horizontal theta advection as one PyTorch regional helper."""
    ru_edge = ru_solve[:, :n_vert_levels]
    sign_ru = fct_sign_pm(ru_edge)
    if cell_owned_vertical_velocity_horizontal_advection_theta_horizontal_advection_ab:
        return _cell_owned_edge_advective_cell_tendency(
            field=theta_m,
            field_start=0,
            field_stop=n_vert_levels,
            ru_edge=ru_edge,
            edge_adv_cell_relation=edge_adv_cell_relation,
            local_edge_relation=local_edge_relation,
            sign_ru=sign_ru,
        )
    return _edge_advective_cell_tendency(
        field=theta_m,
        field_start=0,
        field_stop=n_vert_levels,
        ru_edge=ru_edge,
        edge_adv_cell_relation=edge_adv_cell_relation,
        local_edge_relation=local_edge_relation,
        sign_ru=sign_ru,
    )


def _theta_horizontal_advection_theta_hadv_section_region(
    *,
    theta_m: torch.Tensor,
    ru_solve: torch.Tensor,
    n_vert_levels: int,
    edge_adv_cell_relation: StaticSlotRelation | tuple[Any, ...],
    local_edge_relation: StaticSlotRelation | tuple[Any, ...],
    include_perturbation_flux: bool,
    cell_edge_relation: StaticSlotRelation | tuple[Any, ...],
    edge_cell_relation: StaticPairRelation | tuple[Any, ...],
    ru: torch.Tensor,
    ru_save: torch.Tensor,
    theta_m_save: torch.Tensor,
    cell_owned_vertical_velocity_horizontal_advection_theta_horizontal_advection_ab: bool = False,
) -> torch.Tensor:
    """Compute dynamics-tendency theta-horizontal-advection horizontal theta advection and RK perturbation flux."""
    tend_theta_patch = _theta_horizontal_advection_theta_advective_cell_tendency_region(
        theta_m=theta_m,
        ru_solve=ru_solve,
        n_vert_levels=n_vert_levels,
        edge_adv_cell_relation=edge_adv_cell_relation,
        local_edge_relation=local_edge_relation,
        cell_owned_vertical_velocity_horizontal_advection_theta_horizontal_advection_ab=cell_owned_vertical_velocity_horizontal_advection_theta_horizontal_advection_ab,
    )
    if include_perturbation_flux:
        tend_theta_patch = _sum_theta_perturbation_flux_slots(
            cell_edge_relation=cell_edge_relation,
            edge_cell_relation=edge_cell_relation,
            ru=ru,
            ru_save=ru_save,
            theta_m_save=theta_m_save,
            initial=tend_theta_patch,
        )
    return tend_theta_patch


def _theta_horizontal_advection_theta_hadv_owned_section_region(
    *,
    theta_m: torch.Tensor,
    ru: torch.Tensor,
    edge_solve_start: int,
    edge_solve_end: int,
    n_vert_levels: int,
    edge_adv_cell_relation: StaticSlotRelation | tuple[Any, ...],
    local_edge_relation: StaticSlotRelation | tuple[Any, ...],
    include_perturbation_flux: bool,
    cell_edge_relation: StaticSlotRelation | tuple[Any, ...],
    edge_cell_relation: StaticPairRelation | tuple[Any, ...],
    ru_save: torch.Tensor,
    theta_m_save: torch.Tensor,
    cell_owned_vertical_velocity_horizontal_advection_theta_horizontal_advection_ab: bool = False,
) -> torch.Tensor:
    """Run theta-horizontal-advection from owning fields so no aliased solve view crosses compilation."""
    edge_slice = slice(int(edge_solve_start), int(edge_solve_end))
    return _theta_horizontal_advection_theta_hadv_section_region(
        theta_m=theta_m,
        ru_solve=ru[edge_slice, :],
        n_vert_levels=n_vert_levels,
        edge_adv_cell_relation=edge_adv_cell_relation,
        local_edge_relation=local_edge_relation,
        include_perturbation_flux=include_perturbation_flux,
        cell_edge_relation=cell_edge_relation,
        edge_cell_relation=edge_cell_relation,
        ru=ru,
        ru_save=ru_save,
        theta_m_save=theta_m_save,
        cell_owned_vertical_velocity_horizontal_advection_theta_horizontal_advection_ab=cell_owned_vertical_velocity_horizontal_advection_theta_horizontal_advection_ab,
    )


def _vertical_velocity_vertical_transport_w_vertical_flux_delta_region(
    *,
    w_solve: torch.Tensor,
    rw_solve: torch.Tensor,
    n_vert_levels: int,
    first_flux: torch.Tensor | None = None,
    last_flux: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute dynamics-tendency vertical-velocity-vertical-transport vertical w-flux divergence on solve-cell rows."""
    n_solve_cells = w_solve.shape[0]
    if n_vert_levels > 1:
        if first_flux is None or (n_vert_levels > 2 and last_flux is None):
            first_flux, last_flux = (
                _vertical_velocity_vertical_transport_w_vertical_boundary_fluxes_region(
                    w_solve=w_solve,
                    rw_solve=rw_solve,
                    n_vert_levels=n_vert_levels,
                )
            )
        if first_flux is None:
            raise ValueError("first_flux is required when n_vert_levels > 1")

        wdwz_mid = None
        if n_vert_levels > 3:
            wdwz_mid = flux3(
                w_solve[:, 0 : n_vert_levels - 3],
                w_solve[:, 1 : n_vert_levels - 2],
                w_solve[:, 2 : n_vert_levels - 1],
                w_solve[:, 3:n_vert_levels],
                0.5
                * (
                    rw_solve[:, 2 : n_vert_levels - 1]
                    + rw_solve[:, 1 : n_vert_levels - 2]
                ),
                1.0,
            )

        return _interior_level_vertical_flux_delta_from_parts(
            first_flux,
            wdwz_mid,
            last_flux,
            n_vert_levels,
        )
    return w_solve.new_zeros((n_solve_cells, 0))


def _vertical_velocity_vertical_transport_w_vertical_boundary_fluxes_region(
    *,
    w_solve: torch.Tensor,
    rw_solve: torch.Tensor,
    n_vert_levels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute only the vertical-velocity-vertical-transport vertical boundary fluxes with no interior reassociation."""
    n_solve_cells = w_solve.shape[0]
    if n_vert_levels <= 1:
        zeros = w_solve.new_zeros((n_solve_cells,))
        return zeros, zeros

    first_flux = (
        0.25 * (rw_solve[:, 1] + rw_solve[:, 0]) * (w_solve[:, 1] + w_solve[:, 0])
    )
    if n_vert_levels <= 2:
        return first_flux, first_flux.new_zeros(first_flux.shape)

    k = n_vert_levels - 1
    last_flux = (
        0.25
        * (rw_solve[:, k] + rw_solve[:, k - 1])
        * (w_solve[:, k] + w_solve[:, k - 1])
    )
    return first_flux, last_flux


def _vertical_velocity_vertical_transport_w_vertical_flux_proof_tensor(
    *,
    w_solve: torch.Tensor,
    rw_solve: torch.Tensor,
    n_vert_levels: int,
) -> torch.Tensor:
    """Reconstruct the vertical-velocity-vertical-transport Fortran ``wdwz`` scratch array for debug proof."""
    n_cells = w_solve.shape[0]
    wdwz = w_solve.new_zeros((n_cells, n_vert_levels + 1))
    if n_vert_levels <= 1:
        return wdwz
    first_flux, last_flux = (
        _vertical_velocity_vertical_transport_w_vertical_boundary_fluxes_region(
            w_solve=w_solve,
            rw_solve=rw_solve,
            n_vert_levels=n_vert_levels,
        )
    )
    wdwz[:, 1] = first_flux
    if n_vert_levels > 3:
        wdwz[:, 2 : n_vert_levels - 1] = flux3(
            w_solve[:, 0 : n_vert_levels - 3],
            w_solve[:, 1 : n_vert_levels - 2],
            w_solve[:, 2 : n_vert_levels - 1],
            w_solve[:, 3:n_vert_levels],
            0.5
            * (rw_solve[:, 2 : n_vert_levels - 1] + rw_solve[:, 1 : n_vert_levels - 2]),
            1.0,
        )
    if n_vert_levels > 2:
        wdwz[:, n_vert_levels - 1] = last_flux
    return wdwz


def _flux3_abs_term_sum(
    q_im2: torch.Tensor,
    q_im1: torch.Tensor,
    q_i: torch.Tensor,
    q_ip1: torch.Tensor,
    ua: torch.Tensor,
    coef3: float,
) -> torch.Tensor:
    """Conservative source-derived absolute scale for MPAS ``flux3``."""
    ua_abs = torch.abs(ua)
    flux4_terms = (
        ua_abs
        * (
            7.0 * torch.abs(q_i)
            + 7.0 * torch.abs(q_im1)
            + torch.abs(q_ip1)
            + torch.abs(q_im2)
        )
        / 12.0
    )
    upwind_terms = (
        abs(float(coef3))
        * ua_abs
        * (
            torch.abs(q_ip1)
            + torch.abs(q_im2)
            + 3.0 * torch.abs(q_i)
            + 3.0 * torch.abs(q_im1)
        )
        / 12.0
    )
    return flux4_terms + upwind_terms


def _vertical_velocity_vertical_transport_w_vertical_flux_abs_term_sum_proof_tensor(
    *,
    w_solve: torch.Tensor,
    rw_solve: torch.Tensor,
    n_vert_levels: int,
) -> torch.Tensor:
    """Return a source-derived absolute scale for the vertical-velocity-vertical-transport ``wdwz`` scratch."""
    n_cells = w_solve.shape[0]
    out = w_solve.new_zeros((n_cells, n_vert_levels + 1))
    if n_vert_levels <= 1:
        return out
    out[:, 1] = 0.25 * (
        torch.abs(rw_solve[:, 1] * w_solve[:, 1])
        + torch.abs(rw_solve[:, 1] * w_solve[:, 0])
        + torch.abs(rw_solve[:, 0] * w_solve[:, 1])
        + torch.abs(rw_solve[:, 0] * w_solve[:, 0])
    )
    if n_vert_levels > 3:
        out[:, 2 : n_vert_levels - 1] = _flux3_abs_term_sum(
            w_solve[:, 0 : n_vert_levels - 3],
            w_solve[:, 1 : n_vert_levels - 2],
            w_solve[:, 2 : n_vert_levels - 1],
            w_solve[:, 3:n_vert_levels],
            0.5
            * (rw_solve[:, 2 : n_vert_levels - 1] + rw_solve[:, 1 : n_vert_levels - 2]),
            1.0,
        )
    if n_vert_levels > 2:
        k = n_vert_levels - 1
        out[:, k] = 0.25 * (
            torch.abs(rw_solve[:, k] * w_solve[:, k])
            + torch.abs(rw_solve[:, k] * w_solve[:, k - 1])
            + torch.abs(rw_solve[:, k - 1] * w_solve[:, k])
            + torch.abs(rw_solve[:, k - 1] * w_solve[:, k - 1])
        )
    return out


def _vertical_velocity_vertical_transport_w_pointwise_update_region(
    *,
    w_solve: torch.Tensor,
    pp_solve: torch.Tensor,
    cqw_solve: torch.Tensor,
    rho_zz_solve: torch.Tensor,
    dpdz_solve: torch.Tensor,
    wdwz_delta: torch.Tensor,
    tend_w_solve_inner: torch.Tensor,
    tend_w_euler_solve_inner: torch.Tensor,
    inv_area_cell_solve: torch.Tensor,
    rdzu: torch.Tensor,
    rdzw: torch.Tensor,
    fzm: torch.Tensor,
    fzp: torch.Tensor,
    n_vert_levels: int,
    include_pressure_gradient: bool,
    include_vertical_mix: bool,
    v_mom_eddy_visc2: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply dynamics-tendency vertical-velocity-vertical-transport pointwise w updates from a precomputed flux delta."""
    tend_w_inner = (
        tend_w_solve_inner * inv_area_cell_solve.unsqueeze(1)
        - rdzu[1:n_vert_levels] * wdwz_delta
    )
    tend_w_euler_pressure_inner = tend_w_euler_solve_inner

    if include_pressure_gradient:
        tend_w_euler_pressure_inner = tend_w_euler_pressure_inner - cqw_solve[
            :, 1:n_vert_levels
        ] * (
            rdzu[1:n_vert_levels]
            * (pp_solve[:, 1:n_vert_levels] - pp_solve[:, 0 : n_vert_levels - 1])
            - (
                fzm[1:n_vert_levels] * dpdz_solve[:, 1:n_vert_levels]
                + fzp[1:n_vert_levels] * dpdz_solve[:, 0 : n_vert_levels - 1]
            )
        )

    tend_w_euler_mix_inner = tend_w_euler_pressure_inner
    if include_vertical_mix:
        rho_half = 0.5 * (
            rho_zz_solve[:, 1:n_vert_levels] + rho_zz_solve[:, 0 : n_vert_levels - 1]
        )
        w_up = (w_solve[:, 2 : n_vert_levels + 1] - w_solve[:, 1:n_vert_levels]) * rdzw[
            1:n_vert_levels
        ]
        w_down = (
            w_solve[:, 1:n_vert_levels] - w_solve[:, 0 : n_vert_levels - 1]
        ) * rdzw[0 : n_vert_levels - 1]
        tend_w_euler_mix_inner = (
            tend_w_euler_mix_inner
            + v_mom_eddy_visc2 * rho_half * (w_up - w_down) * rdzu[1:n_vert_levels]
        )

    return (
        tend_w_inner,
        tend_w_inner + tend_w_euler_mix_inner,
        tend_w_euler_pressure_inner,
        tend_w_euler_mix_inner,
    )


def _vertical_velocity_vertical_transport_w_vertical_section_region(
    *,
    w_solve: torch.Tensor,
    rw_solve: torch.Tensor,
    pp_solve: torch.Tensor,
    cqw_solve: torch.Tensor,
    rho_zz_solve: torch.Tensor,
    dpdz_solve: torch.Tensor,
    tend_w_solve_inner: torch.Tensor,
    tend_w_euler_solve_inner: torch.Tensor,
    inv_area_cell_solve: torch.Tensor,
    rdzu: torch.Tensor,
    rdzw: torch.Tensor,
    fzm: torch.Tensor,
    fzp: torch.Tensor,
    n_vert_levels: int,
    include_pressure_gradient: bool,
    include_vertical_mix: bool,
    v_mom_eddy_visc2: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute dynamics-tendency vertical-velocity-vertical-transport w vertical tendency/finalization as one numeric region."""
    wdwz_delta = _vertical_velocity_vertical_transport_w_vertical_flux_delta_region(
        w_solve=w_solve,
        rw_solve=rw_solve,
        n_vert_levels=n_vert_levels,
    )
    pointwise = _vertical_velocity_vertical_transport_w_pointwise_update_region(
        w_solve=w_solve,
        pp_solve=pp_solve,
        cqw_solve=cqw_solve,
        rho_zz_solve=rho_zz_solve,
        dpdz_solve=dpdz_solve,
        wdwz_delta=wdwz_delta,
        tend_w_solve_inner=tend_w_solve_inner,
        tend_w_euler_solve_inner=tend_w_euler_solve_inner,
        inv_area_cell_solve=inv_area_cell_solve,
        rdzu=rdzu,
        rdzw=rdzw,
        fzm=fzm,
        fzp=fzp,
        n_vert_levels=n_vert_levels,
        include_pressure_gradient=include_pressure_gradient,
        include_vertical_mix=include_vertical_mix,
        v_mom_eddy_visc2=v_mom_eddy_visc2,
    )
    return pointwise[1], pointwise[3]


def _vertical_velocity_horizontal_mixing_vertical_velocity_vertical_transport_w_tendency_region(
    *,
    cell_edge_relation: StaticSlotRelation | tuple[Any, ...],
    edge_cell_relation: StaticPairRelation | tuple[Any, ...],
    solve_cell_pair_relation: StaticPairRelation | tuple[Any, ...],
    edge_sign_del2: torch.Tensor,
    edge_sign_del4: torch.Tensor | None,
    rho_edge: torch.Tensor,
    w: torch.Tensor,
    kdiff: torch.Tensor,
    delsq_w_base: torch.Tensor,
    n_vert_levels: int,
    patch_row_start: int,
    patch_row_end: int,
    solve_relative_start: int,
    solve_relative_end: int,
    solve_flux_valid_all_by_slot: tuple[bool, ...] | None,
    tend_w_solve_inner: torch.Tensor,
    rw_solve: torch.Tensor,
    pp_solve: torch.Tensor,
    cqw_solve: torch.Tensor,
    rho_zz_solve: torch.Tensor,
    dpdz_solve: torch.Tensor,
    inv_area_cell_solve: torch.Tensor,
    rdzu: torch.Tensor,
    rdzw: torch.Tensor,
    fzm: torch.Tensor,
    fzp: torch.Tensor,
    include_pressure_gradient: bool,
    include_vertical_mix: bool,
    v_mom_eddy_visc2: float,
    kdiff_patch: torch.Tensor | None = None,
    mesh_scaling_del2: torch.Tensor | None = None,
    mesh_scaling_del2_slots: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse the contiguous RK1 vertical-velocity-horizontal-mixing/vertical-velocity-vertical-transport w tendency chain around its live state."""
    tend_w_euler_cell_patch = (
        _vertical_velocity_horizontal_mixing_w_horizontal_mix_section_region(
            cell_edge_relation=cell_edge_relation,
            edge_cell_relation=edge_cell_relation,
            solve_cell_pair_relation=solve_cell_pair_relation,
            edge_sign_del2=edge_sign_del2,
            edge_sign_del4=edge_sign_del4,
            rho_edge=rho_edge,
            w=w,
            kdiff=kdiff,
            delsq_w_base=delsq_w_base,
            n_vert_levels=n_vert_levels,
            patch_row_start=patch_row_start,
            patch_row_end=patch_row_end,
            solve_relative_start=solve_relative_start,
            solve_relative_end=solve_relative_end,
            solve_flux_valid_all_by_slot=solve_flux_valid_all_by_slot,
            kdiff_patch=kdiff_patch,
            mesh_scaling_del2=mesh_scaling_del2,
            mesh_scaling_del2_slots=mesh_scaling_del2_slots,
        )
    )
    euler_solve_inner = tend_w_euler_cell_patch[
        solve_relative_start:solve_relative_end,
        1:n_vert_levels,
    ]
    tend_w_final, euler_final = (
        _vertical_velocity_vertical_transport_w_vertical_section_region(
            w_solve=w[
                patch_row_start + solve_relative_start : patch_row_start
                + solve_relative_end,
                :,
            ],
            rw_solve=rw_solve,
            pp_solve=pp_solve,
            cqw_solve=cqw_solve,
            rho_zz_solve=rho_zz_solve,
            dpdz_solve=dpdz_solve,
            tend_w_solve_inner=tend_w_solve_inner,
            tend_w_euler_solve_inner=euler_solve_inner,
            inv_area_cell_solve=inv_area_cell_solve,
            rdzu=rdzu,
            rdzw=rdzw,
            fzm=fzm,
            fzp=fzp,
            n_vert_levels=n_vert_levels,
            include_pressure_gradient=include_pressure_gradient,
            include_vertical_mix=include_vertical_mix,
            v_mom_eddy_visc2=v_mom_eddy_visc2,
        )
    )
    tend_w_euler_cell_patch = _replace_active_rows_cols_numeric(
        tend_w_euler_cell_patch,
        euler_final,
        row_start=solve_relative_start,
        row_end=solve_relative_end,
        col_start=1,
        col_end=n_vert_levels,
    )
    return tend_w_final, tend_w_euler_cell_patch


def _theta_vertical_advection_diabatic_theta_vertical_flux_delta_region(
    *,
    theta_m_solve: torch.Tensor,
    theta_m_save_solve: torch.Tensor,
    rw_solve: torch.Tensor,
    rw_save_solve: torch.Tensor,
    fzm: torch.Tensor,
    fzp: torch.Tensor,
    coef_3rd_order: float,
    n_vert_levels: int,
    first_flux: torch.Tensor | None = None,
    last_flux: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute dynamics-tendency theta-vertical-advection-diabatic vertical theta-flux divergence on solve-cell rows."""
    if n_vert_levels > 1:
        if first_flux is None or (n_vert_levels > 2 and last_flux is None):
            first_flux, last_flux = (
                _theta_vertical_advection_diabatic_theta_vertical_boundary_fluxes_region(
                    theta_m_solve=theta_m_solve,
                    theta_m_save_solve=theta_m_save_solve,
                    rw_solve=rw_solve,
                    rw_save_solve=rw_save_solve,
                    fzm=fzm,
                    fzp=fzp,
                    n_vert_levels=n_vert_levels,
                )
            )
        if first_flux is None:
            raise ValueError("first_flux is required when n_vert_levels > 1")

        wdtz_mid = None
        if n_vert_levels > 3:
            wdtz_mid = flux3(
                theta_m_solve[:, 0 : n_vert_levels - 3],
                theta_m_solve[:, 1 : n_vert_levels - 2],
                theta_m_solve[:, 2 : n_vert_levels - 1],
                theta_m_solve[:, 3:n_vert_levels],
                rw_solve[:, 2 : n_vert_levels - 1],
                coef_3rd_order,
            )
            wdtz_mid = wdtz_mid + (
                rw_save_solve[:, 2 : n_vert_levels - 1]
                - rw_solve[:, 2 : n_vert_levels - 1]
            ) * (
                fzm[2 : n_vert_levels - 1]
                * theta_m_save_solve[:, 2 : n_vert_levels - 1]
                + fzp[2 : n_vert_levels - 1]
                * theta_m_save_solve[:, 1 : n_vert_levels - 2]
            )

        return _full_level_vertical_flux_delta_from_parts(
            first_flux,
            wdtz_mid,
            last_flux,
            n_vert_levels,
        )
    return theta_m_solve.new_zeros((theta_m_solve.shape[0], n_vert_levels))


def _theta_vertical_advection_diabatic_theta_vertical_boundary_fluxes_region(
    *,
    theta_m_solve: torch.Tensor,
    theta_m_save_solve: torch.Tensor,
    rw_solve: torch.Tensor,
    rw_save_solve: torch.Tensor,
    fzm: torch.Tensor,
    fzp: torch.Tensor,
    n_vert_levels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute only the theta-vertical-advection-diabatic vertical boundary fluxes with no interior reassociation."""
    n_solve_cells = theta_m_solve.shape[0]
    if n_vert_levels <= 1:
        zeros = theta_m_solve.new_zeros((n_solve_cells,))
        return zeros, zeros

    k = 1
    first_flux = rw_solve[:, k] * (
        fzm[k] * theta_m_solve[:, k] + fzp[k] * theta_m_solve[:, k - 1]
    ) + (rw_save_solve[:, k] - rw_solve[:, k]) * (
        fzm[k] * theta_m_save_solve[:, k] + fzp[k] * theta_m_save_solve[:, k - 1]
    )
    if n_vert_levels <= 2:
        return first_flux, first_flux.new_zeros(first_flux.shape)

    k = n_vert_levels - 1
    last_flux = rw_save_solve[:, k] * (
        fzm[k] * theta_m_solve[:, k] + fzp[k] * theta_m_solve[:, k - 1]
    )
    return first_flux, last_flux


def _theta_vertical_advection_diabatic_theta_vertical_flux_proof_tensor(
    *,
    theta_m_solve: torch.Tensor,
    theta_m_save_solve: torch.Tensor,
    rw_solve: torch.Tensor,
    rw_save_solve: torch.Tensor,
    fzm: torch.Tensor,
    fzp: torch.Tensor,
    coef_3rd_order: float,
    n_vert_levels: int,
) -> torch.Tensor:
    """Reconstruct the theta-vertical-advection-diabatic Fortran ``wdtz`` scratch array for debug proof."""
    n_cells = theta_m_solve.shape[0]
    wdtz = theta_m_solve.new_zeros((n_cells, n_vert_levels + 1))
    if n_vert_levels <= 1:
        return wdtz
    first_flux, last_flux = (
        _theta_vertical_advection_diabatic_theta_vertical_boundary_fluxes_region(
            theta_m_solve=theta_m_solve,
            theta_m_save_solve=theta_m_save_solve,
            rw_solve=rw_solve,
            rw_save_solve=rw_save_solve,
            fzm=fzm,
            fzp=fzp,
            n_vert_levels=n_vert_levels,
        )
    )
    wdtz[:, 1] = first_flux
    if n_vert_levels > 3:
        wdtz_mid = flux3(
            theta_m_solve[:, 0 : n_vert_levels - 3],
            theta_m_solve[:, 1 : n_vert_levels - 2],
            theta_m_solve[:, 2 : n_vert_levels - 1],
            theta_m_solve[:, 3:n_vert_levels],
            rw_solve[:, 2 : n_vert_levels - 1],
            coef_3rd_order,
        )
        wdtz[:, 2 : n_vert_levels - 1] = wdtz_mid + (
            rw_save_solve[:, 2 : n_vert_levels - 1] - rw_solve[:, 2 : n_vert_levels - 1]
        ) * (
            fzm[2 : n_vert_levels - 1] * theta_m_save_solve[:, 2 : n_vert_levels - 1]
            + fzp[2 : n_vert_levels - 1] * theta_m_save_solve[:, 1 : n_vert_levels - 2]
        )
    if n_vert_levels > 2:
        wdtz[:, n_vert_levels - 1] = last_flux
    return wdtz


def _theta_vertical_advection_diabatic_theta_vertical_flux_abs_term_sum_proof_tensor(
    *,
    theta_m_solve: torch.Tensor,
    theta_m_save_solve: torch.Tensor,
    rw_solve: torch.Tensor,
    rw_save_solve: torch.Tensor,
    fzm: torch.Tensor,
    fzp: torch.Tensor,
    coef_3rd_order: float,
    n_vert_levels: int,
) -> torch.Tensor:
    """Return a source-derived absolute scale for the theta-vertical-advection-diabatic ``wdtz`` scratch."""
    n_cells = theta_m_solve.shape[0]
    out = theta_m_solve.new_zeros((n_cells, n_vert_levels + 1))
    if n_vert_levels <= 1:
        return out
    k = 1
    rw_diff = rw_save_solve[:, k] - rw_solve[:, k]
    out[:, k] = (
        torch.abs(rw_solve[:, k] * fzm[k] * theta_m_solve[:, k])
        + torch.abs(rw_solve[:, k] * fzp[k] * theta_m_solve[:, k - 1])
        + torch.abs(rw_diff * fzm[k] * theta_m_save_solve[:, k])
        + torch.abs(rw_diff * fzp[k] * theta_m_save_solve[:, k - 1])
    )
    if n_vert_levels > 3:
        flux_terms = _flux3_abs_term_sum(
            theta_m_solve[:, 0 : n_vert_levels - 3],
            theta_m_solve[:, 1 : n_vert_levels - 2],
            theta_m_solve[:, 2 : n_vert_levels - 1],
            theta_m_solve[:, 3:n_vert_levels],
            rw_solve[:, 2 : n_vert_levels - 1],
            coef_3rd_order,
        )
        rw_diff_mid = (
            rw_save_solve[:, 2 : n_vert_levels - 1] - rw_solve[:, 2 : n_vert_levels - 1]
        )
        correction_terms = torch.abs(
            rw_diff_mid
            * fzm[2 : n_vert_levels - 1]
            * theta_m_save_solve[:, 2 : n_vert_levels - 1]
        ) + torch.abs(
            rw_diff_mid
            * fzp[2 : n_vert_levels - 1]
            * theta_m_save_solve[:, 1 : n_vert_levels - 2]
        )
        out[:, 2 : n_vert_levels - 1] = flux_terms + correction_terms
    if n_vert_levels > 2:
        k = n_vert_levels - 1
        out[:, k] = torch.abs(
            rw_save_solve[:, k] * fzm[k] * theta_m_solve[:, k]
        ) + torch.abs(rw_save_solve[:, k] * fzp[k] * theta_m_solve[:, k - 1])
    return out


def _theta_vertical_advection_diabatic_theta_pointwise_update_region(
    *,
    theta_m_solve: torch.Tensor,
    rho_zz_solve: torch.Tensor,
    rt_diabatic_tend_solve: torch.Tensor,
    tend_theta_solve: torch.Tensor,
    tend_rho_solve: torch.Tensor,
    wdtz_delta: torch.Tensor,
    inv_area_cell_solve: torch.Tensor,
    rdzw: torch.Tensor,
    n_vert_levels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply dynamics-tendency theta-vertical-advection-diabatic pointwise theta update from a precomputed flux delta."""
    tend_theta_base = (
        tend_theta_solve * inv_area_cell_solve.unsqueeze(1) - rdzw[:] * wdtz_delta
    )
    rthdynten_patch = (tend_theta_base - tend_rho_solve * theta_m_solve) / rho_zz_solve
    tend_theta_dia_patch = tend_theta_base + rho_zz_solve * rt_diabatic_tend_solve
    return tend_theta_dia_patch, rthdynten_patch


def _theta_vertical_advection_diabatic_theta_vertical_section_region(
    *,
    theta_m_solve: torch.Tensor,
    theta_m_save_solve: torch.Tensor,
    rw_solve: torch.Tensor,
    rw_save_solve: torch.Tensor,
    rho_zz_solve: torch.Tensor,
    rt_diabatic_tend_solve: torch.Tensor,
    tend_theta_solve: torch.Tensor,
    tend_rho_solve: torch.Tensor,
    inv_area_cell_solve: torch.Tensor,
    rdzw: torch.Tensor,
    fzm: torch.Tensor,
    fzp: torch.Tensor,
    coef_3rd_order: float,
    n_vert_levels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute dynamics-tendency theta-vertical-advection-diabatic theta vertical tendency and diagnostic as one region."""
    wdtz_delta = _theta_vertical_advection_diabatic_theta_vertical_flux_delta_region(
        theta_m_solve=theta_m_solve,
        theta_m_save_solve=theta_m_save_solve,
        rw_solve=rw_solve,
        rw_save_solve=rw_save_solve,
        fzm=fzm,
        fzp=fzp,
        coef_3rd_order=coef_3rd_order,
        n_vert_levels=n_vert_levels,
    )
    return _theta_vertical_advection_diabatic_theta_pointwise_update_region(
        theta_m_solve=theta_m_solve,
        rho_zz_solve=rho_zz_solve,
        rt_diabatic_tend_solve=rt_diabatic_tend_solve,
        tend_theta_solve=tend_theta_solve,
        tend_rho_solve=tend_rho_solve,
        wdtz_delta=wdtz_delta,
        inv_area_cell_solve=inv_area_cell_solve,
        rdzw=rdzw,
        n_vert_levels=n_vert_levels,
    )


def _theta_vertical_mixing_theta_vertical_mix_update_region(
    *,
    theta_work_solve: torch.Tensor,
    rho_zz_solve: torch.Tensor,
    tend_theta_euler_solve_inner: torch.Tensor,
    zgrid_solve: torch.Tensor,
    v_theta_eddy_visc2: float,
    prandtl_inv: float,
    n_vert_levels: int,
) -> torch.Tensor:
    """Apply dynamics-tendency theta-vertical-mixing vertical theta-mixing on active solve-cell columns."""
    if n_vert_levels <= 2:
        return tend_theta_euler_solve_inner[:, 0:0]

    z1 = zgrid_solve[:, 0 : n_vert_levels - 2]
    z2 = zgrid_solve[:, 1 : n_vert_levels - 1]
    z3 = zgrid_solve[:, 2:n_vert_levels]
    z4 = zgrid_solve[:, 3 : n_vert_levels + 1]

    zm = 0.5 * (z1 + z2)
    z0 = 0.5 * (z2 + z3)
    zp = 0.5 * (z3 + z4)
    theta_diff = (
        theta_work_solve[:, 2:n_vert_levels]
        - theta_work_solve[:, 1 : n_vert_levels - 1]
    ) / (zp - z0) - (
        theta_work_solve[:, 1 : n_vert_levels - 1]
        - theta_work_solve[:, 0 : n_vert_levels - 2]
    ) / (z0 - zm)
    theta_mix = (
        v_theta_eddy_visc2
        * prandtl_inv
        * rho_zz_solve[:, 1 : n_vert_levels - 1]
        * theta_diff
        / (0.5 * (zp - zm))
    )
    return tend_theta_euler_solve_inner + theta_mix


def _theta_finalization_theta_final_section_region(
    *,
    tend_theta_solve: torch.Tensor,
    tend_theta_euler_solve: torch.Tensor,
    tend_rtheta_physics_solve: torch.Tensor | None = None,
) -> torch.Tensor:
    """Add dynamics-tendency theta Euler/physics tendencies in a pure numeric final region."""
    tend_theta_total = tend_theta_solve + tend_theta_euler_solve
    if tend_rtheta_physics_solve is not None:
        tend_theta_total = tend_theta_total + tend_rtheta_physics_solve
    return tend_theta_total


def _theta_vertical_mixing_theta_finalization_theta_final_section_region(
    *,
    tend_theta_solve: torch.Tensor,
    tend_theta_euler_cell_patch: torch.Tensor | None,
    tend_theta_euler_cell_base: torch.Tensor | None,
    tend_theta_euler_solve_base: torch.Tensor,
    theta_work_solve: torch.Tensor | None,
    rho_zz_solve: torch.Tensor | None,
    zgrid_solve: torch.Tensor | None,
    tend_rtheta_physics_solve: torch.Tensor | None,
    apply_vertical_mix: bool,
    cell_solve_rel_start: int,
    cell_solve_rel_end: int,
    v_theta_eddy_visc2: float,
    prandtl_inv: float,
    n_vert_levels: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply optional theta-vertical-mixing mixing and complete the theta-finalization theta tendency."""
    if apply_vertical_mix:
        if tend_theta_euler_cell_patch is None:
            tend_theta_euler_cell_patch = tend_theta_euler_cell_base
        if (
            tend_theta_euler_cell_patch is None
            or theta_work_solve is None
            or rho_zz_solve is None
            or zgrid_solve is None
        ):
            raise ValueError(
                "dynamics-tendency theta-vertical-mixing vertical mixing requires complete cell fields"
            )
        theta_mix_patch = _theta_vertical_mixing_theta_vertical_mix_update_region(
            theta_work_solve=theta_work_solve,
            rho_zz_solve=rho_zz_solve,
            tend_theta_euler_solve_inner=tend_theta_euler_cell_patch[
                cell_solve_rel_start:cell_solve_rel_end,
                1 : n_vert_levels - 1,
            ],
            zgrid_solve=zgrid_solve,
            v_theta_eddy_visc2=v_theta_eddy_visc2,
            prandtl_inv=prandtl_inv,
            n_vert_levels=n_vert_levels,
        )
        tend_theta_euler_cell_patch = _replace_active_rows_cols_numeric(
            tend_theta_euler_cell_patch,
            theta_mix_patch,
            row_start=cell_solve_rel_start,
            row_end=cell_solve_rel_end,
            col_start=1,
            col_end=n_vert_levels - 1,
        )
    tend_theta_euler_solve = (
        tend_theta_euler_cell_patch[
            cell_solve_rel_start:cell_solve_rel_end,
            :,
        ]
        if tend_theta_euler_cell_patch is not None
        else tend_theta_euler_solve_base
    )
    tend_theta_solve = _theta_finalization_theta_final_section_region(
        tend_theta_solve=tend_theta_solve,
        tend_theta_euler_solve=tend_theta_euler_solve,
        tend_rtheta_physics_solve=tend_rtheta_physics_solve,
    )
    return tend_theta_solve, tend_theta_euler_cell_patch


def _theta_horizontal_advection_theta_finalization_theta_tendency_region(
    *,
    theta_m: torch.Tensor,
    ru: torch.Tensor,
    edge_solve_start: int,
    edge_solve_end: int,
    n_vert_levels: int,
    edge_adv_cell_relation: StaticSlotRelation | tuple[Any, ...],
    local_edge_relation: StaticSlotRelation | tuple[Any, ...],
    include_perturbation_flux: bool,
    solve_cell_edge_relation: StaticSlotRelation | tuple[Any, ...],
    solve_edge_cell_relation: StaticPairRelation | tuple[Any, ...],
    ru_save: torch.Tensor,
    theta_m_save: torch.Tensor,
    cell_owned_vertical_velocity_horizontal_advection_theta_horizontal_advection_ab: bool,
    include_horizontal_mix: bool,
    mix_cell_edge_relation: StaticSlotRelation | tuple[Any, ...] | None,
    mix_edge_cell_relation: StaticPairRelation | tuple[Any, ...] | None,
    solve_cell_pair_relation: StaticPairRelation | tuple[Any, ...] | None,
    edge_sign_del2: torch.Tensor | None,
    edge_sign_del4: torch.Tensor | None,
    rho_edge: torch.Tensor,
    kdiff: torch.Tensor,
    delsq_theta_base: torch.Tensor,
    pr_scale: torch.Tensor | None,
    patch_row_start: int,
    patch_row_end: int,
    solve_relative_start: int,
    solve_relative_end: int,
    solve_flux_valid_all_by_slot: tuple[bool, ...] | None,
    theta_m_solve: torch.Tensor,
    theta_m_save_solve: torch.Tensor,
    rw_solve: torch.Tensor,
    rw_save_solve: torch.Tensor,
    rho_zz_solve: torch.Tensor,
    rt_diabatic_tend_solve: torch.Tensor,
    tend_rho_solve: torch.Tensor,
    inv_area_cell_solve: torch.Tensor,
    rdzw: torch.Tensor,
    fzm: torch.Tensor,
    fzp: torch.Tensor,
    coef_3rd_order: float,
    tend_theta_euler_cell_base: torch.Tensor | None,
    tend_theta_euler_solve_base: torch.Tensor,
    theta_work_solve: torch.Tensor | None,
    zgrid_solve: torch.Tensor | None,
    tend_rtheta_physics_solve: torch.Tensor | None,
    apply_vertical_mix: bool,
    v_theta_eddy_visc2: float,
    prandtl_inv: float,
    kdiff_patch: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Fuse the contiguous theta-horizontal-advection-theta-finalization theta chain around its three live outputs."""
    tend_theta_solve = _theta_horizontal_advection_theta_hadv_owned_section_region(
        theta_m=theta_m,
        ru=ru,
        edge_solve_start=edge_solve_start,
        edge_solve_end=edge_solve_end,
        n_vert_levels=n_vert_levels,
        edge_adv_cell_relation=edge_adv_cell_relation,
        local_edge_relation=local_edge_relation,
        include_perturbation_flux=include_perturbation_flux,
        cell_edge_relation=solve_cell_edge_relation,
        edge_cell_relation=solve_edge_cell_relation,
        ru_save=ru_save,
        theta_m_save=theta_m_save,
        cell_owned_vertical_velocity_horizontal_advection_theta_horizontal_advection_ab=cell_owned_vertical_velocity_horizontal_advection_theta_horizontal_advection_ab,
    )
    tend_theta_euler_cell_patch: torch.Tensor | None = None
    if include_horizontal_mix:
        if (
            mix_cell_edge_relation is None
            or mix_edge_cell_relation is None
            or solve_cell_pair_relation is None
            or edge_sign_del2 is None
            or pr_scale is None
        ):
            raise ValueError(
                "theta-horizontal-advection-theta-finalization horizontal mixing requires its static relations and weights"
            )
        tend_theta_euler_cell_patch = (
            _theta_horizontal_mixing_theta_horizontal_mix_section_region(
                cell_edge_relation=mix_cell_edge_relation,
                edge_cell_relation=mix_edge_cell_relation,
                solve_cell_pair_relation=solve_cell_pair_relation,
                edge_sign_del2=edge_sign_del2,
                edge_sign_del4=edge_sign_del4,
                rho_edge=rho_edge,
                theta_m=theta_m,
                kdiff=kdiff,
                delsq_theta_base=delsq_theta_base,
                pr_scale=pr_scale,
                patch_row_start=patch_row_start,
                patch_row_end=patch_row_end,
                solve_relative_start=solve_relative_start,
                solve_relative_end=solve_relative_end,
                solve_flux_valid_all_by_slot=solve_flux_valid_all_by_slot,
                kdiff_patch=kdiff_patch,
            )
        )
    tend_theta_solve, rthdynten_solve = (
        _theta_vertical_advection_diabatic_theta_vertical_section_region(
            theta_m_solve=theta_m_solve,
            theta_m_save_solve=theta_m_save_solve,
            rw_solve=rw_solve,
            rw_save_solve=rw_save_solve,
            rho_zz_solve=rho_zz_solve,
            rt_diabatic_tend_solve=rt_diabatic_tend_solve,
            tend_theta_solve=tend_theta_solve,
            tend_rho_solve=tend_rho_solve,
            inv_area_cell_solve=inv_area_cell_solve,
            rdzw=rdzw,
            fzm=fzm,
            fzp=fzp,
            coef_3rd_order=coef_3rd_order,
            n_vert_levels=n_vert_levels,
        )
    )
    tend_theta_solve, tend_theta_euler_cell_patch = (
        _theta_vertical_mixing_theta_finalization_theta_final_section_region(
            tend_theta_solve=tend_theta_solve,
            tend_theta_euler_cell_patch=tend_theta_euler_cell_patch,
            tend_theta_euler_cell_base=tend_theta_euler_cell_base,
            tend_theta_euler_solve_base=tend_theta_euler_solve_base,
            theta_work_solve=theta_work_solve,
            rho_zz_solve=rho_zz_solve if apply_vertical_mix else None,
            zgrid_solve=zgrid_solve,
            tend_rtheta_physics_solve=tend_rtheta_physics_solve,
            apply_vertical_mix=apply_vertical_mix,
            cell_solve_rel_start=solve_relative_start,
            cell_solve_rel_end=solve_relative_end,
            v_theta_eddy_visc2=v_theta_eddy_visc2,
            prandtl_inv=prandtl_inv,
            n_vert_levels=n_vert_levels,
        )
    )
    return tend_theta_solve, tend_theta_euler_cell_patch, rthdynten_solve


def _sum_w_mix_slots(
    *,
    cell_edge_relation: StaticSlotRelation | tuple[Any, ...],
    edge_cell_relation: StaticPairRelation | tuple[Any, ...],
    edge_sign: torch.Tensor,
    rho_edge: torch.Tensor,
    w: torch.Tensor,
    kdiff: torch.Tensor,
    n_vert_levels: int,
    kdiff_patch: torch.Tensor | None = None,
    patch_row_start: int = 0,
    patch_row_end: int = 0,
    mesh_scaling_del2: torch.Tensor | None = None,
    mesh_scaling_del2_slots: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate dynamics-tendency w horizontal-mixing del2 and tendency slots together."""
    safe_edges = _dynamics_tendency_slot_field(cell_edge_relation, 0, "indices")
    flux_valid = _dynamics_tendency_slot_field(cell_edge_relation, 1, "valid")
    flux_valid_all_by_slot = _dynamics_tendency_slot_field(
        cell_edge_relation, 6, "valid_all_by_slot"
    )
    cell1 = _dynamics_tendency_pair_field(edge_cell_relation, 0, "first")
    cell2 = _dynamics_tendency_pair_field(edge_cell_relation, 1, "second")
    if flux_valid is None:
        flux_valid = _dynamics_tendency_pair_field(edge_cell_relation, 2, "valid")
    if flux_valid is None:
        raise ValueError("dynamics-tendency w mix requires a valid flux mask")
    n_cells = safe_edges.shape[0]
    if (
        _dynamics_tendency_can_use_no_ad_full_slot_reduce(
            w,
            safe_edges,
            cell1,
            cell2,
            flux_valid,
            edge_sign,
            rho_edge,
            kdiff,
            mesh_scaling_del2,
            mesh_scaling_del2_slots,
        )
        and safe_edges.shape[1] > 1
    ):
        rho_batch = gather_rows(rho_edge, safe_edges)
        w_cell1_batch = gather_rows(w, cell1)
        w_cell2_batch = gather_rows(w, cell2)
        if kdiff_patch is not None:
            kdiff_cell1_batch, kdiff_cell2_batch = (
                _dynamics_tendency_pair_gather_patch_or_base(
                    edge_cell_relation,
                    kdiff,
                    kdiff_patch,
                    patch_row_start=patch_row_start,
                    patch_row_end=patch_row_end,
                )
            )
        else:
            kdiff_cell1_batch = gather_rows(kdiff, cell1)
            kdiff_cell2_batch = gather_rows(kdiff, cell2)
        if mesh_scaling_del2_slots is None:
            if mesh_scaling_del2 is None:
                raise ValueError(
                    "mesh_scaling_del2 or mesh_scaling_del2_slots is required"
                )
            mesh_scaling_batch = gather_rows(mesh_scaling_del2, safe_edges)
        else:
            mesh_scaling_batch = mesh_scaling_del2_slots
        base_flux = (
            edge_sign.unsqueeze(2)
            * (
                rho_batch[:, :, 1:n_vert_levels]
                + rho_batch[:, :, 0 : n_vert_levels - 1]
            )
            * (
                w_cell2_batch[:, :, 1:n_vert_levels]
                - w_cell1_batch[:, :, 1:n_vert_levels]
            )
        )
        valid_all = (
            flux_valid_all_by_slot is not None
            and len(flux_valid_all_by_slot) == safe_edges.shape[1]
            and all(flux_valid_all_by_slot)
        )
        if not valid_all:
            base_flux = base_flux * flux_valid.unsqueeze(2).to(dtype=w.dtype)
        scaled_flux = (
            base_flux
            * mesh_scaling_batch.unsqueeze(2)
            * 0.25
            * (
                kdiff_cell1_batch[:, :, 1:n_vert_levels]
                + kdiff_cell2_batch[:, :, 1:n_vert_levels]
                + kdiff_cell1_batch[:, :, 0 : n_vert_levels - 1]
                + kdiff_cell2_batch[:, :, 0 : n_vert_levels - 1]
            )
        )
        return _sum_fortran_order(base_flux, dim=1), _sum_fortran_order(
            scaled_flux, dim=1
        )
    delsq_total: torch.Tensor | None = None
    tend_total: torch.Tensor | None = None
    gather_batch = _dynamics_tendency_slot_gather_batch(w, safe_edges.shape[1])
    if gather_batch == 1:
        for slot in range(safe_edges.shape[1]):
            rho_slot = gather_rows(rho_edge, safe_edges[:, slot])
            w_cell1 = gather_rows(w, cell1[:, slot])
            w_cell2 = gather_rows(w, cell2[:, slot])
            if kdiff_patch is not None:
                kdiff_cell1_slot, kdiff_cell2_slot = (
                    _dynamics_tendency_pair_gather_slot_patch_or_base(
                        edge_cell_relation,
                        kdiff,
                        kdiff_patch,
                        patch_row_start=patch_row_start,
                        patch_row_end=patch_row_end,
                        slot=slot,
                    )
                )
            else:
                kdiff_cell1_slot = gather_rows(kdiff, cell1[:, slot])
                kdiff_cell2_slot = gather_rows(kdiff, cell2[:, slot])
            if mesh_scaling_del2_slots is None:
                if mesh_scaling_del2 is None:
                    raise ValueError(
                        "mesh_scaling_del2 or mesh_scaling_del2_slots is required"
                    )
                mesh_scaling_slot = gather_rows(mesh_scaling_del2, safe_edges[:, slot])
            else:
                mesh_scaling_slot = mesh_scaling_del2_slots[:, slot]
            base_flux = (
                edge_sign[:, slot].view(n_cells, 1)
                * (rho_slot[:, 1:n_vert_levels] + rho_slot[:, 0 : n_vert_levels - 1])
                * (w_cell2[:, 1:n_vert_levels] - w_cell1[:, 1:n_vert_levels])
            )
            valid_all = (
                flux_valid_all_by_slot is not None and flux_valid_all_by_slot[slot]
            )
            if not valid_all:
                base_flux = base_flux * flux_valid[:, slot].view(n_cells, 1).to(
                    dtype=w.dtype
                )
            scaled_flux = (
                base_flux
                * mesh_scaling_slot.view(n_cells, 1)
                * 0.25
                * (
                    kdiff_cell1_slot[:, 1:n_vert_levels]
                    + kdiff_cell2_slot[:, 1:n_vert_levels]
                    + kdiff_cell1_slot[:, 0 : n_vert_levels - 1]
                    + kdiff_cell2_slot[:, 0 : n_vert_levels - 1]
                )
            )
            delsq_total = _accumulate_ordered_slot(delsq_total, base_flux)
            tend_total = _accumulate_ordered_slot(tend_total, scaled_flux)
        if delsq_total is None or tend_total is None:
            empty = w.new_zeros((n_cells, max(n_vert_levels - 1, 0)))
            return empty, empty.clone()
        return delsq_total, tend_total
    for batch_start, batch_stop in _dynamics_tendency_slot_batches(
        safe_edges.shape[1], gather_batch
    ):
        rho_batch = _dynamics_tendency_gather_slot_batch(
            rho_edge, safe_edges, batch_start, batch_stop
        )
        w_cell1_batch = _dynamics_tendency_gather_slot_batch(
            w, cell1, batch_start, batch_stop
        )
        w_cell2_batch = _dynamics_tendency_gather_slot_batch(
            w, cell2, batch_start, batch_stop
        )
        if kdiff_patch is not None:
            kdiff_cell1_batch, kdiff_cell2_batch = (
                _dynamics_tendency_pair_gather_patch_or_base(
                    edge_cell_relation,
                    kdiff,
                    kdiff_patch,
                    patch_row_start=patch_row_start,
                    patch_row_end=patch_row_end,
                    slot_start=batch_start,
                    slot_stop=batch_stop,
                )
            )
        else:
            kdiff_cell1_batch = _dynamics_tendency_gather_slot_batch(
                kdiff,
                cell1,
                batch_start,
                batch_stop,
            )
            kdiff_cell2_batch = _dynamics_tendency_gather_slot_batch(
                kdiff,
                cell2,
                batch_start,
                batch_stop,
            )
        if mesh_scaling_del2_slots is None:
            if mesh_scaling_del2 is None:
                raise ValueError(
                    "mesh_scaling_del2 or mesh_scaling_del2_slots is required"
                )
            mesh_scaling_batch = _dynamics_tendency_gather_slot_batch(
                mesh_scaling_del2,
                safe_edges,
                batch_start,
                batch_stop,
            )
        else:
            mesh_scaling_batch = mesh_scaling_del2_slots[:, batch_start:batch_stop]
        for local_slot in range(batch_stop - batch_start):
            slot = batch_start + local_slot
            rho_slot = rho_batch[:, local_slot, :]
            w_cell1 = w_cell1_batch[:, local_slot, :]
            w_cell2 = w_cell2_batch[:, local_slot, :]
            kdiff_cell1_slot = kdiff_cell1_batch[:, local_slot, :]
            kdiff_cell2_slot = kdiff_cell2_batch[:, local_slot, :]
            base_flux = (
                edge_sign[:, slot].view(n_cells, 1)
                * (rho_slot[:, 1:n_vert_levels] + rho_slot[:, 0 : n_vert_levels - 1])
                * (w_cell2[:, 1:n_vert_levels] - w_cell1[:, 1:n_vert_levels])
            )
            valid_all = (
                flux_valid_all_by_slot is not None and flux_valid_all_by_slot[slot]
            )
            if not valid_all:
                base_flux = base_flux * flux_valid[:, slot].view(n_cells, 1).to(
                    dtype=w.dtype
                )
            scaled_flux = (
                base_flux
                * mesh_scaling_batch[:, local_slot].view(n_cells, 1)
                * 0.25
                * (
                    kdiff_cell1_slot[:, 1:n_vert_levels]
                    + kdiff_cell2_slot[:, 1:n_vert_levels]
                    + kdiff_cell1_slot[:, 0 : n_vert_levels - 1]
                    + kdiff_cell2_slot[:, 0 : n_vert_levels - 1]
                )
            )
            delsq_total = _accumulate_ordered_slot(delsq_total, base_flux)
            tend_total = _accumulate_ordered_slot(tend_total, scaled_flux)
    shape = (n_cells, max(n_vert_levels - 1, 0))
    if delsq_total is None:
        delsq_total = w.new_zeros(shape)
    if tend_total is None:
        tend_total = w.new_zeros(shape)
    return delsq_total, tend_total


def _sum_theta_mix_slots(
    *,
    cell_edge_relation: StaticSlotRelation | tuple[Any, ...],
    edge_cell_relation: StaticPairRelation | tuple[Any, ...],
    edge_sign: torch.Tensor,
    rho_edge: torch.Tensor,
    theta_m: torch.Tensor,
    kdiff: torch.Tensor,
    pr_scale: torch.Tensor,
    kdiff_patch: torch.Tensor | None = None,
    patch_row_start: int = 0,
    patch_row_end: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate dynamics-tendency theta horizontal-mixing del2 and tendency slots together."""
    safe_edges = _dynamics_tendency_slot_field(cell_edge_relation, 0, "indices")
    flux_valid = _dynamics_tendency_slot_field(cell_edge_relation, 1, "valid")
    flux_valid_all_by_slot = _dynamics_tendency_slot_field(
        cell_edge_relation, 6, "valid_all_by_slot"
    )
    cell1 = _dynamics_tendency_pair_field(edge_cell_relation, 0, "first")
    cell2 = _dynamics_tendency_pair_field(edge_cell_relation, 1, "second")
    if flux_valid is None:
        flux_valid = _dynamics_tendency_pair_field(edge_cell_relation, 2, "valid")
    if flux_valid is None:
        raise ValueError("dynamics-tendency theta mix requires a valid flux mask")
    n_cells = safe_edges.shape[0]
    if (
        _dynamics_tendency_can_use_no_ad_full_slot_reduce(
            theta_m,
            safe_edges,
            cell1,
            cell2,
            flux_valid,
            edge_sign,
            rho_edge,
            kdiff,
            pr_scale,
        )
        and safe_edges.shape[1] > 1
    ):
        theta_cell1_batch = gather_rows(theta_m, cell1)
        theta_cell2_batch = gather_rows(theta_m, cell2)
        rho_batch = gather_rows(rho_edge, safe_edges)
        if kdiff_patch is not None:
            kdiff_cell1_batch, kdiff_cell2_batch = (
                _dynamics_tendency_pair_gather_patch_or_base(
                    edge_cell_relation,
                    kdiff,
                    kdiff_patch,
                    patch_row_start=patch_row_start,
                    patch_row_end=patch_row_end,
                )
            )
        else:
            kdiff_cell1_batch = gather_rows(kdiff, cell1)
            kdiff_cell2_batch = gather_rows(kdiff, cell2)
        base_flux = (
            edge_sign.unsqueeze(2) * (theta_cell2_batch - theta_cell1_batch) * rho_batch
        )
        valid_all = (
            flux_valid_all_by_slot is not None
            and len(flux_valid_all_by_slot) == safe_edges.shape[1]
            and all(flux_valid_all_by_slot)
        )
        if not valid_all:
            base_flux = base_flux * flux_valid.unsqueeze(2).to(dtype=theta_m.dtype)
        scaled_flux = (
            base_flux
            * 0.5
            * (kdiff_cell1_batch + kdiff_cell2_batch)
            * pr_scale.unsqueeze(2)
        )
        return _sum_fortran_order(base_flux, dim=1), _sum_fortran_order(
            scaled_flux, dim=1
        )
    delsq_total: torch.Tensor | None = None
    tend_total: torch.Tensor | None = None
    gather_batch = _dynamics_tendency_slot_gather_batch(theta_m, safe_edges.shape[1])
    if gather_batch == 1:
        for slot in range(safe_edges.shape[1]):
            theta_cell1 = gather_rows(theta_m, cell1[:, slot])
            theta_cell2 = gather_rows(theta_m, cell2[:, slot])
            rho_slot = gather_rows(rho_edge, safe_edges[:, slot])
            if kdiff_patch is not None:
                kdiff_cell1_slot, kdiff_cell2_slot = (
                    _dynamics_tendency_pair_gather_slot_patch_or_base(
                        edge_cell_relation,
                        kdiff,
                        kdiff_patch,
                        patch_row_start=patch_row_start,
                        patch_row_end=patch_row_end,
                        slot=slot,
                    )
                )
            else:
                kdiff_cell1_slot = gather_rows(kdiff, cell1[:, slot])
                kdiff_cell2_slot = gather_rows(kdiff, cell2[:, slot])
            base_flux = (
                edge_sign[:, slot].view(n_cells, 1)
                * (theta_cell2 - theta_cell1)
                * rho_slot
            )
            valid_all = (
                flux_valid_all_by_slot is not None and flux_valid_all_by_slot[slot]
            )
            if not valid_all:
                base_flux = base_flux * flux_valid[:, slot].view(n_cells, 1).to(
                    dtype=theta_m.dtype
                )
            scaled_flux = (
                base_flux
                * 0.5
                * (kdiff_cell1_slot + kdiff_cell2_slot)
                * pr_scale[:, slot].view(n_cells, 1)
            )
            delsq_total = _accumulate_ordered_slot(delsq_total, base_flux)
            tend_total = _accumulate_ordered_slot(tend_total, scaled_flux)
        if delsq_total is None or tend_total is None:
            empty = theta_m.new_zeros((n_cells, theta_m.shape[1]))
            return empty, empty.clone()
        return delsq_total, tend_total
    for batch_start, batch_stop in _dynamics_tendency_slot_batches(
        safe_edges.shape[1], gather_batch
    ):
        theta_cell1_batch = _dynamics_tendency_gather_slot_batch(
            theta_m, cell1, batch_start, batch_stop
        )
        theta_cell2_batch = _dynamics_tendency_gather_slot_batch(
            theta_m, cell2, batch_start, batch_stop
        )
        rho_batch = _dynamics_tendency_gather_slot_batch(
            rho_edge, safe_edges, batch_start, batch_stop
        )
        if kdiff_patch is not None:
            kdiff_cell1_batch, kdiff_cell2_batch = (
                _dynamics_tendency_pair_gather_patch_or_base(
                    edge_cell_relation,
                    kdiff,
                    kdiff_patch,
                    patch_row_start=patch_row_start,
                    patch_row_end=patch_row_end,
                    slot_start=batch_start,
                    slot_stop=batch_stop,
                )
            )
        else:
            kdiff_cell1_batch = _dynamics_tendency_gather_slot_batch(
                kdiff,
                cell1,
                batch_start,
                batch_stop,
            )
            kdiff_cell2_batch = _dynamics_tendency_gather_slot_batch(
                kdiff,
                cell2,
                batch_start,
                batch_stop,
            )
        for local_slot in range(batch_stop - batch_start):
            slot = batch_start + local_slot
            theta_cell1 = theta_cell1_batch[:, local_slot, :]
            theta_cell2 = theta_cell2_batch[:, local_slot, :]
            rho_slot = rho_batch[:, local_slot, :]
            kdiff_cell1_slot = kdiff_cell1_batch[:, local_slot, :]
            kdiff_cell2_slot = kdiff_cell2_batch[:, local_slot, :]
            base_flux = (
                edge_sign[:, slot].view(n_cells, 1)
                * (theta_cell2 - theta_cell1)
                * rho_slot
            )
            valid_all = (
                flux_valid_all_by_slot is not None and flux_valid_all_by_slot[slot]
            )
            if not valid_all:
                base_flux = base_flux * flux_valid[:, slot].view(n_cells, 1).to(
                    dtype=theta_m.dtype
                )
            scaled_flux = (
                base_flux
                * 0.5
                * (kdiff_cell1_slot + kdiff_cell2_slot)
                * pr_scale[:, slot].view(n_cells, 1)
            )
            delsq_total = _accumulate_ordered_slot(delsq_total, base_flux)
            tend_total = _accumulate_ordered_slot(tend_total, scaled_flux)
    shape = (n_cells, theta_m.shape[1])
    if delsq_total is None:
        delsq_total = theta_m.new_zeros(shape)
    if tend_total is None:
        tend_total = theta_m.new_zeros(shape)
    return delsq_total, tend_total


def _replace_contiguous_rows_numeric(
    base: torch.Tensor,
    row_start: int,
    row_end: int,
    values: torch.Tensor,
) -> torch.Tensor:
    """Replace a static contiguous row range using tensor-only composition."""
    row_start = int(row_start)
    row_end = int(row_end)
    if row_start < 0 or row_end < row_start or row_end > base.shape[0]:
        raise ValueError(
            f"invalid numeric row slice [{row_start}:{row_end}] "
            f"for dim-0 size {base.shape[0]}"
        )
    if row_end - row_start != values.shape[0]:
        raise ValueError(
            f"numeric row slice length {row_end - row_start} "
            f"does not match values rows {values.shape[0]}"
        )
    if row_start == 0 and row_end == base.shape[0]:
        return values
    return torch.cat((base[:row_start, ...], values, base[row_end:, ...]), dim=0)


def _vertical_velocity_horizontal_mixing_w_horizontal_mix_section_region(
    *,
    cell_edge_relation: StaticSlotRelation | tuple[Any, ...],
    edge_cell_relation: StaticPairRelation | tuple[Any, ...],
    solve_cell_pair_relation: StaticPairRelation | tuple[Any, ...],
    edge_sign_del2: torch.Tensor,
    edge_sign_del4: torch.Tensor | None,
    rho_edge: torch.Tensor,
    w: torch.Tensor,
    kdiff: torch.Tensor,
    delsq_w_base: torch.Tensor,
    n_vert_levels: int,
    patch_row_start: int,
    patch_row_end: int,
    solve_relative_start: int,
    solve_relative_end: int,
    solve_flux_valid_all_by_slot: tuple[bool, ...] | None,
    kdiff_patch: torch.Tensor | None = None,
    mesh_scaling_del2: torch.Tensor | None = None,
    mesh_scaling_del2_slots: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the complete active-cell dynamics-tendency vertical-velocity-horizontal-mixing horizontal w-mixing section."""
    delsq_w_patch, w_turb_tend_sum = _sum_w_mix_slots(
        cell_edge_relation=cell_edge_relation,
        edge_cell_relation=edge_cell_relation,
        edge_sign=edge_sign_del2,
        rho_edge=rho_edge,
        w=w,
        kdiff=kdiff,
        kdiff_patch=kdiff_patch,
        patch_row_start=patch_row_start,
        patch_row_end=patch_row_end,
        mesh_scaling_del2=mesh_scaling_del2,
        mesh_scaling_del2_slots=mesh_scaling_del2_slots,
        n_vert_levels=n_vert_levels,
    )
    # _sum_w_mix_slots returns an owned tensor and the optional row replacement
    # below is functional. Avoid a full active-cell zero allocation and copy.
    tend_w_inner = w_turb_tend_sum
    if edge_sign_del4 is not None:
        tend_w_del4 = _subtract_weighted_pair_delta_slots_fortran_order(
            initial=tend_w_inner[solve_relative_start:solve_relative_end, :],
            field=delsq_w_base,
            cell_pair_relation=solve_cell_pair_relation,
            flux_valid_all_by_slot=solve_flux_valid_all_by_slot,
            edge_sign=edge_sign_del4,
            field_patch=delsq_w_patch,
            patch_row_start=patch_row_start,
            patch_row_end=patch_row_end,
            col_start=1,
            col_stop=delsq_w_base.shape[1] - 1,
            field_patch_col_start=1,
        )
        tend_w_inner = _replace_contiguous_rows_numeric(
            tend_w_inner,
            solve_relative_start,
            solve_relative_end,
            tend_w_del4,
        )
    boundary = tend_w_inner.new_zeros((tend_w_inner.shape[0], 1))
    return torch.cat((boundary, tend_w_inner, boundary), dim=1)


def _theta_horizontal_mixing_theta_horizontal_mix_section_region(
    *,
    cell_edge_relation: StaticSlotRelation | tuple[Any, ...],
    edge_cell_relation: StaticPairRelation | tuple[Any, ...],
    solve_cell_pair_relation: StaticPairRelation | tuple[Any, ...],
    edge_sign_del2: torch.Tensor,
    edge_sign_del4: torch.Tensor | None,
    rho_edge: torch.Tensor,
    theta_m: torch.Tensor,
    kdiff: torch.Tensor,
    delsq_theta_base: torch.Tensor,
    pr_scale: torch.Tensor,
    patch_row_start: int,
    patch_row_end: int,
    solve_relative_start: int,
    solve_relative_end: int,
    solve_flux_valid_all_by_slot: tuple[bool, ...] | None,
    kdiff_patch: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the complete active-cell dynamics-tendency theta-horizontal-mixing horizontal theta-mixing section."""
    delsq_theta_patch, theta_turb_tend_sum = _sum_theta_mix_slots(
        cell_edge_relation=cell_edge_relation,
        edge_cell_relation=edge_cell_relation,
        edge_sign=edge_sign_del2,
        rho_edge=rho_edge,
        theta_m=theta_m,
        kdiff=kdiff,
        kdiff_patch=kdiff_patch,
        patch_row_start=patch_row_start,
        patch_row_end=patch_row_end,
        pr_scale=pr_scale,
    )
    # The reducer result is already owned; the optional replacement never mutates
    # it. Retain that storage instead of allocating and adding a full zero field.
    tend_theta_euler_patch = theta_turb_tend_sum
    if edge_sign_del4 is not None:
        tend_theta_del4 = _subtract_weighted_pair_delta_slots_fortran_order(
            initial=tend_theta_euler_patch[solve_relative_start:solve_relative_end, :],
            field=delsq_theta_base,
            cell_pair_relation=solve_cell_pair_relation,
            flux_valid_all_by_slot=solve_flux_valid_all_by_slot,
            edge_sign=edge_sign_del4,
            field_patch=delsq_theta_patch,
            patch_row_start=patch_row_start,
            patch_row_end=patch_row_end,
            col_start=0,
            col_stop=delsq_theta_base.shape[1],
            field_patch_col_start=0,
        )
        tend_theta_euler_patch = _replace_contiguous_rows_numeric(
            tend_theta_euler_patch,
            solve_relative_start,
            solve_relative_end,
            tend_theta_del4,
        )
    return tend_theta_euler_patch


def _sum_theta_perturbation_flux_slots(
    *,
    cell_edge_relation: StaticSlotRelation | tuple[Any, ...],
    edge_cell_relation: StaticPairRelation | tuple[Any, ...],
    ru: torch.Tensor,
    ru_save: torch.Tensor,
    theta_m_save: torch.Tensor,
    initial: torch.Tensor | None = None,
) -> torch.Tensor:
    """Accumulate or apply dynamics-tendency theta perturbation flux in Fortran slot order.

    When ``initial`` is supplied, each edge contribution is subtracted from the
    running horizontal-advection tendency immediately. This preserves the
    statement association of the RK>1 Fortran loop; subtracting a separately
    reduced edge total introduces a one-rounding drift at the RK2 boundary.
    """
    safe_edges = _dynamics_tendency_slot_field(cell_edge_relation, 0, "indices")
    edge_sign = _dynamics_tendency_slot_field(cell_edge_relation, 2, "sign")
    signed_dv_edge = _dynamics_tendency_slot_field(cell_edge_relation, 3, "weights")
    flux_valid_all_by_slot = _dynamics_tendency_slot_field(
        cell_edge_relation, 6, "valid_all_by_slot"
    )
    cell1 = _dynamics_tendency_pair_field(edge_cell_relation, 0, "first")
    cell2 = _dynamics_tendency_pair_field(edge_cell_relation, 1, "second")
    flux_valid = _dynamics_tendency_pair_field(edge_cell_relation, 2, "valid")
    if edge_sign is None or signed_dv_edge is None:
        raise ValueError(
            "dynamics-tendency theta perturbation flux requires signed metric weights"
        )
    if flux_valid is None:
        flux_valid = torch.ones_like(safe_edges, dtype=torch.bool)
    n_cells = safe_edges.shape[0]
    if (
        initial is None
        and _dynamics_tendency_can_use_no_ad_full_slot_reduce(
            theta_m_save,
            safe_edges,
            cell1,
            cell2,
            flux_valid,
            edge_sign,
            ru,
            ru_save,
            signed_dv_edge,
        )
        and safe_edges.shape[1] > 1
    ):
        theta_cell1_batch = gather_rows(theta_m_save, cell1)
        theta_cell2_batch = gather_rows(theta_m_save, cell2)
        ru_save_batch = gather_rows(ru_save, safe_edges)
        ru_batch = gather_rows(ru, safe_edges)
        metric_batch = signed_dv_edge
        theta_face = 0.5 * (theta_cell2_batch + theta_cell1_batch)
        contrib = metric_batch.unsqueeze(2) * (ru_save_batch - ru_batch) * theta_face
        valid_all = (
            flux_valid_all_by_slot is not None
            and len(flux_valid_all_by_slot) == safe_edges.shape[1]
            and all(flux_valid_all_by_slot)
        )
        if not valid_all:
            contrib = contrib * flux_valid.unsqueeze(2).to(dtype=theta_m_save.dtype)
        return _sum_fortran_order(contrib, dim=1)
    total: torch.Tensor | None = initial
    for batch_start, batch_stop in _dynamics_tendency_slot_batches(
        safe_edges.shape[1],
        _dynamics_tendency_slot_gather_batch(theta_m_save, safe_edges.shape[1]),
    ):
        theta_cell1_batch = _dynamics_tendency_gather_slot_batch(
            theta_m_save, cell1, batch_start, batch_stop
        )
        theta_cell2_batch = _dynamics_tendency_gather_slot_batch(
            theta_m_save, cell2, batch_start, batch_stop
        )
        ru_save_batch = _dynamics_tendency_gather_slot_batch(
            ru_save, safe_edges, batch_start, batch_stop
        )
        ru_batch = _dynamics_tendency_gather_slot_batch(
            ru, safe_edges, batch_start, batch_stop
        )
        metric_batch = signed_dv_edge[:, batch_start:batch_stop]
        for local_slot in range(batch_stop - batch_start):
            slot = batch_start + local_slot
            theta_face = 0.5 * (
                theta_cell2_batch[:, local_slot, :]
                + theta_cell1_batch[:, local_slot, :]
            )
            contrib = (
                metric_batch[:, local_slot].view(n_cells, 1)
                * (ru_save_batch[:, local_slot, :] - ru_batch[:, local_slot, :])
                * theta_face
            )
            valid_all = (
                flux_valid_all_by_slot is not None and flux_valid_all_by_slot[slot]
            )
            if not valid_all:
                contrib = contrib * flux_valid[:, slot].view(n_cells, 1).to(
                    dtype=theta_m_save.dtype
                )
            if initial is None:
                total = _accumulate_ordered_slot(total, contrib)
            elif valid_all:
                total = total - contrib
            else:
                assert total is not None  # initial supplied a running tendency.
                total = torch.where(
                    flux_valid[:, slot].view(n_cells, 1),
                    total - contrib,
                    total,
                )
    if total is None:
        return theta_m_save.new_zeros((n_cells, theta_m_save.shape[1]))
    return total


def _theta_perturbation_flux_proof_slots(
    *,
    cell_edge_relation: StaticSlotRelation,
    edge_cell_relation: StaticPairRelation,
    ru: torch.Tensor,
    ru_save: torch.Tensor,
    theta_m_save: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return signed and absolute dynamics-tendency theta perturbation-flux proof terms."""
    safe_edges = _dynamics_tendency_slot_field(cell_edge_relation, 0, "indices")
    signed_dv_edge = _dynamics_tendency_slot_field(cell_edge_relation, 3, "weights")
    flux_valid_all_by_slot = _dynamics_tendency_slot_field(
        cell_edge_relation, 6, "valid_all_by_slot"
    )
    cell1 = _dynamics_tendency_pair_field(edge_cell_relation, 0, "first")
    cell2 = _dynamics_tendency_pair_field(edge_cell_relation, 1, "second")
    flux_valid = _dynamics_tendency_pair_field(edge_cell_relation, 2, "valid")
    if signed_dv_edge is None:
        raise ValueError(
            "dynamics-tendency theta perturbation abs proof requires signed metric weights"
        )
    if flux_valid is None:
        flux_valid = torch.ones_like(safe_edges, dtype=torch.bool)
    n_cells = safe_edges.shape[0]
    n_slots = safe_edges.shape[1]
    n_levels = theta_m_save.shape[1]
    terms = theta_m_save.new_zeros((n_cells, n_slots, n_levels))
    abs_total: torch.Tensor | None = None
    for batch_start, batch_stop in _dynamics_tendency_slot_batches(
        n_slots,
        _dynamics_tendency_slot_gather_batch(theta_m_save, n_slots),
    ):
        theta_cell1_batch = _dynamics_tendency_gather_slot_batch(
            theta_m_save, cell1, batch_start, batch_stop
        )
        theta_cell2_batch = _dynamics_tendency_gather_slot_batch(
            theta_m_save, cell2, batch_start, batch_stop
        )
        ru_save_batch = _dynamics_tendency_gather_slot_batch(
            ru_save, safe_edges, batch_start, batch_stop
        )
        ru_batch = _dynamics_tendency_gather_slot_batch(
            ru, safe_edges, batch_start, batch_stop
        )
        metric_batch = signed_dv_edge[:, batch_start:batch_stop]
        for local_slot in range(batch_stop - batch_start):
            slot = batch_start + local_slot
            theta_face = 0.5 * (
                theta_cell2_batch[:, local_slot, :]
                + theta_cell1_batch[:, local_slot, :]
            )
            contrib = -(
                metric_batch[:, local_slot].view(n_cells, 1)
                * (ru_save_batch[:, local_slot, :] - ru_batch[:, local_slot, :])
                * theta_face
            )
            valid_all = (
                flux_valid_all_by_slot is not None and flux_valid_all_by_slot[slot]
            )
            if not valid_all:
                contrib = contrib * flux_valid[:, slot].view(n_cells, 1).to(
                    dtype=theta_m_save.dtype
                )
            terms[:, slot, :] = contrib
            abs_total = _accumulate_ordered_slot(abs_total, torch.abs(contrib))
    if abs_total is None:
        abs_total = theta_m_save.new_zeros((n_cells, n_levels))
    return terms, abs_total


def _dynamics_tendency_vertical_velocity_horizontal_advection_hadv_proof_taps(
    *,
    w: torch.Tensor,
    ru: torch.Tensor,
    fzm: torch.Tensor,
    fzp: torch.Tensor,
    solve_edge_slice: slice,
    n_vert_levels: int,
    edge_adv_cell_relation: StaticSlotRelation,
    local_edge_relation: StaticSlotRelation,
) -> dict[str, Any]:
    """Reconstruct vertical-velocity-horizontal-advection diagnostic proof terms independently of the numeric helper."""
    ru_edge = (
        fzm[1:n_vert_levels].view(1, -1) * ru[solve_edge_slice, 1:n_vert_levels]
        + fzp[1:n_vert_levels].view(1, -1) * ru[solve_edge_slice, 0 : n_vert_levels - 1]
    )
    ru_edge_abs_term_sum = torch.abs(fzm[1:n_vert_levels]).view(1, -1) * torch.abs(
        ru[solve_edge_slice, 1:n_vert_levels]
    ) + torch.abs(fzp[1:n_vert_levels]).view(1, -1) * torch.abs(
        ru[solve_edge_slice, 0 : n_vert_levels - 1]
    )
    (
        ru_edge_w,
        ru_edge_w_abs_term_sum,
        flux_arr,
        flux_abs_term_sum,
        tend_w_abs_term_sum,
        tend_w_terms,
        tend_w_terms_abs_term_sum,
    ) = _edge_advective_cell_tendency_proof_tensors(
        field=w,
        field_start=1,
        field_stop=n_vert_levels,
        ru_edge=ru_edge,
        ru_edge_abs_term_sum=ru_edge_abs_term_sum,
        edge_adv_cell_relation=edge_adv_cell_relation,
        local_edge_relation=local_edge_relation,
        sign_ru=fct_sign_pm(ru_edge),
        full_level_count=n_vert_levels + 1,
    )
    return {
        "vertical_velocity_horizontal_advection_ru_edge_w": ru_edge_w,
        "vertical_velocity_horizontal_advection_ru_edge_w_abs_term_sum": ru_edge_w_abs_term_sum,
        "vertical_velocity_horizontal_advection_flux_arr": flux_arr,
        "vertical_velocity_horizontal_advection_flux_abs_term_sum": flux_abs_term_sum,
        "vertical_velocity_horizontal_advection_tend_w_abs_term_sum": tend_w_abs_term_sum,
        "vertical_velocity_horizontal_advection_tend_w_terms": tend_w_terms,
        "vertical_velocity_horizontal_advection_tend_w_terms_abs_term_sum": tend_w_terms_abs_term_sum,
    }


def _dynamics_tendency_theta_horizontal_advection_hadv_proof_taps(
    *,
    theta_m: torch.Tensor,
    theta_m_save: torch.Tensor,
    ru: torch.Tensor,
    ru_save: torch.Tensor,
    solve_edge_slice: slice,
    n_vert_levels: int,
    rk_step: int,
    edge_adv_cell_relation: StaticSlotRelation,
    local_edge_relation: StaticSlotRelation,
    cell_edge_relation: StaticSlotRelation,
    edge_cell_relation: StaticPairRelation,
) -> dict[str, Any]:
    """Reconstruct theta-horizontal-advection diagnostic proof terms independently of the numeric helper."""
    ru_edge = ru[solve_edge_slice, :n_vert_levels]
    (
        _ru_edge_unused,
        _ru_edge_abs_unused,
        flux_arr,
        flux_abs_term_sum,
        tend_theta_abs_term_sum,
        tend_theta_terms,
        _tend_theta_terms_abs_unused,
    ) = _edge_advective_cell_tendency_proof_tensors(
        field=theta_m,
        field_start=0,
        field_stop=n_vert_levels,
        ru_edge=ru_edge,
        edge_adv_cell_relation=edge_adv_cell_relation,
        local_edge_relation=local_edge_relation,
        sign_ru=fct_sign_pm(ru_edge),
        full_level_count=n_vert_levels,
    )
    if rk_step > 1:
        perturbation_terms, perturbation_abs = _theta_perturbation_flux_proof_slots(
            cell_edge_relation=cell_edge_relation,
            edge_cell_relation=edge_cell_relation,
            ru=ru,
            ru_save=ru_save,
            theta_m_save=theta_m_save,
        )
    else:
        local_edges = _dynamics_tendency_slot_field(local_edge_relation, 0, "indices")
        if local_edges is None:
            raise ValueError(
                "dynamics-tendency theta-horizontal-advection proof requires a local edge relation"
            )
        perturbation_terms = theta_m.new_zeros(
            (local_edges.shape[0], local_edges.shape[1], n_vert_levels)
        )
        perturbation_abs = theta_m.new_zeros((local_edges.shape[0], n_vert_levels))
    return {
        "theta_horizontal_advection_flux_arr": flux_arr,
        "theta_horizontal_advection_flux_abs_term_sum": flux_abs_term_sum,
        "theta_horizontal_advection_tend_theta_abs_term_sum": tend_theta_abs_term_sum,
        "theta_horizontal_advection_perturbation_abs_term_sum": perturbation_abs,
        "theta_horizontal_advection_tend_theta_terms": tend_theta_terms,
        "theta_horizontal_advection_perturbation_terms": perturbation_terms,
    }


def _sum_weighted_index_slots(
    *,
    field: torch.Tensor,
    indices: torch.Tensor | None = None,
    weights: torch.Tensor,
    index_relation: StaticSlotRelation | None = None,
    field_patch: torch.Tensor | None = None,
    patch_row_start: int = 0,
    patch_row_end: int = 0,
) -> torch.Tensor:
    """Accumulate weighted index stencils without full row-slot-level tensors."""
    if index_relation is not None:
        indices = _dynamics_tendency_slot_field(index_relation, 0, "indices")
    if indices is None:
        raise ValueError(
            "dynamics-tendency weighted index slots require indices or index_relation"
        )
    n_rows = indices.shape[0]
    if (
        _dynamics_tendency_can_use_no_ad_full_slot_reduce(
            field, indices, weights, field_patch
        )
        and indices.shape[1] > 1
    ):
        if index_relation is None:
            values = _dynamics_tendency_gather_rows_patch_or_base(
                field,
                indices,
                field_patch,
                patch_row_start,
                patch_row_end,
            )
        else:
            values = _dynamics_tendency_slot_gather_patch_or_base(
                index_relation,
                field,
                field_patch,
                patch_row_start=patch_row_start,
                patch_row_end=patch_row_end,
            )
        return _sum_fortran_order(weights.unsqueeze(2) * values, dim=1)
    total: torch.Tensor | None = None
    for batch_start, batch_stop in _dynamics_tendency_slot_batches(
        indices.shape[1], _dynamics_tendency_slot_gather_batch(field, indices.shape[1])
    ):
        if index_relation is not None:
            values_batch = _dynamics_tendency_slot_gather_patch_or_base(
                index_relation,
                field,
                field_patch,
                patch_row_start=patch_row_start,
                patch_row_end=patch_row_end,
                slot_start=batch_start,
                slot_stop=batch_stop,
            )
        elif field_patch is None:
            values_batch = _dynamics_tendency_gather_slot_batch(
                field, indices, batch_start, batch_stop
            )
        else:
            values_batch = _dynamics_tendency_gather_rows_patch_or_base(
                field,
                indices[:, batch_start:batch_stop],
                field_patch,
                patch_row_start,
                patch_row_end,
            )
        for local_slot in range(batch_stop - batch_start):
            slot = batch_start + local_slot
            contrib = weights[:, slot].view(n_rows, 1) * values_batch[:, local_slot, :]
            total = _accumulate_ordered_slot(total, contrib)
    if total is None:
        return field.new_zeros((n_rows, field.shape[1]))
    return total


def _sum_weighted_cell_delta_slots(
    *,
    field: torch.Tensor,
    cell1: torch.Tensor,
    cell2: torch.Tensor,
    flux_valid: torch.Tensor,
    flux_valid_all_by_slot: tuple[bool, ...] | None = None,
    edge_sign: torch.Tensor,
    col_start: int,
    col_stop: int,
    field_patch: torch.Tensor | None = None,
    patch_row_start: int = 0,
    patch_row_end: int = 0,
) -> torch.Tensor:
    """Compatibility wrapper for raw cell-pair delta reductions."""
    relation = StaticPairRelation(
        "dynamics_tendency.weighted_cell_delta.raw",
        first=cell1,
        second=cell2,
        valid=flux_valid,
    )
    return _sum_weighted_pair_delta_slots(
        field=field,
        cell_pair_relation=relation,
        flux_valid_all_by_slot=flux_valid_all_by_slot,
        edge_sign=edge_sign,
        col_start=col_start,
        col_stop=col_stop,
        field_patch=field_patch,
        patch_row_start=patch_row_start,
        patch_row_end=patch_row_end,
    )


def _sum_weighted_pair_delta_slots(
    *,
    field: torch.Tensor,
    cell_pair_relation: StaticPairRelation | tuple[Any, ...],
    flux_valid_all_by_slot: tuple[bool, ...] | None = None,
    edge_sign: torch.Tensor,
    col_start: int,
    col_stop: int,
    field_patch: torch.Tensor | None = None,
    patch_row_start: int = 0,
    patch_row_end: int = 0,
    field_patch_col_start: int = 0,
) -> torch.Tensor:
    """Accumulate weighted pair deltas with relation-owned patch gathers."""
    cell1 = _dynamics_tendency_pair_field(cell_pair_relation, 0, "first")
    cell2 = _dynamics_tendency_pair_field(cell_pair_relation, 1, "second")
    flux_valid = _dynamics_tendency_pair_field(cell_pair_relation, 2, "valid")
    if flux_valid is None:
        raise ValueError(
            "dynamics-tendency weighted pair delta requires a relation validity mask"
        )
    n_cells = cell1.shape[0]
    if (
        _dynamics_tendency_can_use_no_ad_full_slot_reduce(
            field,
            cell1,
            cell2,
            flux_valid,
            edge_sign,
            field_patch,
        )
        and cell1.shape[1] > 1
    ):
        field_cell1, field_cell2 = _dynamics_tendency_gather_pair_patch_or_base_cols(
            cell_pair_relation,
            field,
            field_patch,
            patch_row_start=patch_row_start,
            patch_row_end=patch_row_end,
            col_start=col_start,
            col_stop=col_stop,
            patch_col_start=field_patch_col_start,
        )
        contrib = edge_sign.unsqueeze(2) * (field_cell2 - field_cell1)
        valid_all = (
            flux_valid_all_by_slot is not None
            and len(flux_valid_all_by_slot) == cell1.shape[1]
            and all(flux_valid_all_by_slot)
        )
        if not valid_all:
            contrib = contrib * flux_valid.unsqueeze(2).to(dtype=field.dtype)
        return _sum_fortran_order(contrib, dim=1)
    total: torch.Tensor | None = None
    for batch_start, batch_stop in _dynamics_tendency_slot_batches(
        cell1.shape[1], _dynamics_tendency_slot_gather_batch(field, cell1.shape[1])
    ):
        field_cell1_batch, field_cell2_batch = (
            _dynamics_tendency_gather_pair_patch_or_base_cols(
                cell_pair_relation,
                field,
                field_patch,
                patch_row_start=patch_row_start,
                patch_row_end=patch_row_end,
                col_start=col_start,
                col_stop=col_stop,
                patch_col_start=field_patch_col_start,
                slot_start=batch_start,
                slot_stop=batch_stop,
            )
        )
        for local_slot in range(batch_stop - batch_start):
            slot = batch_start + local_slot
            contrib = edge_sign[:, slot].view(n_cells, 1) * (
                field_cell2_batch[:, local_slot, :]
                - field_cell1_batch[:, local_slot, :]
            )
            valid_all = (
                flux_valid_all_by_slot is not None and flux_valid_all_by_slot[slot]
            )
            if not valid_all:
                contrib = contrib * flux_valid[:, slot].view(n_cells, 1).to(
                    dtype=field.dtype
                )
            total = _accumulate_ordered_slot(total, contrib)
    if total is None:
        return field.new_zeros((n_cells, max(col_stop - col_start, 0)))
    return total


def _sum_weighted_pair_delta_slots_level(
    *,
    field: torch.Tensor,
    cell_pair_relation: StaticPairRelation | tuple[Any, ...],
    flux_valid_all_by_slot: tuple[bool, ...] | None = None,
    edge_sign: torch.Tensor,
    field_patch: torch.Tensor | None = None,
    patch_row_start: int = 0,
    patch_row_end: int = 0,
) -> torch.Tensor:
    """Level-shaped dynamics-tendency pair-delta reducer with a stable compile role."""
    cell1 = _dynamics_tendency_pair_field(cell_pair_relation, 0, "first")
    cell2 = _dynamics_tendency_pair_field(cell_pair_relation, 1, "second")
    flux_valid = _dynamics_tendency_pair_field(cell_pair_relation, 2, "valid")
    if flux_valid is None:
        raise ValueError(
            "dynamics-tendency weighted pair delta requires a relation validity mask"
        )
    n_cells = cell1.shape[0]
    col_start = 0
    col_stop = field.shape[1]
    if (
        _dynamics_tendency_can_use_no_ad_full_slot_reduce(
            field,
            cell1,
            cell2,
            flux_valid,
            edge_sign,
            field_patch,
        )
        and cell1.shape[1] > 1
    ):
        field_cell1, field_cell2 = _dynamics_tendency_gather_pair_patch_or_base_cols(
            cell_pair_relation,
            field,
            field_patch,
            patch_row_start=patch_row_start,
            patch_row_end=patch_row_end,
            col_start=col_start,
            col_stop=col_stop,
            patch_col_start=0,
        )
        contrib = edge_sign.unsqueeze(2) * (field_cell2 - field_cell1)
        valid_all = (
            flux_valid_all_by_slot is not None
            and len(flux_valid_all_by_slot) == cell1.shape[1]
            and all(flux_valid_all_by_slot)
        )
        if not valid_all:
            contrib = contrib * flux_valid.unsqueeze(2).to(dtype=field.dtype)
        return _sum_fortran_order(contrib, dim=1)
    total: torch.Tensor | None = None
    for batch_start, batch_stop in _dynamics_tendency_slot_batches(
        cell1.shape[1], _dynamics_tendency_slot_gather_batch(field, cell1.shape[1])
    ):
        field_cell1_batch, field_cell2_batch = (
            _dynamics_tendency_gather_pair_patch_or_base_cols(
                cell_pair_relation,
                field,
                field_patch,
                patch_row_start=patch_row_start,
                patch_row_end=patch_row_end,
                col_start=col_start,
                col_stop=col_stop,
                patch_col_start=0,
                slot_start=batch_start,
                slot_stop=batch_stop,
            )
        )
        for local_slot in range(batch_stop - batch_start):
            slot = batch_start + local_slot
            contrib = edge_sign[:, slot].view(n_cells, 1) * (
                field_cell2_batch[:, local_slot, :]
                - field_cell1_batch[:, local_slot, :]
            )
            valid_all = (
                flux_valid_all_by_slot is not None and flux_valid_all_by_slot[slot]
            )
            if not valid_all:
                contrib = contrib * flux_valid[:, slot].view(n_cells, 1).to(
                    dtype=field.dtype
                )
            total = _accumulate_ordered_slot(total, contrib)
    if total is None:
        return field.new_zeros((n_cells, col_stop))
    return total


def _subtract_weighted_pair_delta_slots_fortran_order(
    *,
    initial: torch.Tensor,
    field: torch.Tensor,
    cell_pair_relation: StaticPairRelation | tuple[Any, ...],
    flux_valid_all_by_slot: tuple[bool, ...] | None,
    edge_sign: torch.Tensor,
    field_patch: torch.Tensor | None,
    patch_row_start: int,
    patch_row_end: int,
    col_start: int,
    col_stop: int,
    field_patch_col_start: int,
) -> torch.Tensor:
    """Apply del4 edge terms in the statement order used by MPAS-Atmosphere.

    The upstream MPAS loop starts with the accumulated del2 tendency and
    subtracts each edge contribution immediately.  Summing the edge terms first
    and subtracting the total changes the binary64 association and is therefore
    not numerically equivalent under the release contract.
    """
    cell1 = _dynamics_tendency_pair_field(cell_pair_relation, 0, "first")
    flux_valid = _dynamics_tendency_pair_field(cell_pair_relation, 2, "valid")
    if flux_valid is None:
        raise ValueError(
            "dynamics-tendency weighted pair delta requires a relation validity mask"
        )
    if initial.shape != (cell1.shape[0], col_stop - col_start):
        raise ValueError(
            "dynamics-tendency ordered del4 initial shape does not match the selected rows/levels"
        )

    result = initial
    for batch_start, batch_stop in _dynamics_tendency_slot_batches(
        cell1.shape[1], _dynamics_tendency_slot_gather_batch(field, cell1.shape[1])
    ):
        field_cell1_batch, field_cell2_batch = (
            _dynamics_tendency_gather_pair_patch_or_base_cols(
                cell_pair_relation,
                field,
                field_patch,
                patch_row_start=patch_row_start,
                patch_row_end=patch_row_end,
                col_start=col_start,
                col_stop=col_stop,
                patch_col_start=field_patch_col_start,
                slot_start=batch_start,
                slot_stop=batch_stop,
            )
        )
        for local_slot in range(batch_stop - batch_start):
            slot = batch_start + local_slot
            contribution = edge_sign[:, slot].view(cell1.shape[0], 1) * (
                field_cell2_batch[:, local_slot, :]
                - field_cell1_batch[:, local_slot, :]
            )
            valid_all = (
                flux_valid_all_by_slot is not None and flux_valid_all_by_slot[slot]
            )
            if valid_all:
                result = result - contribution
            else:
                result = torch.where(
                    flux_valid[:, slot].view(cell1.shape[0], 1),
                    result - contribution,
                    result,
                )
    return result


def _sum_weighted_pair_delta_slots_inner_interface(
    *,
    field: torch.Tensor,
    cell_pair_relation: StaticPairRelation | tuple[Any, ...],
    flux_valid_all_by_slot: tuple[bool, ...] | None = None,
    edge_sign: torch.Tensor,
    field_patch: torch.Tensor | None = None,
    patch_row_start: int = 0,
    patch_row_end: int = 0,
    field_patch_col_start: int = 1,
) -> torch.Tensor:
    """Interface-shaped dynamics-tendency pair-delta reducer with a stable compile role."""
    cell1 = _dynamics_tendency_pair_field(cell_pair_relation, 0, "first")
    cell2 = _dynamics_tendency_pair_field(cell_pair_relation, 1, "second")
    flux_valid = _dynamics_tendency_pair_field(cell_pair_relation, 2, "valid")
    if flux_valid is None:
        raise ValueError(
            "dynamics-tendency weighted pair delta requires a relation validity mask"
        )
    n_cells = cell1.shape[0]
    col_start = 1
    col_stop = field.shape[1] - 1
    if (
        _dynamics_tendency_can_use_no_ad_full_slot_reduce(
            field,
            cell1,
            cell2,
            flux_valid,
            edge_sign,
            field_patch,
        )
        and cell1.shape[1] > 1
    ):
        field_cell1, field_cell2 = _dynamics_tendency_gather_pair_patch_or_base_cols(
            cell_pair_relation,
            field,
            field_patch,
            patch_row_start=patch_row_start,
            patch_row_end=patch_row_end,
            col_start=col_start,
            col_stop=col_stop,
            patch_col_start=field_patch_col_start,
        )
        contrib = edge_sign.unsqueeze(2) * (field_cell2 - field_cell1)
        valid_all = (
            flux_valid_all_by_slot is not None
            and len(flux_valid_all_by_slot) == cell1.shape[1]
            and all(flux_valid_all_by_slot)
        )
        if not valid_all:
            contrib = contrib * flux_valid.unsqueeze(2).to(dtype=field.dtype)
        return _sum_fortran_order(contrib, dim=1)
    total: torch.Tensor | None = None
    for batch_start, batch_stop in _dynamics_tendency_slot_batches(
        cell1.shape[1], _dynamics_tendency_slot_gather_batch(field, cell1.shape[1])
    ):
        field_cell1_batch, field_cell2_batch = (
            _dynamics_tendency_gather_pair_patch_or_base_cols(
                cell_pair_relation,
                field,
                field_patch,
                patch_row_start=patch_row_start,
                patch_row_end=patch_row_end,
                col_start=col_start,
                col_stop=col_stop,
                patch_col_start=field_patch_col_start,
                slot_start=batch_start,
                slot_stop=batch_stop,
            )
        )
        for local_slot in range(batch_stop - batch_start):
            slot = batch_start + local_slot
            contrib = edge_sign[:, slot].view(n_cells, 1) * (
                field_cell2_batch[:, local_slot, :]
                - field_cell1_batch[:, local_slot, :]
            )
            valid_all = (
                flux_valid_all_by_slot is not None and flux_valid_all_by_slot[slot]
            )
            if not valid_all:
                contrib = contrib * flux_valid[:, slot].view(n_cells, 1).to(
                    dtype=field.dtype
                )
            total = _accumulate_ordered_slot(total, contrib)
    if total is None:
        return field.new_zeros((n_cells, max(col_stop - col_start, 0)))
    return total


_DYNAMICS_TENDENCY_CONNECTIVITY_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
_DYNAMICS_TENDENCY_CONNECTIVITY_PAYLOAD_NAMES = (
    "edge_idx_all",
    "cell_idx",
    "solve_cells",
    "solve_edge_idx",
    "solve_edge_slice",
    "cell_edge_valid",
    "cell_edge_valid_all_by_slot",
    "safe_cell_edges",
    "cell_edge_sign",
    "cell_signed_dv",
    "cell_edge_relation",
    "cell_flux_edge_relation",
    "solve_edge_valid",
    "solve_edge_valid_all_by_slot",
    "solve_safe_edges",
    "solve_edge_sign",
    "solve_signed_dv",
    "solve_cell_edge_relation",
    "solve_flux_valid",
    "solve_flux_valid_all_by_slot",
    "solve_cell1",
    "solve_cell2",
    "solve_flux_cell_relation",
    "edge_safe_adv_cells",
    "edge_adv_coef",
    "edge_adv_coef_3rd",
    "edge_adv_cell_relation",
    "edge_eoe_valid",
    "edge_eoe_valid_all_by_slot",
    "edge_eoe_safe",
    "edge_eoe_weights",
    "edge_neighbor_relation",
    "solve_local_edges",
    "solve_edge_local_valid_all_by_slot",
    "solve_local_edge_relation",
    "cell_flux_valid",
    "cell_flux_valid_all_by_slot",
    "cell_flux1",
    "cell_flux2",
    "cell_flux_pair_relation",
    "edge_all_pair_valid",
    "edge_all_c1",
    "edge_all_c2",
    "edge_all_pair_relation",
    "edge_all_vertex1",
    "edge_all_vertex2",
    "edge_all_vertex_relation",
    "solve_edge_pair_valid",
    "edge_c1",
    "edge_c2",
    "solve_edge_pair_relation",
    "solve_edge_vertex1",
    "solve_edge_vertex2",
    "solve_edge_vertex_relation",
    "solve_edge_mask",
    "solve_edge_mask_all",
    "solve_edge_valid_weight",
    "cell_dv_over_dc",
    "cell_mesh_scaling_del2",
    "solve_del4_dv_over_dc",
    "cell_w_mix_edge_sign",
    "cell_theta_mix_edge_sign",
    "cell_div_del2_edge_sign",
    "solve_del4_mesh_scaling",
    "solve_del4_inv_dc",
    "solve_del4_min_inv_dv_dc",
    "edge_all_inv_dc",
    "edge_all_min_inv_dv_dc",
    "vort_vertices",
    "vertex_edges",
    "vertex_edge_relation",
    "vertex_del2_edge_sign",
    "cell_slot_count",
)

# The numeric body reads these values through the corresponding flattened
# relation tuple. Keeping a second outer alias only adds Dynamo identity guards.
_DYNAMICS_TENDENCY_RELATION_OWNED_PAYLOAD_ALIAS_NAMES = frozenset(
    {
        "cell_edge_valid",
        "cell_edge_valid_all_by_slot",
        "safe_cell_edges",
        "cell_edge_sign",
        "cell_signed_dv",
        "solve_edge_valid",
        "solve_edge_valid_all_by_slot",
        "solve_safe_edges",
        "solve_edge_sign",
        "solve_signed_dv",
        "solve_flux_valid",
        "solve_cell1",
        "solve_cell2",
        "edge_safe_adv_cells",
        "edge_adv_coef",
        "edge_adv_coef_3rd",
        "edge_eoe_valid",
        "edge_eoe_valid_all_by_slot",
        "edge_eoe_safe",
        "edge_eoe_weights",
        "solve_local_edges",
        "solve_edge_local_valid_all_by_slot",
        "cell_flux_valid",
        "cell_flux_valid_all_by_slot",
        "cell_flux1",
        "cell_flux2",
        "edge_all_pair_valid",
        "edge_all_c1",
        "edge_all_c2",
        "edge_all_vertex1",
        "edge_all_vertex2",
        "solve_edge_pair_valid",
        "edge_c1",
        "edge_c2",
        "solve_edge_vertex1",
        "solve_edge_vertex2",
        "solve_edge_mask",
        "solve_edge_mask_all",
        "solve_edge_valid_weight",
    }
)
_DYNAMICS_TENDENCY_NUMERIC_CONNECTIVITY_PAYLOAD_NAMES = tuple(
    name
    for name in _DYNAMICS_TENDENCY_CONNECTIVITY_PAYLOAD_NAMES
    if name not in _DYNAMICS_TENDENCY_RELATION_OWNED_PAYLOAD_ALIAS_NAMES
)
_DYNAMICS_TENDENCY_CONNECTIVITY_TENSOR_NAMES = (
    "edgesOnCell",
    "nEdgesOnCell",
    "edgesOnCell_sign",
    "cellsOnEdge",
    "verticesOnEdge",
    "edgesOnEdge",
    "nAdvCellsForEdge",
    "nEdgesOnEdge",
    "advCellsForEdge",
    "adv_coefs",
    "adv_coefs_3rd",
    "weightsOnEdge",
    "edgesOnVertex",
    "edgesOnVertex_sign",
    "dvEdge",
    "dcEdge",
    "invDcEdge",
    "invDvEdge",
    "invAreaCell",
    "invAreaTriangle",
    "meshScalingDel2",
    "meshScalingDel4",
)


def _dynamics_tendency_tensor_cache_key(tensor: torch.Tensor) -> tuple[Any, ...] | None:
    return _tensor_version_cache_key(tensor)


def _dynamics_tendency_connectivity_payload_cache_key(
    **kwargs: Any,
) -> tuple[Any, ...] | None:
    """Return the mutation-sensitive cache key for a dynamics-tendency static payload."""
    tensors = tuple(
        kwargs.get(name) for name in _DYNAMICS_TENDENCY_CONNECTIVITY_TENSOR_NAMES
    )
    if not _static_connectivity_cache_allowed(tensors):
        return None

    tensor_keys: list[tuple[Any, ...]] = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            tensor_key = _dynamics_tendency_tensor_cache_key(tensor)
            if tensor_key is None:
                return None
            tensor_keys.append(tensor_key)

    return (
        kwargs["device"],
        kwargs["nCells"],
        kwargs["nEdges"],
        kwargs["maxEdges"],
        kwargs["maxEdges2"],
        kwargs["cell_start"],
        kwargs["cell_end"],
        kwargs["cell_solve_start"],
        kwargs["cell_solve_end"],
        kwargs["edge_start"],
        kwargs["edge_end"],
        kwargs["edge_solve_start"],
        kwargs["edge_solve_end"],
        kwargs.get("vertex_start"),
        kwargs.get("vertex_end"),
        kwargs.get("vertexDegree"),
        bool(kwargs.get("build_ordered_ad_transpose", False)),
        bool(kwargs.get("build_edge_advective_ordered_ad_transpose", False)),
        *tensor_keys,
    )


def _dynamics_tendency_static_all_true(
    mask: torch.Tensor, *, allow_cuda_sync: bool = False
) -> bool:
    """Return setup-time all-true metadata for static topology masks."""
    return bool(_static_mask_all_true(mask, allow_cuda_sync=allow_cuda_sync))


def _dynamics_tendency_slot_all_true(
    mask: torch.Tensor,
    *,
    allow_cuda_sync: bool = False,
) -> tuple[bool, ...] | None:
    """Return setup-time per-slot all-true metadata for static topology masks."""
    return _static_slot_all_true(mask, allow_cuda_sync=allow_cuda_sync)


def _dynamics_tendency_zero_row_patch_like(
    field: torch.Tensor,
    row_start: int,
    row_end: int,
) -> torch.Tensor:
    return field.new_zeros((int(row_end) - int(row_start), *field.shape[1:]))


class _DynamicsTendencyDebugTapObserver:
    """Capture optional diagnostic tensors from dynamics-tendency section calls.

    The observer derives diagnostics from section inputs and outputs without
    changing the tendency calculation.
    """

    def __init__(
        self,
        call_arguments: Mapping[str, Any],
        taps: dict[str, dict[str, Any]],
    ) -> None:
        self.a = call_arguments
        self.taps = taps
        self.rk_step = int(call_arguments["rk_step"])
        self.n_levels = int(call_arguments["nVertLevels"])
        self.cell_start = int(call_arguments["cellStart"]) - 1
        self.cell_end = int(call_arguments["cellEnd"])
        self.cell_solve_start = int(call_arguments["cellSolveStart"]) - 1
        self.cell_solve_end = int(call_arguments["cellSolveEnd"])
        self.edge_start = int(call_arguments["edgeStart"]) - 1
        self.edge_end = int(call_arguments["edgeEnd"])
        self.edge_solve_start = int(call_arguments["edgeSolveStart"]) - 1
        self.edge_solve_end = int(call_arguments["edgeSolveEnd"])
        self.vertex_start = int(call_arguments["vertexStart"]) - 1
        self.vertex_end = int(call_arguments["vertexEnd"])
        self.cell_solve_rel_start = self.cell_solve_start - self.cell_start
        self.cell_solve_rel_end = self.cell_solve_end - self.cell_start
        self.edge_solve_rel_start = self.edge_solve_start - self.edge_start
        self.edge_solve_rel_end = self.edge_solve_end - self.edge_start
        u = self._tensor("u")
        self.device = u.device
        self.dtype = u.dtype

        self.kdiff = self._input_or_zeros(
            "kdiff_in", (int(call_arguments["nCells"]) + 1, self.n_levels)
        )
        self.tend_u_euler = self._input_or_zeros(
            "tend_u_euler_in", (int(call_arguments["nEdges"]) + 1, self.n_levels)
        )
        self.tend_w_euler = self._input_or_zeros(
            "tend_w_euler_in",
            (int(call_arguments["nCells"]) + 1, self.n_levels + 1),
        )
        self.tend_theta_euler = self._input_or_zeros(
            "tend_theta_euler_in",
            (int(call_arguments["nCells"]) + 1, self.n_levels),
        )
        self.tend_rho = self._input_or_zeros(
            "tend_rho_in", (int(call_arguments["nCells"]) + 1, self.n_levels)
        )
        self.tend_u = self._zeros((int(call_arguments["nEdges"]) + 1, self.n_levels))
        self.tend_w = self._zeros(
            (int(call_arguments["nCells"]) + 1, self.n_levels + 1)
        )
        self.tend_theta = self._zeros(
            (int(call_arguments["nCells"]) + 1, self.n_levels)
        )
        self.h_divergence = self._zeros(
            (int(call_arguments["nCells"]) + 1, self.n_levels)
        )
        self.rthdynten = self._zeros((int(call_arguments["nCells"]) + 1, self.n_levels))
        self.dpdz = self._zeros((int(call_arguments["nCells"]) + 1, self.n_levels))
        self.delsq_w = self._zeros(
            (int(call_arguments["nCells"]) + 1, self.n_levels + 1)
        )
        self.delsq_theta = self._zeros(
            (int(call_arguments["nCells"]) + 1, self.n_levels)
        )
        self.tend_u_euler_edge_patch: torch.Tensor | None = None
        self.kdiff_cell_patch: torch.Tensor | None = None
        self.h_divergence_cell_patch: torch.Tensor | None = None
        self.tend_rho_cell_patch: torch.Tensor | None = None
        self.dpdz_cell_patch: torch.Tensor | None = None
        self.tend_w_euler_cell_patch: torch.Tensor | None = None
        self.tend_theta_euler_cell_patch: torch.Tensor | None = None
        self.rthdynten_solve_patch: torch.Tensor | None = None
        self.normal_velocity_tendency_hdiv_proof_taps: dict[str, Any] = {}
        self.vertical_velocity_horizontal_advection_proof_taps: dict[str, Any] = {}
        self.theta_horizontal_advection_proof_taps: dict[str, Any] = {}
        self._recorded_t04 = False
        self._recorded_t07 = False
        self._recorded_t11 = False

    def _tensor(self, name: str) -> torch.Tensor:
        value = self.a[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"dynamics-tendency debug tap input {name!r} must be a tensor"
            )
        return value

    def _zeros(self, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.zeros(shape, dtype=self.dtype, device=self.device)

    def _input_or_zeros(self, name: str, shape: tuple[int, ...]) -> torch.Tensor:
        value = self.a.get(name)
        return value if isinstance(value, torch.Tensor) else self._zeros(shape)

    @staticmethod
    def _row(base: torch.Tensor, start: int, end: int, patch: object) -> torch.Tensor:
        if not isinstance(patch, torch.Tensor):
            return base
        out = base.clone()
        out[start:end, ...] = patch
        return out

    @staticmethod
    def _row_cols(
        base: torch.Tensor,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        patch: object,
    ) -> torch.Tensor:
        if not isinstance(patch, torch.Tensor):
            return base
        out = base.clone()
        out[row_start:row_end, col_start:col_end, ...] = patch
        return out

    def _record(self, name: str, **arrays: torch.Tensor) -> None:
        self.taps[name] = {key: value.clone() for key, value in arrays.items()}

    def __call__(
        self,
        name: str,
        _args: tuple[object, ...],
        kwargs: Mapping[str, Any],
        result: object,
        _elapsed: float,
    ) -> None:
        getattr(self, f"_observe_{name}")(kwargs, result)

    def _observe_mixing_initialization_mixing_initialization(
        self, _kwargs: Mapping[str, Any], result: object
    ) -> None:
        tend_u_euler_patch, kdiff_patch, _h_mom, _h_theta = cast(
            tuple[torch.Tensor, torch.Tensor, float, float], result
        )
        if self.rk_step == 1:
            self.tend_u_euler_edge_patch = tend_u_euler_patch
            self.kdiff_cell_patch = kdiff_patch
        self._record(
            "T01_post_kdiff",
            kdiff=self._row(
                self.kdiff, self.cell_start, self.cell_end, self.kdiff_cell_patch
            ),
        )

    def _observe_density_tendency_density(
        self, kwargs: Mapping[str, Any], result: object
    ) -> None:
        hdiv_patch, tend_rho_patch, dpdz_patch = cast(
            tuple[torch.Tensor, torch.Tensor, torch.Tensor], result
        )
        self.h_divergence_cell_patch = hdiv_patch
        (
            hdiv_sum,
            hdiv_abs,
            hdiv_count,
            hdiv_terms,
        ) = _normal_velocity_tendency_h_divergence_proof_tensors(
            ru=kwargs["ru"],
            dv_edge=kwargs["dv_edge"],
            cell_edge_relation=kwargs["cell_edge_relation"],
        )
        self.normal_velocity_tendency_hdiv_proof_taps = {
            "h_divergence_edge_sum": hdiv_sum,
            "h_divergence_abs_term_sum": hdiv_abs,
            "h_divergence_term_count": hdiv_count,
            "normal_velocity_tendency_hdiv_terms": hdiv_terms,
        }
        if self.rk_step == 1:
            self.tend_rho_cell_patch = tend_rho_patch
            self.dpdz_cell_patch = dpdz_patch
        self._record(
            "T02_post_density",
            h_divergence=self._row(
                self.h_divergence,
                self.cell_start,
                self.cell_end,
                self.h_divergence_cell_patch,
            ),
            tend_rho=self._row(
                self.tend_rho,
                self.cell_start,
                self.cell_end,
                self.tend_rho_cell_patch,
            ),
            dpdz=self._row(
                self.dpdz, self.cell_start, self.cell_end, self.dpdz_cell_patch
            ),
        )

    def _observe_normal_velocity_tendency_normal_velocity(
        self, kwargs: Mapping[str, Any], result: object
    ) -> None:
        tend_u_patch, tend_u_euler_patch = cast(
            tuple[torch.Tensor, torch.Tensor], result
        )
        if self.rk_step == 1:
            if self.tend_u_euler_edge_patch is None:
                self.tend_u_euler_edge_patch = self.tend_u_euler[
                    self.edge_start : self.edge_end, :
                ]
            self.tend_u_euler_edge_patch = _dynamics_tendency_replace_patch_rows(
                self.tend_u_euler_edge_patch,
                self.edge_solve_rel_start,
                self.edge_solve_rel_end,
                tend_u_euler_patch,
            )
        solve_slice = slice(
            int(kwargs["edge_solve_start"]), int(kwargs["edge_solve_end"])
        )
        normal_velocity_tendency_proof_taps = _normal_velocity_tendency_proof_tensors(
            edge_cell_relation=kwargs["edge_cell_relation"],
            edge_neighbor_relation=kwargs["edge_neighbor_relation"],
            u=kwargs["u"],
            u_edge=kwargs["u"][solve_slice, :],
            rw=kwargs["rw"],
            ke=kwargs["ke"],
            h_divergence=kwargs["h_divergence"],
            h_divergence_cell_patch=kwargs["h_divergence_cell_patch"],
            h_divergence_abs_term_sum_cell_patch=self.normal_velocity_tendency_hdiv_proof_taps[
                "h_divergence_abs_term_sum"
            ],
            pv_edge=kwargs["pv_edge"],
            pv_edge_solve=kwargs["pv_edge"][solve_slice, :],
            rho_edge_solve=kwargs["rho_edge"][solve_slice, :],
            inv_dc_edge_solve=kwargs["inv_dc_edge"][solve_slice],
            rdzw=kwargs["rdzw"],
            fzm=kwargs["fzm"],
            fzp=kwargs["fzp"],
            n_vert_levels=kwargs["n_vert_levels"],
            include_pressure_gradient=kwargs["include_pressure_gradient"],
            zz=kwargs["zz"],
            pp=kwargs["pp"],
            dpdz=kwargs["dpdz"],
            dpdz_cell_patch=kwargs["dpdz_cell_patch"],
            cell_patch_row_start=kwargs["cell_patch_row_start"],
            cell_patch_row_end=kwargs["cell_patch_row_end"],
            inv_area_cell=self._tensor("invAreaCell"),
            cqu_solve=kwargs["cqu"][solve_slice, :],
            zxu_solve=kwargs["zxu"][solve_slice, :],
        )
        self._record(
            "T03_post_u_base",
            tend_u=self._row(
                self.tend_u,
                self.edge_solve_start,
                self.edge_solve_end,
                tend_u_patch,
            ),
            tend_u_euler=self._row(
                self.tend_u_euler,
                self.edge_solve_start,
                self.edge_solve_end,
                self.tend_u_euler_edge_patch,
            ),
            h_divergence=self._row(
                self.h_divergence,
                self.cell_start,
                self.cell_end,
                self.h_divergence_cell_patch,
            ),
            **self.normal_velocity_tendency_hdiv_proof_taps,
            **normal_velocity_tendency_proof_taps,
        )

    def _record_t04(self) -> None:
        if self._recorded_t04:
            return
        self._recorded_t04 = True
        self._record(
            "T04_post_u_mixing",
            tend_u_euler=self._row(
                self.tend_u_euler,
                self.edge_start,
                self.edge_end,
                self.tend_u_euler_edge_patch,
            ),
        )

    def _observe_normal_velocity_mixing_u_mixing(
        self, _kwargs: Mapping[str, Any], result: object
    ) -> None:
        self.tend_u_euler_edge_patch = result  # type: ignore[assignment]
        self._record_t04()

    def _observe_normal_velocity_finalization_u_final(
        self, _kwargs: Mapping[str, Any], result: object
    ) -> None:
        self._record_t04()
        self._record(
            "T05_post_u_final",
            tend_u=self._row(
                self.tend_u,
                self.edge_solve_start,
                self.edge_solve_end,
                result,
            ),
        )

    def _observe_vertical_velocity_horizontal_advection_w_hadv(
        self, kwargs: Mapping[str, Any], result: object
    ) -> None:
        self.vertical_velocity_horizontal_advection_proof_taps = (
            _dynamics_tendency_vertical_velocity_horizontal_advection_hadv_proof_taps(
                w=self._tensor("w"),
                ru=self._tensor("ru"),
                fzm=kwargs["fzm"],
                fzp=kwargs["fzp"],
                solve_edge_slice=slice(self.edge_solve_start, self.edge_solve_end),
                n_vert_levels=kwargs["n_vert_levels"],
                edge_adv_cell_relation=kwargs["edge_adv_cell_relation"],
                local_edge_relation=kwargs["local_edge_relation"],
            )
        )
        self._record(
            "T06_post_w_hadv",
            tend_w=self._row_cols(
                self.tend_w,
                self.cell_solve_start,
                self.cell_solve_end,
                1,
                self.n_levels,
                result,
            ),
            **self.vertical_velocity_horizontal_advection_proof_taps,
        )

    def _record_t07(self) -> None:
        if self._recorded_t07:
            return
        self._recorded_t07 = True
        delsq_w_patch = getattr(self, "delsq_w_cell_patch", None)
        self._record(
            "T07_post_w_hmix",
            delsq_w=self._row(
                self.delsq_w,
                self.cell_start,
                self.cell_end,
                delsq_w_patch,
            )[:, : self.n_levels],
            tend_w_euler=self._row(
                self.tend_w_euler,
                self.cell_start,
                self.cell_end,
                self.tend_w_euler_cell_patch,
            ),
        )

    def _observe_vertical_velocity_horizontal_mixing_w_horizontal_mix(
        self, kwargs: Mapping[str, Any], result: object
    ) -> None:
        delsq_w_inner_patch, _ = _sum_w_mix_slots(
            cell_edge_relation=kwargs["cell_edge_relation"],
            edge_cell_relation=kwargs["edge_cell_relation"],
            edge_sign=kwargs["edge_sign_del2"],
            rho_edge=kwargs["rho_edge"],
            w=kwargs["w"],
            kdiff=kwargs["kdiff"],
            kdiff_patch=kwargs["kdiff_patch"],
            patch_row_start=kwargs["patch_row_start"],
            patch_row_end=kwargs["patch_row_end"],
            mesh_scaling_del2=kwargs["mesh_scaling_del2"],
            mesh_scaling_del2_slots=kwargs["mesh_scaling_del2_slots"],
            n_vert_levels=kwargs["n_vert_levels"],
        )
        self.tend_w_euler_cell_patch = cast(torch.Tensor, result)
        row_patch = self._zeros((self.cell_end - self.cell_start, self.n_levels + 1))
        row_patch[:, 1 : self.n_levels] = delsq_w_inner_patch
        self.delsq_w_cell_patch = row_patch
        self._record_t07()

    def _observe_vertical_velocity_vertical_transport_w_vertical(
        self, kwargs: Mapping[str, Any], result: object
    ) -> None:
        self._record_t07()
        wdwz_delta = _vertical_velocity_vertical_transport_w_vertical_flux_delta_region(
            w_solve=kwargs["w_solve"],
            rw_solve=kwargs["rw_solve"],
            n_vert_levels=kwargs["n_vert_levels"],
        )
        (
            tend_w_vertical_inner_patch,
            tend_w_final_inner_patch,
            tend_w_euler_pressure_inner_patch,
            tend_w_euler_final_inner_patch,
        ) = _vertical_velocity_vertical_transport_w_pointwise_update_region(
            w_solve=kwargs["w_solve"],
            pp_solve=kwargs["pp_solve"],
            cqw_solve=kwargs["cqw_solve"],
            rho_zz_solve=kwargs["rho_zz_solve"],
            dpdz_solve=kwargs["dpdz_solve"],
            wdwz_delta=wdwz_delta,
            tend_w_solve_inner=kwargs["tend_w_solve_inner"],
            tend_w_euler_solve_inner=kwargs["tend_w_euler_solve_inner"],
            inv_area_cell_solve=kwargs["inv_area_cell_solve"],
            rdzu=kwargs["rdzu"],
            rdzw=kwargs["rdzw"],
            fzm=kwargs["fzm"],
            fzp=kwargs["fzp"],
            n_vert_levels=kwargs["n_vert_levels"],
            include_pressure_gradient=kwargs["include_pressure_gradient"],
            include_vertical_mix=kwargs["include_vertical_mix"],
            v_mom_eddy_visc2=kwargs["v_mom_eddy_visc2"],
        )
        live_final, live_euler = cast(tuple[torch.Tensor, torch.Tensor], result)
        if not torch.equal(live_final, tend_w_final_inner_patch) or not torch.equal(
            live_euler, tend_w_euler_final_inner_patch
        ):
            raise RuntimeError(
                "dynamics-tendency vertical-velocity-vertical-transport proof reconstruction changed live outputs"
            )
        before_w = kwargs["tend_w_solve_inner"]
        before_euler = kwargs["tend_w_euler_solve_inner"]
        w = self._tensor("w")
        n = self.n_levels
        vertical_velocity_vertical_transport_wdwz = (
            _vertical_velocity_vertical_transport_w_vertical_flux_proof_tensor(
                w_solve=kwargs["w_solve"],
                rw_solve=kwargs["rw_solve"],
                n_vert_levels=n,
            )
        )
        vertical_velocity_vertical_transport_wdwz_abs = _vertical_velocity_vertical_transport_w_vertical_flux_abs_term_sum_proof_tensor(
            w_solve=kwargs["w_solve"],
            rw_solve=kwargs["rw_solve"],
            n_vert_levels=n,
        )
        shape = (self.cell_solve_end - self.cell_solve_start, n + 1)
        vertical_velocity_vertical_transport_hadv_area = w.new_zeros(shape)
        vertical_velocity_vertical_transport_hadv_abs = w.new_zeros(shape)
        vertical_velocity_vertical_transport_vertical = w.new_zeros(shape)
        vertical_velocity_vertical_transport_vertical_abs = w.new_zeros(shape)
        vertical_velocity_vertical_transport_pressure = w.new_zeros(shape)
        vertical_velocity_vertical_transport_pressure_abs = w.new_zeros(shape)
        vertical_velocity_vertical_transport_tend_w_abs = w.new_zeros(shape)
        vertical_velocity_vertical_transport_euler_abs = w.new_zeros(shape)
        inv_area = kwargs["inv_area_cell_solve"]
        vertical_velocity_vertical_transport_hadv_area[:, 1:n] = (
            before_w * inv_area.unsqueeze(1)
        )
        vertical_velocity_vertical_transport_hadv_abs[:, 1:n] = torch.abs(
            self.vertical_velocity_horizontal_advection_proof_taps[
                "vertical_velocity_horizontal_advection_tend_w_abs_term_sum"
            ][:, 1:n]
            * torch.abs(inv_area).unsqueeze(1)
        )
        rdzu = kwargs["rdzu"]
        vertical_velocity_vertical_transport_vertical[:, 1:n] = -rdzu[1:n].unsqueeze(
            0
        ) * (
            vertical_velocity_vertical_transport_wdwz[:, 2 : n + 1]
            - vertical_velocity_vertical_transport_wdwz[:, 1:n]
        )
        vertical_velocity_vertical_transport_vertical_abs[:, 1:n] = torch.abs(
            rdzu[1:n].unsqueeze(0)
        ) * (
            vertical_velocity_vertical_transport_wdwz_abs[:, 2 : n + 1]
            + vertical_velocity_vertical_transport_wdwz_abs[:, 1:n]
        )
        vertical_velocity_vertical_transport_tend_w_abs[:, 1:n] = (
            vertical_velocity_vertical_transport_hadv_abs[:, 1:n]
            + vertical_velocity_vertical_transport_vertical_abs[:, 1:n]
        )
        if self.rk_step == 1:
            vertical_velocity_vertical_transport_pressure[:, 1:n] = (
                tend_w_euler_pressure_inner_patch - before_euler
            )
            cqw_inner = kwargs["cqw_solve"][:, 1:n]
            pp_solve = kwargs["pp_solve"]
            dpdz_solve = kwargs["dpdz_solve"]
            fzm = kwargs["fzm"]
            fzp = kwargs["fzp"]
            vertical_velocity_vertical_transport_pressure_abs[:, 1:n] = torch.abs(
                cqw_inner
            ) * (
                torch.abs(rdzu[1:n].unsqueeze(0) * pp_solve[:, 1:n])
                + torch.abs(rdzu[1:n].unsqueeze(0) * pp_solve[:, : n - 1])
                + torch.abs(fzm[1:n].unsqueeze(0) * dpdz_solve[:, 1:n])
                + torch.abs(fzp[1:n].unsqueeze(0) * dpdz_solve[:, : n - 1])
            )
            vertical_velocity_vertical_transport_euler_abs[:, 1:n] = (
                torch.abs(before_euler)
                + vertical_velocity_vertical_transport_pressure_abs[:, 1:n]
            )
        vertical_velocity_vertical_transport_proof_taps = {
            "vertical_velocity_horizontal_advection_tend_w_abs_term_sum": self.vertical_velocity_horizontal_advection_proof_taps[
                "vertical_velocity_horizontal_advection_tend_w_abs_term_sum"
            ],
            "vertical_velocity_vertical_transport_wdwz": vertical_velocity_vertical_transport_wdwz,
            "vertical_velocity_vertical_transport_wdwz_abs_term_sum": vertical_velocity_vertical_transport_wdwz_abs,
            "vertical_velocity_vertical_transport_hadv_area": vertical_velocity_vertical_transport_hadv_area,
            "vertical_velocity_vertical_transport_hadv_area_abs_term_sum": vertical_velocity_vertical_transport_hadv_abs,
            "vertical_velocity_vertical_transport_vertical_tendency": vertical_velocity_vertical_transport_vertical,
            "vertical_velocity_vertical_transport_vertical_tendency_abs_term_sum": vertical_velocity_vertical_transport_vertical_abs,
            "vertical_velocity_vertical_transport_pressure_gradient": vertical_velocity_vertical_transport_pressure,
            "vertical_velocity_vertical_transport_pressure_gradient_abs_term_sum": vertical_velocity_vertical_transport_pressure_abs,
            "vertical_velocity_vertical_transport_tend_w_abs_term_sum": vertical_velocity_vertical_transport_tend_w_abs,
            "vertical_velocity_vertical_transport_tend_w_euler_abs_term_sum": vertical_velocity_vertical_transport_euler_abs,
        }
        pressure_cell_patch = self.tend_w_euler_cell_patch
        if self.rk_step == 1:
            if pressure_cell_patch is None:
                pressure_cell_patch = self.tend_w_euler[
                    self.cell_start : self.cell_end, :
                ]
            pressure_cell_patch = _dynamics_tendency_replace_patch_rows_cols(
                pressure_cell_patch,
                self.cell_solve_rel_start,
                self.cell_solve_rel_end,
                1,
                n,
                tend_w_euler_pressure_inner_patch,
            )
            self.tend_w_euler_cell_patch = _dynamics_tendency_replace_patch_rows_cols(
                pressure_cell_patch,
                self.cell_solve_rel_start,
                self.cell_solve_rel_end,
                1,
                n,
                tend_w_euler_final_inner_patch,
            )
        self._record(
            "T08_post_w_vert",
            tend_w=self._row_cols(
                self.tend_w,
                self.cell_solve_start,
                self.cell_solve_end,
                1,
                n,
                tend_w_vertical_inner_patch,
            ),
            tend_w_euler=self._row(
                self.tend_w_euler,
                self.cell_start,
                self.cell_end,
                pressure_cell_patch,
            ),
            **vertical_velocity_vertical_transport_proof_taps,
        )
        self._record(
            "T09_post_w_final",
            tend_w=self._row_cols(
                self.tend_w,
                self.cell_solve_start,
                self.cell_solve_end,
                1,
                n,
                tend_w_final_inner_patch,
            ),
        )

    def _observe_theta_horizontal_advection_theta_hadv(
        self, kwargs: Mapping[str, Any], result: object
    ) -> None:
        self.theta_horizontal_advection_proof_taps = (
            _dynamics_tendency_theta_horizontal_advection_hadv_proof_taps(
                theta_m=self._tensor("theta_m"),
                theta_m_save=self._tensor("theta_m_save"),
                ru=self._tensor("ru"),
                ru_save=self._tensor("ru_save"),
                solve_edge_slice=slice(self.edge_solve_start, self.edge_solve_end),
                n_vert_levels=self.n_levels,
                rk_step=self.rk_step,
                edge_adv_cell_relation=kwargs["edge_adv_cell_relation"],
                local_edge_relation=kwargs["local_edge_relation"],
                cell_edge_relation=kwargs["cell_edge_relation"],
                edge_cell_relation=kwargs["edge_cell_relation"],
            )
        )
        self.tend_theta_solve_patch = result
        self._record(
            "T10_post_theta_hadv",
            tend_theta=self._row(
                self.tend_theta,
                self.cell_solve_start,
                self.cell_solve_end,
                result,
            ),
            **self.theta_horizontal_advection_proof_taps,
        )

    def _record_t11(self) -> None:
        if self._recorded_t11:
            return
        self._recorded_t11 = True
        self._record(
            "T11_post_theta_hmix",
            delsq_theta=self._row(
                self.delsq_theta,
                self.cell_start,
                self.cell_end,
                getattr(self, "delsq_theta_cell_patch", None),
            ),
            tend_theta_euler=self._row(
                self.tend_theta_euler,
                self.cell_start,
                self.cell_end,
                self.tend_theta_euler_cell_patch,
            ),
        )

    def _observe_theta_horizontal_mixing_theta_horizontal_mix(
        self, kwargs: Mapping[str, Any], result: object
    ) -> None:
        self.delsq_theta_cell_patch, _ = _sum_theta_mix_slots(
            cell_edge_relation=kwargs["cell_edge_relation"],
            edge_cell_relation=kwargs["edge_cell_relation"],
            edge_sign=kwargs["edge_sign_del2"],
            rho_edge=kwargs["rho_edge"],
            theta_m=kwargs["theta_m"],
            kdiff=kwargs["kdiff"],
            kdiff_patch=kwargs["kdiff_patch"],
            patch_row_start=kwargs["patch_row_start"],
            patch_row_end=kwargs["patch_row_end"],
            pr_scale=kwargs["pr_scale"],
        )
        self.tend_theta_euler_cell_patch = cast(torch.Tensor, result)
        self._record_t11()

    def _observe_theta_vertical_advection_diabatic_theta_vertical(
        self, kwargs: Mapping[str, Any], result: object
    ) -> None:
        self._record_t11()
        tend_theta_patch, self.rthdynten_solve_patch = cast(
            tuple[torch.Tensor, torch.Tensor], result
        )
        n = self.n_levels
        before = kwargs["tend_theta_solve"]
        theta_vertical_advection_diabatic_wdtz = (
            _theta_vertical_advection_diabatic_theta_vertical_flux_proof_tensor(
                theta_m_solve=kwargs["theta_m_solve"],
                theta_m_save_solve=kwargs["theta_m_save_solve"],
                rw_solve=kwargs["rw_solve"],
                rw_save_solve=kwargs["rw_save_solve"],
                fzm=kwargs["fzm"],
                fzp=kwargs["fzp"],
                coef_3rd_order=kwargs["coef_3rd_order"],
                n_vert_levels=n,
            )
        )
        theta_vertical_advection_diabatic_wdtz_abs = _theta_vertical_advection_diabatic_theta_vertical_flux_abs_term_sum_proof_tensor(
            theta_m_solve=kwargs["theta_m_solve"],
            theta_m_save_solve=kwargs["theta_m_save_solve"],
            rw_solve=kwargs["rw_solve"],
            rw_save_solve=kwargs["rw_save_solve"],
            fzm=kwargs["fzm"],
            fzp=kwargs["fzp"],
            coef_3rd_order=kwargs["coef_3rd_order"],
            n_vert_levels=n,
        )
        inv_area = kwargs["inv_area_cell_solve"]
        theta_vertical_advection_diabatic_hadv_area = before * inv_area.unsqueeze(1)
        theta_vertical_advection_diabatic_hadv_abs = torch.abs(
            (
                self.theta_horizontal_advection_proof_taps[
                    "theta_horizontal_advection_tend_theta_abs_term_sum"
                ]
                + self.theta_horizontal_advection_proof_taps[
                    "theta_horizontal_advection_perturbation_abs_term_sum"
                ]
            )
            * torch.abs(inv_area).unsqueeze(1)
        )
        rdzw = kwargs["rdzw"]
        theta_vertical_advection_diabatic_vertical = -rdzw[:n].unsqueeze(0) * (
            theta_vertical_advection_diabatic_wdtz[:, 1 : n + 1]
            - theta_vertical_advection_diabatic_wdtz[:, :n]
        )
        theta_vertical_advection_diabatic_vertical_abs = torch.abs(
            rdzw[:n].unsqueeze(0)
        ) * (
            theta_vertical_advection_diabatic_wdtz_abs[:, 1 : n + 1]
            + theta_vertical_advection_diabatic_wdtz_abs[:, :n]
        )
        numerator = (
            theta_vertical_advection_diabatic_hadv_area
            + theta_vertical_advection_diabatic_vertical
            - kwargs["tend_rho_solve"] * kwargs["theta_m_solve"]
        )
        diabatic = kwargs["rho_zz_solve"] * kwargs["rt_diabatic_tend_solve"]
        numerator_abs = (
            theta_vertical_advection_diabatic_hadv_abs
            + theta_vertical_advection_diabatic_vertical_abs
            + torch.abs(kwargs["tend_rho_solve"] * kwargs["theta_m_solve"])
        )
        diabatic_abs = torch.abs(diabatic)
        tend_abs = (
            theta_vertical_advection_diabatic_hadv_abs
            + theta_vertical_advection_diabatic_vertical_abs
            + diabatic_abs
        )
        rth_abs = numerator_abs / torch.clamp(
            torch.abs(kwargs["rho_zz_solve"]),
            min=torch.finfo(kwargs["rho_zz_solve"].dtype).tiny,
        )
        proof = {
            "theta_horizontal_advection_tend_theta_abs_term_sum": self.theta_horizontal_advection_proof_taps[
                "theta_horizontal_advection_tend_theta_abs_term_sum"
            ],
            "theta_horizontal_advection_perturbation_abs_term_sum": self.theta_horizontal_advection_proof_taps[
                "theta_horizontal_advection_perturbation_abs_term_sum"
            ],
            "theta_vertical_advection_diabatic_wdtz": theta_vertical_advection_diabatic_wdtz,
            "theta_vertical_advection_diabatic_wdtz_abs_term_sum": theta_vertical_advection_diabatic_wdtz_abs,
            "theta_vertical_advection_diabatic_hadv_area": theta_vertical_advection_diabatic_hadv_area,
            "theta_vertical_advection_diabatic_hadv_area_abs_term_sum": theta_vertical_advection_diabatic_hadv_abs,
            "theta_vertical_advection_diabatic_vertical_tendency": theta_vertical_advection_diabatic_vertical,
            "theta_vertical_advection_diabatic_vertical_tendency_abs_term_sum": theta_vertical_advection_diabatic_vertical_abs,
            "theta_vertical_advection_diabatic_rthdynten_numerator": numerator,
            "theta_vertical_advection_diabatic_rthdynten_numerator_abs_term_sum": numerator_abs,
            "theta_vertical_advection_diabatic_diabatic_increment": diabatic,
            "theta_vertical_advection_diabatic_diabatic_abs_term_sum": diabatic_abs,
            "theta_vertical_advection_diabatic_tend_theta_abs_term_sum": tend_abs,
            "theta_vertical_advection_diabatic_rthdynten_abs_term_sum": rth_abs,
        }
        self.tend_theta_solve_patch = tend_theta_patch
        self._record(
            "T12_post_theta_vert",
            tend_theta=self._row(
                self.tend_theta,
                self.cell_solve_start,
                self.cell_solve_end,
                tend_theta_patch,
            ),
            rthdynten=self._row(
                self.rthdynten,
                self.cell_solve_start,
                self.cell_solve_end,
                self.rthdynten_solve_patch,
            ),
            **proof,
        )

    def _observe_theta_vertical_mixing_theta_finalization_theta_final(
        self, _kwargs: Mapping[str, Any], result: object
    ) -> None:
        tend_theta_patch, self.tend_theta_euler_cell_patch = cast(
            tuple[torch.Tensor, torch.Tensor | None], result
        )
        euler_solve = (
            self.tend_theta_euler_cell_patch[
                self.cell_solve_start - self.cell_start : self.cell_solve_end
                - self.cell_start,
                :,
            ]
            if self.tend_theta_euler_cell_patch is not None
            else _kwargs["tend_theta_euler_solve_base"]
        )
        self._record(
            "T13_post_theta_final",
            tend_theta=self._row(
                self.tend_theta,
                self.cell_solve_start,
                self.cell_solve_end,
                tend_theta_patch,
            ),
            tend_theta_euler=self._row(
                self.tend_theta_euler,
                self.cell_solve_start,
                self.cell_solve_end,
                euler_solve,
            ),
            rthdynten=self._row(
                self.rthdynten,
                self.cell_solve_start,
                self.cell_solve_end,
                self.rthdynten_solve_patch,
            ),
        )


def _dynamics_tendency_pair_local_metadata(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    row_start: int | None,
    row_end: int | None,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    bool | None,
    bool | None,
]:
    """Precompute relation-local row maps for patch-or-base pair gathers."""
    if row_start is None or row_end is None:
        return None, None, None, None, None, None
    row_start = int(row_start)
    row_end = int(row_end)
    row_span = max(row_end - row_start - 1, 0)
    first_valid = (first >= row_start) & (first < row_end)
    second_valid = (second >= row_start) & (second < row_end)
    first_local = _dynamics_tendency_long_index(
        (first - row_start).clamp(min=0, max=row_span)
    )
    second_local = _dynamics_tendency_long_index(
        (second - row_start).clamp(min=0, max=row_span)
    )
    first_all = _dynamics_tendency_static_all_true(first_valid, allow_cuda_sync=True)
    second_all = _dynamics_tendency_static_all_true(second_valid, allow_cuda_sync=True)
    return first_local, second_local, first_valid, second_valid, first_all, second_all


def _dynamics_tendency_slot_local_metadata(
    indices: torch.Tensor,
    *,
    row_start: int | None,
    row_end: int | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, bool | None]:
    """Precompute relation-local row maps for patch-or-base slot gathers."""
    if row_start is None or row_end is None:
        return None, None, None
    row_start = int(row_start)
    row_end = int(row_end)
    row_span = max(row_end - row_start - 1, 0)
    local_valid = (indices >= row_start) & (indices < row_end)
    local_indices = _dynamics_tendency_long_index(
        (indices - row_start).clamp(min=0, max=row_span)
    )
    local_all = _dynamics_tendency_static_all_true(local_valid, allow_cuda_sync=True)
    return local_indices, local_valid, local_all


def _build_dynamics_tendency_connectivity_payload(
    *,
    device: torch.device,
    nCells: int,
    nEdges: int,
    maxEdges: int,
    maxEdges2: int,
    cell_start: int,
    cell_end: int,
    cell_solve_start: int,
    cell_solve_end: int,
    edge_start: int,
    edge_end: int,
    edge_solve_start: int,
    edge_solve_end: int,
    edgesOnCell: torch.Tensor,
    nEdgesOnCell: torch.Tensor,
    edgesOnCell_sign: torch.Tensor,
    cellsOnEdge: torch.Tensor,
    verticesOnEdge: torch.Tensor,
    edgesOnEdge: torch.Tensor,
    nAdvCellsForEdge: torch.Tensor,
    nEdgesOnEdge: torch.Tensor,
    advCellsForEdge: torch.Tensor,
    adv_coefs: torch.Tensor,
    adv_coefs_3rd: torch.Tensor,
    weightsOnEdge: torch.Tensor,
    vertex_start: int | None = None,
    vertex_end: int | None = None,
    vertexDegree: int | None = None,
    edgesOnVertex: torch.Tensor | None = None,
    edgesOnVertex_sign: torch.Tensor | None = None,
    dvEdge: torch.Tensor | None = None,
    dcEdge: torch.Tensor | None = None,
    invDcEdge: torch.Tensor | None = None,
    invDvEdge: torch.Tensor | None = None,
    invAreaCell: torch.Tensor | None = None,
    invAreaTriangle: torch.Tensor | None = None,
    meshScalingDel2: torch.Tensor | None = None,
    meshScalingDel4: torch.Tensor | None = None,
    build_ordered_ad_transpose: bool = False,
    build_edge_advective_ordered_ad_transpose: bool = False,
) -> dict[str, Any]:
    """Build reusable dynamics-tendency connectivity scaffolding for one mesh/domain slice."""
    cell_idx = torch.arange(cell_start, cell_end, device=device, dtype=torch.int64)
    solve_cells = torch.arange(
        cell_solve_start, cell_solve_end, device=device, dtype=torch.int64
    )
    solve_edge_idx = torch.arange(
        edge_solve_start, edge_solve_end, device=device, dtype=torch.int64
    )
    solve_edge_slice = slice(edge_solve_start, edge_solve_end)
    cell_slot_count = _active_slot_count(nEdgesOnCell[cell_idx], maxEdges, device)
    solve_slot_count = _active_slot_count(nEdgesOnCell[solve_cells], maxEdges, device)
    common_cell_slot_count = max(cell_slot_count, solve_slot_count)
    cell_slots = torch.arange(common_cell_slot_count, device=device, dtype=torch.int64)

    cell_edges_1based = edgesOnCell[cell_idx, :common_cell_slot_count]
    cell_edge_valid = (
        (cell_slots[None, :] < nEdgesOnCell[cell_idx][:, None])
        & (cell_edges_1based >= 1)
        & (cell_edges_1based <= nEdges)
    )
    safe_cell_edges = _dynamics_tendency_long_index(
        torch.where(
            cell_edge_valid, cell_edges_1based - 1, torch.zeros_like(cell_edges_1based)
        )
    )
    cell_edge_sign = edgesOnCell_sign[cell_idx, :common_cell_slot_count]

    solve_edges_1based = edgesOnCell[solve_cells, :common_cell_slot_count]
    solve_edge_valid = (
        (cell_slots[None, :] < nEdgesOnCell[solve_cells][:, None])
        & (solve_edges_1based >= 1)
        & (solve_edges_1based <= nEdges)
    )
    solve_safe_edges = _dynamics_tendency_long_index(
        torch.where(
            solve_edge_valid,
            solve_edges_1based - 1,
            torch.zeros_like(solve_edges_1based),
        )
    )
    solve_edge_sign = edgesOnCell_sign[solve_cells, :common_cell_slot_count]
    solve_edge_cells = gather_rows(cellsOnEdge, solve_safe_edges)
    solve_cell1_1based = solve_edge_cells[:, :, 0]
    solve_cell2_1based = solve_edge_cells[:, :, 1]
    solve_flux_valid = solve_edge_valid & (
        (solve_cell1_1based >= 1)
        & (solve_cell1_1based <= nCells)
        & (solve_cell2_1based >= 1)
        & (solve_cell2_1based <= nCells)
    )
    solve_cell1 = _dynamics_tendency_long_index(
        torch.where(
            solve_flux_valid,
            solve_cell1_1based - 1,
            torch.zeros_like(solve_cell1_1based),
        )
    )
    solve_cell2 = _dynamics_tendency_long_index(
        torch.where(
            solve_flux_valid,
            solve_cell2_1based - 1,
            torch.zeros_like(solve_cell2_1based),
        )
    )

    edge_num_adv_cells = nAdvCellsForEdge[solve_edge_slice]
    full_adv_slot_count = advCellsForEdge.shape[1]
    if (
        edge_num_adv_cells.numel() > 0
        and device.type == "cpu"
        and not _compile_active()
    ):
        adv_slot_count = min(
            full_adv_slot_count, int(edge_num_adv_cells.max().detach().cpu())
        )
    else:
        adv_slot_count = full_adv_slot_count
    adv_slots = torch.arange(adv_slot_count, device=device, dtype=torch.int64)
    edge_adv_cells = advCellsForEdge[solve_edge_slice, :adv_slot_count]
    adv_valid = adv_slots[None, :] < edge_num_adv_cells[:, None]
    adv_valid_weight = adv_valid.to(dtype=DTYPE)
    edge_safe_adv_cells = _dynamics_tendency_long_index(
        torch.where(
            adv_valid,
            edge_adv_cells - 1,
            torch.zeros_like(edge_adv_cells),
        )
    )
    edge_adv_coef = adv_coefs[solve_edge_slice, :adv_slot_count] * adv_valid_weight
    edge_adv_coef_3rd = (
        adv_coefs_3rd[solve_edge_slice, :adv_slot_count] * adv_valid_weight
    )

    edge_num_neighbors = nEdgesOnEdge[solve_edge_slice]
    full_neighbor_slot_count = edgesOnEdge.shape[1]
    if (
        edge_num_neighbors.numel() > 0
        and device.type == "cpu"
        and not _compile_active()
    ):
        neighbor_slot_count = min(
            full_neighbor_slot_count, int(edge_num_neighbors.max().detach().cpu())
        )
    else:
        neighbor_slot_count = full_neighbor_slot_count
    edge_neighbor_slots = torch.arange(
        neighbor_slot_count, device=device, dtype=torch.int64
    )
    edge_neighbors_1based = edgesOnEdge[solve_edge_slice, :neighbor_slot_count]
    edge_eoe_valid = (
        (edge_neighbor_slots[None, :] < edge_num_neighbors[:, None])
        & (edge_neighbors_1based >= 1)
        & (edge_neighbors_1based <= nEdges)
    )
    edge_eoe_safe = _dynamics_tendency_long_index(
        torch.where(
            edge_eoe_valid,
            edge_neighbors_1based - 1,
            torch.zeros_like(edge_neighbors_1based),
        )
    )
    edge_eoe_weights = weightsOnEdge[solve_edge_slice, :neighbor_slot_count]
    edge_eoe_weights = edge_eoe_weights * edge_eoe_valid.to(
        dtype=edge_eoe_weights.dtype
    )

    solve_edge_local_valid = (
        solve_edge_valid
        & (solve_safe_edges >= edge_solve_start)
        & (solve_safe_edges < edge_solve_end)
    )
    solve_local_edges = _dynamics_tendency_long_index(
        torch.where(
            solve_edge_local_valid,
            solve_safe_edges - edge_solve_start,
            torch.zeros_like(solve_safe_edges),
        )
    )

    cell_flux_cells = gather_rows(cellsOnEdge, safe_cell_edges)
    cell_flux1_1based = cell_flux_cells[:, :, 0]
    cell_flux2_1based = cell_flux_cells[:, :, 1]
    cell_flux_valid = cell_edge_valid & (
        (cell_flux1_1based >= 1)
        & (cell_flux1_1based <= nCells)
        & (cell_flux2_1based >= 1)
        & (cell_flux2_1based <= nCells)
    )
    cell_flux1 = _dynamics_tendency_long_index(
        torch.where(
            cell_flux_valid, cell_flux1_1based - 1, torch.zeros_like(cell_flux1_1based)
        )
    )
    cell_flux2 = _dynamics_tendency_long_index(
        torch.where(
            cell_flux_valid, cell_flux2_1based - 1, torch.zeros_like(cell_flux2_1based)
        )
    )

    edge_cell1_1based = cellsOnEdge[solve_edge_slice, 0]
    edge_cell2_1based = cellsOnEdge[solve_edge_slice, 1]
    solve_edge_pair_valid = (
        (edge_cell1_1based >= 1)
        & (edge_cell1_1based <= nCells)
        & (edge_cell2_1based >= 1)
        & (edge_cell2_1based <= nCells)
    )
    edge_c1 = _dynamics_tendency_long_index(
        torch.where(
            solve_edge_pair_valid,
            edge_cell1_1based - 1,
            torch.zeros_like(edge_cell1_1based),
        )
    )
    edge_c2 = _dynamics_tendency_long_index(
        torch.where(
            solve_edge_pair_valid,
            edge_cell2_1based - 1,
            torch.zeros_like(edge_cell2_1based),
        )
    )
    solve_edge_vertex1 = _dynamics_tendency_long_index(
        verticesOnEdge[solve_edge_slice, 0] - 1
    )
    solve_edge_vertex2 = _dynamics_tendency_long_index(
        verticesOnEdge[solve_edge_slice, 1] - 1
    )
    solve_edge_mask = solve_edge_pair_valid.unsqueeze(1).to(dtype=DTYPE)
    solve_edge_mask_all = _dynamics_tendency_static_all_true(
        solve_edge_pair_valid,
        allow_cuda_sync=True,
    )
    solve_edge_valid_weight = solve_edge_local_valid.to(dtype=DTYPE)

    if dvEdge is not None:
        cell_signed_dv = cell_edge_sign * dvEdge[safe_cell_edges]
        solve_signed_dv = solve_edge_sign * dvEdge[solve_safe_edges]
    else:
        cell_signed_dv = None
        solve_signed_dv = None

    if dvEdge is not None and invDcEdge is not None:
        cell_dv_over_dc = dvEdge[safe_cell_edges] * invDcEdge[safe_cell_edges]
        if meshScalingDel4 is not None:
            solve_del4_dv_over_dc = (
                meshScalingDel4[solve_safe_edges]
                * dvEdge[solve_safe_edges]
                * invDcEdge[solve_safe_edges]
            )
        else:
            solve_del4_dv_over_dc = None
    else:
        cell_dv_over_dc = None
        solve_del4_dv_over_dc = None

    if meshScalingDel2 is not None:
        cell_mesh_scaling_del2 = meshScalingDel2[safe_cell_edges]
    else:
        cell_mesh_scaling_del2 = None

    if invAreaCell is not None:
        cell_inv_area = invAreaCell[cell_idx]
    else:
        cell_inv_area = None

    if cell_dv_over_dc is not None and cell_inv_area is not None:
        cell_area_slots = cell_inv_area.unsqueeze(1)
        cell_w_mix_edge_sign = (
            (0.5 * cell_area_slots) * cell_edge_sign * cell_dv_over_dc
        )
        cell_theta_mix_edge_sign = cell_area_slots * cell_edge_sign * cell_dv_over_dc
    else:
        cell_theta_mix_edge_sign = None
        cell_w_mix_edge_sign = None

    if dvEdge is not None and cell_inv_area is not None:
        cell_div_del2_edge_sign = (
            cell_inv_area.unsqueeze(1)
            * dvEdge[safe_cell_edges]
            * cell_edge_sign
            * cell_edge_valid.to(dtype=DTYPE)
        )
    else:
        cell_div_del2_edge_sign = None

    if meshScalingDel4 is not None and invDcEdge is not None and invDvEdge is not None:
        solve_del4_mesh_scaling = meshScalingDel4[solve_edge_idx]
        solve_del4_inv_dc = invDcEdge[solve_edge_idx]
        solve_del4_min_inv_dv_dc = torch.minimum(
            invDvEdge[solve_edge_idx],
            4.0 * invDcEdge[solve_edge_idx],
        )
    else:
        solve_del4_mesh_scaling = None
        solve_del4_inv_dc = None
        solve_del4_min_inv_dv_dc = None

    edge_idx_all = torch.arange(edge_start, edge_end, device=device, dtype=torch.int64)
    edge_all_cell1_1based = cellsOnEdge[edge_idx_all, 0]
    edge_all_cell2_1based = cellsOnEdge[edge_idx_all, 1]
    edge_all_pair_valid = (
        (edge_all_cell1_1based >= 1)
        & (edge_all_cell1_1based <= nCells)
        & (edge_all_cell2_1based >= 1)
        & (edge_all_cell2_1based <= nCells)
    )
    edge_all_c1 = _dynamics_tendency_long_index(
        torch.where(
            edge_all_pair_valid,
            edge_all_cell1_1based - 1,
            torch.zeros_like(edge_all_cell1_1based),
        )
    )
    edge_all_c2 = _dynamics_tendency_long_index(
        torch.where(
            edge_all_pair_valid,
            edge_all_cell2_1based - 1,
            torch.zeros_like(edge_all_cell2_1based),
        )
    )
    edge_all_vertex1 = _dynamics_tendency_long_index(
        verticesOnEdge[edge_idx_all, 0] - 1
    )
    edge_all_vertex2 = _dynamics_tendency_long_index(
        verticesOnEdge[edge_idx_all, 1] - 1
    )

    if (
        vertex_start is not None
        and vertex_end is not None
        and vertexDegree is not None
        and edgesOnVertex is not None
    ):
        vort_vertices = torch.arange(
            int(vertex_start), int(vertex_end), device=device, dtype=torch.int64
        )
        vertex_edges = _dynamics_tendency_long_index(
            edgesOnVertex[vort_vertices, : int(vertexDegree)] - 1
        )
        if (
            edgesOnVertex_sign is not None
            and invAreaTriangle is not None
            and dcEdge is not None
        ):
            vertex_del2_edge_sign = (
                invAreaTriangle[vort_vertices].unsqueeze(1)
                * dcEdge[vertex_edges]
                * edgesOnVertex_sign[vort_vertices, : int(vertexDegree)]
            )
        else:
            vertex_del2_edge_sign = None
    else:
        vort_vertices = None
        vertex_edges = None
        vertex_del2_edge_sign = None

    cell_edge_valid_all_by_slot = _dynamics_tendency_slot_all_true(
        cell_edge_valid,
        allow_cuda_sync=True,
    )
    solve_edge_valid_all_by_slot = _dynamics_tendency_slot_all_true(
        solve_edge_valid,
        allow_cuda_sync=True,
    )
    solve_flux_valid_all_by_slot = _dynamics_tendency_slot_all_true(
        solve_flux_valid,
        allow_cuda_sync=True,
    )
    edge_eoe_valid_all_by_slot = _dynamics_tendency_slot_all_true(
        edge_eoe_valid,
        allow_cuda_sync=True,
    )
    edge_adv_valid_all_by_slot = _dynamics_tendency_slot_all_true(
        adv_valid,
        allow_cuda_sync=True,
    )
    solve_edge_local_valid_all_by_slot = _dynamics_tendency_slot_all_true(
        solve_edge_local_valid,
        allow_cuda_sync=True,
    )
    cell_flux_valid_all_by_slot = _dynamics_tendency_slot_all_true(
        cell_flux_valid,
        allow_cuda_sync=True,
    )
    solve_flux_valid_all = _dynamics_tendency_static_all_true(
        solve_flux_valid,
        allow_cuda_sync=True,
    )
    cell_flux_valid_all = _dynamics_tendency_static_all_true(
        cell_flux_valid,
        allow_cuda_sync=True,
    )
    edge_all_pair_valid_all = _dynamics_tendency_static_all_true(
        edge_all_pair_valid,
        allow_cuda_sync=True,
    )
    if invDcEdge is not None and invDvEdge is not None:
        edge_all_inv_dc = invDcEdge[edge_idx_all]
        edge_all_min_inv_dv_dc = torch.minimum(
            invDvEdge[edge_idx_all],
            4.0 * edge_all_inv_dc,
        )
    else:
        edge_all_inv_dc = None
        edge_all_min_inv_dv_dc = None

    (
        solve_cell1_local,
        solve_cell2_local,
        solve_cell1_local_valid,
        solve_cell2_local_valid,
        solve_cell1_local_all,
        solve_cell2_local_all,
    ) = _dynamics_tendency_pair_local_metadata(
        solve_cell1,
        solve_cell2,
        row_start=cell_start,
        row_end=cell_end,
    )
    (
        cell_flux1_local,
        cell_flux2_local,
        cell_flux1_local_valid,
        cell_flux2_local_valid,
        cell_flux1_local_all,
        cell_flux2_local_all,
    ) = _dynamics_tendency_pair_local_metadata(
        cell_flux1,
        cell_flux2,
        row_start=cell_start,
        row_end=cell_end,
    )
    (
        edge_all_c1_local,
        edge_all_c2_local,
        edge_all_c1_local_valid,
        edge_all_c2_local_valid,
        edge_all_c1_local_all,
        edge_all_c2_local_all,
    ) = _dynamics_tendency_pair_local_metadata(
        edge_all_c1,
        edge_all_c2,
        row_start=cell_start,
        row_end=cell_end,
    )
    (
        edge_all_vertex1_local,
        edge_all_vertex2_local,
        edge_all_vertex1_local_valid,
        edge_all_vertex2_local_valid,
        edge_all_vertex1_local_all,
        edge_all_vertex2_local_all,
    ) = _dynamics_tendency_pair_local_metadata(
        edge_all_vertex1,
        edge_all_vertex2,
        row_start=vertex_start,
        row_end=vertex_end,
    )
    (
        edge_c1_local,
        edge_c2_local,
        edge_c1_local_valid,
        edge_c2_local_valid,
        edge_c1_local_all,
        edge_c2_local_all,
    ) = _dynamics_tendency_pair_local_metadata(
        edge_c1,
        edge_c2,
        row_start=cell_start,
        row_end=cell_end,
    )
    (
        solve_edge_vertex1_local,
        solve_edge_vertex2_local,
        solve_edge_vertex1_local_valid,
        solve_edge_vertex2_local_valid,
        solve_edge_vertex1_local_all,
        solve_edge_vertex2_local_all,
    ) = _dynamics_tendency_pair_local_metadata(
        solve_edge_vertex1,
        solve_edge_vertex2,
        row_start=vertex_start,
        row_end=vertex_end,
    )
    cell_edge_local, cell_edge_local_valid, cell_edge_local_all = (
        _dynamics_tendency_slot_local_metadata(
            safe_cell_edges,
            row_start=edge_start,
            row_end=edge_end,
        )
    )
    solve_edge_slot_local, solve_edge_slot_local_valid, solve_edge_slot_local_all = (
        _dynamics_tendency_slot_local_metadata(
            solve_safe_edges,
            row_start=edge_start,
            row_end=edge_end,
        )
    )
    vertex_edge_local: torch.Tensor | None = None
    vertex_edge_local_valid: torch.Tensor | None = None
    vertex_edge_local_all: bool | None = None
    if vertex_edges is not None:
        vertex_edge_local, vertex_edge_local_valid, vertex_edge_local_all = (
            _dynamics_tendency_slot_local_metadata(
                vertex_edges,
                row_start=edge_start,
                row_end=edge_end,
            )
        )
    if build_ordered_ad_transpose:
        edge_neighbor_transpose_destinations, edge_neighbor_transpose_valid = (
            build_ordered_output_transpose(
                edge_eoe_safe,
                nEdges + 1,
                valid=edge_eoe_valid,
            )
        )
        edge_c1_transpose_destinations, edge_c1_transpose_valid = (
            build_ordered_output_transpose(
                edge_c1,
                nCells + 1,
                valid=solve_edge_pair_valid,
            )
        )
        edge_c2_transpose_destinations, edge_c2_transpose_valid = (
            build_ordered_output_transpose(
                edge_c2,
                nCells + 1,
                valid=solve_edge_pair_valid,
            )
        )
    else:
        edge_neighbor_transpose_destinations = None
        edge_neighbor_transpose_valid = None
        edge_c1_transpose_destinations = None
        edge_c1_transpose_valid = None
        edge_c2_transpose_destinations = None
        edge_c2_transpose_valid = None
    if build_edge_advective_ordered_ad_transpose:
        edge_adv_transpose_destinations, edge_adv_transpose_valid = (
            build_ordered_output_transpose(
                edge_safe_adv_cells,
                nCells + 1,
                valid=adv_valid,
            )
        )
        solve_local_transpose_destinations, solve_local_transpose_valid = (
            build_ordered_output_transpose(
                solve_local_edges,
                edge_solve_end - edge_solve_start,
                valid=solve_edge_local_valid,
            )
        )
    else:
        edge_adv_transpose_destinations = None
        edge_adv_transpose_valid = None
        solve_local_transpose_destinations = None
        solve_local_transpose_valid = None
    cell_edge_relation = StaticSlotRelation(
        "dynamics_tendency.cell_edges",
        indices=safe_cell_edges,
        valid=cell_edge_valid,
        sign=cell_edge_sign,
        weights=cell_signed_dv,
        valid_all_by_slot=cell_edge_valid_all_by_slot,
        local_indices=cell_edge_local,
        local_valid=cell_edge_local_valid,
        local_all=cell_edge_local_all,
    )
    cell_flux_edge_relation = StaticSlotRelation(
        "dynamics_tendency.cell_flux_edges",
        indices=safe_cell_edges,
        valid=cell_flux_valid,
        sign=cell_edge_sign,
        valid_all_by_slot=cell_flux_valid_all_by_slot,
        local_indices=cell_edge_local,
        local_valid=cell_edge_local_valid,
        local_all=cell_edge_local_all,
    )
    solve_cell_edge_relation = StaticSlotRelation(
        "dynamics_tendency.solve_cell_edges",
        indices=solve_safe_edges,
        valid=solve_edge_valid,
        sign=solve_edge_sign,
        weights=solve_signed_dv,
        valid_all_by_slot=solve_edge_valid_all_by_slot,
        local_indices=solve_edge_slot_local,
        local_valid=solve_edge_slot_local_valid,
        local_all=solve_edge_slot_local_all,
    )
    solve_local_edge_relation = StaticSlotRelation(
        "dynamics_tendency.solve_local_edges",
        indices=solve_local_edges,
        valid=solve_edge_local_valid,
        sign=solve_edge_sign,
        valid_all_by_slot=solve_edge_local_valid_all_by_slot,
        valid_weight=solve_edge_valid_weight,
        transpose_destinations=solve_local_transpose_destinations,
        transpose_valid=solve_local_transpose_valid,
    )
    edge_adv_cell_relation = StaticSlotRelation(
        "dynamics_tendency.edge_adv_cells",
        indices=edge_safe_adv_cells,
        valid=adv_valid,
        coefficients=edge_adv_coef,
        coefficients_3rd=edge_adv_coef_3rd,
        valid_all_by_slot=edge_adv_valid_all_by_slot,
        valid_weight=adv_valid_weight,
        transpose_destinations=edge_adv_transpose_destinations,
        transpose_valid=edge_adv_transpose_valid,
    )
    edge_neighbor_relation = StaticSlotRelation(
        "dynamics_tendency.edge_neighbors",
        indices=edge_eoe_safe,
        valid=edge_eoe_valid,
        weights=edge_eoe_weights,
        valid_all_by_slot=edge_eoe_valid_all_by_slot,
        transpose_destinations=edge_neighbor_transpose_destinations,
        transpose_valid=edge_neighbor_transpose_valid,
    )
    solve_flux_cell_relation = StaticPairRelation(
        "dynamics_tendency.solve_flux_cells",
        first=solve_cell1,
        second=solve_cell2,
        valid=solve_flux_valid,
        valid_all=solve_flux_valid_all,
        first_local=solve_cell1_local,
        second_local=solve_cell2_local,
        first_local_valid=solve_cell1_local_valid,
        second_local_valid=solve_cell2_local_valid,
        first_local_all=solve_cell1_local_all,
        second_local_all=solve_cell2_local_all,
    )
    cell_flux_pair_relation = StaticPairRelation(
        "dynamics_tendency.cell_flux_cells",
        first=cell_flux1,
        second=cell_flux2,
        valid=cell_flux_valid,
        valid_all=cell_flux_valid_all,
        first_local=cell_flux1_local,
        second_local=cell_flux2_local,
        first_local_valid=cell_flux1_local_valid,
        second_local_valid=cell_flux2_local_valid,
        first_local_all=cell_flux1_local_all,
        second_local_all=cell_flux2_local_all,
    )
    edge_all_pair_relation = StaticPairRelation(
        "dynamics_tendency.edge_all_cells",
        first=edge_all_c1,
        second=edge_all_c2,
        valid=edge_all_pair_valid,
        valid_all=edge_all_pair_valid_all,
        first_local=edge_all_c1_local,
        second_local=edge_all_c2_local,
        first_local_valid=edge_all_c1_local_valid,
        second_local_valid=edge_all_c2_local_valid,
        first_local_all=edge_all_c1_local_all,
        second_local_all=edge_all_c2_local_all,
    )
    edge_all_vertex_relation = StaticPairRelation(
        "dynamics_tendency.edge_all_vertices",
        first=edge_all_vertex1,
        second=edge_all_vertex2,
        valid=edge_all_pair_valid,
        valid_all=edge_all_pair_valid_all,
        first_local=edge_all_vertex1_local,
        second_local=edge_all_vertex2_local,
        first_local_valid=edge_all_vertex1_local_valid,
        second_local_valid=edge_all_vertex2_local_valid,
        first_local_all=edge_all_vertex1_local_all,
        second_local_all=edge_all_vertex2_local_all,
    )
    solve_edge_pair_relation = StaticPairRelation(
        "dynamics_tendency.solve_edge_cells",
        first=edge_c1,
        second=edge_c2,
        valid=solve_edge_pair_valid,
        valid_2d=solve_edge_mask,
        valid_all=solve_edge_mask_all,
        first_local=edge_c1_local,
        second_local=edge_c2_local,
        first_local_valid=edge_c1_local_valid,
        second_local_valid=edge_c2_local_valid,
        first_local_all=edge_c1_local_all,
        second_local_all=edge_c2_local_all,
        first_transpose_destinations=edge_c1_transpose_destinations,
        first_transpose_valid=edge_c1_transpose_valid,
        second_transpose_destinations=edge_c2_transpose_destinations,
        second_transpose_valid=edge_c2_transpose_valid,
    )
    solve_edge_vertex_relation = StaticPairRelation(
        "dynamics_tendency.solve_edge_vertices",
        first=solve_edge_vertex1,
        second=solve_edge_vertex2,
        valid=solve_edge_pair_valid,
        valid_all=solve_edge_mask_all,
        first_local=solve_edge_vertex1_local,
        second_local=solve_edge_vertex2_local,
        first_local_valid=solve_edge_vertex1_local_valid,
        second_local_valid=solve_edge_vertex2_local_valid,
        first_local_all=solve_edge_vertex1_local_all,
        second_local_all=solve_edge_vertex2_local_all,
    )
    vertex_edge_relation = None
    if vertex_edges is not None:
        vertex_edge_relation = StaticSlotRelation(
            "dynamics_tendency.vertex_edges",
            indices=vertex_edges,
            weights=vertex_del2_edge_sign,
            local_indices=vertex_edge_local,
            local_valid=vertex_edge_local_valid,
            local_all=vertex_edge_local_all,
        )

    return {
        "edge_idx_all": edge_idx_all,
        "cell_idx": cell_idx,
        "solve_cells": solve_cells,
        "solve_edge_idx": solve_edge_idx,
        "solve_edge_slice": solve_edge_slice,
        "cell_edge_valid": cell_edge_valid,
        "cell_edge_valid_all_by_slot": cell_edge_valid_all_by_slot,
        "safe_cell_edges": safe_cell_edges,
        "cell_edge_sign": cell_edge_sign,
        "cell_signed_dv": cell_signed_dv,
        "cell_edge_relation": cell_edge_relation,
        "cell_flux_edge_relation": cell_flux_edge_relation,
        "solve_edge_valid": solve_edge_valid,
        "solve_edge_valid_all_by_slot": solve_edge_valid_all_by_slot,
        "solve_safe_edges": solve_safe_edges,
        "solve_edge_sign": solve_edge_sign,
        "solve_signed_dv": solve_signed_dv,
        "solve_cell_edge_relation": solve_cell_edge_relation,
        "solve_flux_valid": solve_flux_valid,
        "solve_flux_valid_all_by_slot": solve_flux_valid_all_by_slot,
        "solve_cell1": solve_cell1,
        "solve_cell2": solve_cell2,
        "solve_flux_cell_relation": solve_flux_cell_relation,
        "edge_safe_adv_cells": edge_safe_adv_cells,
        "edge_adv_coef": edge_adv_coef,
        "edge_adv_coef_3rd": edge_adv_coef_3rd,
        "edge_adv_cell_relation": edge_adv_cell_relation,
        "edge_eoe_valid": edge_eoe_valid,
        "edge_eoe_valid_all_by_slot": edge_eoe_valid_all_by_slot,
        "edge_eoe_safe": edge_eoe_safe,
        "edge_eoe_weights": edge_eoe_weights,
        "edge_neighbor_relation": edge_neighbor_relation,
        "solve_local_edges": solve_local_edges,
        "solve_edge_local_valid_all_by_slot": solve_edge_local_valid_all_by_slot,
        "solve_local_edge_relation": solve_local_edge_relation,
        "cell_flux_valid": cell_flux_valid,
        "cell_flux_valid_all_by_slot": cell_flux_valid_all_by_slot,
        "cell_flux1": cell_flux1,
        "cell_flux2": cell_flux2,
        "cell_flux_pair_relation": cell_flux_pair_relation,
        "edge_all_pair_valid": edge_all_pair_valid,
        "edge_all_c1": edge_all_c1,
        "edge_all_c2": edge_all_c2,
        "edge_all_pair_relation": edge_all_pair_relation,
        "edge_all_vertex1": edge_all_vertex1,
        "edge_all_vertex2": edge_all_vertex2,
        "edge_all_vertex_relation": edge_all_vertex_relation,
        "solve_edge_pair_valid": solve_edge_pair_valid,
        "edge_c1": edge_c1,
        "edge_c2": edge_c2,
        "solve_edge_pair_relation": solve_edge_pair_relation,
        "solve_edge_vertex1": solve_edge_vertex1,
        "solve_edge_vertex2": solve_edge_vertex2,
        "solve_edge_vertex_relation": solve_edge_vertex_relation,
        "solve_edge_mask": solve_edge_mask,
        "solve_edge_mask_all": solve_edge_mask_all,
        "solve_edge_valid_weight": solve_edge_valid_weight,
        "cell_dv_over_dc": cell_dv_over_dc,
        "cell_mesh_scaling_del2": cell_mesh_scaling_del2,
        "solve_del4_dv_over_dc": solve_del4_dv_over_dc,
        "cell_w_mix_edge_sign": cell_w_mix_edge_sign,
        "cell_theta_mix_edge_sign": cell_theta_mix_edge_sign,
        "cell_div_del2_edge_sign": cell_div_del2_edge_sign,
        "solve_del4_mesh_scaling": solve_del4_mesh_scaling,
        "solve_del4_inv_dc": solve_del4_inv_dc,
        "solve_del4_min_inv_dv_dc": solve_del4_min_inv_dv_dc,
        "edge_all_inv_dc": edge_all_inv_dc,
        "edge_all_min_inv_dv_dc": edge_all_min_inv_dv_dc,
        "vort_vertices": vort_vertices,
        "vertex_edges": vertex_edges,
        "vertex_edge_relation": vertex_edge_relation,
        "vertex_del2_edge_sign": vertex_del2_edge_sign,
        "cell_slot_count": common_cell_slot_count,
    }


def _dynamics_tendency_connectivity_payload(**kwargs: Any) -> dict[str, Any]:
    """Return cached dynamics-tendency connectivity scaffolding when topology is static."""
    key = _dynamics_tendency_connectivity_payload_cache_key(**kwargs)
    if key is None:
        return _build_dynamics_tendency_connectivity_payload(**kwargs)  # type: ignore[arg-type]
    cached = _DYNAMICS_TENDENCY_CONNECTIVITY_CACHE.get(key)
    if cached is not None:
        return cached
    payload = _build_dynamics_tendency_connectivity_payload(**kwargs)  # type: ignore[arg-type]
    if len(_DYNAMICS_TENDENCY_CONNECTIVITY_CACHE) >= 4:
        _DYNAMICS_TENDENCY_CONNECTIVITY_CACHE.pop(
            next(iter(_DYNAMICS_TENDENCY_CONNECTIVITY_CACHE))
        )
    _DYNAMICS_TENDENCY_CONNECTIVITY_CACHE[key] = payload
    return payload


@dataclass(frozen=True)
class DynamicsTendencySectionOps:
    """Immutable call table for the canonical mixing-initialization-theta-finalization numeric composition."""

    mixing_initialization_mixing_initialization: Callable[..., Any]
    density_tendency_density: Callable[..., Any]
    normal_velocity_tendency_normal_velocity: Callable[..., Any]
    normal_velocity_mixing_u_mixing: Callable[..., Any]
    normal_velocity_finalization_u_final: Callable[..., Any]
    vertical_velocity_horizontal_advection_w_hadv: Callable[..., Any]
    vertical_velocity_horizontal_mixing_w_horizontal_mix: Callable[..., Any]
    vertical_velocity_vertical_transport_w_vertical: Callable[..., Any]
    theta_horizontal_advection_theta_hadv: Callable[..., Any]
    theta_horizontal_mixing_theta_horizontal_mix: Callable[..., Any]
    theta_vertical_advection_diabatic_theta_vertical: Callable[..., Any]
    theta_vertical_mixing_theta_finalization_theta_final: Callable[..., Any]
    normal_velocity_mixing_normal_velocity_finalization_u_tendency: (
        Callable[..., Any] | None
    ) = None
    vertical_velocity_horizontal_mixing_vertical_velocity_vertical_transport_w_tendency: (
        Callable[..., Any] | None
    ) = None
    theta_horizontal_advection_theta_finalization_theta_tendency: (
        Callable[..., Any] | None
    ) = None


DYNAMICS_TENDENCY_SECTION_CALL_SEQUENCE = (
    "mixing_initialization_mixing_initialization",
    "density_tendency_density",
    "normal_velocity_tendency_normal_velocity",
    "normal_velocity_mixing_u_mixing",
    "normal_velocity_finalization_u_final",
    "vertical_velocity_horizontal_advection_w_hadv",
    "vertical_velocity_horizontal_mixing_w_horizontal_mix",
    "vertical_velocity_vertical_transport_w_vertical",
    "theta_horizontal_advection_theta_hadv",
    "theta_horizontal_mixing_theta_horizontal_mix",
    "theta_vertical_advection_diabatic_theta_vertical",
    "theta_vertical_mixing_theta_finalization_theta_final",
)


_CANONICAL_DYNAMICS_TENDENCY_SECTION_OPS = DynamicsTendencySectionOps(
    mixing_initialization_mixing_initialization=_mixing_initialization_mixing_initialization_region,
    density_tendency_density=_density_tendency_density_section_region,
    normal_velocity_tendency_normal_velocity=_normal_velocity_tendency_normal_velocity_owned_section_region,
    normal_velocity_mixing_u_mixing=_normal_velocity_mixing_u_mixing_section_region,
    normal_velocity_finalization_u_final=_normal_velocity_finalization_u_final_section_region,
    vertical_velocity_horizontal_advection_w_hadv=_vertical_velocity_horizontal_advection_w_hadv_owned_section_region,
    vertical_velocity_horizontal_mixing_w_horizontal_mix=_vertical_velocity_horizontal_mixing_w_horizontal_mix_section_region,
    vertical_velocity_vertical_transport_w_vertical=_vertical_velocity_vertical_transport_w_vertical_section_region,
    theta_horizontal_advection_theta_hadv=_theta_horizontal_advection_theta_hadv_owned_section_region,
    theta_horizontal_mixing_theta_horizontal_mix=_theta_horizontal_mixing_theta_horizontal_mix_section_region,
    theta_vertical_advection_diabatic_theta_vertical=_theta_vertical_advection_diabatic_theta_vertical_section_region,
    theta_vertical_mixing_theta_finalization_theta_final=_theta_vertical_mixing_theta_finalization_theta_final_section_region,
    normal_velocity_mixing_normal_velocity_finalization_u_tendency=_normal_velocity_mixing_normal_velocity_finalization_u_tendency_region,
    vertical_velocity_horizontal_mixing_vertical_velocity_vertical_transport_w_tendency=_vertical_velocity_horizontal_mixing_vertical_velocity_vertical_transport_w_tendency_region,
    theta_horizontal_advection_theta_finalization_theta_tendency=_theta_horizontal_advection_theta_finalization_theta_tendency_region,
)

_DYNAMICS_TENDENCY_NORMAL_VELOCITY_MIXING_NORMAL_VELOCITY_FINALIZATION_COMPILE_NAME = "dynamics_tendency_normal_velocity_mixing_normal_velocity_finalization_u_tendency_region"
_DYNAMICS_TENDENCY_VERTICAL_VELOCITY_HORIZONTAL_MIXING_VERTICAL_VELOCITY_VERTICAL_TRANSPORT_COMPILE_NAME = "dynamics_tendency_vertical_velocity_horizontal_mixing_vertical_velocity_vertical_transport_w_tendency_region"
_DYNAMICS_TENDENCY_THETA_HORIZONTAL_ADVECTION_THETA_FINALIZATION_COMPILE_NAME = "dynamics_tendency_theta_horizontal_advection_theta_finalization_theta_tendency_region"

_DYNAMICS_TENDENCY_SECTION_COMPILE_NAMES = MappingProxyType(
    {
        "normal_velocity_tendency_normal_velocity": "dynamics_tendency_normal_velocity_tendency_normal_velocity_section",
        "normal_velocity_mixing_u_mixing": "dynamics_tendency_normal_velocity_mixing_u_mixing_section",
        "normal_velocity_finalization_u_final": "dynamics_tendency_normal_velocity_finalization_u_final_section",
        "vertical_velocity_horizontal_advection_w_hadv": "dynamics_tendency_vertical_velocity_horizontal_advection_w_hadv_section",
        "vertical_velocity_horizontal_mixing_w_horizontal_mix": "dynamics_tendency_vertical_velocity_horizontal_mixing_w_horizontal_mix_section",
        "vertical_velocity_vertical_transport_w_vertical": "dynamics_tendency_vertical_velocity_vertical_transport_w_vertical_section",
        "theta_horizontal_advection_theta_hadv": "dynamics_tendency_theta_horizontal_advection_theta_hadv_section",
        "theta_horizontal_mixing_theta_horizontal_mix": "dynamics_tendency_theta_horizontal_mixing_theta_horizontal_mix_section",
        "theta_vertical_advection_diabatic_theta_vertical": "dynamics_tendency_theta_vertical_advection_diabatic_theta_vertical_section",
        "theta_vertical_mixing_theta_finalization_theta_final": "dynamics_tendency_theta_vertical_mixing_theta_finalization_theta_final_section",
    }
)

_DYNAMICS_TENDENCY_SECTION_PROFILE_NAMES = MappingProxyType(
    {
        "density_tendency_density": "mpas.dynamics_tendency.density_divergence",
        "normal_velocity_tendency_normal_velocity": "mpas.dynamics_tendency.normal_velocity_tendency",
        "normal_velocity_mixing_u_mixing": "mpas.dynamics_tendency.normal_velocity_mixing",
        "normal_velocity_finalization_u_final": "mpas.dynamics_tendency.normal_velocity_finalization",
        "vertical_velocity_horizontal_advection_w_hadv": "mpas.dynamics_tendency.vertical_velocity_horizontal_advection",
        "vertical_velocity_horizontal_mixing_w_horizontal_mix": "mpas.dynamics_tendency.vertical_velocity_horizontal_mixing",
        "vertical_velocity_vertical_transport_w_vertical": "mpas.dynamics_tendency.vertical_velocity_vertical_transport",
        "theta_horizontal_advection_theta_hadv": "mpas.dynamics_tendency.theta_horizontal_advection",
        "theta_horizontal_mixing_theta_horizontal_mix": "mpas.dynamics_tendency.theta_horizontal_mixing",
        "theta_vertical_advection_diabatic_theta_vertical": "mpas.dynamics_tendency.theta_vertical_advection_diabatic",
        "theta_vertical_mixing_theta_finalization_theta_final": "mpas.dynamics_tendency.theta_vertical_mixing_and_finalization",
    }
)


def _observe_dynamics_tendency_vertical_velocity_horizontal_mixing_vertical_velocity_vertical_transport_parent(
    inner: Callable[..., Any],
    observer: Callable[
        [str, tuple[object, ...], Mapping[str, Any], object, float], None
    ],
) -> Callable[..., Any]:
    """Expose vertical-velocity-horizontal-mixing/vertical-velocity-vertical-transport proof taps around the production parent without replacing it."""

    @wraps(inner)
    def observed(*args, **kwargs):
        started = perf_counter()
        result = inner(*args, **kwargs)
        elapsed = perf_counter() - started
        vertical_velocity_horizontal_mixing_keys = (
            "cell_edge_relation",
            "edge_cell_relation",
            "solve_cell_pair_relation",
            "edge_sign_del2",
            "edge_sign_del4",
            "rho_edge",
            "w",
            "kdiff",
            "delsq_w_base",
            "n_vert_levels",
            "patch_row_start",
            "patch_row_end",
            "solve_relative_start",
            "solve_relative_end",
            "solve_flux_valid_all_by_slot",
            "kdiff_patch",
            "mesh_scaling_del2",
            "mesh_scaling_del2_slots",
        )
        vertical_velocity_horizontal_mixing_kwargs = {
            name: kwargs[name] for name in vertical_velocity_horizontal_mixing_keys
        }
        vertical_velocity_horizontal_mixing_result = (
            _vertical_velocity_horizontal_mixing_w_horizontal_mix_section_region(
                **vertical_velocity_horizontal_mixing_kwargs
            )
        )
        observer(
            "vertical_velocity_horizontal_mixing_w_horizontal_mix",
            (),
            vertical_velocity_horizontal_mixing_kwargs,
            vertical_velocity_horizontal_mixing_result,
            elapsed,
        )
        row_start = kwargs["patch_row_start"] + kwargs["solve_relative_start"]
        row_end = kwargs["patch_row_start"] + kwargs["solve_relative_end"]
        n_levels = kwargs["n_vert_levels"]
        vertical_velocity_vertical_transport_kwargs = {
            "w_solve": kwargs["w"][row_start:row_end, :],
            "rw_solve": kwargs["rw_solve"],
            "pp_solve": kwargs["pp_solve"],
            "cqw_solve": kwargs["cqw_solve"],
            "rho_zz_solve": kwargs["rho_zz_solve"],
            "dpdz_solve": kwargs["dpdz_solve"],
            "tend_w_solve_inner": kwargs["tend_w_solve_inner"],
            "tend_w_euler_solve_inner": vertical_velocity_horizontal_mixing_result[
                kwargs["solve_relative_start"] : kwargs["solve_relative_end"],
                1:n_levels,
            ],
            "inv_area_cell_solve": kwargs["inv_area_cell_solve"],
            "rdzu": kwargs["rdzu"],
            "rdzw": kwargs["rdzw"],
            "fzm": kwargs["fzm"],
            "fzp": kwargs["fzp"],
            "n_vert_levels": n_levels,
            "include_pressure_gradient": kwargs["include_pressure_gradient"],
            "include_vertical_mix": kwargs["include_vertical_mix"],
            "v_mom_eddy_visc2": kwargs["v_mom_eddy_visc2"],
        }
        tend_w_final, euler_cell_final = result
        vertical_velocity_vertical_transport_result = (
            tend_w_final,
            euler_cell_final[
                kwargs["solve_relative_start"] : kwargs["solve_relative_end"],
                1:n_levels,
            ],
        )
        observer(
            "vertical_velocity_vertical_transport_w_vertical",
            (),
            vertical_velocity_vertical_transport_kwargs,
            vertical_velocity_vertical_transport_result,
            0.0,
        )
        return result

    return observed


def _observe_dynamics_tendency_theta_horizontal_advection_theta_finalization_parent(
    inner: Callable[..., Any],
    observer: Callable[
        [str, tuple[object, ...], Mapping[str, Any], object, float], None
    ],
) -> Callable[..., Any]:
    """Expose theta proof taps without replacing the production parent call."""

    @wraps(inner)
    def observed(*args, **kwargs):
        started = perf_counter()
        result = inner(*args, **kwargs)
        elapsed = perf_counter() - started

        theta_horizontal_advection_keys = (
            "theta_m",
            "ru",
            "edge_solve_start",
            "edge_solve_end",
            "n_vert_levels",
            "edge_adv_cell_relation",
            "local_edge_relation",
            "include_perturbation_flux",
            "solve_cell_edge_relation",
            "solve_edge_cell_relation",
            "ru_save",
            "theta_m_save",
            "cell_owned_vertical_velocity_horizontal_advection_theta_horizontal_advection_ab",
        )
        theta_horizontal_advection_kwargs = {
            name: kwargs[name] for name in theta_horizontal_advection_keys
        }
        theta_horizontal_advection_kwargs["cell_edge_relation"] = (
            theta_horizontal_advection_kwargs.pop("solve_cell_edge_relation")
        )
        theta_horizontal_advection_kwargs["edge_cell_relation"] = (
            theta_horizontal_advection_kwargs.pop("solve_edge_cell_relation")
        )
        theta_horizontal_advection_result = (
            _theta_horizontal_advection_theta_hadv_owned_section_region(
                **theta_horizontal_advection_kwargs
            )
        )
        observer(
            "theta_horizontal_advection_theta_hadv",
            (),
            theta_horizontal_advection_kwargs,
            theta_horizontal_advection_result,
            elapsed,
        )

        theta_horizontal_mixing_result = None
        if kwargs["include_horizontal_mix"]:
            theta_horizontal_mixing_kwargs = {
                "cell_edge_relation": kwargs["mix_cell_edge_relation"],
                "edge_cell_relation": kwargs["mix_edge_cell_relation"],
                "solve_cell_pair_relation": kwargs["solve_cell_pair_relation"],
                "edge_sign_del2": kwargs["edge_sign_del2"],
                "edge_sign_del4": kwargs["edge_sign_del4"],
                "rho_edge": kwargs["rho_edge"],
                "theta_m": kwargs["theta_m"],
                "kdiff": kwargs["kdiff"],
                "delsq_theta_base": kwargs["delsq_theta_base"],
                "pr_scale": kwargs["pr_scale"],
                "patch_row_start": kwargs["patch_row_start"],
                "patch_row_end": kwargs["patch_row_end"],
                "solve_relative_start": kwargs["solve_relative_start"],
                "solve_relative_end": kwargs["solve_relative_end"],
                "solve_flux_valid_all_by_slot": kwargs["solve_flux_valid_all_by_slot"],
                "kdiff_patch": kwargs["kdiff_patch"],
            }
            theta_horizontal_mixing_result = (
                _theta_horizontal_mixing_theta_horizontal_mix_section_region(
                    **theta_horizontal_mixing_kwargs
                )
            )
            observer(
                "theta_horizontal_mixing_theta_horizontal_mix",
                (),
                theta_horizontal_mixing_kwargs,
                theta_horizontal_mixing_result,
                0.0,
            )

        theta_vertical_advection_diabatic_keys = (
            "theta_m_solve",
            "theta_m_save_solve",
            "rw_solve",
            "rw_save_solve",
            "rho_zz_solve",
            "rt_diabatic_tend_solve",
            "tend_rho_solve",
            "inv_area_cell_solve",
            "rdzw",
            "fzm",
            "fzp",
            "coef_3rd_order",
            "n_vert_levels",
        )
        theta_vertical_advection_diabatic_kwargs = {
            name: kwargs[name] for name in theta_vertical_advection_diabatic_keys
        }
        theta_vertical_advection_diabatic_kwargs["tend_theta_solve"] = (
            theta_horizontal_advection_result
        )
        theta_vertical_advection_diabatic_result = (
            _theta_vertical_advection_diabatic_theta_vertical_section_region(
                **theta_vertical_advection_diabatic_kwargs
            )
        )
        observer(
            "theta_vertical_advection_diabatic_theta_vertical",
            (),
            theta_vertical_advection_diabatic_kwargs,
            theta_vertical_advection_diabatic_result,
            0.0,
        )

        theta_vertical_mixing_keys = (
            "tend_theta_euler_cell_base",
            "tend_theta_euler_solve_base",
            "theta_work_solve",
            "rho_zz_solve",
            "zgrid_solve",
            "tend_rtheta_physics_solve",
            "apply_vertical_mix",
            "v_theta_eddy_visc2",
            "prandtl_inv",
            "n_vert_levels",
        )
        theta_vertical_mixing_kwargs = {
            name: kwargs[name] for name in theta_vertical_mixing_keys
        }
        theta_vertical_mixing_kwargs.update(
            {
                "tend_theta_solve": theta_vertical_advection_diabatic_result[0],
                "tend_theta_euler_cell_patch": theta_horizontal_mixing_result,
                "cell_solve_rel_start": kwargs["solve_relative_start"],
                "cell_solve_rel_end": kwargs["solve_relative_end"],
            }
        )
        theta_vertical_mixing_result = (
            _theta_vertical_mixing_theta_finalization_theta_final_section_region(
                **theta_vertical_mixing_kwargs
            )
        )
        observer(
            "theta_vertical_mixing_theta_finalization_theta_final",
            (),
            theta_vertical_mixing_kwargs,
            theta_vertical_mixing_result,
            0.0,
        )

        expected = (
            theta_vertical_mixing_result[0],
            theta_vertical_mixing_result[1],
            theta_vertical_advection_diabatic_result[1],
        )
        for actual_value, expected_value in zip(result, expected):
            if actual_value is None or expected_value is None:
                if actual_value is not expected_value:
                    raise RuntimeError(
                        "dynamics-tendency theta-horizontal-advection-theta-finalization parent changed a nullable live output"
                    )
            elif not torch.equal(actual_value, expected_value):
                raise RuntimeError(
                    "dynamics-tendency theta-horizontal-advection-theta-finalization parent differs from its section sequence"
                )
        return result

    return observed


def build_dynamics_tendency_section_ops(
    *,
    compile_region_names: tuple[str, ...] = (),
    compile_mode: str = "reduce-overhead",
    compile_backend: str = "inductor",
    stage_key: tuple[object, ...] = (),
    profile_sections: bool = False,
    section_observer: (
        Callable[[str, tuple[object, ...], Mapping[str, Any], object, float], None]
        | None
    ) = None,
) -> DynamicsTendencySectionOps:
    """Build runner-owned wrappers around the one canonical dynamics-tendency equation table."""
    requested = frozenset(compile_region_names)
    allowed = frozenset(_DYNAMICS_TENDENCY_SECTION_COMPILE_NAMES.values()) | {
        _DYNAMICS_TENDENCY_NORMAL_VELOCITY_MIXING_NORMAL_VELOCITY_FINALIZATION_COMPILE_NAME,
        _DYNAMICS_TENDENCY_VERTICAL_VELOCITY_HORIZONTAL_MIXING_VERTICAL_VELOCITY_VERTICAL_TRANSPORT_COMPILE_NAME,
        _DYNAMICS_TENDENCY_THETA_HORIZONTAL_ADVECTION_THETA_FINALIZATION_COMPILE_NAME,
    }
    unknown = sorted(requested - allowed)
    if unknown:
        raise ValueError(
            f"unknown dynamics-tendency section compile targets: {unknown}"
        )
    if not requested and not profile_sections and section_observer is None:
        return _CANONICAL_DYNAMICS_TENDENCY_SECTION_OPS

    from mpas_pytorch.dynamics.compile_utils import maybe_compile_kernel

    compile_normal_velocity_mixing_normal_velocity_finalization = (
        _DYNAMICS_TENDENCY_NORMAL_VELOCITY_MIXING_NORMAL_VELOCITY_FINALIZATION_COMPILE_NAME
        in requested
    )
    if (
        compile_normal_velocity_mixing_normal_velocity_finalization
        and requested.intersection(
            {
                _DYNAMICS_TENDENCY_SECTION_COMPILE_NAMES[
                    "normal_velocity_mixing_u_mixing"
                ],
            }
        )
    ):
        raise ValueError(
            "normal-velocity-mixing+normal-velocity-finalization parent compilation cannot be mixed with the normal-velocity-mixing child target"
        )
    compile_vertical_velocity_horizontal_mixing_vertical_velocity_vertical_transport = (
        _DYNAMICS_TENDENCY_VERTICAL_VELOCITY_HORIZONTAL_MIXING_VERTICAL_VELOCITY_VERTICAL_TRANSPORT_COMPILE_NAME
        in requested
    )
    if (
        compile_vertical_velocity_horizontal_mixing_vertical_velocity_vertical_transport
        and requested.intersection(
            {
                _DYNAMICS_TENDENCY_SECTION_COMPILE_NAMES[
                    "vertical_velocity_horizontal_mixing_w_horizontal_mix"
                ],
                _DYNAMICS_TENDENCY_SECTION_COMPILE_NAMES[
                    "vertical_velocity_vertical_transport_w_vertical"
                ],
            }
        )
    ):
        raise ValueError(
            "vertical-velocity-horizontal-mixing+vertical-velocity-vertical-transport parent compilation cannot be mixed with vertical-velocity-horizontal-mixing/vertical-velocity-vertical-transport child targets"
        )
    compile_theta_horizontal_advection_theta_finalization = (
        _DYNAMICS_TENDENCY_THETA_HORIZONTAL_ADVECTION_THETA_FINALIZATION_COMPILE_NAME
        in requested
    )
    if compile_theta_horizontal_advection_theta_finalization and requested.intersection(
        {
            _DYNAMICS_TENDENCY_SECTION_COMPILE_NAMES[
                "theta_horizontal_advection_theta_hadv"
            ],
            _DYNAMICS_TENDENCY_SECTION_COMPILE_NAMES[
                "theta_horizontal_mixing_theta_horizontal_mix"
            ],
            _DYNAMICS_TENDENCY_SECTION_COMPILE_NAMES[
                "theta_vertical_advection_diabatic_theta_vertical"
            ],
            _DYNAMICS_TENDENCY_SECTION_COMPILE_NAMES[
                "theta_vertical_mixing_theta_finalization_theta_final"
            ],
        }
    ):
        raise ValueError(
            "theta-horizontal-advection-theta-finalization parent compilation cannot be mixed with theta child targets"
        )

    resolved: dict[str, Callable[..., Any]] = {}
    for field_name in DYNAMICS_TENDENCY_SECTION_CALL_SEQUENCE:
        fn = getattr(_CANONICAL_DYNAMICS_TENDENCY_SECTION_OPS, field_name)
        compile_name = _DYNAMICS_TENDENCY_SECTION_COMPILE_NAMES.get(field_name)
        if compile_name in requested:
            fn = maybe_compile_kernel(
                compile_name,
                fn,
                enabled=True,
                mode=compile_mode,
                backend=compile_backend,
                specialization_key=(
                    "dynamics_tendency_section_plan",
                    stage_key,
                    field_name,
                ),
            )
        if section_observer is not None:
            inner = fn

            @wraps(inner)
            def observed(
                *args,
                _inner=inner,
                _field_name=field_name,
                **kwargs,
            ):
                started = perf_counter()
                result = _inner(*args, **kwargs)
                elapsed = perf_counter() - started
                section_observer(_field_name, args, kwargs, result, elapsed)
                return result

            fn = observed
        profile_name = _DYNAMICS_TENDENCY_SECTION_PROFILE_NAMES.get(field_name)
        if profile_sections and profile_name is not None:
            inner = fn

            @wraps(inner)
            def profiled(*args, _inner=inner, _name=profile_name, **kwargs):
                with profile_section(_name, enabled=True):
                    return _inner(*args, **kwargs)

            fn = profiled
        resolved[field_name] = fn
    normal_velocity_mixing_normal_velocity_finalization_u_tendency: (
        Callable[..., Any] | None
    )
    if profile_sections:
        normal_velocity_mixing_normal_velocity_finalization_u_tendency = None
    else:
        normal_velocity_mixing_normal_velocity_finalization_u_tendency = (
            _normal_velocity_mixing_normal_velocity_finalization_u_tendency_region
        )
        if compile_normal_velocity_mixing_normal_velocity_finalization:
            normal_velocity_mixing_normal_velocity_finalization_u_tendency = maybe_compile_kernel(
                _DYNAMICS_TENDENCY_NORMAL_VELOCITY_MIXING_NORMAL_VELOCITY_FINALIZATION_COMPILE_NAME,
                normal_velocity_mixing_normal_velocity_finalization_u_tendency,
                enabled=True,
                mode=compile_mode,
                backend=compile_backend,
                specialization_key=(
                    "dynamics_tendency_section_plan",
                    stage_key,
                    "normal_velocity_mixing_normal_velocity_finalization",
                ),
            )
        elif requested.intersection(
            {
                _DYNAMICS_TENDENCY_SECTION_COMPILE_NAMES[
                    "normal_velocity_mixing_u_mixing"
                ],
            }
        ):
            normal_velocity_mixing_normal_velocity_finalization_u_tendency = None
        if (
            section_observer is not None
            and normal_velocity_mixing_normal_velocity_finalization_u_tendency
            is not None
        ):
            inner = normal_velocity_mixing_normal_velocity_finalization_u_tendency

            @wraps(inner)
            def observed_normal_velocity_mixing_normal_velocity_finalization(
                *args, _inner=inner, **kwargs
            ):
                started = perf_counter()
                result = _inner(*args, **kwargs)
                elapsed = perf_counter() - started
                section_observer(
                    "normal_velocity_mixing_u_mixing", args, kwargs, result[0], elapsed
                )
                section_observer(
                    "normal_velocity_finalization_u_final", args, kwargs, result[1], 0.0
                )
                return result

            normal_velocity_mixing_normal_velocity_finalization_u_tendency = (
                observed_normal_velocity_mixing_normal_velocity_finalization
            )
    vertical_velocity_horizontal_mixing_vertical_velocity_vertical_transport_w_tendency: (
        Callable[..., Any] | None
    )
    if profile_sections:
        vertical_velocity_horizontal_mixing_vertical_velocity_vertical_transport_w_tendency = None
    else:
        vertical_velocity_horizontal_mixing_vertical_velocity_vertical_transport_w_tendency = _vertical_velocity_horizontal_mixing_vertical_velocity_vertical_transport_w_tendency_region
        if compile_vertical_velocity_horizontal_mixing_vertical_velocity_vertical_transport:
            vertical_velocity_horizontal_mixing_vertical_velocity_vertical_transport_w_tendency = maybe_compile_kernel(
                _DYNAMICS_TENDENCY_VERTICAL_VELOCITY_HORIZONTAL_MIXING_VERTICAL_VELOCITY_VERTICAL_TRANSPORT_COMPILE_NAME,
                vertical_velocity_horizontal_mixing_vertical_velocity_vertical_transport_w_tendency,
                enabled=True,
                mode=compile_mode,
                backend=compile_backend,
                specialization_key=(
                    "dynamics_tendency_section_plan",
                    stage_key,
                    "vertical_velocity_horizontal_mixing_vertical_velocity_vertical_transport",
                ),
            )
        elif requested.intersection(
            {
                _DYNAMICS_TENDENCY_SECTION_COMPILE_NAMES[
                    "vertical_velocity_horizontal_mixing_w_horizontal_mix"
                ],
                _DYNAMICS_TENDENCY_SECTION_COMPILE_NAMES[
                    "vertical_velocity_vertical_transport_w_vertical"
                ],
            }
        ):
            vertical_velocity_horizontal_mixing_vertical_velocity_vertical_transport_w_tendency = None
        if (
            section_observer is not None
            and vertical_velocity_horizontal_mixing_vertical_velocity_vertical_transport_w_tendency
            is not None
        ):
            vertical_velocity_horizontal_mixing_vertical_velocity_vertical_transport_w_tendency = _observe_dynamics_tendency_vertical_velocity_horizontal_mixing_vertical_velocity_vertical_transport_parent(
                vertical_velocity_horizontal_mixing_vertical_velocity_vertical_transport_w_tendency,
                section_observer,
            )
    theta_horizontal_advection_theta_finalization_theta_tendency: (
        Callable[..., Any] | None
    )
    if profile_sections:
        theta_horizontal_advection_theta_finalization_theta_tendency = None
    else:
        theta_horizontal_advection_theta_finalization_theta_tendency = (
            _theta_horizontal_advection_theta_finalization_theta_tendency_region
        )
        if compile_theta_horizontal_advection_theta_finalization:
            theta_horizontal_advection_theta_finalization_theta_tendency = maybe_compile_kernel(
                _DYNAMICS_TENDENCY_THETA_HORIZONTAL_ADVECTION_THETA_FINALIZATION_COMPILE_NAME,
                theta_horizontal_advection_theta_finalization_theta_tendency,
                enabled=True,
                mode=compile_mode,
                backend=compile_backend,
                specialization_key=(
                    "dynamics_tendency_section_plan",
                    stage_key,
                    "theta_horizontal_advection_theta_finalization",
                ),
            )
        elif requested.intersection(
            {
                _DYNAMICS_TENDENCY_SECTION_COMPILE_NAMES[
                    "theta_horizontal_advection_theta_hadv"
                ],
                _DYNAMICS_TENDENCY_SECTION_COMPILE_NAMES[
                    "theta_horizontal_mixing_theta_horizontal_mix"
                ],
                _DYNAMICS_TENDENCY_SECTION_COMPILE_NAMES[
                    "theta_vertical_advection_diabatic_theta_vertical"
                ],
                _DYNAMICS_TENDENCY_SECTION_COMPILE_NAMES[
                    "theta_vertical_mixing_theta_finalization_theta_final"
                ],
            }
        ):
            theta_horizontal_advection_theta_finalization_theta_tendency = None
        if (
            section_observer is not None
            and theta_horizontal_advection_theta_finalization_theta_tendency is not None
        ):
            theta_horizontal_advection_theta_finalization_theta_tendency = _observe_dynamics_tendency_theta_horizontal_advection_theta_finalization_parent(
                theta_horizontal_advection_theta_finalization_theta_tendency,
                section_observer,
            )
    return DynamicsTendencySectionOps(
        **resolved,
        normal_velocity_mixing_normal_velocity_finalization_u_tendency=normal_velocity_mixing_normal_velocity_finalization_u_tendency,
        vertical_velocity_horizontal_mixing_vertical_velocity_vertical_transport_w_tendency=vertical_velocity_horizontal_mixing_vertical_velocity_vertical_transport_w_tendency,
        theta_horizontal_advection_theta_finalization_theta_tendency=theta_horizontal_advection_theta_finalization_theta_tendency,
    )


def _observe_dynamics_tendency_section_ops(
    section_ops: DynamicsTendencySectionOps,
    observer: Callable[
        [str, tuple[object, ...], Mapping[str, Any], object, float], None
    ],
) -> DynamicsTendencySectionOps:
    """Attach a diagnostic observer without changing the numeric call table."""
    resolved: dict[str, Callable[..., Any]] = {}
    for field_name in DYNAMICS_TENDENCY_SECTION_CALL_SEQUENCE:
        inner = getattr(section_ops, field_name)

        @wraps(inner)
        def observed(*args, _inner=inner, _field_name=field_name, **kwargs):
            started = perf_counter()
            result = _inner(*args, **kwargs)
            elapsed = perf_counter() - started
            observer(_field_name, args, kwargs, result, elapsed)
            return result

        resolved[field_name] = observed
    normal_velocity_mixing_normal_velocity_finalization_u_tendency = (
        section_ops.normal_velocity_mixing_normal_velocity_finalization_u_tendency
    )
    if normal_velocity_mixing_normal_velocity_finalization_u_tendency is not None:
        inner = normal_velocity_mixing_normal_velocity_finalization_u_tendency

        @wraps(inner)
        def observed_normal_velocity_mixing_normal_velocity_finalization(
            *args, _inner=inner, **kwargs
        ):
            started = perf_counter()
            result = _inner(*args, **kwargs)
            elapsed = perf_counter() - started
            observer(
                "normal_velocity_mixing_u_mixing", args, kwargs, result[0], elapsed
            )
            observer(
                "normal_velocity_finalization_u_final", args, kwargs, result[1], 0.0
            )
            return result

        normal_velocity_mixing_normal_velocity_finalization_u_tendency = (
            observed_normal_velocity_mixing_normal_velocity_finalization
        )
    vertical_velocity_horizontal_mixing_vertical_velocity_vertical_transport_w_tendency = section_ops.vertical_velocity_horizontal_mixing_vertical_velocity_vertical_transport_w_tendency
    if (
        vertical_velocity_horizontal_mixing_vertical_velocity_vertical_transport_w_tendency
        is not None
    ):
        vertical_velocity_horizontal_mixing_vertical_velocity_vertical_transport_w_tendency = _observe_dynamics_tendency_vertical_velocity_horizontal_mixing_vertical_velocity_vertical_transport_parent(
            vertical_velocity_horizontal_mixing_vertical_velocity_vertical_transport_w_tendency,
            observer,
        )
    theta_horizontal_advection_theta_finalization_theta_tendency = (
        section_ops.theta_horizontal_advection_theta_finalization_theta_tendency
    )
    if theta_horizontal_advection_theta_finalization_theta_tendency is not None:
        theta_horizontal_advection_theta_finalization_theta_tendency = _observe_dynamics_tendency_theta_horizontal_advection_theta_finalization_parent(
            theta_horizontal_advection_theta_finalization_theta_tendency,
            observer,
        )
    return DynamicsTendencySectionOps(
        **resolved,
        normal_velocity_mixing_normal_velocity_finalization_u_tendency=normal_velocity_mixing_normal_velocity_finalization_u_tendency,
        vertical_velocity_horizontal_mixing_vertical_velocity_vertical_transport_w_tendency=vertical_velocity_horizontal_mixing_vertical_velocity_vertical_transport_w_tendency,
        theta_horizontal_advection_theta_finalization_theta_tendency=theta_horizontal_advection_theta_finalization_theta_tendency,
    )


def prebind_dynamics_tendency_connectivity_payload(
    call_arguments: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the whole-stage static topology payload outside the numeric graph."""
    args = call_arguments
    device = args["u"].device  # type: ignore[union-attr]
    build_ordered_ad_transpose = any(
        isinstance(args.get(name), torch.Tensor)
        and _tensor_carries_autodiff(args[name])  # type: ignore[arg-type]
        for name in _DYNAMICS_TENDENCY_DYNAMIC_ARGUMENT_NAMES
    )
    return _dynamics_tendency_connectivity_payload(
        device=device,
        nCells=int(args["nCells"]),
        nEdges=int(args["nEdges"]),
        maxEdges=int(args["maxEdges"]),
        maxEdges2=int(args["maxEdges2"]),
        cell_start=int(args["cellStart"]) - 1,
        cell_end=int(args["cellEnd"]),
        cell_solve_start=int(args["cellSolveStart"]) - 1,
        cell_solve_end=int(args["cellSolveEnd"]),
        edge_start=int(args["edgeStart"]) - 1,
        edge_end=int(args["edgeEnd"]),
        edge_solve_start=int(args["edgeSolveStart"]) - 1,
        edge_solve_end=int(args["edgeSolveEnd"]),
        edgesOnCell=args["edgesOnCell"],
        nEdgesOnCell=args["nEdgesOnCell"],
        edgesOnCell_sign=args["edgesOnCell_sign"],
        cellsOnEdge=args["cellsOnEdge"],
        verticesOnEdge=args["verticesOnEdge"],
        edgesOnEdge=args["edgesOnEdge"],
        nAdvCellsForEdge=args["nAdvCellsForEdge"],
        nEdgesOnEdge=args["nEdgesOnEdge"],
        advCellsForEdge=args["advCellsForEdge"],
        adv_coefs=args["adv_coefs"],
        adv_coefs_3rd=args["adv_coefs_3rd"],
        weightsOnEdge=args["weightsOnEdge"],
        vertex_start=int(args["vertexStart"]) - 1,
        vertex_end=int(args["vertexEnd"]),
        vertexDegree=int(args["vertexDegree"]),
        edgesOnVertex=args["edgesOnVertex"],
        edgesOnVertex_sign=args["edgesOnVertex_sign"],
        dvEdge=args["dvEdge"],
        dcEdge=args["dcEdge"],
        invDcEdge=None,
        invDvEdge=None,
        invAreaCell=None,
        invAreaTriangle=args["invAreaTriangle"],
        meshScalingDel2=None,
        meshScalingDel4=None,
        build_ordered_ad_transpose=build_ordered_ad_transpose,
        build_edge_advective_ordered_ad_transpose=build_ordered_ad_transpose,
    )


def _dynamics_tendency_connectivity_values(
    payload: Mapping[str, Any] | tuple[Any, ...],
) -> tuple[Any, ...]:
    """Return the fixed-order connectivity values used by the numeric body."""
    if isinstance(payload, tuple):
        if len(payload) != len(_DYNAMICS_TENDENCY_CONNECTIVITY_PAYLOAD_NAMES):
            raise ValueError(
                "invalid whole-dynamics-tendency connectivity tuple length: "
                f"expected {len(_DYNAMICS_TENDENCY_CONNECTIVITY_PAYLOAD_NAMES)}, got {len(payload)}"
            )
        return payload
    missing = [
        name
        for name in _DYNAMICS_TENDENCY_CONNECTIVITY_PAYLOAD_NAMES
        if name not in payload
    ]
    if missing:
        raise KeyError(
            f"whole-dynamics-tendency connectivity payload is missing fields: {missing}"
        )
    return tuple(
        payload[name] for name in _DYNAMICS_TENDENCY_CONNECTIVITY_PAYLOAD_NAMES
    )


def _dynamics_tendency_numeric_connectivity_values(
    payload: tuple[Any, ...],
) -> tuple[Any, ...]:
    """Validate the compact, relation-owned whole-stage connectivity tuple."""
    if not isinstance(payload, tuple):
        raise TypeError(
            "whole-dynamics-tendency numeric connectivity payload must be a tuple"
        )
    if len(payload) != len(_DYNAMICS_TENDENCY_NUMERIC_CONNECTIVITY_PAYLOAD_NAMES):
        raise ValueError(
            "invalid whole-dynamics-tendency numeric connectivity tuple length: "
            f"expected {len(_DYNAMICS_TENDENCY_NUMERIC_CONNECTIVITY_PAYLOAD_NAMES)}, "
            f"got {len(payload)}"
        )
    return payload


def _normalize_dynamics_tendency_numeric_connectivity_payload(
    payload: Mapping[str, Any] | tuple[Any, ...],
) -> tuple[Any, ...]:
    """Normalize public connectivity into the fixed numeric-body tuple."""
    if isinstance(payload, Mapping):
        values_by_name = {
            name: _dynamics_tendency_numeric_relation(value)
            for name, value in zip(
                _DYNAMICS_TENDENCY_CONNECTIVITY_PAYLOAD_NAMES,
                _dynamics_tendency_connectivity_values(payload),
            )
        }
        numeric = tuple(
            values_by_name[name]
            for name in _DYNAMICS_TENDENCY_NUMERIC_CONNECTIVITY_PAYLOAD_NAMES
        )
        return _dynamics_tendency_numeric_connectivity_values(numeric)
    if not isinstance(payload, tuple):
        raise TypeError(
            "whole-dynamics-tendency connectivity payload must be a mapping or tuple"
        )
    if len(payload) == len(_DYNAMICS_TENDENCY_CONNECTIVITY_PAYLOAD_NAMES):
        values_by_name = {
            name: _dynamics_tendency_numeric_relation(value)
            for name, value in zip(
                _DYNAMICS_TENDENCY_CONNECTIVITY_PAYLOAD_NAMES, payload
            )
        }
        payload = tuple(
            values_by_name[name]
            for name in _DYNAMICS_TENDENCY_NUMERIC_CONNECTIVITY_PAYLOAD_NAMES
        )
    else:
        payload = tuple(_dynamics_tendency_numeric_relation(value) for value in payload)
    return _dynamics_tendency_numeric_connectivity_values(payload)


def _resolve_dynamics_tendency_public_section_ops(
    call_arguments: Mapping[str, Any],
) -> DynamicsTendencySectionOps:
    """Resolve direct-call compile/profile wrappers before numeric composition."""
    requested = frozenset(call_arguments.get("_compile_region_names") or ()) & (
        _DYNAMICS_TENDENCY_REGION_COMPILE_TARGETS
        - {_DYNAMICS_TENDENCY_WHOLE_STAGE_COMPILE_TARGET}
    )
    section_ops = call_arguments.get("_section_ops")
    if section_ops is None:
        return build_dynamics_tendency_section_ops(
            compile_region_names=tuple(sorted(requested)),
            compile_mode=str(call_arguments["_compile_mode"]),
            compile_backend=str(call_arguments["_compile_backend"]),
            stage_key=(
                "direct_call",
                *_dynamics_tendency_whole_stage_specialization_key(call_arguments),
            ),
            profile_sections=bool(call_arguments.get("_profile_sections"))
            and not _compile_active(),
        )
    if not isinstance(section_ops, DynamicsTendencySectionOps):
        raise TypeError("_section_ops must be a DynamicsTendencySectionOps instance")
    if requested:
        raise ValueError(
            "prebuilt dynamics-tendency section operators cannot be combined with inline "
            f"compile targets: {sorted(requested)}"
        )
    return section_ops


def _dynamics_tendency_canonical_composition(
    # Dimensions (scalars)
    nCells: int,
    nEdges: int,
    nVertices: int,
    nVertLevels: int,
    nCellsSolve: int,
    nEdgesSolve: int,
    vertexDegree: int,
    maxEdges: int,
    maxEdges2: int,
    num_scalars: int,
    moist_start: int,
    moist_end: int,
    # Time stepping
    rk_step: int,
    dt: float,
    # Configuration scalars
    coef_3rd_order: float,
    c_s: float,
    config_horiz_mixing: str,
    config_del4u_div_factor: float,
    config_h_mom_eddy_visc2: float,
    config_v_mom_eddy_visc2: float,
    config_h_theta_eddy_visc2: float,
    config_v_theta_eddy_visc2: float,
    config_h_theta_eddy_visc4: float,
    config_h_mom_eddy_visc4: float,
    config_visc4_2dsmag: float,
    config_len_disp: float,
    cf1: float,
    cf2: float,
    cf3: float,
    r_earth: float,
    config_mpas_cam_coef: float,
    config_mix_full: int,
    config_rayleigh_damp_u: int,
    config_rayleigh_damp_u_timescale_days: float,
    config_number_rayleigh_damp_u_levels: int,
    config_number_cam_damping_levels: int,
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
    # 1D arrays - mesh metric
    fEdge: torch.Tensor,  # (nEdges+1,)
    dvEdge: torch.Tensor,  # (nEdges+1,)
    dcEdge: torch.Tensor,  # (nEdges+1,)
    invDcEdge: torch.Tensor,  # (nEdges+1,)
    invDvEdge: torch.Tensor,  # (nEdges+1,)
    invAreaCell: torch.Tensor,  # (nCells+1,)
    invAreaTriangle: torch.Tensor,  # (nVertices+1,)
    meshScalingDel2: torch.Tensor,  # (nEdges+1,)
    meshScalingDel4: torch.Tensor,  # (nEdges+1,)
    latCell: torch.Tensor,  # (nCells+1,)
    latEdge: torch.Tensor,  # (nEdges+1,)
    angleEdge: torch.Tensor,  # (nEdges+1,)
    # 1D arrays - vertical
    rdzu: torch.Tensor,  # (nVertLevels,)
    rdzw: torch.Tensor,  # (nVertLevels,)
    fzm: torch.Tensor,  # (nVertLevels,)
    fzp: torch.Tensor,  # (nVertLevels,)
    u_init: torch.Tensor,  # (nVertLevels,)
    v_init: torch.Tensor,  # (nVertLevels,)
    qv_init: torch.Tensor,  # (nVertLevels,)
    # Connectivity arrays
    cellsOnEdge: torch.Tensor,  # (nEdges+1, 2) int32, 1-based
    verticesOnEdge: torch.Tensor,  # (nEdges+1, 2) int32, 1-based
    edgesOnCell: torch.Tensor,  # (nCells+1, maxEdges) int32, 1-based
    edgesOnEdge: torch.Tensor,  # (nEdges+1, maxEdges2) int32, 1-based
    cellsOnCell: torch.Tensor,  # (nCells+1, maxEdges) int32, 1-based
    edgesOnVertex: torch.Tensor,  # (nVertices+1, vertexDegree) int32, 1-based
    nEdgesOnCell: torch.Tensor,  # (nCells+1,) int32
    nEdgesOnEdge: torch.Tensor,  # (nEdges+1,) int32
    advCellsForEdge: torch.Tensor,  # (nEdges+1, 15) int32, 1-based
    nAdvCellsForEdge: torch.Tensor,  # (nEdges+1,) int32
    adv_coefs: torch.Tensor,  # (nEdges+1, 15)
    adv_coefs_3rd: torch.Tensor,  # (nEdges+1, 15)
    weightsOnEdge: torch.Tensor,  # (nEdges+1, maxEdges2)
    deriv_two: torch.Tensor,  # (nEdges+1, 2, 15)
    defc_a: torch.Tensor,  # (nCells+1, maxEdges)
    defc_b: torch.Tensor,  # (nCells+1, maxEdges)
    edgesOnCell_sign: torch.Tensor,  # (nCells+1, maxEdges)
    edgesOnVertex_sign: torch.Tensor,  # (nVertices+1, vertexDegree)
    # 2D state arrays
    u: torch.Tensor,  # (nEdges+1, nVertLevels)
    v: torch.Tensor,  # (nEdges+1, nVertLevels)
    w: torch.Tensor,  # (nCells+1, nVertLevels+1)
    ru: torch.Tensor,  # (nEdges+1, nVertLevels)
    rw: torch.Tensor,  # (nCells+1, nVertLevels+1)
    ru_save: torch.Tensor,  # (nEdges+1, nVertLevels)
    rw_save: torch.Tensor,  # (nCells+1, nVertLevels+1)
    theta_m: torch.Tensor,  # (nCells+1, nVertLevels)
    theta_m_save: torch.Tensor,  # (nCells+1, nVertLevels)
    rho_zz: torch.Tensor,  # (nCells+1, nVertLevels)
    rho_edge: torch.Tensor,  # (nEdges+1, nVertLevels)
    exner: torch.Tensor,  # (nCells+1, nVertLevels)
    divergence: torch.Tensor,  # (nCells+1, nVertLevels)
    vorticity: torch.Tensor,  # (nVertices+1, nVertLevels)
    ke: torch.Tensor,  # (nCells+1, nVertLevels)
    pv_edge: torch.Tensor,  # (nEdges+1, nVertLevels)
    pp: torch.Tensor,  # (nCells+1, nVertLevels)
    rr: torch.Tensor,  # (nCells+1, nVertLevels)
    rr_save: torch.Tensor,  # (nCells+1, nVertLevels)
    rb: torch.Tensor,  # (nCells+1, nVertLevels)
    pressure_b: torch.Tensor,  # (nCells+1, nVertLevels)
    zgrid: torch.Tensor,  # (nCells+1, nVertLevels+1)
    zz: torch.Tensor,  # (nCells+1, nVertLevels)
    zxu: torch.Tensor,  # (nEdges+1, nVertLevels)
    cqw: torch.Tensor,  # (nCells+1, nVertLevels)
    cqu: torch.Tensor,  # (nEdges+1, nVertLevels)
    rt_diabatic_tend: torch.Tensor,  # (nCells+1, nVertLevels)
    scalars: torch.Tensor,  # (nCells+1, nVertLevels, num_scalars)
    t_init: torch.Tensor,  # (nCells+1, nVertLevels)
    tend_w_pgf: torch.Tensor | None,  # legacy, currently unused
    tend_w_buoy: torch.Tensor | None,  # legacy, currently unused
    ur_cell: torch.Tensor,  # (nCells+1, nVertLevels)
    vr_cell: torch.Tensor,  # (nCells+1, nVertLevels)
    qtot: torch.Tensor,  # (nCells+1, nVertLevels) Pre-computed by the caller
    # Optional input arrays that persist across RK steps
    # These are computed on rk_step==1 and reused on rk_step>1
    kdiff_in: torch.Tensor | None = None,  # (nCells+1, nVertLevels)
    # Configuration flags selected at runtime.
    config_curvature_u: bool = False,  # Include curvature terms for tend_u
    config_dynamics_tendency_cell_owned_vertical_velocity_horizontal_advection_theta_horizontal_advection_ab: bool = False,
    tend_u_euler_in: torch.Tensor | None = None,  # (nEdges+1, nVertLevels)
    tend_w_euler_in: torch.Tensor | None = None,  # (nCells+1, nVertLevels+1)
    tend_theta_euler_in: torch.Tensor | None = None,  # (nCells+1, nVertLevels)
    tend_rho_in: torch.Tensor | None = None,  # (nCells+1, nVertLevels)
    tend_ru_physics: torch.Tensor | None = None,  # (nEdges+1, nVertLevels)
    tend_rtheta_physics: torch.Tensor | None = None,  # (nCells+1, nVertLevels)
    tend_rho_physics: torch.Tensor | None = None,  # (nCells+1, nVertLevels)
    _connectivity_payload: tuple[Any, ...] | None = None,
    _section_ops: DynamicsTendencySectionOps | None = None,
) -> tuple[torch.Tensor | None, ...]:
    """
    Compute dynamical tendencies for RK integration.

    This is the largest kernel in the MPAS dynamics core, computing all
    tendencies for explicit time stepping including:
    - Horizontal mass flux divergence (tend_rho)
    - Horizontal momentum advection, Coriolis, and pressure gradient (tend_u)
    - Vertical momentum advection, pressure gradient, and buoyancy (tend_w)
    - Potential temperature advection and diabatic heating (tend_theta)
    - Horizontal and vertical mixing for all prognostic variables

    The computation follows the split-explicit time integration where:
    - rk_step=1: Initialize Euler tendencies for mixing (computed once per large step)
    - rk_step>1: Reuse Euler tendencies, update advective tendencies

    Index Convention:
        Fortran uses 1-based indexing: k=1..nVertLevels, iCell=1..nCells
        PyTorch uses 0-based indexing: k=0..nVertLevels-1, iCell=0..nCells-1
        Fortran loop `do k = 2, nVertLevels` maps to Python `for k in range(1, nVertLevels)`

    Args:
        nCells, nEdges, nVertices, nVertLevels: Mesh dimensions
        nCellsSolve, nEdgesSolve: Number of owned cells/edges to solve
        vertexDegree, maxEdges, maxEdges2: Connectivity dimensions
        num_scalars, moist_start, moist_end: Scalar tracer info
        rk_step: Runge-Kutta step number (1, 2, or 3)
        dt: Large timestep
        coef_3rd_order: 3rd-order flux coefficient
        c_s: Smagorinsky constant
        config_*: Physics configuration parameters
        cell/vertex/edgeStart/End: Loop bounds (1-based)
        fEdge, dvEdge, etc.: Mesh metric arrays
        u, v, w, ru, rw, etc.: State arrays
        Various tendency and diagnostic arrays
        qtot: Pre-computed total moisture mixing ratio (nCells+1, nVertLevels).
              Computed by caller from scalars, matching Fortran architecture.
    Returns:
        Fixed compact-patch tuple. Public full-field packing is owned by
        :func:`atm_compute_dyn_tend_work` or the runner.
    """
    # Infer device from input tensors for device-consistent tensor creation
    device = u.device if isinstance(u, torch.Tensor) else torch.device("cpu")
    # Constants
    gravity = GRAVITY
    omega = OMEGA
    prandtl = PRANDTL
    prandtl_inv = 1.0 / prandtl
    invDt = 1.0 / dt
    inv_r_earth = 1.0 / r_earth
    seconds_per_day = 86400.0

    v_mom_eddy_visc2 = config_v_mom_eddy_visc2
    v_theta_eddy_visc2 = config_v_theta_eddy_visc2

    # Convert Fortran 1-based bounds to Python 0-based
    cell_start = cellStart - 1
    cell_end = cellEnd
    vertex_start = vertexStart - 1
    vertex_end = vertexEnd
    edge_start = edgeStart - 1
    edge_end = edgeEnd
    cell_solve_start = cellSolveStart - 1
    cell_solve_end = cellSolveEnd
    edge_solve_start = edgeSolveStart - 1
    edge_solve_end = edgeSolveEnd

    def _output_zero_base(shape):
        return torch.zeros((1, *tuple(shape)[1:]), dtype=DTYPE, device=device)

    def _slot_patch_rows_are_local(*relations):
        for relation in relations:
            if (
                relation is None
                or _dynamics_tendency_slot_field(relation, 10, "local_all") is not True
            ):
                return False
        return True

    def _pair_patch_rows_are_local(*relations):
        for relation in relations:
            if not (
                relation is not None
                and _dynamics_tendency_pair_field(relation, 9, "first_local_all")
                is True
                and _dynamics_tendency_pair_field(relation, 10, "second_local_all")
                is True
            ):
                return False
        return True

    def _scratch_base_or_compact(shape, *, compact_safe: bool):
        if compact_safe:
            return torch.empty((1, *tuple(shape)[1:]), dtype=DTYPE, device=device)
        return torch.zeros(shape, dtype=DTYPE, device=device)

    # Initialize output arrays.
    # For rk_step > 1, some arrays persist from rk_step==1 and must be passed in.
    tend_u = _output_zero_base((nEdges + 1, nVertLevels))
    tend_theta = _output_zero_base((nCells + 1, nVertLevels))
    tend_w = _output_zero_base((nCells + 1, nVertLevels + 1))
    h_divergence = _output_zero_base((nCells + 1, nVertLevels))

    # These arrays persist across RK steps - use input if provided, else initialize
    if kdiff_in is not None:
        kdiff = kdiff_in
    else:
        kdiff = _output_zero_base((nCells + 1, nVertLevels))

    if tend_u_euler_in is not None:
        tend_u_euler = tend_u_euler_in
    else:
        tend_u_euler = _output_zero_base((nEdges + 1, nVertLevels))

    if tend_w_euler_in is not None:
        tend_w_euler = tend_w_euler_in
    else:
        tend_w_euler = _output_zero_base((nCells + 1, nVertLevels + 1))

    if tend_theta_euler_in is not None:
        tend_theta_euler = tend_theta_euler_in
    else:
        tend_theta_euler = _output_zero_base((nCells + 1, nVertLevels))

    if tend_rho_in is not None:
        tend_rho = tend_rho_in
    else:
        tend_rho = _output_zero_base((nCells + 1, nVertLevels))

    dpdz = torch.zeros((nCells + 1, nVertLevels), dtype=DTYPE, device=device)
    tend_u_solve_patch = None
    tend_w_solve_inner_patch = None
    tend_theta_solve_patch = None
    rthdynten_solve_patch = None
    tend_u_euler_edge_patch = None
    tend_w_euler_cell_patch = None
    tend_theta_euler_cell_patch = None
    kdiff_cell_patch = None
    h_divergence_cell_patch = None
    tend_rho_cell_patch = None
    dpdz_cell_patch = None

    section_ops = _section_ops
    if section_ops is None:
        raise ValueError(
            "dynamics-tendency canonical composition requires prebound section operators"
        )

    # The runner/public boundary owns topology construction. The composition only
    # consumes the immutable payload selected for this mesh and execution plan.
    if _connectivity_payload is None:
        raise ValueError(
            "dynamics-tendency canonical composition requires prebound connectivity"
        )
    conn = _connectivity_payload
    (
        edge_idx_all,
        cell_idx,
        solve_cells,
        solve_edge_idx,
        solve_edge_slice,
        cell_edge_relation,
        cell_flux_edge_relation,
        solve_cell_edge_relation,
        solve_flux_valid_all_by_slot,
        solve_flux_cell_relation,
        edge_adv_cell_relation,
        edge_neighbor_relation,
        solve_local_edge_relation,
        cell_flux_pair_relation,
        edge_all_pair_relation,
        edge_all_vertex_relation,
        solve_edge_pair_relation,
        solve_edge_vertex_relation,
        _cell_dv_over_dc,
        _cell_mesh_scaling_del2,
        _solve_del4_dv_over_dc,
        _cell_w_mix_edge_sign,
        _cell_theta_mix_edge_sign,
        cell_div_del2_edge_sign,
        solve_del4_mesh_scaling,
        solve_del4_inv_dc,
        solve_del4_min_inv_dv_dc,
        edge_all_inv_dc,
        edge_all_min_inv_dv_dc,
        vort_vertices_payload,
        vertex_edges_payload,
        vertex_edge_relation,
        vertex_del2_edge_sign,
        cell_slot_count,
    ) = _dynamics_tendency_numeric_connectivity_values(conn)
    cell_edge_valid = None
    safe_cell_edges = None
    cell_edge_sign = None
    cell_signed_dv = None
    solve_edge_valid = None
    solve_safe_edges = None
    solve_edge_sign = None
    solve_signed_dv = None
    solve_flux_valid = None
    edge_adv_coef = None
    edge_adv_coef_3rd = None
    edge_eoe_valid = None
    edge_eoe_weights = None
    cell_flux_valid = None
    edge_all_pair_valid = None
    solve_edge_pair_valid = None
    edge_c1 = None
    edge_c2 = None
    solve_edge_mask = None
    solve_edge_mask_all = False
    solve_edge_valid_weight = None
    solve_edge_mask_all = bool(solve_edge_mask_all)
    if cell_edge_relation is not None:
        safe_cell_edges = _dynamics_tendency_slot_field(
            cell_edge_relation, 0, "indices"
        )
        relation_valid = _dynamics_tendency_slot_field(cell_edge_relation, 1, "valid")
        relation_sign = _dynamics_tendency_slot_field(cell_edge_relation, 2, "sign")
        relation_weights = _dynamics_tendency_slot_field(
            cell_edge_relation, 3, "weights"
        )
        cell_edge_valid = (
            relation_valid if relation_valid is not None else cell_edge_valid
        )
        cell_edge_sign = relation_sign if relation_sign is not None else cell_edge_sign
        cell_signed_dv = (
            relation_weights if relation_weights is not None else cell_signed_dv
        )

    if cell_flux_edge_relation is not None:
        relation_valid = _dynamics_tendency_slot_field(
            cell_flux_edge_relation, 1, "valid"
        )
        cell_flux_valid = (
            relation_valid if relation_valid is not None else cell_flux_valid
        )
    else:
        cell_flux_edge_relation = cell_edge_relation

    if solve_cell_edge_relation is not None:
        solve_safe_edges = _dynamics_tendency_slot_field(
            solve_cell_edge_relation, 0, "indices"
        )
        relation_valid = _dynamics_tendency_slot_field(
            solve_cell_edge_relation, 1, "valid"
        )
        relation_sign = _dynamics_tendency_slot_field(
            solve_cell_edge_relation, 2, "sign"
        )
        relation_weights = _dynamics_tendency_slot_field(
            solve_cell_edge_relation, 3, "weights"
        )
        solve_edge_valid = (
            relation_valid if relation_valid is not None else solve_edge_valid
        )
        solve_edge_sign = (
            relation_sign if relation_sign is not None else solve_edge_sign
        )
        solve_signed_dv = (
            relation_weights if relation_weights is not None else solve_signed_dv
        )

    if solve_local_edge_relation is not None:
        relation_sign = _dynamics_tendency_slot_field(
            solve_local_edge_relation, 2, "sign"
        )
        relation_valid_weight = _dynamics_tendency_slot_field(
            solve_local_edge_relation, 7, "valid_weight"
        )
        solve_edge_sign = (
            relation_sign if relation_sign is not None else solve_edge_sign
        )
        solve_edge_valid_weight = (
            relation_valid_weight
            if relation_valid_weight is not None
            else solve_edge_valid_weight
        )

    if edge_adv_cell_relation is not None:
        relation_coefficients = _dynamics_tendency_slot_field(
            edge_adv_cell_relation, 4, "coefficients"
        )
        relation_coefficients_3rd = _dynamics_tendency_slot_field(
            edge_adv_cell_relation, 5, "coefficients_3rd"
        )
        edge_adv_coef = (
            relation_coefficients
            if relation_coefficients is not None
            else edge_adv_coef
        )
        edge_adv_coef_3rd = (
            relation_coefficients_3rd
            if relation_coefficients_3rd is not None
            else edge_adv_coef_3rd
        )

    if edge_neighbor_relation is not None:
        relation_valid = _dynamics_tendency_slot_field(
            edge_neighbor_relation, 1, "valid"
        )
        relation_weights = _dynamics_tendency_slot_field(
            edge_neighbor_relation, 3, "weights"
        )
        edge_eoe_valid = (
            relation_valid if relation_valid is not None else edge_eoe_valid
        )
        edge_eoe_weights = (
            relation_weights if relation_weights is not None else edge_eoe_weights
        )

    if solve_flux_cell_relation is not None:
        relation_valid = _dynamics_tendency_pair_field(
            solve_flux_cell_relation, 2, "valid"
        )
        solve_flux_valid = (
            relation_valid if relation_valid is not None else solve_flux_valid
        )

    if cell_flux_pair_relation is not None:
        relation_valid = _dynamics_tendency_pair_field(
            cell_flux_pair_relation, 2, "valid"
        )
        cell_flux_valid = (
            relation_valid if relation_valid is not None else cell_flux_valid
        )

    if edge_all_pair_relation is not None:
        relation_valid = _dynamics_tendency_pair_field(
            edge_all_pair_relation, 2, "valid"
        )
        edge_all_pair_valid = (
            relation_valid if relation_valid is not None else edge_all_pair_valid
        )

    if solve_edge_pair_relation is not None:
        edge_c1 = _dynamics_tendency_pair_field(solve_edge_pair_relation, 0, "first")
        edge_c2 = _dynamics_tendency_pair_field(solve_edge_pair_relation, 1, "second")
        relation_valid = _dynamics_tendency_pair_field(
            solve_edge_pair_relation, 2, "valid"
        )
        relation_valid_2d = _dynamics_tendency_pair_field(
            solve_edge_pair_relation, 3, "valid_2d"
        )
        relation_valid_all = _dynamics_tendency_pair_field(
            solve_edge_pair_relation, 4, "valid_all"
        )
        solve_edge_pair_valid = (
            relation_valid if relation_valid is not None else solve_edge_pair_valid
        )
        solve_edge_mask = (
            relation_valid_2d if relation_valid_2d is not None else solve_edge_mask
        )
        solve_edge_mask_all = (
            bool(relation_valid_all)
            if relation_valid_all is not None
            else solve_edge_mask_all
        )

    if safe_cell_edges is None or cell_edge_valid is None or cell_edge_sign is None:
        raise ValueError("dynamics-tendency requires the prebound cell-edge relation")

    # Intermediate arrays (needed for del4 diffusion).  When static relations prove
    # all later gathers are satisfied from active patches, keep only a tiny typed
    # base for shape/dtype/device metadata instead of a full mesh-sized zero field.
    delsq_u_compact_safe = _slot_patch_rows_are_local(
        cell_edge_relation,
        vertex_edge_relation,
    )
    delsq_divergence_compact_safe = _pair_patch_rows_are_local(solve_edge_pair_relation)
    delsq_vorticity_compact_safe = _pair_patch_rows_are_local(
        solve_edge_vertex_relation
    )
    delsq_theta_compact_safe = _pair_patch_rows_are_local(solve_flux_cell_relation)
    delsq_w_compact_safe = delsq_theta_compact_safe
    delsq_u = _scratch_base_or_compact(
        (nEdges + 1, nVertLevels), compact_safe=delsq_u_compact_safe
    )
    delsq_divergence = _scratch_base_or_compact(
        (nCells + 1, nVertLevels), compact_safe=delsq_divergence_compact_safe
    )
    delsq_vorticity = _scratch_base_or_compact(
        (nVertices + 1, nVertLevels), compact_safe=delsq_vorticity_compact_safe
    )
    delsq_theta = _scratch_base_or_compact(
        (nCells + 1, nVertLevels), compact_safe=delsq_theta_compact_safe
    )
    delsq_w = _scratch_base_or_compact(
        (nCells + 1, nVertLevels + 1), compact_safe=delsq_w_compact_safe
    )

    # =========================================================================
    # SECTION 1: rk_step == 1 initialization

    # =========================================================================

    (
        tend_u_euler_initialized_patch,
        kdiff_initialized_patch,
        h_mom_eddy_visc4,
        h_theta_eddy_visc4,
    ) = section_ops.mixing_initialization_mixing_initialization(
        rk_step=rk_step,
        edge_count=int(edge_end) - int(edge_start),
        n_vert_levels=nVertLevels,
        config_horiz_mixing=config_horiz_mixing,
        cell_edge_relation=cell_edge_relation,
        defc_a_cell=defc_a[cell_idx, :cell_slot_count],
        defc_b_cell=defc_b[cell_idx, :cell_slot_count],
        u=u,
        v=v,
        c_s=c_s,
        config_len_disp=config_len_disp,
        inv_dt=invDt,
        config_h_theta_eddy_visc2=config_h_theta_eddy_visc2,
        config_visc4_2dsmag=config_visc4_2dsmag,
        config_h_mom_eddy_visc4=config_h_mom_eddy_visc4,
        config_h_theta_eddy_visc4=config_h_theta_eddy_visc4,
        config_mpas_cam_coef=config_mpas_cam_coef,
        config_number_cam_damping_levels=config_number_cam_damping_levels,
    )
    if rk_step == 1:
        if tend_u_euler_initialized_patch is None or kdiff_initialized_patch is None:
            raise RuntimeError(
                "dynamics-tendency mixing-initialization omitted first-stage mixing outputs"
            )
        tend_u_euler_edge_patch = tend_u_euler_initialized_patch
        kdiff_cell_patch = kdiff_initialized_patch

    # =========================================================================
    # SECTION 2: Tendency for density (horizontal mass flux divergence)

    # =========================================================================

    h_div_patch, tend_rho_patch, dpdz_patch = section_ops.density_tendency_density(
        ru=ru,
        rw_cells=rw[cell_idx, :],
        rb_cells=rb[cell_idx, :],
        rr_save_cells=rr_save[cell_idx, :],
        qtot_cells=qtot[cell_idx, :],
        dv_edge=dvEdge,
        inv_area_cell=invAreaCell[cell_idx],
        rdzw=rdzw,
        cell_edge_relation=cell_edge_relation,
        gravity=gravity,
        include_density_update=rk_step == 1,
        tend_rho_physics_cells=(
            tend_rho_physics[cell_idx, :] if tend_rho_physics is not None else None
        ),
    )
    h_divergence_cell_patch = h_div_patch
    if rk_step == 1:
        if tend_rho_patch is None or dpdz_patch is None:
            raise RuntimeError(
                "dynamics-tendency density-tendency omitted first-stage density outputs"
            )
        tend_rho_cell_patch = tend_rho_patch
        dpdz_cell_patch = dpdz_patch

    # =========================================================================
    # SECTION 3: Compute u (normal) velocity tendency

    # =========================================================================

    # Horizontal pressure gradient, vertical transport, nonlinear Coriolis/PV,
    # KE gradient, divergence damping, and optional curvature for tend_u.
    # .
    normal_velocity_tendency_normal_velocity_section = (
        section_ops.normal_velocity_tendency_normal_velocity
    )

    tend_u_patch, tend_u_euler_patch = normal_velocity_tendency_normal_velocity_section(
        edge_cell_relation=solve_edge_pair_relation,
        edge_neighbor_relation=edge_neighbor_relation,
        edge_solve_start=edge_solve_start,
        edge_solve_end=edge_solve_end,
        tend_u_euler_source_row_start=(
            edge_start if tend_u_euler_edge_patch is not None else 0
        ),
        u=u,
        rw=rw,
        ke=ke,
        h_divergence=h_divergence,
        h_divergence_cell_patch=h_divergence_cell_patch,
        pv_edge=pv_edge,
        rho_edge=rho_edge,
        inv_dc_edge=invDcEdge,
        rdzw=rdzw,
        fzm=fzm,
        fzp=fzp,
        solve_edge_mask=solve_edge_mask,
        solve_edge_mask_all=solve_edge_mask_all,
        n_vert_levels=nVertLevels,
        tend_u_euler_source=(
            tend_u_euler_edge_patch
            if tend_u_euler_edge_patch is not None
            else tend_u_euler
        ),
        include_pressure_gradient=rk_step == 1,
        zz=zz,
        pp=pp,
        dpdz=dpdz,
        dpdz_cell_patch=dpdz_cell_patch,
        cell_patch_row_start=cell_start,
        cell_patch_row_end=cell_end,
        cqu=cqu,
        zxu=zxu,
        include_curvature=bool(config_curvature_u),
        w=w,
        angle_edge=angleEdge,
        lat_edge=latEdge,
        omega=omega,
        inv_r_earth=inv_r_earth,
    )

    if rk_step == 1:
        if tend_u_euler_edge_patch is None:
            tend_u_euler_edge_patch = tend_u_euler[edge_start:edge_end, :]
        edge_solve_rel_start = edge_solve_start - edge_start
        edge_solve_rel_end = edge_solve_end - edge_start
        if (
            edge_solve_rel_start == 0
            and edge_solve_rel_end == tend_u_euler_edge_patch.shape[0]
        ):
            tend_u_euler_edge_patch = tend_u_euler_patch
        else:
            updated_tend_u_euler_edge_patch = tend_u_euler_edge_patch.clone()
            updated_tend_u_euler_edge_patch[
                edge_solve_rel_start:edge_solve_rel_end, :
            ] = tend_u_euler_patch
            tend_u_euler_edge_patch = updated_tend_u_euler_edge_patch

    tend_u_solve_patch = tend_u_patch
    apply_rayleigh = config_rayleigh_damp_u == 1
    rayleigh_coef_inverse = 0.0
    if apply_rayleigh:
        rayleigh_coef_inverse = 1.0 / (
            float(config_number_rayleigh_damp_u_levels)
            * (config_rayleigh_damp_u_timescale_days * seconds_per_day)
        )
    normal_velocity_mixing_normal_velocity_finalization_completed = False
    # =========================================================================
    # SECTION 4: Horizontal mixing for u (del2 and del4)

    # =========================================================================

    if rk_step == 1:
        # Compute del2(u) and add 2nd-order diffusion to tend_u_euler

        diff_edges = edge_idx_all

        if isinstance(edge_all_inv_dc, torch.Tensor) and isinstance(
            edge_all_min_inv_dv_dc,
            torch.Tensor,
        ):
            r_dc = edge_all_inv_dc
            r_dv = edge_all_min_inv_dv_dc
        else:
            r_dc = invDcEdge[diff_edges]
            r_dv = torch.minimum(invDvEdge[diff_edges], 4.0 * invDcEdge[diff_edges])
        if edge_all_pair_relation is None:
            raise RuntimeError(
                "dynamics-tendency normal-velocity-mixing del2 diffusion requires static edge-cell pairs"
            )
        if edge_all_vertex_relation is None:
            raise RuntimeError(
                "dynamics-tendency normal-velocity-mixing del2 diffusion requires static edge-vertex pairs"
            )
        if tend_u_euler_edge_patch is None:
            tend_u_euler_edge_patch = tend_u_euler[edge_start:edge_end, :]
        include_del4 = h_mom_eddy_visc4 > 0.0
        vertex_edge_relation_arg = None
        cell_edge_relation_arg = None
        solve_edge_pair_relation_arg = None
        solve_edge_vertex_relation_arg = None
        vertex_edge_sign_arg = None
        cell_div_edge_sign_arg = None
        del4_r_dc = None
        del4_r_dv = None

        # 4th-order mixing (if h_mom_eddy_visc4 > 0)

        if include_del4:
            # Compute del2(vorticity) at vertices

            if not (
                isinstance(vort_vertices_payload, torch.Tensor)
                and isinstance(vertex_edges_payload, torch.Tensor)
                and isinstance(vertex_del2_edge_sign, torch.Tensor)
                and vertex_edge_relation is not None
            ):
                raise RuntimeError(
                    "dynamics-tendency normal-velocity-mixing del4 requires prebound vertex-edge topology"
                )
            vertex_edge_sign_arg = vertex_del2_edge_sign
            vertex_edge_relation_arg = vertex_edge_relation

            # Compute del2(divergence) at cells

            div_cells = cell_idx
            safe_edges = safe_cell_edges
            if isinstance(cell_div_del2_edge_sign, torch.Tensor):
                cell_div_edge_sign_arg = cell_div_del2_edge_sign
            else:
                cell_div_edge_sign_arg = (
                    invAreaCell[div_cells].unsqueeze(1)
                    * dvEdge[safe_edges]
                    * cell_edge_sign
                    * cell_edge_valid.to(dtype=DTYPE)
                )
            if cell_edge_relation is None:
                raise RuntimeError(
                    "dynamics-tendency normal-velocity-mixing del4 requires prebound cell-edge topology"
                )
            cell_edge_relation_arg = cell_edge_relation

            # Apply del4 diffusion to tend_u_euler

            if (
                isinstance(solve_del4_mesh_scaling, torch.Tensor)
                and isinstance(solve_del4_inv_dc, torch.Tensor)
                and isinstance(solve_del4_min_inv_dv_dc, torch.Tensor)
            ):
                u_mix_scale = solve_del4_mesh_scaling * h_mom_eddy_visc4
                del4_r_dc = u_mix_scale * config_del4u_div_factor * solve_del4_inv_dc
                del4_r_dv = u_mix_scale * solve_del4_min_inv_dv_dc
            else:
                del4_edges = solve_edge_idx
                u_mix_scale = meshScalingDel4[del4_edges] * h_mom_eddy_visc4
                del4_r_dc = (
                    u_mix_scale * config_del4u_div_factor * invDcEdge[del4_edges]
                )
                del4_r_dv = u_mix_scale * torch.minimum(
                    invDvEdge[del4_edges], 4.0 * invDcEdge[del4_edges]
                )

            if solve_edge_pair_relation is None or solve_edge_vertex_relation is None:
                raise RuntimeError(
                    "dynamics-tendency normal-velocity-mixing del4 requires prebound edge pair topology"
                )
            solve_edge_pair_relation_arg = solve_edge_pair_relation
            solve_edge_vertex_relation_arg = solve_edge_vertex_relation

        # Vertical mixing for u (2nd order in z-space)

        include_vertical_mix = v_mom_eddy_visc2 > 0.0
        u_mix_solve = None
        rho_edge_solve_patch = None
        zgrid_cell1 = None
        zgrid_cell2 = None
        if include_del4 or include_vertical_mix or apply_rayleigh:
            rho_edge_solve_patch = rho_edge[solve_edge_slice, :]
        if include_vertical_mix:
            mix_edges = solve_edge_idx
            if config_mix_full == 1:  # Mix full state
                u_mix_solve = u[mix_edges, :]
            else:  # Mix perturbation from initial state
                cos_edge = torch.cos(angleEdge[mix_edges]).unsqueeze(1)
                sin_edge = torch.sin(angleEdge[mix_edges]).unsqueeze(1)
                u_mix_solve = (
                    u[mix_edges, :]
                    - u_init.unsqueeze(0) * cos_edge
                    - v_init.unsqueeze(0) * sin_edge
                )
            zgrid_cell1 = zgrid[edge_c1, :]
            zgrid_cell2 = zgrid[edge_c2, :]
        edge_solve_rel_start = edge_solve_start - edge_start
        edge_solve_rel_end = edge_solve_end - edge_start
        normal_velocity_mixing_normal_velocity_finalization_u_tendency = (
            section_ops.normal_velocity_mixing_normal_velocity_finalization_u_tendency
        )
        if normal_velocity_mixing_normal_velocity_finalization_u_tendency is not None:
            tend_u_euler_edge_patch, tend_u_solve_patch = (
                normal_velocity_mixing_normal_velocity_finalization_u_tendency(
                    edge_cell_relation=edge_all_pair_relation,
                    edge_vertex_relation=edge_all_vertex_relation,
                    divergence=divergence,
                    vorticity=vorticity,
                    kdiff=kdiff,
                    kdiff_patch=kdiff_cell_patch,
                    patch_row_start=cell_start,
                    patch_row_end=cell_end,
                    rho_edge_patch=rho_edge[diff_edges, :],
                    tend_u_euler_edge_patch=tend_u_euler_edge_patch,
                    inv_dc_edge_patch=r_dc,
                    min_inv_dv_dc_edge_patch=r_dv,
                    mesh_scaling_del2_patch=meshScalingDel2[diff_edges],
                    edge_valid=edge_all_pair_valid,
                    include_del4=include_del4,
                    vertex_edge_relation=vertex_edge_relation_arg,
                    cell_edge_relation=cell_edge_relation_arg,
                    solve_edge_pair_relation=solve_edge_pair_relation_arg,
                    solve_edge_vertex_relation=solve_edge_vertex_relation_arg,
                    delsq_u=delsq_u,
                    delsq_divergence=delsq_divergence,
                    delsq_vorticity=delsq_vorticity,
                    vertex_edge_sign=vertex_edge_sign_arg,
                    cell_div_edge_sign=cell_div_edge_sign_arg,
                    r_dc_solve=del4_r_dc,
                    r_dv_solve=del4_r_dv,
                    rho_edge_solve_patch=rho_edge_solve_patch,
                    solve_edge_valid=solve_edge_pair_valid,
                    edge_patch_row_start=edge_start,
                    edge_patch_row_end=edge_end,
                    cell_patch_row_start=cell_start,
                    cell_patch_row_end=cell_end,
                    vertex_patch_row_start=vertex_start,
                    vertex_patch_row_end=vertex_end,
                    edge_solve_rel_start=edge_solve_rel_start,
                    edge_solve_rel_end=edge_solve_rel_end,
                    include_vertical_mix=include_vertical_mix,
                    u_mix_solve=u_mix_solve,
                    zgrid_cell1=zgrid_cell1,
                    zgrid_cell2=zgrid_cell2,
                    v_mom_eddy_visc2=v_mom_eddy_visc2,
                    tend_u_solve=tend_u_solve_patch,
                    u_solve_rayleigh=(
                        u[solve_edge_slice, :] if apply_rayleigh else None
                    ),
                    tend_ru_physics_solve=(
                        tend_ru_physics[solve_edge_slice, :]
                        if tend_ru_physics is not None
                        else None
                    ),
                    apply_rayleigh=apply_rayleigh,
                    rayleigh_coef_inverse=rayleigh_coef_inverse,
                    rayleigh_damp_level_count=(
                        config_number_rayleigh_damp_u_levels if apply_rayleigh else 0
                    ),
                    n_vert_levels=nVertLevels,
                )
            )
            normal_velocity_mixing_normal_velocity_finalization_completed = True
        else:
            normal_velocity_mixing_u_mixing_section = (
                section_ops.normal_velocity_mixing_u_mixing
            )
            tend_u_euler_edge_patch = normal_velocity_mixing_u_mixing_section(
                edge_cell_relation=edge_all_pair_relation,
                edge_vertex_relation=edge_all_vertex_relation,
                divergence=divergence,
                vorticity=vorticity,
                kdiff=kdiff,
                kdiff_patch=kdiff_cell_patch,
                patch_row_start=cell_start,
                patch_row_end=cell_end,
                rho_edge_patch=rho_edge[diff_edges, :],
                tend_u_euler_edge_patch=tend_u_euler_edge_patch,
                inv_dc_edge_patch=r_dc,
                min_inv_dv_dc_edge_patch=r_dv,
                mesh_scaling_del2_patch=meshScalingDel2[diff_edges],
                edge_valid=edge_all_pair_valid,
                include_del4=include_del4,
                vertex_edge_relation=vertex_edge_relation_arg,
                cell_edge_relation=cell_edge_relation_arg,
                solve_edge_pair_relation=solve_edge_pair_relation_arg,
                solve_edge_vertex_relation=solve_edge_vertex_relation_arg,
                delsq_u=delsq_u,
                delsq_divergence=delsq_divergence,
                delsq_vorticity=delsq_vorticity,
                vertex_edge_sign=vertex_edge_sign_arg,
                cell_div_edge_sign=cell_div_edge_sign_arg,
                r_dc_solve=del4_r_dc,
                r_dv_solve=del4_r_dv,
                rho_edge_solve_patch=rho_edge_solve_patch,
                solve_edge_valid=solve_edge_pair_valid,
                edge_patch_row_start=edge_start,
                edge_patch_row_end=edge_end,
                cell_patch_row_start=cell_start,
                cell_patch_row_end=cell_end,
                vertex_patch_row_start=vertex_start,
                vertex_patch_row_end=vertex_end,
                edge_solve_rel_start=edge_solve_rel_start,
                edge_solve_rel_end=edge_solve_rel_end,
                include_vertical_mix=include_vertical_mix,
                u_mix_solve=u_mix_solve,
                zgrid_cell1=zgrid_cell1,
                zgrid_cell2=zgrid_cell2,
                v_mom_eddy_visc2=v_mom_eddy_visc2,
                n_vert_levels=nVertLevels,
            )

    # =========================================================================
    # SECTION 5: Rayleigh damping on u

    # =========================================================================

    # Add Euler tendency to tend_u (no physics tendency in checkpoint)

    if not normal_velocity_mixing_normal_velocity_finalization_completed:
        if tend_u_solve_patch is None:
            tend_u_solve_patch = tend_u[edge_solve_start:edge_solve_end, :]
        tend_u_euler_solve_patch = (
            tend_u_euler_edge_patch[
                edge_solve_start - edge_start : edge_solve_end - edge_start,
                :,
            ]
            if tend_u_euler_edge_patch is not None
            else tend_u_euler[edge_solve_start:edge_solve_end, :]
        )
        normal_velocity_finalization_u_final_section = (
            section_ops.normal_velocity_finalization_u_final
        )
        tend_u_solve_patch = normal_velocity_finalization_u_final_section(
            tend_u_solve=tend_u_solve_patch,
            tend_u_euler_solve=tend_u_euler_solve_patch,
            rho_edge_solve=(rho_edge[solve_edge_slice, :] if apply_rayleigh else None),
            u_solve=(u[solve_edge_slice, :] if apply_rayleigh else None),
            tend_ru_physics_solve=(
                tend_ru_physics[solve_edge_slice, :]
                if tend_ru_physics is not None
                else None
            ),
            apply_rayleigh=apply_rayleigh,
            rayleigh_coef_inverse=rayleigh_coef_inverse,
            rayleigh_damp_level_count=(
                config_number_rayleigh_damp_u_levels if apply_rayleigh else 0
            ),
            n_vert_levels=nVertLevels,
        )

    # =========================================================================
    # SECTION 6: RHS for w - horizontal advection

    # =========================================================================

    w_cells = solve_cells
    vertical_velocity_horizontal_advection_w_hadv_section = (
        section_ops.vertical_velocity_horizontal_advection_w_hadv
    )
    tend_w_patch = vertical_velocity_horizontal_advection_w_hadv_section(
        w=w,
        ru=ru,
        edge_solve_start=edge_solve_start,
        edge_solve_end=edge_solve_end,
        fzm=fzm,
        fzp=fzp,
        n_vert_levels=nVertLevels,
        edge_adv_cell_relation=edge_adv_cell_relation,
        local_edge_relation=solve_local_edge_relation,
        include_curvature=bool(config_curvature_u),
        ur_cell_solve=ur_cell[w_cells, :],
        vr_cell_solve=vr_cell[w_cells, :],
        rho_zz_solve=rho_zz[w_cells, :],
        lat_cell_solve=latCell[w_cells],
        omega=omega,
        r_earth=r_earth,
        cell_owned_vertical_velocity_horizontal_advection_theta_horizontal_advection_ab=bool(
            config_dynamics_tendency_cell_owned_vertical_velocity_horizontal_advection_theta_horizontal_advection_ab
        ),
    )
    tend_w_solve_inner_patch = tend_w_patch
    # =========================================================================
    # SECTION 7: Horizontal mixing for w

    # =========================================================================

    vertical_velocity_horizontal_mixing_vertical_velocity_vertical_transport_w_tendency = section_ops.vertical_velocity_horizontal_mixing_vertical_velocity_vertical_transport_w_tendency
    if rk_step == 1:
        safe_edges = safe_cell_edges
        cell_mesh_del2_slots = None
        edge_sign = (
            0.5
            * invAreaCell[cell_idx].unsqueeze(1)
            * cell_edge_sign
            * dvEdge[safe_edges]
            * invDcEdge[safe_edges]
        )
        mesh_scaling_del2_arg = meshScalingDel2
        edge_sign_del4 = None
        if h_mom_eddy_visc4 > 0.0:
            r_area_cell = h_mom_eddy_visc4 * invAreaCell[solve_cells]
            edge_sign_del4 = (
                meshScalingDel4[solve_safe_edges]
                * r_area_cell.unsqueeze(1)
                * dvEdge[solve_safe_edges]
                * solve_edge_sign
                * invDcEdge[solve_safe_edges]
            )
        if (
            vertical_velocity_horizontal_mixing_vertical_velocity_vertical_transport_w_tendency
            is None
        ):
            vertical_velocity_horizontal_mixing_w_horizontal_mix_section = (
                section_ops.vertical_velocity_horizontal_mixing_w_horizontal_mix
            )
            tend_w_euler_cell_patch = (
                vertical_velocity_horizontal_mixing_w_horizontal_mix_section(
                    cell_edge_relation=cell_flux_edge_relation,
                    edge_cell_relation=cell_flux_pair_relation,
                    solve_cell_pair_relation=solve_flux_cell_relation,
                    edge_sign_del2=edge_sign,
                    edge_sign_del4=edge_sign_del4,
                    rho_edge=rho_edge,
                    w=w,
                    kdiff=kdiff,
                    delsq_w_base=delsq_w,
                    n_vert_levels=nVertLevels,
                    patch_row_start=cell_start,
                    patch_row_end=cell_end,
                    solve_relative_start=cell_solve_start - cell_start,
                    solve_relative_end=cell_solve_end - cell_start,
                    solve_flux_valid_all_by_slot=solve_flux_valid_all_by_slot,
                    kdiff_patch=kdiff_cell_patch,
                    mesh_scaling_del2=mesh_scaling_del2_arg,
                    mesh_scaling_del2_slots=cell_mesh_del2_slots,
                )
            )
    # =========================================================================
    # SECTION 8: Vertical advection, pressure gradient and buoyancy for w

    # =========================================================================

    if tend_w_solve_inner_patch is None:
        tend_w_solve_inner_patch = tend_w[
            cell_solve_start:cell_solve_end, 1:nVertLevels
        ]
    cell_slice = slice(cell_solve_start, cell_solve_end)
    cell_solve_rel_start = cell_solve_start - cell_start
    cell_solve_rel_end = cell_solve_end - cell_start
    if tend_w_euler_cell_patch is None:
        tend_w_euler_solve_inner_patch = tend_w_euler[cell_slice, 1:nVertLevels]
    else:
        tend_w_euler_solve_inner_patch = tend_w_euler_cell_patch[
            cell_solve_rel_start:cell_solve_rel_end,
            1:nVertLevels,
        ]
    dpdz_solve_patch = (
        dpdz_cell_patch[cell_solve_rel_start:cell_solve_rel_end, :]
        if dpdz_cell_patch is not None
        else dpdz[cell_slice, :]
    )
    if (
        rk_step == 1
        and vertical_velocity_horizontal_mixing_vertical_velocity_vertical_transport_w_tendency
        is not None
    ):
        tend_w_solve_inner_patch, tend_w_euler_cell_patch = (
            vertical_velocity_horizontal_mixing_vertical_velocity_vertical_transport_w_tendency(
                cell_edge_relation=cell_flux_edge_relation,
                edge_cell_relation=cell_flux_pair_relation,
                solve_cell_pair_relation=solve_flux_cell_relation,
                edge_sign_del2=edge_sign,
                edge_sign_del4=edge_sign_del4,
                rho_edge=rho_edge,
                w=w,
                kdiff=kdiff,
                delsq_w_base=delsq_w,
                n_vert_levels=nVertLevels,
                patch_row_start=cell_start,
                patch_row_end=cell_end,
                solve_relative_start=cell_solve_rel_start,
                solve_relative_end=cell_solve_rel_end,
                solve_flux_valid_all_by_slot=solve_flux_valid_all_by_slot,
                tend_w_solve_inner=tend_w_solve_inner_patch,
                rw_solve=rw[cell_slice, :],
                pp_solve=pp[cell_slice, :],
                cqw_solve=cqw[cell_slice, :],
                rho_zz_solve=rho_zz[cell_slice, :],
                dpdz_solve=dpdz_solve_patch,
                inv_area_cell_solve=invAreaCell[cell_slice],
                rdzu=rdzu,
                rdzw=rdzw,
                fzm=fzm,
                fzp=fzp,
                include_pressure_gradient=True,
                include_vertical_mix=v_mom_eddy_visc2 > 0.0,
                v_mom_eddy_visc2=v_mom_eddy_visc2,
                kdiff_patch=kdiff_cell_patch,
                mesh_scaling_del2=mesh_scaling_del2_arg,
                mesh_scaling_del2_slots=cell_mesh_del2_slots,
            )
        )
    else:
        vertical_velocity_vertical_transport_w_vertical_section = (
            section_ops.vertical_velocity_vertical_transport_w_vertical
        )
        tend_w_solve_inner_patch, tend_w_euler_solve_inner_patch = (
            vertical_velocity_vertical_transport_w_vertical_section(
                w_solve=w[cell_slice, :],
                rw_solve=rw[cell_slice, :],
                pp_solve=pp[cell_slice, :],
                cqw_solve=cqw[cell_slice, :],
                rho_zz_solve=rho_zz[cell_slice, :],
                dpdz_solve=dpdz_solve_patch,
                tend_w_solve_inner=tend_w_solve_inner_patch,
                tend_w_euler_solve_inner=tend_w_euler_solve_inner_patch,
                inv_area_cell_solve=invAreaCell[cell_slice],
                rdzu=rdzu,
                rdzw=rdzw,
                fzm=fzm,
                fzp=fzp,
                n_vert_levels=nVertLevels,
                include_pressure_gradient=rk_step == 1,
                include_vertical_mix=rk_step == 1 and v_mom_eddy_visc2 > 0.0,
                v_mom_eddy_visc2=v_mom_eddy_visc2,
            )
        )
        if rk_step == 1:
            if tend_w_euler_cell_patch is None:
                tend_w_euler_cell_patch = tend_w_euler[cell_start:cell_end, :]
            tend_w_euler_cell_patch = _dynamics_tendency_replace_patch_rows_cols(
                tend_w_euler_cell_patch,
                cell_solve_rel_start,
                cell_solve_rel_end,
                1,
                nVertLevels,
                tend_w_euler_solve_inner_patch,
            )

    # =========================================================================
    # SECTION 9: RHS for theta - horizontal advection

    # =========================================================================

    theta_horizontal_advection_theta_finalization_theta_tendency = (
        section_ops.theta_horizontal_advection_theta_finalization_theta_tendency
    )
    if theta_horizontal_advection_theta_finalization_theta_tendency is None:
        theta_horizontal_advection_theta_hadv_section = (
            section_ops.theta_horizontal_advection_theta_hadv
        )
        tend_theta_patch = theta_horizontal_advection_theta_hadv_section(
            theta_m=theta_m,
            ru=ru,
            edge_solve_start=edge_solve_start,
            edge_solve_end=edge_solve_end,
            n_vert_levels=nVertLevels,
            edge_adv_cell_relation=edge_adv_cell_relation,
            local_edge_relation=solve_local_edge_relation,
            include_perturbation_flux=rk_step > 1,
            cell_edge_relation=solve_cell_edge_relation,
            edge_cell_relation=solve_flux_cell_relation,
            ru_save=ru_save,
            theta_m_save=theta_m_save,
            cell_owned_vertical_velocity_horizontal_advection_theta_horizontal_advection_ab=bool(
                config_dynamics_tendency_cell_owned_vertical_velocity_horizontal_advection_theta_horizontal_advection_ab
            ),
        )
        tend_theta_solve_patch = tend_theta_patch
    else:
        cell_slice = slice(cell_solve_start, cell_solve_end)
        cell_solve_rel_start = cell_solve_start - cell_start
        cell_solve_rel_end = cell_solve_end - cell_start
        tend_rho_solve_patch = (
            tend_rho_cell_patch[cell_solve_rel_start:cell_solve_rel_end, :]
            if tend_rho_cell_patch is not None
            else tend_rho[cell_slice, :]
        )
        theta_edge_sign = None
        theta_edge_sign_del4 = None
        theta_pr_scale = None
        if rk_step == 1:
            theta_edge_sign = (
                invAreaCell[cell_idx].unsqueeze(1)
                * cell_edge_sign
                * dvEdge[safe_cell_edges]
                * invDcEdge[safe_cell_edges]
            )
            theta_pr_scale = prandtl_inv * meshScalingDel2[safe_cell_edges]
            if h_theta_eddy_visc4 > 0.0:
                r_area_cell = (
                    h_theta_eddy_visc4 * prandtl_inv * invAreaCell[solve_cells]
                )
                theta_edge_sign_del4 = (
                    meshScalingDel4[solve_safe_edges]
                    * r_area_cell.unsqueeze(1)
                    * dvEdge[solve_safe_edges]
                    * solve_edge_sign
                    * invDcEdge[solve_safe_edges]
                )
        apply_theta_vertical_mix = rk_step == 1 and v_theta_eddy_visc2 > 0.0
        theta_work_solve = None
        if apply_theta_vertical_mix:
            if config_mix_full == 1:
                theta_work_solve = theta_m[cell_slice, :]
            else:
                theta_work_solve = theta_m[cell_slice, :] - t_init[cell_slice, :]
        (
            tend_theta_solve_patch,
            tend_theta_euler_cell_patch,
            rthdynten_solve_patch,
        ) = theta_horizontal_advection_theta_finalization_theta_tendency(
            theta_m=theta_m,
            ru=ru,
            edge_solve_start=edge_solve_start,
            edge_solve_end=edge_solve_end,
            n_vert_levels=nVertLevels,
            edge_adv_cell_relation=edge_adv_cell_relation,
            local_edge_relation=solve_local_edge_relation,
            include_perturbation_flux=rk_step > 1,
            solve_cell_edge_relation=solve_cell_edge_relation,
            solve_edge_cell_relation=solve_flux_cell_relation,
            ru_save=ru_save,
            theta_m_save=theta_m_save,
            cell_owned_vertical_velocity_horizontal_advection_theta_horizontal_advection_ab=bool(
                config_dynamics_tendency_cell_owned_vertical_velocity_horizontal_advection_theta_horizontal_advection_ab
            ),
            include_horizontal_mix=rk_step == 1,
            mix_cell_edge_relation=(cell_flux_edge_relation if rk_step == 1 else None),
            mix_edge_cell_relation=(cell_flux_pair_relation if rk_step == 1 else None),
            solve_cell_pair_relation=(
                solve_flux_cell_relation if rk_step == 1 else None
            ),
            edge_sign_del2=theta_edge_sign,
            edge_sign_del4=theta_edge_sign_del4,
            rho_edge=rho_edge,
            kdiff=kdiff,
            delsq_theta_base=delsq_theta,
            pr_scale=theta_pr_scale,
            patch_row_start=cell_start,
            patch_row_end=cell_end,
            solve_relative_start=cell_solve_rel_start,
            solve_relative_end=cell_solve_rel_end,
            solve_flux_valid_all_by_slot=solve_flux_valid_all_by_slot,
            theta_m_solve=theta_m[cell_slice, :],
            theta_m_save_solve=theta_m_save[cell_slice, :],
            rw_solve=rw[cell_slice, :],
            rw_save_solve=rw_save[cell_slice, :],
            rho_zz_solve=rho_zz[cell_slice, :],
            rt_diabatic_tend_solve=rt_diabatic_tend[cell_slice, :],
            tend_rho_solve=tend_rho_solve_patch,
            inv_area_cell_solve=invAreaCell[cell_slice],
            rdzw=rdzw,
            fzm=fzm,
            fzp=fzp,
            coef_3rd_order=coef_3rd_order,
            tend_theta_euler_cell_base=(
                tend_theta_euler[cell_start:cell_end, :]
                if apply_theta_vertical_mix
                else None
            ),
            tend_theta_euler_solve_base=tend_theta_euler[cell_slice, :],
            theta_work_solve=theta_work_solve,
            zgrid_solve=(zgrid[cell_slice, :] if apply_theta_vertical_mix else None),
            tend_rtheta_physics_solve=(
                tend_rtheta_physics[cell_slice, :]
                if tend_rtheta_physics is not None
                else None
            ),
            apply_vertical_mix=apply_theta_vertical_mix,
            v_theta_eddy_visc2=v_theta_eddy_visc2,
            prandtl_inv=prandtl_inv,
            kdiff_patch=kdiff_cell_patch,
        )
    # =========================================================================
    # SECTION 10: Horizontal mixing for theta_m

    # =========================================================================
    if (
        rk_step == 1
        and theta_horizontal_advection_theta_finalization_theta_tendency is None
    ):
        edge_sign = (
            invAreaCell[cell_idx].unsqueeze(1)
            * cell_edge_sign
            * dvEdge[safe_cell_edges]
            * invDcEdge[safe_cell_edges]
        )
        pr_scale = prandtl_inv * meshScalingDel2[safe_cell_edges]
        edge_sign_del4 = None
        if h_theta_eddy_visc4 > 0.0:
            r_area_cell = h_theta_eddy_visc4 * prandtl_inv * invAreaCell[solve_cells]
            edge_sign_del4 = (
                meshScalingDel4[solve_safe_edges]
                * r_area_cell.unsqueeze(1)
                * dvEdge[solve_safe_edges]
                * solve_edge_sign
                * invDcEdge[solve_safe_edges]
            )
        theta_horizontal_mixing_theta_horizontal_mix_section = (
            section_ops.theta_horizontal_mixing_theta_horizontal_mix
        )
        tend_theta_euler_cell_patch = (
            theta_horizontal_mixing_theta_horizontal_mix_section(
                cell_edge_relation=cell_flux_edge_relation,
                edge_cell_relation=cell_flux_pair_relation,
                solve_cell_pair_relation=solve_flux_cell_relation,
                edge_sign_del2=edge_sign,
                edge_sign_del4=edge_sign_del4,
                rho_edge=rho_edge,
                theta_m=theta_m,
                kdiff=kdiff,
                delsq_theta_base=delsq_theta,
                patch_row_start=cell_start,
                patch_row_end=cell_end,
                solve_relative_start=cell_solve_start - cell_start,
                solve_relative_end=cell_solve_end - cell_start,
                solve_flux_valid_all_by_slot=solve_flux_valid_all_by_slot,
                kdiff_patch=kdiff_cell_patch,
                pr_scale=pr_scale,
            )
        )

    # =========================================================================
    # SECTION 11: Vertical advection plus diabatic term for theta

    # =========================================================================

    if tend_theta_solve_patch is None:
        tend_theta_solve_patch = tend_theta[cell_solve_start:cell_solve_end, :]
    cell_slice = slice(cell_solve_start, cell_solve_end)
    cell_solve_rel_start = cell_solve_start - cell_start
    cell_solve_rel_end = cell_solve_end - cell_start
    tend_rho_solve_patch = (
        tend_rho_cell_patch[cell_solve_rel_start:cell_solve_rel_end, :]
        if tend_rho_cell_patch is not None
        else tend_rho[cell_slice, :]
    )
    if theta_horizontal_advection_theta_finalization_theta_tendency is None:
        theta_vertical_advection_diabatic_theta_vertical_section = (
            section_ops.theta_vertical_advection_diabatic_theta_vertical
        )
        tend_theta_solve_patch, rthdynten_solve_patch = (
            theta_vertical_advection_diabatic_theta_vertical_section(
                theta_m_solve=theta_m[cell_slice, :],
                theta_m_save_solve=theta_m_save[cell_slice, :],
                rw_solve=rw[cell_slice, :],
                rw_save_solve=rw_save[cell_slice, :],
                rho_zz_solve=rho_zz[cell_slice, :],
                rt_diabatic_tend_solve=rt_diabatic_tend[cell_slice, :],
                tend_theta_solve=tend_theta_solve_patch,
                tend_rho_solve=tend_rho_solve_patch,
                inv_area_cell_solve=invAreaCell[cell_slice],
                rdzw=rdzw,
                fzm=fzm,
                fzp=fzp,
                coef_3rd_order=coef_3rd_order,
                n_vert_levels=nVertLevels,
            )
        )

    # =========================================================================
    # SECTION 12: Vertical mixing for theta (2nd order)

    # =========================================================================

    apply_theta_vertical_mix = rk_step == 1 and v_theta_eddy_visc2 > 0.0
    theta_work_solve = None
    if apply_theta_vertical_mix:
        if config_mix_full == 1:
            theta_work_solve = theta_m[cell_slice, :]
        else:
            theta_work_solve = theta_m[cell_slice, :] - t_init[cell_slice, :]

    # Add Euler tendency to tend_theta (no physics tendency in checkpoint)

    if tend_theta_solve_patch is None:
        tend_theta_solve_patch = tend_theta[cell_solve_start:cell_solve_end, :]
    if theta_horizontal_advection_theta_finalization_theta_tendency is None:
        theta_vertical_mixing_theta_finalization_theta_final_section = (
            section_ops.theta_vertical_mixing_theta_finalization_theta_final
        )
        tend_theta_solve_patch, tend_theta_euler_cell_patch = (
            theta_vertical_mixing_theta_finalization_theta_final_section(
                tend_theta_solve=tend_theta_solve_patch,
                tend_theta_euler_cell_patch=tend_theta_euler_cell_patch,
                tend_theta_euler_cell_base=(
                    tend_theta_euler[cell_start:cell_end, :]
                    if apply_theta_vertical_mix
                    else None
                ),
                tend_theta_euler_solve_base=tend_theta_euler[cell_slice, :],
                theta_work_solve=theta_work_solve,
                rho_zz_solve=(
                    rho_zz[cell_slice, :] if apply_theta_vertical_mix else None
                ),
                zgrid_solve=(
                    zgrid[cell_slice, :] if apply_theta_vertical_mix else None
                ),
                tend_rtheta_physics_solve=(
                    tend_rtheta_physics[cell_slice, :]
                    if tend_rtheta_physics is not None
                    else None
                ),
                apply_vertical_mix=apply_theta_vertical_mix,
                cell_solve_rel_start=cell_solve_rel_start,
                cell_solve_rel_end=cell_solve_rel_end,
                v_theta_eddy_visc2=v_theta_eddy_visc2,
                prandtl_inv=prandtl_inv,
                n_vert_levels=nVertLevels,
            )
        )

    if tend_w_solve_inner_patch is None:
        raise RuntimeError(
            "dynamics-tendency omitted its compact interior vertical tendency"
        )
    return (
        tend_u_solve_patch,
        tend_rho_cell_patch,
        tend_theta_solve_patch,
        tend_w_solve_inner_patch,
        tend_u_euler_edge_patch,
        tend_w_euler_cell_patch,
        tend_theta_euler_cell_patch,
        h_divergence_cell_patch,
        kdiff_cell_patch,
        rthdynten_solve_patch,
    )


@wraps(_dynamics_tendency_canonical_composition)
def atm_compute_dyn_tend_work(*args, **kwargs):
    """Run canonical dynamics-tendency math and pack its compact outputs at the public boundary."""
    debug_taps = kwargs.pop("_debug_taps", None)
    workspace = kwargs.pop("workspace", None)
    compile_region_names = kwargs.pop("_compile_region_names", ())
    compile_mode = kwargs.pop("_compile_mode", "reduce-overhead")
    compile_backend = kwargs.pop("_compile_backend", "inductor")
    profile_sections = bool(kwargs.pop("_profile_sections", False))
    bound = _DYNAMICS_TENDENCY_WORK_SIGNATURE.bind(*args, **kwargs)
    bound.apply_defaults()
    _validate_dynamics_tendency_numeric_plan_inputs(bound.arguments)
    if bound.arguments.get("_connectivity_payload") is None:
        profile_topology = profile_sections and not _compile_active()
        topology_context = (
            profile_section("mpas.dynamics_tendency.topology", enabled=True)
            if profile_topology
            else _NoProfileSection()
        )
        with topology_context:
            bound.arguments["_connectivity_payload"] = (
                prebind_dynamics_tendency_connectivity_payload(bound.arguments)
            )
    bound.arguments["_connectivity_payload"] = (
        _normalize_dynamics_tendency_numeric_connectivity_payload(
            bound.arguments["_connectivity_payload"]
        )
    )
    orchestration_arguments = dict(bound.arguments)
    orchestration_arguments.update(
        {
            "_compile_region_names": compile_region_names,
            "_compile_mode": compile_mode,
            "_compile_backend": compile_backend,
            "_profile_sections": profile_sections,
        }
    )
    section_ops = _resolve_dynamics_tendency_public_section_ops(orchestration_arguments)
    if debug_taps is not None:
        if not isinstance(debug_taps, dict):
            raise TypeError("_debug_taps must be a write-only dict")
        observer = _DynamicsTendencyDebugTapObserver(bound.arguments, debug_taps)
        section_ops = _observe_dynamics_tendency_section_ops(section_ops, observer)
    bound.arguments["_section_ops"] = section_ops
    compact_outputs = _dynamics_tendency_canonical_composition(
        *bound.args, **bound.kwargs
    )
    return pack_dynamics_tendency_numeric_outputs(
        compact_outputs,
        bound.arguments,
        workspace=workspace,
    )


_DYNAMICS_TENDENCY_WORK_SIGNATURE = inspect.signature(
    _dynamics_tendency_canonical_composition
)
_DYNAMICS_TENDENCY_WORK_ARGUMENT_NAMES = tuple(
    _DYNAMICS_TENDENCY_WORK_SIGNATURE.parameters
)


def _validate_dynamics_tendency_numeric_plan_inputs(
    call_arguments: Mapping[str, Any],
) -> None:
    """Validate orchestration contracts once, before entering compiled math."""
    tensor_names = (
        "u",
        "v",
        "w",
        "ru",
        "rw",
        "theta_m",
        "rho_zz",
        "rho_edge",
        "exner",
        "pp",
        "rr",
        "qtot",
    )
    missing = [name for name in tensor_names if name not in call_arguments]
    if missing:
        raise KeyError(
            f"dynamics-tendency numeric plan is missing validation inputs: {missing}"
        )
    require_consistent_device(
        "DynamicsTendencyNumericPlan",
        {name: call_arguments[name] for name in tensor_names},
    )
    u = call_arguments["u"]
    qtot = call_arguments["qtot"]
    if not isinstance(u, torch.Tensor) or not isinstance(qtot, torch.Tensor):
        raise TypeError(
            "dynamics-tendency numeric plan requires tensor u and qtot inputs"
        )
    n_cells = int(call_arguments["nCells"])
    n_vert_levels = int(call_arguments["nVertLevels"])
    expected_qtot_shape = (n_cells + 1, n_vert_levels)
    if tuple(qtot.shape) != expected_qtot_shape:
        raise ValueError(
            "qtot shape mismatch. Expected "
            f"{expected_qtot_shape}, got {tuple(qtot.shape)}."
        )
    cell_owned = bool(
        call_arguments.get(
            "config_dynamics_tendency_cell_owned_vertical_velocity_horizontal_advection_theta_horizontal_advection_ab",
            False,
        )
    )
    if cell_owned and u.device.type != "cuda":
        raise ValueError(
            "config_dynamics_tendency_cell_owned_vertical_velocity_horizontal_advection_theta_horizontal_advection_ab is GPU A/B-only; do not build "
            "a CPU dynamics-tendency numeric plan for it."
        )
    if cell_owned and is_grad_mode():
        raise ValueError(
            "config_dynamics_tendency_cell_owned_vertical_velocity_horizontal_advection_theta_horizontal_advection_ab is a forward-only GPU A/B "
            "experiment; do not build an AD-active dynamics-tendency numeric plan for it."
        )


@dataclass(frozen=True)
class DynamicsTendencyNumericPlan:
    """Runner-owned whole-dynamics-tendency callable with immutable inputs prebound."""

    key: tuple[object, ...]
    static_argument_names: tuple[str, ...]
    dynamic_argument_names: tuple[str, ...]
    static_arguments: Mapping[str, Any]
    numeric_callable: Callable[..., dict | tuple[torch.Tensor | None, ...]]

    def dynamic_arguments(self, call_arguments: Mapping[str, Any]) -> dict[str, Any]:
        missing = [
            name for name in self.dynamic_argument_names if name not in call_arguments
        ]
        if missing:
            raise KeyError(
                f"dynamics-tendency numeric call is missing dynamic arguments: {missing}"
            )
        return {name: call_arguments[name] for name in self.dynamic_argument_names}

    def dynamic_values(self, call_arguments: Mapping[str, Any]) -> tuple[Any, ...]:
        missing = [
            name for name in self.dynamic_argument_names if name not in call_arguments
        ]
        if missing:
            raise KeyError(
                f"dynamics-tendency numeric call is missing dynamic arguments: {missing}"
            )
        return tuple(call_arguments[name] for name in self.dynamic_argument_names)


def _bind_dynamics_tendency_numeric_callable(
    static_arguments: Mapping[str, Any],
) -> Callable[..., dict | tuple[torch.Tensor | None, ...]]:
    """Build a dynamic-only dynamics-tendency entry point with static values baked in."""
    parameters = _DYNAMICS_TENDENCY_WORK_ARGUMENT_NAMES
    dynamic_names = tuple(_DYNAMICS_TENDENCY_DYNAMIC_ARGUMENT_NAMES)
    dynamic_set = set(dynamic_names)
    missing_static = [
        name
        for name in parameters
        if name not in dynamic_set and name not in static_arguments
    ]
    extra_static = sorted(set(static_arguments) - set(parameters))
    if missing_static or extra_static:
        raise ValueError(
            "whole-dynamics-tendency numeric binding does not match the canonical function signature: "
            f"missing_static={missing_static}, extra_static={extra_static}"
        )

    namespace: dict[str, Any] = {
        "_dynamics_tendency_body": _dynamics_tendency_canonical_composition
    }
    call_names: list[str] = []
    for name in parameters:
        if name in dynamic_set:
            call_names.append(name)
            continue
        bound_name = f"_dynamics_tendency_static_{name}"
        namespace[bound_name] = static_arguments[name]
        call_names.append(bound_name)

    # This generated binder contains no equations. It gives Dynamo an explicit
    # dynamic signature without a functools.partial keyword dictionary while the
    # canonical dynamics-tendency body remains the only maintained implementation.
    source = (
        f"def _dynamics_tendency_bound_numeric_body({', '.join(dynamic_names)}):\n"
        f"    return _dynamics_tendency_body({', '.join(call_names)})\n"
    )
    exec(compile(source, __file__, "exec"), namespace)  # noqa: S102
    numeric_callable = namespace["_dynamics_tendency_bound_numeric_body"]
    if not callable(numeric_callable):
        raise TypeError("failed to construct whole-dynamics-tendency numeric callable")
    numeric_callable.__module__ = __name__
    numeric_callable.__qualname__ = "_dynamics_tendency_bound_numeric_body"
    return cast(
        Callable[..., dict[Any, Any] | tuple[torch.Tensor | None, ...]],
        numeric_callable,
    )


def _dynamics_tendency_static_argument_fingerprint(
    static_arguments: Mapping[str, Any],
) -> tuple[Any, ...]:
    """Return a mutation-sensitive fingerprint for dynamics-tendency immutable inputs."""

    def value_fingerprint(name: str, value: object) -> object:
        if isinstance(value, torch.Tensor):
            tensor_key = _tensor_version_cache_key(value)
            if tensor_key is None:
                raise ValueError(
                    f"dynamics-tendency static tensor {name!r} cannot be cache-keyed safely"
                )
            return tensor_key
        if isinstance(value, tuple):
            return tuple(
                value_fingerprint(f"{name}[{index}]", item)
                for index, item in enumerate(value)
            )
        if isinstance(value, StaticPairRelation):
            return (
                "StaticPairRelation",
                value.name,
                value_fingerprint(f"{name}.first", value.first),
                value_fingerprint(f"{name}.second", value.second),
                value_fingerprint(f"{name}.valid", value.valid),
                value_fingerprint(f"{name}.valid_2d", value.valid_2d),
                value.valid_all,
                value_fingerprint(f"{name}.first_local", value.first_local),
                value_fingerprint(f"{name}.second_local", value.second_local),
                value_fingerprint(f"{name}.first_local_valid", value.first_local_valid),
                value_fingerprint(
                    f"{name}.second_local_valid", value.second_local_valid
                ),
                value.first_local_all,
                value.second_local_all,
                value_fingerprint(f"{name}.active", value.active),
                value_fingerprint(f"{name}.active_3d", value.active_3d),
                value.active_all,
            )
        if isinstance(value, StaticSlotRelation):
            return (
                "StaticSlotRelation",
                value.name,
                value_fingerprint(f"{name}.indices", value.indices),
                value_fingerprint(f"{name}.valid", value.valid),
                value_fingerprint(f"{name}.sign", value.sign),
                value_fingerprint(f"{name}.weights", value.weights),
                value_fingerprint(f"{name}.coefficients", value.coefficients),
                value_fingerprint(f"{name}.coefficients_3rd", value.coefficients_3rd),
                value.valid_all_by_slot,
                value_fingerprint(f"{name}.valid_weight", value.valid_weight),
                value_fingerprint(f"{name}.local_indices", value.local_indices),
                value_fingerprint(f"{name}.local_valid", value.local_valid),
                value.local_all,
            )
        if isinstance(value, DynamicsTendencySectionOps):
            return (
                "DynamicsTendencySectionOps",
                tuple(
                    (
                        name,
                        getattr(value, name).__module__,
                        getattr(value, name).__qualname__,
                        id(getattr(value, name)),
                    )
                    for name in DYNAMICS_TENDENCY_SECTION_CALL_SEQUENCE
                ),
            )
        if isinstance(value, (str, int, float, bool, type(None), frozenset, slice)):
            if isinstance(value, slice):
                return ("slice", value.start, value.stop, value.step)
            return value
        return ("identity", type(value).__qualname__, id(value))

    fingerprint: list[object] = []
    for name in sorted(static_arguments):
        value = static_arguments[name]
        if not isinstance(
            value,
            (
                torch.Tensor,
                str,
                int,
                float,
                bool,
                type(None),
                tuple,
                frozenset,
                slice,
                DynamicsTendencySectionOps,
            ),
        ):
            raise TypeError(
                f"dynamics-tendency static argument {name!r} has unsupported plan-owned type "
                f"{type(value).__name__}"
            )
        fingerprint.append((name, value_fingerprint(name, value)))
    return tuple(fingerprint)


def build_dynamics_tendency_numeric_plan(
    call_arguments: Mapping[str, Any],
    *,
    stage_key: tuple[object, ...],
    require_canonical_section_ops: bool = True,
) -> DynamicsTendencyNumericPlan:
    """Bind immutable whole-dynamics-tendency inputs and expose a dynamic-tensor call."""
    call_arguments = dict(call_arguments)
    debug_taps = call_arguments.pop("_debug_taps", None)
    if debug_taps is not None:
        raise ValueError(
            "dynamics-tendency debug taps are public-wrapper diagnostics and cannot enter "
            "a numeric plan"
        )
    orchestration_control_names = {
        "workspace",
        "_compile_region_names",
        "_compile_mode",
        "_compile_backend",
        "_profile_sections",
        "_whole_stage_compile_body",
        "_return_compact_patches",
    }
    present_controls = sorted(orchestration_control_names & call_arguments.keys())
    if present_controls:
        raise ValueError(
            "whole-dynamics-tendency numeric plan requires orchestration controls outside the graph: "
            f"{present_controls}"
        )
    _validate_dynamics_tendency_numeric_plan_inputs(call_arguments)
    if call_arguments.get("_connectivity_payload") is None:
        raise ValueError(
            "dynamics-tendency numeric plan requires prebound connectivity"
        )
    dynamic_names = tuple(_DYNAMICS_TENDENCY_DYNAMIC_ARGUMENT_NAMES)
    missing_dynamic = [name for name in dynamic_names if name not in call_arguments]
    if missing_dynamic:
        raise KeyError(
            f"dynamics-tendency plan is missing dynamic arguments: {missing_dynamic}"
        )
    static_arguments = {
        name: value
        for name, value in call_arguments.items()
        if name not in set(dynamic_names)
    }
    for name, parameter in _DYNAMICS_TENDENCY_WORK_SIGNATURE.parameters.items():
        if (
            name not in set(dynamic_names)
            and name not in static_arguments
            and parameter.default is not inspect.Parameter.empty
        ):
            static_arguments[name] = parameter.default
    if static_arguments.get("_section_ops") is None:
        static_arguments["_section_ops"] = _CANONICAL_DYNAMICS_TENDENCY_SECTION_OPS
    for name in _DYNAMICS_TENDENCY_UNUSED_SOURCE_ARGUMENT_NAMES:
        static_arguments[name] = None
    if (
        require_canonical_section_ops
        and static_arguments.get("_section_ops")
        is not _CANONICAL_DYNAMICS_TENDENCY_SECTION_OPS
    ):
        raise ValueError(
            "whole-dynamics-tendency numeric plan requires orchestration controls outside the graph: "
            f"{{'_section_ops': {static_arguments.get('_section_ops')!r}}}"
        )
    connectivity_payload = static_arguments.get("_connectivity_payload")
    if connectivity_payload is None:
        raise ValueError(
            "dynamics-tendency numeric plan requires prebound connectivity"
        )
    static_arguments["_connectivity_payload"] = (
        _normalize_dynamics_tendency_numeric_connectivity_payload(connectivity_payload)
    )
    payload_source_arguments = {
        name: static_arguments[name]
        for name in _DYNAMICS_TENDENCY_PAYLOAD_OWNED_SOURCE_ARGUMENT_NAMES
    }
    payload_source_fingerprint = _dynamics_tendency_static_argument_fingerprint(
        payload_source_arguments
    )
    for name in _DYNAMICS_TENDENCY_PAYLOAD_OWNED_SOURCE_ARGUMENT_NAMES:
        static_arguments[name] = None
    static_fingerprint = _dynamics_tendency_static_argument_fingerprint(
        static_arguments
    )
    key = (
        "DynamicsTendencyNumericPlan",
        tuple(stage_key),
        ("payload_source_fingerprint", payload_source_fingerprint),
        static_fingerprint,
    )
    frozen_static = MappingProxyType(dict(static_arguments))
    numeric_callable = _bind_dynamics_tendency_numeric_callable(frozen_static)
    return DynamicsTendencyNumericPlan(
        key=key,
        static_argument_names=tuple(sorted(frozen_static)),
        dynamic_argument_names=dynamic_names,
        static_arguments=frozen_static,
        numeric_callable=numeric_callable,
    )


def pack_dynamics_tendency_numeric_outputs(
    compact_outputs: tuple[torch.Tensor | None, ...],
    call_arguments: Mapping[str, Any],
    *,
    workspace: object | None = None,
    take_unchanged_ownership: bool = False,
) -> dict[str, Any]:
    """Pack active dynamics-tendency patches into owned public arrays outside compilation."""
    if not isinstance(compact_outputs, tuple) or len(compact_outputs) != len(
        _DYNAMICS_TENDENCY_COMPACT_OUTPUT_NAMES
    ):
        raise TypeError(
            "whole-dynamics-tendency numeric output must be the fixed compact patch tuple"
        )
    patches = dict(zip(_DYNAMICS_TENDENCY_COMPACT_OUTPUT_NAMES, compact_outputs))
    u = call_arguments["u"]
    if not isinstance(u, torch.Tensor):
        raise TypeError(
            "whole-dynamics-tendency output packing requires tensor input 'u'"
        )
    n_cells = int(call_arguments["nCells"])
    n_edges = int(call_arguments["nEdges"])
    n_levels = int(call_arguments["nVertLevels"])
    cell_start = int(call_arguments["cellStart"]) - 1
    cell_end = int(call_arguments["cellEnd"])
    cell_solve_start = int(call_arguments["cellSolveStart"]) - 1
    cell_solve_end = int(call_arguments["cellSolveEnd"])
    edge_start = int(call_arguments["edgeStart"]) - 1
    edge_end = int(call_arguments["edgeEnd"])
    edge_solve_start = int(call_arguments["edgeSolveStart"]) - 1
    edge_solve_end = int(call_arguments["edgeSolveEnd"])

    workspace_dynamics_tendency = getattr(workspace, "dynamics_tendency", None)
    workspace_enabled = (
        workspace_dynamics_tendency is not None
        and not is_grad_mode()
        and not any(
            isinstance(value, torch.Tensor) and _tensor_carries_autodiff(value)
            for value in (*compact_outputs, *call_arguments.values())
        )
    )

    def pack_rows(
        name: str | None,
        shape: tuple[int, ...],
        workspace_name: str,
        patch_name: str,
        row_start: int,
        row_end: int,
    ) -> torch.Tensor:
        value = call_arguments.get(name) if name is not None else None
        patch = patches[patch_name]
        candidate = getattr(workspace_dynamics_tendency, workspace_name, None)
        if workspace_enabled and (
            isinstance(candidate, torch.Tensor)
            and tuple(candidate.shape) == shape
            and candidate.dtype == u.dtype
            and candidate.device == u.device
        ):
            if isinstance(value, torch.Tensor):
                candidate.copy_(value)
            else:
                candidate.zero_()
            if patch is not None:
                candidate[row_start:row_end, ...] = patch
            return candidate
        if patch is None:
            if isinstance(value, torch.Tensor):
                if (
                    take_unchanged_ownership
                    and not torch.is_grad_enabled()
                    and not _tensor_carries_autodiff(value)
                ):
                    return value
                return value.clone()
            return u.new_zeros(shape)
        expected_shape = (row_end - row_start, *shape[1:])
        if tuple(patch.shape) != expected_shape:
            raise ValueError(
                f"whole-dynamics-tendency compact patch {patch_name!r} has shape "
                f"{tuple(patch.shape)}; expected {expected_shape}"
            )
        if isinstance(value, torch.Tensor):
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"whole-dynamics-tendency public base {name!r} has shape {tuple(value.shape)}; "
                    f"expected {shape}"
                )
            return _replace_row_slice(value, row_start, row_end, patch)
        pieces: list[torch.Tensor] = []
        if row_start:
            pieces.append(u.new_zeros((row_start, *shape[1:])))
        pieces.append(patch)
        if row_end < shape[0]:
            pieces.append(u.new_zeros((shape[0] - row_end, *shape[1:])))
        if len(pieces) == 1:
            return patch.clone()
        return torch.cat(tuple(pieces), dim=0)

    def pack_zero_padded_interior_rows(
        shape: tuple[int, int],
        workspace_name: str,
        patch_name: str,
        row_start: int,
        row_end: int,
    ) -> torch.Tensor:
        """Pack compact interior columns once at the public output boundary."""
        patch = patches[patch_name]
        if patch is None:
            raise ValueError(
                f"whole-dynamics-tendency compact output omitted {patch_name!r}"
            )
        expected_shape = (row_end - row_start, max(shape[1] - 2, 0))
        if tuple(patch.shape) != expected_shape:
            raise ValueError(
                f"whole-dynamics-tendency compact patch {patch_name!r} has shape "
                f"{tuple(patch.shape)}; expected {expected_shape}"
            )
        candidate = getattr(workspace_dynamics_tendency, workspace_name, None)
        if workspace_enabled and (
            isinstance(candidate, torch.Tensor)
            and tuple(candidate.shape) == shape
            and candidate.dtype == u.dtype
            and candidate.device == u.device
        ):
            candidate.zero_()
            candidate[row_start:row_end, 1:-1] = patch
            return candidate
        return F.pad(
            patch,
            (1, 1, row_start, shape[0] - row_end),
            mode="constant",
            value=0.0,
        )

    rk_step = int(call_arguments["rk_step"])
    required_patches = {
        "tend_u_solve_patch",
        "tend_theta_solve_patch",
        "tend_w_solve_inner_patch",
        "h_divergence_cell_patch",
        "rthdynten_solve_patch",
    }
    if rk_step == 1:
        required_patches.update(
            {
                "tend_rho_cell_patch",
                "tend_u_euler_edge_patch",
                "tend_w_euler_cell_patch",
                "tend_theta_euler_cell_patch",
                "kdiff_cell_patch",
            }
        )
    missing_patches = sorted(name for name in required_patches if patches[name] is None)
    if missing_patches:
        raise ValueError(
            "whole-dynamics-tendency compact output omitted required active patches for "
            f"RK step {rk_step}: {missing_patches}"
        )

    tend_u = pack_rows(
        None,
        (n_edges + 1, n_levels),
        "tend_u",
        "tend_u_solve_patch",
        edge_solve_start,
        edge_solve_end,
    )
    tend_rho = pack_rows(
        "tend_rho_in",
        (n_cells + 1, n_levels),
        "tend_rho",
        "tend_rho_cell_patch",
        cell_start,
        cell_end,
    )
    tend_theta = pack_rows(
        None,
        (n_cells + 1, n_levels),
        "tend_theta",
        "tend_theta_solve_patch",
        cell_solve_start,
        cell_solve_end,
    )
    tend_w = pack_zero_padded_interior_rows(
        (n_cells + 1, n_levels + 1),
        "tend_w",
        "tend_w_solve_inner_patch",
        cell_solve_start,
        cell_solve_end,
    )
    tend_u_euler = pack_rows(
        "tend_u_euler_in",
        (n_edges + 1, n_levels),
        "tend_u_euler",
        "tend_u_euler_edge_patch",
        edge_start,
        edge_end,
    )
    tend_w_euler = pack_rows(
        "tend_w_euler_in",
        (n_cells + 1, n_levels + 1),
        "tend_w_euler",
        "tend_w_euler_cell_patch",
        cell_start,
        cell_end,
    )
    tend_theta_euler = pack_rows(
        "tend_theta_euler_in",
        (n_cells + 1, n_levels),
        "tend_theta_euler",
        "tend_theta_euler_cell_patch",
        cell_start,
        cell_end,
    )
    h_divergence = pack_rows(
        None,
        (n_cells + 1, n_levels),
        "h_divergence",
        "h_divergence_cell_patch",
        cell_start,
        cell_end,
    )
    kdiff = pack_rows(
        "kdiff_in",
        (n_cells + 1, n_levels),
        "kdiff",
        "kdiff_cell_patch",
        cell_start,
        cell_end,
    )
    rthdynten = pack_rows(
        None,
        (n_cells + 1, n_levels),
        "rthdynten",
        "rthdynten_solve_patch",
        cell_solve_start,
        cell_solve_end,
    )
    return {
        "tend_u": tend_u,
        "tend_rho": tend_rho,
        "tend_theta": tend_theta,
        "tend_w": tend_w,
        "tend_u_euler": tend_u_euler,
        "tend_w_euler": tend_w_euler,
        "tend_theta_euler": tend_theta_euler,
        "h_divergence": h_divergence,
        "kdiff": kdiff,
        "rthdynten": rthdynten,
    }


def _dynamics_tendency_whole_stage_specialization_key(
    kwargs: Mapping[str, Any],
) -> tuple[Any, ...]:
    """Static key for the compiled dynamics-tendency entry-to-exit numeric body."""

    u = kwargs.get("u")
    theta_m = kwargs.get("theta_m")
    qtot = kwargs.get("qtot")
    device = getattr(u, "device", torch.device("cpu"))
    dtype = getattr(u, "dtype", DTYPE)
    return (
        "dims",
        int(kwargs["nCells"]),
        int(kwargs["nEdges"]),
        int(kwargs["nVertices"]),
        int(kwargs["nVertLevels"]),
        int(kwargs["nCellsSolve"]),
        int(kwargs["nEdgesSolve"]),
        int(kwargs["maxEdges"]),
        int(kwargs["maxEdges2"]),
        "rk_step",
        int(kwargs["rk_step"]),
        "num_scalars",
        int(kwargs["num_scalars"]),
        "moist",
        int(kwargs["moist_start"]),
        int(kwargs["moist_end"]),
        "mixing",
        str(kwargs["config_horiz_mixing"]),
        "mix_full",
        int(kwargs["config_mix_full"]),
        "rayleigh",
        int(kwargs["config_rayleigh_damp_u"]),
        int(kwargs["config_number_rayleigh_damp_u_levels"]),
        int(kwargs["config_number_cam_damping_levels"]),
        "curvature",
        bool(kwargs.get("config_curvature_u", False)),
        "cell_owned_ab",
        bool(
            kwargs.get(
                "config_dynamics_tendency_cell_owned_vertical_velocity_horizontal_advection_theta_horizontal_advection_ab",
                False,
            )
        ),
        "device",
        str(device),
        "dtype",
        str(dtype),
        "theta_m_alias_save",
        theta_m is kwargs.get("theta_m_save"),
        "qtot_shape",
        tuple(qtot.shape) if isinstance(qtot, torch.Tensor) else None,
    )


def dyn_tend(
    state: dict,
    state_prev: dict,
    mesh: MESHState,
    vert_coef: torch.Tensor,
) -> dict:
    """
    High-level wrapper for atm_compute_dyn_tend_work - INTENTIONALLY NOT IMPLEMENTED.

    This wrapper is a placeholder for a potential future API simplification.
    The low-level atm_compute_dyn_tend_work() function is used directly in
    execute_rk_step() to maintain explicit control over all parameters and
    avoid the complexity of automatic parameter extraction from state dicts.

    Design Decision:
        The explicit parameter passing pattern in atm_compute_dyn_tend_work()
        provides better type safety, clearer documentation, and easier debugging
        compared to a wrapper that would need to extract ~60+ parameters from
        nested dictionaries. This is intentional, not a TODO.

    Usage:
        Use atm_compute_dyn_tend_work() directly with explicit parameters.
        See time_integration.py execute_rk_step() for the expected call pattern.

    Raises:
        NotImplementedError: Always - use atm_compute_dyn_tend_work() instead.
    """
    raise NotImplementedError(
        "dyn_tend() wrapper is intentionally not implemented. "
        "Use atm_compute_dyn_tend_work() directly with explicit parameters. "
        "See execute_rk_step() for the expected call pattern."
    )
