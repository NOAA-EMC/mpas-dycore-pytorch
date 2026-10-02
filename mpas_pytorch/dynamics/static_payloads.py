"""Runtime-owned static dycore payload cache.

The hot dycore kernels build topology payloads from mesh connectivity and
namelist-static loop bounds.  Those payloads should be owned by the runner
configuration and passed through RK-stage shallow copies, not hidden in
module-global caches as the primary execution model.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import torch

_MISSING = object()

from mpas_pytorch.dynamics.stencil_ops import (
    _compile_active,
    _sum_weighted_stencil_slots,
    _tensor_carries_autodiff,
    _tensor_version_cache_key,
    gather_rows,
)
from mpas_pytorch.mesh_connectivity import MeshConnectivity

_MESH_CONNECTIVITY_REQUIRED_KEYS = (
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
)
_USE_RELATION_WEIGHTS = object()


def build_ordered_output_transpose(
    indices: torch.Tensor,
    n_source_rows: int,
    *,
    valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a source-owned, stable transpose of a static gather relation.

    Destination rows are ordered exactly as they occur in each forward slot.
    The representation lets a VJP gather contributions without scatter/atomics
    while retaining duplicate-index accumulation order.
    """
    if indices.ndim not in (1, 2):
        raise ValueError(
            "ordered output transpose requires rank-1 or rank-2 indices; "
            f"got shape {tuple(indices.shape)}"
        )
    n_source_rows = int(n_source_rows)
    if n_source_rows <= 0:
        raise ValueError("ordered output transpose requires positive source rows")
    if valid is not None and tuple(valid.shape) != tuple(indices.shape):
        raise ValueError(
            "ordered output transpose valid-mask shape mismatch: "
            f"indices={tuple(indices.shape)}, valid={tuple(valid.shape)}"
        )

    slots = 1 if indices.ndim == 1 else int(indices.shape[1])
    per_slot: list[tuple[torch.Tensor, torch.Tensor, int]] = []
    max_duplicates = 0
    for slot in range(slots):
        slot_indices = indices if indices.ndim == 1 else indices[:, slot]
        slot_valid = (
            None if valid is None else (valid if valid.ndim == 1 else valid[:, slot])
        )
        if slot_valid is None:
            output_rows = torch.arange(
                slot_indices.shape[0], device=indices.device, dtype=torch.int64
            )
            sources = slot_indices.to(dtype=torch.int64)
        else:
            output_rows = torch.nonzero(slot_valid, as_tuple=False).reshape(-1)
            sources = slot_indices[slot_valid].to(dtype=torch.int64)
        if sources.numel() and (
            bool(torch.any(sources < 0)) or bool(torch.any(sources >= n_source_rows))
        ):
            raise ValueError("ordered output transpose source index is out of range")
        counts = torch.bincount(sources, minlength=n_source_rows)
        slot_max = int(counts.max().detach().cpu()) if counts.numel() else 0
        max_duplicates = max(max_duplicates, slot_max)
        per_slot.append((sources, output_rows.to(dtype=torch.int64), slot_max))

    width = max(max_duplicates, 1)
    shape = (
        (n_source_rows, width) if indices.ndim == 1 else (n_source_rows, slots, width)
    )
    destinations = torch.zeros(shape, device=indices.device, dtype=torch.int64)
    transpose_valid = torch.zeros(shape, device=indices.device, dtype=torch.bool)
    for slot, (sources, output_rows, _slot_max) in enumerate(per_slot):
        if sources.numel() == 0:
            continue
        order = torch.argsort(sources, stable=True)
        ordered_sources = sources[order]
        ordered_destinations = output_rows[order]
        counts = torch.bincount(ordered_sources, minlength=n_source_rows)
        offsets = torch.cumsum(counts, dim=0) - counts
        rank = torch.arange(
            ordered_sources.numel(), device=indices.device, dtype=torch.int64
        ) - torch.repeat_interleave(offsets, counts)
        if indices.ndim == 1:
            destinations[ordered_sources, rank] = ordered_destinations
            transpose_valid[ordered_sources, rank] = True
        else:
            destinations[ordered_sources, slot, rank] = ordered_destinations
            transpose_valid[ordered_sources, slot, rank] = True
    return destinations, transpose_valid


def _uses_static_autograd(
    field: torch.Tensor, *static_tensors: torch.Tensor | None
) -> bool:
    """Return True when a relation can use its explicit static-topology VJP."""
    if _compile_active():
        return False
    if not _tensor_carries_autodiff(field):
        return False
    return not any(
        isinstance(tensor, torch.Tensor) and _tensor_carries_autodiff(tensor)
        for tensor in static_tensors
    )


def _zero_tangent(
    shape: tuple[int, ...],
    reference: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    return reference.new_zeros(shape, dtype=dtype)


def _mask_static_values(
    values: torch.Tensor,
    valid: torch.Tensor | None,
    *,
    valid_all: bool | None,
) -> torch.Tensor:
    return _fill_static_invalid(values, valid, 0.0, valid_all=valid_all)


def _fill_static_invalid(
    values: torch.Tensor,
    valid: torch.Tensor | None,
    fill_value: float | torch.Tensor,
    *,
    valid_all: bool | None,
) -> torch.Tensor:
    if valid is None or valid_all is True:
        return values
    mask = valid
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    fill = _fill_like(values, fill_value)
    return torch.where(mask, values, fill)


class _StaticPairGatherFunction(torch.autograd.Function):
    """Gather a static pair relation with an explicit segment-sum transpose."""

    @staticmethod
    def forward(
        field: torch.Tensor,
        first: torch.Tensor,
        second: torch.Tensor,
        valid: torch.Tensor | None,
        valid_all: bool | None,
        fill_value: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        first_values = _fill_static_invalid(
            gather_rows(field, first), valid, fill_value, valid_all=valid_all
        )
        second_values = _fill_static_invalid(
            gather_rows(field, second), valid, fill_value, valid_all=valid_all
        )
        return first_values, second_values

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        field, first, second, valid, valid_all, _fill_value = inputs
        del output
        valid_save = (
            valid
            if valid is not None
            else torch.empty(0, dtype=torch.bool, device=field.device)
        )
        ctx.save_for_backward(first, second, valid_save)
        ctx.save_for_forward(first, second, valid_save)
        ctx.has_valid = valid is not None
        ctx.valid_all = valid_all
        ctx.field_shape = tuple(field.shape)
        ctx.field_dtype = field.dtype

    @staticmethod
    def backward(
        ctx, grad_first: torch.Tensor | None, grad_second: torch.Tensor | None
    ):
        first, second, valid_save = ctx.saved_tensors
        valid = valid_save if ctx.has_valid else None
        if grad_first is None:
            grad_first = _zero_tangent(
                (*first.shape, *ctx.field_shape[1:]),
                first,
                dtype=ctx.field_dtype,
            )
        if grad_second is None:
            grad_second = _zero_tangent(
                (*second.shape, *ctx.field_shape[1:]),
                second,
                dtype=ctx.field_dtype,
            )
        grad_first = _mask_static_values(grad_first, valid, valid_all=ctx.valid_all)
        grad_second = _mask_static_values(grad_second, valid, valid_all=ctx.valid_all)
        out = grad_first.new_zeros(ctx.field_shape)
        out.index_add_(
            0,
            first.reshape(-1).to(dtype=torch.int64),
            grad_first.reshape(-1, *grad_first.shape[first.ndim :]),
        )
        out.index_add_(
            0,
            second.reshape(-1).to(dtype=torch.int64),
            grad_second.reshape(-1, *grad_second.shape[second.ndim :]),
        )
        return out, None, None, None, None, None

    @staticmethod
    def jvp(
        ctx: Any,
        grad_field: torch.Tensor | None,
        grad_first: torch.Tensor | None,
        grad_second: torch.Tensor | None,
        grad_valid: torch.Tensor | None,
        grad_valid_all: bool | None,
        grad_fill_value: float | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del grad_first, grad_second, grad_valid, grad_valid_all, grad_fill_value
        first, second, valid_save = ctx.saved_tensors
        valid = valid_save if ctx.has_valid else None
        if grad_field is None:
            first_tangent = _zero_tangent(
                (*first.shape, *ctx.field_shape[1:]),
                first,
                dtype=ctx.field_dtype,
            )
            second_tangent = _zero_tangent(
                (*second.shape, *ctx.field_shape[1:]),
                second,
                dtype=ctx.field_dtype,
            )
        else:
            first_tangent = gather_rows(grad_field, first)
            second_tangent = gather_rows(grad_field, second)
        return (
            _mask_static_values(first_tangent, valid, valid_all=ctx.valid_all),
            _mask_static_values(second_tangent, valid, valid_all=ctx.valid_all),
        )


class _StaticSlotGatherFunction(torch.autograd.Function):
    """Gather a static slot relation with fill semantics and explicit VJP."""

    @staticmethod
    def forward(
        field: torch.Tensor,
        indices: torch.Tensor,
        valid: torch.Tensor | None,
        fill_value: float,
    ) -> torch.Tensor:
        values = gather_rows(field, indices)
        if valid is None:
            return values
        return _fill_static_invalid(values, valid, fill_value, valid_all=None)

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        field, indices, valid, _fill_value = inputs
        del output
        valid_save = (
            valid
            if valid is not None
            else torch.empty(0, dtype=torch.bool, device=field.device)
        )
        ctx.save_for_backward(indices, valid_save)
        ctx.save_for_forward(indices, valid_save)
        ctx.has_valid = valid is not None
        ctx.field_shape = tuple(field.shape)
        ctx.field_dtype = field.dtype

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        indices, valid_save = ctx.saved_tensors
        valid = valid_save if ctx.has_valid else None
        grad = _mask_static_values(grad_output, valid, valid_all=None)
        out = grad_output.new_zeros(ctx.field_shape)
        out.index_add_(
            0,
            indices.reshape(-1).to(dtype=torch.int64),
            grad.reshape(-1, *grad.shape[indices.ndim :]),
        )
        return out, None, None, None

    @staticmethod
    def jvp(
        ctx: Any,
        grad_field: torch.Tensor | None,
        grad_indices: torch.Tensor | None,
        grad_valid: torch.Tensor | None,
        grad_fill_value: float | None,
    ) -> torch.Tensor:
        del grad_indices, grad_valid, grad_fill_value
        indices, valid_save = ctx.saved_tensors
        valid = valid_save if ctx.has_valid else None
        if grad_field is None:
            tangent = _zero_tangent(
                (*indices.shape, *ctx.field_shape[1:]),
                indices,
                dtype=ctx.field_dtype,
            )
        else:
            tangent = gather_rows(grad_field, indices)
        return _mask_static_values(tangent, valid, valid_all=None)


class _StaticSlotReduceFunction(torch.autograd.Function):
    """Ordered static slot reduction with an explicit gather transpose."""

    @staticmethod
    def forward(
        field: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor | None,
        valid_mask: torch.Tensor | None,
        slot_factor: torch.Tensor | None,
        valid_all_by_slot: tuple[bool, ...] | None,
    ) -> torch.Tensor:
        return _sum_weighted_stencil_slots(
            field=field,
            indices=indices,
            weights=weights,
            valid_mask=valid_mask,
            valid_all_by_slot=valid_all_by_slot,
            slot_factor=slot_factor,
        )

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        field, indices, weights, valid_mask, slot_factor, valid_all_by_slot = inputs
        del output
        empty_float = field.new_empty(0)
        empty_bool = torch.empty(0, dtype=torch.bool, device=field.device)
        weights_save = weights if weights is not None else empty_float
        valid_save = valid_mask if valid_mask is not None else empty_bool
        slot_factor_save = slot_factor if slot_factor is not None else empty_float
        ctx.save_for_backward(indices, weights_save, valid_save, slot_factor_save)
        ctx.save_for_forward(indices, weights_save, valid_save, slot_factor_save)
        ctx.has_weights = weights is not None
        ctx.has_valid = valid_mask is not None
        ctx.has_slot_factor = slot_factor is not None
        ctx.valid_all_by_slot = valid_all_by_slot
        ctx.field_shape = tuple(field.shape)
        ctx.field_dtype = field.dtype

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        indices, weights_save, valid_save, slot_factor_save = ctx.saved_tensors
        weights = weights_save if ctx.has_weights else None
        valid_mask = valid_save if ctx.has_valid else None
        slot_factor = slot_factor_save if ctx.has_slot_factor else None
        out = grad_output.new_zeros(ctx.field_shape)
        for slot in range(indices.shape[1]):
            contrib = grad_output
            if slot_factor is not None:
                contrib = contrib * slot_factor
            if weights is not None:
                contrib = contrib * weights[:, slot].unsqueeze(1)
            valid_all = (
                ctx.valid_all_by_slot is not None and ctx.valid_all_by_slot[slot]
            )
            if valid_mask is not None and not valid_all:
                contrib = _mask_static_values(
                    contrib,
                    valid_mask[:, slot],
                    valid_all=valid_all,
                )
            out.index_add_(0, indices[:, slot].to(dtype=torch.int64), contrib)
        return out, None, None, None, None, None

    @staticmethod
    def jvp(
        ctx: Any,
        grad_field: torch.Tensor | None,
        grad_indices: torch.Tensor | None,
        grad_weights: torch.Tensor | None,
        grad_valid_mask: torch.Tensor | None,
        grad_slot_factor: torch.Tensor | None,
        grad_valid_all_by_slot: tuple[bool, ...] | None,
    ) -> torch.Tensor:
        del (
            grad_indices,
            grad_weights,
            grad_valid_mask,
            grad_slot_factor,
            grad_valid_all_by_slot,
        )
        indices, weights_save, valid_save, slot_factor_save = ctx.saved_tensors
        weights = weights_save if ctx.has_weights else None
        valid_mask = valid_save if ctx.has_valid else None
        slot_factor = slot_factor_save if ctx.has_slot_factor else None
        if grad_field is None:
            return _zero_tangent(
                (indices.shape[0], ctx.field_shape[1]),
                indices,
                dtype=ctx.field_dtype,
            )
        return _sum_weighted_stencil_slots(
            field=grad_field,
            indices=indices,
            weights=weights,
            valid_mask=valid_mask,
            valid_all_by_slot=ctx.valid_all_by_slot,
            slot_factor=slot_factor,
        )


def static_pair_gather(
    field: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
    valid: torch.Tensor | None,
    valid_all: bool | None,
    *,
    fill_value: float | torch.Tensor = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Functional static pair gather shared by plans and relation objects."""
    if not isinstance(fill_value, torch.Tensor) and _uses_static_autograd(
        field, first, second, valid
    ):
        return cast(
            tuple[torch.Tensor, torch.Tensor],
            _StaticPairGatherFunction.apply(
                field, first, second, valid, valid_all, float(fill_value)
            ),
        )
    first_values = _fill_static_invalid(
        gather_rows(field, first), valid, fill_value, valid_all=valid_all
    )
    second_values = _fill_static_invalid(
        gather_rows(field, second), valid, fill_value, valid_all=valid_all
    )
    return first_values, second_values


def static_slot_reduce(
    field: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor | None,
    valid_mask: torch.Tensor | None,
    valid_all_by_slot: tuple[bool, ...] | None,
    *,
    slot_factor: torch.Tensor | None = None,
) -> torch.Tensor:
    """Functional ordered static slot reduction with explicit AD transpose."""
    if _uses_static_autograd(field, indices, weights, valid_mask, slot_factor):
        return cast(
            torch.Tensor,
            _StaticSlotReduceFunction.apply(
                field,
                indices,
                weights,
                valid_mask,
                slot_factor,
                valid_all_by_slot,
            ),
        )
    return _sum_weighted_stencil_slots(
        field=field,
        indices=indices,
        weights=weights,
        valid_mask=valid_mask,
        valid_all_by_slot=valid_all_by_slot,
        slot_factor=slot_factor,
    )


def static_patch_gather(
    field: torch.Tensor,
    patch: torch.Tensor,
    indices: torch.Tensor,
    local_indices: torch.Tensor,
    local_valid: torch.Tensor,
    local_all: bool | None,
) -> torch.Tensor:
    """Gather a compact row patch with a static base-field fallback."""
    if local_all is True:
        return gather_rows(patch, local_indices)
    base_values = gather_rows(field, indices)
    patch_values = gather_rows(patch, local_indices)
    return _fill_static_invalid(
        patch_values,
        local_valid,
        base_values,
        valid_all=local_all,
    )


@dataclass
class StaticPayloadEntry:
    """One cache entry keyed by the builder's mutation-sensitive payload key."""

    cache_key: Any
    payload: Any


@dataclass(frozen=True)
class StaticRange:
    """Contiguous static row range owned by a dycore topology payload."""

    name: str
    start: int
    stop: int

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop)

    def to(self, device: torch.device | str) -> StaticRange:
        return self

    def snapshot(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "start": self.start,
            "stop": self.stop,
        }


@dataclass(frozen=True)
class StaticPairRelation:
    """Static two-sided cell/edge relation with optional masks and local rows."""

    name: str
    first: torch.Tensor
    second: torch.Tensor
    valid: torch.Tensor | None = None
    valid_2d: torch.Tensor | None = None
    valid_all: bool | None = None
    first_local: torch.Tensor | None = None
    second_local: torch.Tensor | None = None
    first_local_valid: torch.Tensor | None = None
    second_local_valid: torch.Tensor | None = None
    first_local_all: bool | None = None
    second_local_all: bool | None = None
    active: torch.Tensor | None = None
    active_3d: torch.Tensor | None = None
    active_all: bool | None = None
    first_transpose_destinations: torch.Tensor | None = None
    first_transpose_valid: torch.Tensor | None = None
    second_transpose_destinations: torch.Tensor | None = None
    second_transpose_valid: torch.Tensor | None = None

    def gather_pair(
        self, field: torch.Tensor, *, fill_value: float | torch.Tensor = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather both relation sides, replacing invalid rows with ``fill_value``."""
        return static_pair_gather(
            field,
            self.first,
            self.second,
            self.valid,
            self.valid_all,
            fill_value=fill_value,
        )

    def gather_pair_patch_or_base(
        self,
        field: torch.Tensor,
        patch: torch.Tensor | None,
        *,
        patch_row_start: int,
        patch_row_end: int,
        slot_start: int | None = None,
        slot_stop: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather relation sides from an active patch, falling back to base rows.

        ``first_local``/``second_local`` are precomputed by the runner-owned
        payload builder.  That keeps repeated row-range arithmetic and clamp/where
        setup out of hot numeric call sites while preserving the same materialized
        replacement semantics as ``base[patch_row_start:patch_row_end] = patch``.
        """
        if patch is None:
            return self._slot_slice_pair_gather(field, slot_start, slot_stop)
        patch_row_start = int(patch_row_start)
        patch_row_end = int(patch_row_end)
        if patch_row_end <= patch_row_start or patch.shape[0] == 0:
            return self._slot_slice_pair_gather(field, slot_start, slot_stop)

        first = self._slot_slice(self.first, slot_start, slot_stop)
        second = self._slot_slice(self.second, slot_start, slot_stop)
        first_local = self._slot_slice(self.first_local, slot_start, slot_stop)
        second_local = self._slot_slice(self.second_local, slot_start, slot_stop)
        first_local_valid = self._slot_slice(
            self.first_local_valid, slot_start, slot_stop
        )
        second_local_valid = self._slot_slice(
            self.second_local_valid, slot_start, slot_stop
        )
        assert first is not None and second is not None

        return (
            self._gather_side_patch_or_base(
                field,
                patch,
                first,
                first_local,
                first_local_valid,
                self.first_local_all,
                patch_row_start,
                patch_row_end,
            ),
            self._gather_side_patch_or_base(
                field,
                patch,
                second,
                second_local,
                second_local_valid,
                self.second_local_all,
                patch_row_start,
                patch_row_end,
            ),
        )

    def average(
        self, field: torch.Tensor, *, fill_value: float | torch.Tensor = 0.0
    ) -> torch.Tensor:
        """Gather both sides and return their ordered arithmetic mean."""
        first, second = self.gather_pair(field, fill_value=fill_value)
        return 0.5 * (first + second)

    def average_vjp(
        self, grad_output: torch.Tensor, n_source_rows: int
    ) -> torch.Tensor:
        """Explicit transpose of ``average`` with respect to ``field``."""
        grad = self._mask_values(grad_output)
        out = grad_output.new_zeros(
            (int(n_source_rows), *grad_output.shape[self.first.ndim :])
        )
        flat_first = self.first.reshape(-1).to(dtype=torch.int64)
        flat_second = self.second.reshape(-1).to(dtype=torch.int64)
        flat_grad = grad.reshape(-1, *grad.shape[self.first.ndim :])
        out.index_add_(0, flat_first, 0.5 * flat_grad)
        out.index_add_(0, flat_second, 0.5 * flat_grad)
        return out

    def _apply_fill(
        self, values: torch.Tensor, fill_value: float | torch.Tensor
    ) -> torch.Tensor:
        return self._fill_invalid(
            values, self.valid, fill_value, valid_all=self.valid_all
        )

    def _mask_values(self, values: torch.Tensor) -> torch.Tensor:
        return self._fill_invalid(values, self.valid, 0.0, valid_all=self.valid_all)

    def _slot_slice_pair_gather(
        self,
        field: torch.Tensor,
        slot_start: int | None,
        slot_stop: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        first = self._slot_slice(self.first, slot_start, slot_stop)
        second = self._slot_slice(self.second, slot_start, slot_stop)
        assert first is not None and second is not None
        return gather_rows(field, first), gather_rows(field, second)

    @staticmethod
    def _slot_slice(
        value: torch.Tensor | None,
        slot_start: int | None,
        slot_stop: int | None,
    ) -> torch.Tensor | None:
        if value is None or slot_start is None or slot_stop is None or value.ndim < 2:
            return value
        return value[:, int(slot_start) : int(slot_stop)]

    @staticmethod
    def _gather_side_patch_or_base(
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

        return static_patch_gather(
            field,
            patch,
            indices,
            safe_local,
            local_valid,
            local_all,
        )

    @staticmethod
    def _fill_invalid(
        values: torch.Tensor,
        valid: torch.Tensor | None,
        fill_value: float | torch.Tensor,
        *,
        valid_all: bool | None,
    ) -> torch.Tensor:
        if valid is None or valid_all is True:
            return values
        return _fill_static_invalid(values, valid, fill_value, valid_all=valid_all)

    def to(self, device: torch.device | str) -> StaticPairRelation:
        target = torch.device(device)
        return StaticPairRelation(
            name=self.name,
            first=_move_payload(self.first, target),
            second=_move_payload(self.second, target),
            valid=_move_payload(self.valid, target),
            valid_2d=_move_payload(self.valid_2d, target),
            valid_all=self.valid_all,
            first_local=_move_payload(self.first_local, target),
            second_local=_move_payload(self.second_local, target),
            first_local_valid=_move_payload(self.first_local_valid, target),
            second_local_valid=_move_payload(self.second_local_valid, target),
            first_local_all=self.first_local_all,
            second_local_all=self.second_local_all,
            active=_move_payload(self.active, target),
            active_3d=_move_payload(self.active_3d, target),
            active_all=self.active_all,
            first_transpose_destinations=_move_payload(
                self.first_transpose_destinations, target
            ),
            first_transpose_valid=_move_payload(self.first_transpose_valid, target),
            second_transpose_destinations=_move_payload(
                self.second_transpose_destinations, target
            ),
            second_transpose_valid=_move_payload(self.second_transpose_valid, target),
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "first_shape": tuple(self.first.shape),
            "second_shape": tuple(self.second.shape),
            "has_valid": self.valid is not None,
            "valid_all": self.valid_all,
            "has_ordered_transpose": all(
                value is not None
                for value in (
                    self.first_transpose_destinations,
                    self.first_transpose_valid,
                    self.second_transpose_destinations,
                    self.second_transpose_valid,
                )
            ),
        }


@dataclass(frozen=True)
class StaticSlotRelation:
    """Static row-slot relation with optional masks, signs, weights, and metadata."""

    name: str
    indices: torch.Tensor
    valid: torch.Tensor | None = None
    sign: torch.Tensor | None = None
    weights: torch.Tensor | None = None
    coefficients: torch.Tensor | None = None
    coefficients_3rd: torch.Tensor | None = None
    valid_all_by_slot: tuple[bool, ...] | None = None
    valid_weight: torch.Tensor | None = None
    local_indices: torch.Tensor | None = None
    local_valid: torch.Tensor | None = None
    local_all: bool | None = None
    transpose_destinations: torch.Tensor | None = None
    transpose_valid: torch.Tensor | None = None

    def gather(
        self, field: torch.Tensor, *, fill_value: float | torch.Tensor = 0.0
    ) -> torch.Tensor:
        """Gather row-slot values with GraphCast-style fill semantics."""
        if not isinstance(fill_value, torch.Tensor) and _uses_static_autograd(
            field, self.indices, self.valid
        ):
            return cast(
                torch.Tensor,
                _StaticSlotGatherFunction.apply(
                    field,
                    self.indices,
                    self.valid,
                    float(fill_value),
                ),
            )
        values = gather_rows(field, self.indices)
        if self.valid is None:
            return values
        return _fill_static_invalid(values, self.valid, fill_value, valid_all=None)

    def gather_patch_or_base(
        self,
        field: torch.Tensor,
        patch: torch.Tensor | None,
        *,
        patch_row_start: int,
        patch_row_end: int,
        slot_start: int | None = None,
        slot_stop: int | None = None,
    ) -> torch.Tensor:
        """Gather from an active patch, falling back to base rows when needed."""
        if patch is None:
            indices = StaticPairRelation._slot_slice(
                self.indices, slot_start, slot_stop
            )
            assert indices is not None
            return gather_rows(
                field,
                indices,
            )
        patch_row_start = int(patch_row_start)
        patch_row_end = int(patch_row_end)
        indices = StaticPairRelation._slot_slice(self.indices, slot_start, slot_stop)
        local_indices = StaticPairRelation._slot_slice(
            self.local_indices,
            slot_start,
            slot_stop,
        )
        local_valid = StaticPairRelation._slot_slice(
            self.local_valid,
            slot_start,
            slot_stop,
        )
        assert indices is not None
        return StaticPairRelation._gather_side_patch_or_base(
            field,
            patch,
            indices,
            local_indices,
            local_valid,
            self.local_all,
            patch_row_start,
            patch_row_end,
        )

    def reduce(
        self,
        field: torch.Tensor,
        *,
        weights: torch.Tensor | None | object = _USE_RELATION_WEIGHTS,
        valid_mask: torch.Tensor | None = None,
        valid_all_by_slot: tuple[bool, ...] | None = None,
        slot_factor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply this static ordered row-slot reduction to ``field``."""
        if weights is _USE_RELATION_WEIGHTS:
            weights = self.weights
        weights_tensor = cast(torch.Tensor | None, weights)
        if valid_mask is None:
            valid_mask = self.valid
        if valid_all_by_slot is None:
            valid_all_by_slot = self.valid_all_by_slot
        return static_slot_reduce(
            field,
            self.indices,
            weights_tensor,
            valid_mask,
            valid_all_by_slot,
            slot_factor=slot_factor,
        )

    def reduce_vjp(
        self,
        grad_output: torch.Tensor,
        n_source_rows: int,
        *,
        weights: torch.Tensor | None | object = _USE_RELATION_WEIGHTS,
        valid_mask: torch.Tensor | None = None,
        slot_factor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Explicit transpose of ``reduce`` with respect to ``field``."""
        if weights is _USE_RELATION_WEIGHTS:
            weights = self.weights
        weights_tensor = cast(torch.Tensor | None, weights)
        if valid_mask is None:
            valid_mask = self.valid
        out = grad_output.new_zeros((int(n_source_rows), grad_output.shape[-1]))
        for slot in range(self.indices.shape[1]):
            contrib = grad_output
            if slot_factor is not None:
                contrib = contrib * slot_factor
            if weights_tensor is not None:
                contrib = contrib * weights_tensor[:, slot].unsqueeze(1)
            if valid_mask is not None:
                contrib = _mask_static_values(
                    contrib,
                    valid_mask[:, slot],
                    valid_all=None,
                )
            out.index_add_(0, self.indices[:, slot].to(dtype=torch.int64), contrib)
        return out

    def to(self, device: torch.device | str) -> StaticSlotRelation:
        target = torch.device(device)
        return StaticSlotRelation(
            name=self.name,
            indices=_move_payload(self.indices, target),
            valid=_move_payload(self.valid, target),
            sign=_move_payload(self.sign, target),
            weights=_move_payload(self.weights, target),
            coefficients=_move_payload(self.coefficients, target),
            coefficients_3rd=_move_payload(self.coefficients_3rd, target),
            valid_all_by_slot=self.valid_all_by_slot,
            valid_weight=_move_payload(self.valid_weight, target),
            local_indices=_move_payload(self.local_indices, target),
            local_valid=_move_payload(self.local_valid, target),
            local_all=self.local_all,
            transpose_destinations=_move_payload(self.transpose_destinations, target),
            transpose_valid=_move_payload(self.transpose_valid, target),
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "indices_shape": tuple(self.indices.shape),
            "has_valid": self.valid is not None,
            "has_sign": self.sign is not None,
            "has_weights": self.weights is not None,
            "valid_all_by_slot": self.valid_all_by_slot,
            "has_local": self.local_indices is not None
            and self.local_valid is not None,
            "local_all": self.local_all,
            "has_ordered_transpose": (
                self.transpose_destinations is not None
                and self.transpose_valid is not None
            ),
        }


def _fill_like(values: torch.Tensor, fill_value: float | torch.Tensor) -> torch.Tensor:
    if isinstance(fill_value, torch.Tensor):
        return torch.zeros_like(values) + fill_value.to(
            dtype=values.dtype, device=values.device
        )
    return torch.full_like(values, float(fill_value))


@dataclass(frozen=True)
class DycoreStaticPayload(Mapping[str, Any]):
    """Immutable typed view of one runner-owned static topology payload."""

    name: str
    data: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", MappingProxyType(dict(self.data)))

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def to(self, device: torch.device | str) -> DycoreStaticPayload:
        target = torch.device(device)
        return DycoreStaticPayload(
            name=self.name,
            data={
                key: _move_payload(value, target) for key, value in self.data.items()
            },
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "keys": sorted(str(key) for key in self.data),
        }


def _move_payload(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, MeshConnectivity):
        return value.to(device)
    if isinstance(value, (StaticPairRelation, StaticSlotRelation, StaticRange)):
        return value.to(device)
    if isinstance(value, DycoreStaticPayload):
        return value.to(device)
    try:
        from mpas_pytorch.dynamics.pattern_c import PatternCOperator
    except Exception:  # noqa: BLE001, S110
        pass
    else:
        if isinstance(value, PatternCOperator):
            return value.to(device)
    if isinstance(value, dict):
        return {key: _move_payload(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_payload(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_payload(item, device) for item in value)
    return value


class DycoreStaticPayloads:
    """Cache of immutable dycore topology payloads for one runner/config."""

    def __init__(
        self,
        entries: Mapping[str, StaticPayloadEntry] | None = None,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        self._entries: dict[str, StaticPayloadEntry] = dict(entries or {})
        self._validated_static_input_keys: set[Any] = set()
        self._static_scalar_values: dict[Any, Any] = {}
        self._vertical_scan_plan_cache: dict[Any, Any] = {}
        self._numeric_plans: dict[tuple[str, Any], Any] = {}
        self.device = torch.device(device) if device is not None else None

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def clear(self) -> None:
        self._entries.clear()
        self._validated_static_input_keys.clear()
        self._static_scalar_values.clear()
        self._vertical_scan_plan_cache.clear()
        self._numeric_plans.clear()

    def has_validated_static_inputs(self, cache_key: Any) -> bool:
        """Return whether static input validation passed for ``cache_key``."""
        return cache_key in self._validated_static_input_keys

    def mark_validated_static_inputs(self, cache_key: Any) -> None:
        """Record successful static input validation for ``cache_key``."""
        self._validated_static_input_keys.add(cache_key)

    def cached_static_scalar(self, cache_key: Any, default: Any = _MISSING) -> Any:
        """Return a cached static scalar value or a private missing sentinel."""
        return self._static_scalar_values.get(cache_key, default)

    def mark_static_scalar(self, cache_key: Any, value: Any) -> None:
        """Record a Python scalar extracted from immutable runner metadata."""
        self._static_scalar_values[cache_key] = value

    def vertical_scan_plan_cache(self) -> dict[Any, Any]:
        """Return the runner-owned tridiagonal scan metadata cache."""
        return self._vertical_scan_plan_cache

    def get_or_build_numeric_plan(
        self,
        name: str,
        cache_key: Any,
        build: Callable[[], Any],
    ) -> Any:
        """Return one immutable numeric plan for an exact static variant."""
        key = (str(name), cache_key)
        if key not in self._numeric_plans:
            # Ordinary forecast entry points run under inference mode, but the
            # cached plans outlive that call and may later be reused by an AD
            # execution. Derived plan tensors must therefore be regular
            # tensors that autograd may save, just like the cached topology
            # payloads from ``_prebound_static_payload``.
            with torch.inference_mode(False):
                self._numeric_plans[key] = build()
        return self._numeric_plans[key]

    def get_or_build(
        self,
        name: str,
        cache_key: Any,
        build: Callable[[], Any],
    ) -> Any:
        """Return a cached payload for ``name`` or build/store a new one."""
        cached = self._entries.get(name)
        if cached is not None and cached.cache_key == cache_key:
            return cached.payload
        payload = build()
        self._entries[name] = StaticPayloadEntry(cache_key=cache_key, payload=payload)
        return payload

    def get_or_build_typed(
        self,
        name: str,
        cache_key: Any,
        build: Callable[[], Mapping[str, Any]],
    ) -> DycoreStaticPayload:
        """Return a typed immutable payload for hot dycore call sites."""
        payload = self.get_or_build(
            name,
            cache_key,
            lambda: DycoreStaticPayload(name=name, data=build()),
        )
        if isinstance(payload, DycoreStaticPayload):
            return payload
        if not isinstance(payload, Mapping):
            raise TypeError(
                f"static payload {name!r} must be a mapping, got {type(payload).__name__}"
            )
        typed = DycoreStaticPayload(name=name, data=payload)
        self._entries[name] = StaticPayloadEntry(cache_key=cache_key, payload=typed)
        return typed

    def get_or_build_mesh_connectivity(
        self,
        mesh: Mapping[str, torch.Tensor],
        *,
        nCells: int,
        nEdges: int,
        nVertices: int,
        nCellsSolve: int | None = None,
        nEdgesSolve: int | None = None,
        name: str = "mesh_connectivity",
    ) -> MeshConnectivity:
        """Return runner-owned package-level mesh connectivity."""
        cache_key = mesh_connectivity_cache_key(
            mesh,
            nCells=nCells,
            nEdges=nEdges,
            nVertices=nVertices,
            nCellsSolve=nCellsSolve,
            nEdgesSolve=nEdgesSolve,
        )
        if cache_key is None:
            return MeshConnectivity.from_mpas_mesh(
                dict(mesh),
                nCells=nCells,
                nEdges=nEdges,
                nVertices=nVertices,
                nCellsSolve=nCellsSolve,
                nEdgesSolve=nEdgesSolve,
            )
        return cast(
            MeshConnectivity,
            self.get_or_build(
                name,
                cache_key,
                lambda: MeshConnectivity.from_mpas_mesh(
                    dict(mesh),
                    nCells=nCells,
                    nEdges=nEdges,
                    nVertices=nVertices,
                    nCellsSolve=nCellsSolve,
                    nEdgesSolve=nEdgesSolve,
                ),
            ),
        )

    def to(self, device: torch.device | str) -> DycoreStaticPayloads:
        """Return a payload store with cached tensors moved to ``device``.

        Payload cache keys remain mutation-sensitive.  If the caller later asks
        for a payload using source tensors whose cache key differs from the moved
        entries, the normal ``get_or_build`` path safely rebuilds that payload.
        """
        target = torch.device(device)
        moved = {
            name: StaticPayloadEntry(
                cache_key=entry.cache_key,
                payload=_move_payload(entry.payload, target),
            )
            for name, entry in self._entries.items()
        }
        return DycoreStaticPayloads(moved, device=target)

    def snapshot(self) -> dict[str, Any]:
        """Return lightweight cache metadata for manifests/tests."""
        payload_types = {
            name: type(entry.payload).__name__
            for name, entry in sorted(self._entries.items())
        }
        return {
            "type": self.__class__.__name__,
            "device": str(self.device) if self.device is not None else None,
            "payload_names": list(self.names()),
            "payload_count": len(self),
            "payload_types": payload_types,
            "validated_static_input_count": len(self._validated_static_input_keys),
            "static_scalar_count": len(self._static_scalar_values),
            "vertical_scan_plan_count": len(self._vertical_scan_plan_cache),
            "numeric_plan_count": len(self._numeric_plans),
            "numeric_plan_names": sorted({name for name, _ in self._numeric_plans}),
        }


def _legacy_dict_to_store(value: Mapping[str, Any]) -> DycoreStaticPayloads:
    entries: dict[str, StaticPayloadEntry] = {}
    for name, cached in value.items():
        if isinstance(name, str) and isinstance(cached, tuple) and len(cached) == 2:
            entries[name] = StaticPayloadEntry(cache_key=cached[0], payload=cached[1])
    return DycoreStaticPayloads(entries)


def mesh_connectivity_cache_key(
    mesh: Mapping[str, torch.Tensor],
    *,
    nCells: int,
    nEdges: int,
    nVertices: int,
    nCellsSolve: int | None = None,
    nEdgesSolve: int | None = None,
) -> tuple[object, ...] | None:
    """Return a mutation-sensitive key for package-level mesh connectivity."""
    tensor_keys: list[tuple[object, ...]] = []
    for name in _MESH_CONNECTIVITY_REQUIRED_KEYS:
        tensor = mesh.get(name)
        if not isinstance(tensor, torch.Tensor):
            return None
        if _tensor_carries_autodiff(tensor):
            return None
        tensor_key = _tensor_version_cache_key(tensor)
        if tensor_key is None:
            return None
        tensor_keys.append((name, *tensor_key))
    return (
        "mesh_connectivity",
        int(nCells),
        int(nEdges),
        int(nVertices),
        None if nCellsSolve is None else int(nCellsSolve),
        None if nEdgesSolve is None else int(nEdgesSolve),
        *tensor_keys,
    )


def ensure_static_payload_store(
    config: MutableMapping[str, Any],
) -> DycoreStaticPayloads:
    """Attach and return the runner-owned static payload store for ``config``."""
    current = config.get("_dycore_static_payloads")
    if isinstance(current, DycoreStaticPayloads):
        return current
    if isinstance(current, Mapping):
        store = _legacy_dict_to_store(current)
    else:
        store = DycoreStaticPayloads()
    config["_dycore_static_payloads"] = store
    return store


__all__ = [
    "DycoreStaticPayload",
    "DycoreStaticPayloads",
    "StaticPairRelation",
    "StaticPayloadEntry",
    "StaticRange",
    "StaticSlotRelation",
    "ensure_static_payload_store",
    "mesh_connectivity_cache_key",
]
