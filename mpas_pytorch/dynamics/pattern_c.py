"""
Pattern C helpers (Edge -> variable-length advection-cell stencil).

These utilities vectorize the high-order scalar flux reconstruction shared by
explicit-scalar-transport/monotonic-scalar-transport style kernels.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import torch

from mpas_pytorch.autodiff.soft_limiter import fct_sign_pm
from mpas_pytorch.dynamics.stencil_ops import (
    _accumulate_ordered_slot,
    _compile_active,
    _gather_slot_batch,
    _replace_rows,
    _slot_batches,
    _slot_gather_batch,
    _static_connectivity_cache_allowed,
    _static_mask_all_true,
    _tensor_version_cache_key,
    gather_rows,
    sum_fortran_order,
)

_PATTERN_C_CONNECTIVITY_CACHE: dict[tuple[object, ...], dict[str, Any]] = {}


def _move_pattern_c_payload_value(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {
            key: _move_pattern_c_payload_value(item, device)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_move_pattern_c_payload_value(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_pattern_c_payload_value(item, device) for item in value)
    return value


@dataclass(frozen=True)
class PatternCOperator:
    """Runner-owned Pattern-C stencil operator with baked static topology."""

    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))

    @property
    def edge_slice(self) -> slice:
        return cast(slice, self.payload["edge_slice"])

    @property
    def edge_count(self) -> int:
        return int(self.payload["edge_count"])

    @property
    def slot_count(self) -> int:
        return int(self.payload["slot_count"])

    def __call__(
        self,
        *,
        scalar_field: torch.Tensor,
        uh_avg: torch.Tensor,
        one: torch.Tensor | None = None,
    ) -> tuple[slice, torch.Tensor]:
        if one is None:
            one = self.payload.get("one")  # type: ignore[assignment]
        if not isinstance(one, torch.Tensor):
            one = torch.tensor(
                1.0, dtype=scalar_field.dtype, device=scalar_field.device
            )
        return compute_pattern_c_flux_from_payload(
            scalar_field=scalar_field,
            uh_avg=uh_avg,
            one=one,
            connectivity_payload=self.payload,
        )

    def to(self, device: torch.device | str) -> "PatternCOperator":
        target = torch.device(device)
        return PatternCOperator(
            {
                key: _move_pattern_c_payload_value(value, target)
                for key, value in self.payload.items()
            }
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "edge_count": self.edge_count,
            "slot_count": self.slot_count,
            "edge_slice_start": self.edge_slice.start,
            "edge_slice_stop": self.edge_slice.stop,
        }


def _long_index(index: torch.Tensor) -> torch.Tensor:
    return index if index.dtype == torch.int64 else index.to(dtype=torch.int64)


def _safe_zero_invalid(indices0: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return _long_index(torch.where(valid, indices0, torch.zeros_like(indices0)))


def _pattern_c_tensor_cache_key(tensor: torch.Tensor) -> tuple[object, ...] | None:
    return _tensor_version_cache_key(tensor)


def _build_pattern_c_connectivity_payload(
    *,
    scalar_field: torch.Tensor,
    n_adv_cells_for_edge: torch.Tensor,
    adv_cells_for_edge: torch.Tensor,
    adv_coefs: torch.Tensor,
    adv_coefs_3rd: torch.Tensor,
    edge_start: int,
    edge_end: int,
    n_adv_cells_max: int,
) -> dict[str, Any]:
    """Build scalar-independent Pattern C edge stencil scaffolding."""
    edge_slice = slice(edge_start - 1, edge_end)
    n_adv = n_adv_cells_for_edge[edge_slice]
    edge_count = int(n_adv.shape[0])
    empty_rows = torch.empty(0, dtype=torch.int64, device=scalar_field.device)
    if edge_count == 0:
        return {
            "edge_slice": edge_slice,
            "edge_count": edge_count,
            "slot_count": 0,
            "min_valid_slots": 0,
            "slot_cells_matrix": adv_cells_for_edge[edge_slice, :0],
            "slot_valid_matrix": torch.empty(
                edge_count, 0, dtype=torch.bool, device=scalar_field.device
            ),
            "slot_valid_matrix_all": True,
            "adv_coef": adv_coefs[edge_slice, :0],
            "adv_coef_3rd": adv_coefs_3rd[edge_slice, :0],
            "use_full_width_patch_path": False,
            "common_full_width_row_count": 0,
            "fallback_row_count": 0,
            "fallback_rows": empty_rows,
            "one": torch.tensor(
                1.0, dtype=scalar_field.dtype, device=scalar_field.device
            ),
        }

    if scalar_field.device.type == "cpu" and not _compile_active():
        slot_count = min(n_adv_cells_max, int(n_adv.max().detach().cpu()))
        min_valid_slots = min(slot_count, int(n_adv.min().detach().cpu()))
    else:
        slot_count = n_adv_cells_max
        min_valid_slots = 0

    adv_cells = adv_cells_for_edge[edge_slice, :slot_count]
    slot_cells: list[torch.Tensor] = []
    slot_valid_matrix_cols: list[torch.Tensor] = []
    for slot in range(slot_count):
        cells = adv_cells[:, slot] - 1
        if slot >= min_valid_slots:
            valid = slot < n_adv
            cells = _safe_zero_invalid(cells, valid)
            slot_valid_matrix_cols.append(valid)
        else:
            cells = _long_index(cells)
            slot_valid_matrix_cols.append(
                torch.ones(edge_count, dtype=torch.bool, device=scalar_field.device)
            )
        slot_cells.append(cells)
    if slot_count == 0:
        slot_cells_matrix = adv_cells[:, :0].to(dtype=torch.int64)
        slot_valid_matrix = torch.empty(
            edge_count, 0, dtype=torch.bool, device=scalar_field.device
        )
    else:
        slot_cells_matrix = torch.stack(slot_cells, dim=1)
        slot_valid_matrix = torch.stack(slot_valid_matrix_cols, dim=1)
    if slot_valid_matrix.device.type == "cpu":
        slot_valid_matrix_all = bool(
            _static_mask_all_true(slot_valid_matrix, allow_cuda_sync=True)
        )
    else:
        slot_valid_matrix_all = False

    use_full_width_patch_path = False
    common_full_width_row_count = 0
    fallback_row_count = 0
    fallback_rows = empty_rows
    if (
        slot_count > 0
        and min_valid_slots < slot_count
        and scalar_field.device.type == "cpu"
        and not _compile_active()
    ):
        full_width_mask = n_adv == slot_count
        common_full_width_row_count = int(full_width_mask.sum().detach().cpu())
        fallback_row_count = edge_count - common_full_width_row_count
        if common_full_width_row_count > fallback_row_count > 0:
            fallback_rows = (
                torch.nonzero(~full_width_mask, as_tuple=False)
                .flatten()
                .to(dtype=torch.int64)
            )
            use_full_width_patch_path = True

    return {
        "edge_slice": edge_slice,
        "edge_count": edge_count,
        "slot_count": slot_count,
        "min_valid_slots": min_valid_slots,
        "slot_cells_matrix": slot_cells_matrix,
        "slot_valid_matrix": slot_valid_matrix,
        "slot_valid_matrix_all": slot_valid_matrix_all,
        "adv_coef": adv_coefs[edge_slice, :slot_count],
        "adv_coef_3rd": adv_coefs_3rd[edge_slice, :slot_count],
        "use_full_width_patch_path": use_full_width_patch_path,
        "common_full_width_row_count": common_full_width_row_count,
        "fallback_row_count": fallback_row_count,
        "fallback_rows": fallback_rows,
        "one": torch.tensor(1.0, dtype=scalar_field.dtype, device=scalar_field.device),
    }


class _PatternCOrderedViewReduce(torch.autograd.Function):
    """Batched Pattern-C forward with the legacy ordered gather adjoint."""

    @staticmethod
    def forward(
        ctx,
        scalar_field: torch.Tensor,
        uh_sign: torch.Tensor,
        adv_coef: torch.Tensor,
        adv_coef_3rd: torch.Tensor,
        slot_cells_matrix: torch.Tensor,
        slot_valid_matrix: torch.Tensor,
        slot_valid_matrix_all: bool | None,
        apply_valid_masks: bool,
    ) -> torch.Tensor:
        ctx.save_for_backward(
            scalar_field,
            uh_sign,
            adv_coef,
            adv_coef_3rd,
            slot_cells_matrix,
            slot_valid_matrix,
        )
        ctx.save_for_forward(
            scalar_field,
            uh_sign,
            adv_coef,
            adv_coef_3rd,
            slot_cells_matrix,
            slot_valid_matrix,
        )
        ctx.slot_valid_matrix_all = slot_valid_matrix_all
        ctx.apply_valid_masks = apply_valid_masks
        edge_count = slot_cells_matrix.shape[0]
        # Keep CPU eager slot temporaries small enough to reuse cached rows.
        # Each output row retains the same ordered slot chain and adjoint.
        block_width = (
            1024
            if scalar_field.device.type == "cpu" and not _compile_active()
            else max(edge_count, 1)
        )
        output_shape = (edge_count, scalar_field.shape[1], scalar_field.shape[2])
        if edge_count == 0:
            return scalar_field.new_zeros(output_shape)
        output = (
            scalar_field.new_empty(output_shape) if edge_count > block_width else None
        )
        apply_masks = apply_valid_masks and slot_valid_matrix_all is not True
        for start in range(0, edge_count, block_width):
            stop = min(start + block_width, edge_count)
            row_count = stop - start
            result: torch.Tensor | None = None
            for slot in range(slot_cells_matrix.shape[1]):
                values = gather_rows(scalar_field, slot_cells_matrix[start:stop, slot])
                weight = adv_coef[start:stop, slot].view(row_count, 1)
                weight = weight + uh_sign[start:stop] * adv_coef_3rd[
                    start:stop, slot
                ].view(row_count, 1)
                contribution = weight.unsqueeze(2) * values
                if apply_masks:
                    contribution = contribution * slot_valid_matrix[
                        start:stop, slot
                    ].view(row_count, 1, 1).to(dtype=contribution.dtype)
                result = _accumulate_ordered_slot(result, contribution)
            if result is None:
                result = scalar_field.new_zeros(
                    (row_count, scalar_field.shape[1], scalar_field.shape[2])
                )
            if output is None:
                return result
            output[start:stop] = result
        return cast(torch.Tensor, output)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            scalar_field,
            uh_sign,
            adv_coef,
            adv_coef_3rd,
            slot_cells_matrix,
            slot_valid_matrix,
        ) = ctx.saved_tensors
        grad_scalar = (
            torch.zeros_like(scalar_field) if ctx.needs_input_grad[0] else None
        )
        grad_uh = torch.zeros_like(uh_sign) if ctx.needs_input_grad[1] else None
        grad_adv = torch.zeros_like(adv_coef) if ctx.needs_input_grad[2] else None
        grad_adv_3rd = (
            torch.zeros_like(adv_coef_3rd) if ctx.needs_input_grad[3] else None
        )
        apply_masks = ctx.apply_valid_masks and ctx.slot_valid_matrix_all is not True

        # Separate gather nodes in the legacy ordered graph accumulate their
        # shared-index adjoints in reverse graph order. Preserve that contract.
        for slot in range(slot_cells_matrix.shape[1] - 1, -1, -1):
            indices = slot_cells_matrix[:, slot]
            scalar_values = gather_rows(scalar_field, indices)
            slot_grad = grad_output
            if apply_masks:
                slot_grad = slot_grad * slot_valid_matrix[:, slot].view(-1, 1, 1).to(
                    dtype=slot_grad.dtype
                )
            weight = adv_coef[:, slot].view(-1, 1)
            weight = weight + uh_sign * adv_coef_3rd[:, slot].view(-1, 1)
            if grad_scalar is not None:
                grad_scalar.index_add_(
                    0,
                    indices.to(dtype=torch.int64),
                    slot_grad * weight.unsqueeze(2),
                )
            weighted_dot = (slot_grad * scalar_values).sum(dim=2)
            if grad_uh is not None:
                grad_uh = grad_uh + weighted_dot * adv_coef_3rd[:, slot].view(-1, 1)
            if grad_adv is not None:
                grad_adv[:, slot] = weighted_dot.sum(dim=1)
            if grad_adv_3rd is not None:
                grad_adv_3rd[:, slot] = (weighted_dot * uh_sign).sum(dim=1)
        return grad_scalar, grad_uh, grad_adv, grad_adv_3rd, None, None, None, None

    @staticmethod
    def jvp(
        ctx,
        grad_scalar: torch.Tensor | None,
        grad_uh: torch.Tensor | None,
        grad_adv: torch.Tensor | None,
        grad_adv_3rd: torch.Tensor | None,
        grad_indices: torch.Tensor | None,
        grad_valid: torch.Tensor | None,
        grad_valid_all: bool | None,
        grad_apply_masks: bool | None,
    ) -> torch.Tensor:
        del grad_indices, grad_valid, grad_valid_all, grad_apply_masks
        (
            scalar_field,
            uh_sign,
            adv_coef,
            adv_coef_3rd,
            slot_cells_matrix,
            slot_valid_matrix,
        ) = ctx.saved_tensors
        edge_count = slot_cells_matrix.shape[0]
        tangent: torch.Tensor | None = None
        apply_masks = ctx.apply_valid_masks and ctx.slot_valid_matrix_all is not True
        for slot in range(slot_cells_matrix.shape[1]):
            indices = slot_cells_matrix[:, slot]
            scalar_values = gather_rows(scalar_field, indices)
            weight = adv_coef[:, slot].view(edge_count, 1)
            weight = weight + uh_sign * adv_coef_3rd[:, slot].view(edge_count, 1)
            contribution = torch.zeros_like(scalar_values)
            if grad_scalar is not None:
                contribution = contribution + weight.unsqueeze(2) * gather_rows(
                    grad_scalar, indices
                )
            weight_tangent = torch.zeros_like(weight)
            if grad_adv is not None:
                weight_tangent = weight_tangent + grad_adv[:, slot].view(edge_count, 1)
            if grad_uh is not None:
                weight_tangent = weight_tangent + grad_uh * adv_coef_3rd[:, slot].view(
                    edge_count, 1
                )
            if grad_adv_3rd is not None:
                weight_tangent = weight_tangent + uh_sign * grad_adv_3rd[:, slot].view(
                    edge_count, 1
                )
            contribution = contribution + weight_tangent.unsqueeze(2) * scalar_values
            if apply_masks:
                contribution = contribution * slot_valid_matrix[:, slot].view(
                    edge_count, 1, 1
                ).to(dtype=contribution.dtype)
            tangent = _accumulate_ordered_slot(tangent, contribution)
        if tangent is None:
            return cast(
                torch.Tensor,
                scalar_field.new_zeros(
                    (edge_count, scalar_field.shape[1], scalar_field.shape[2])
                ),
            )
        return tangent


def _compute_pattern_c_flux_rows_view_reduce(
    *,
    scalar_field: torch.Tensor,
    uh_sign: torch.Tensor,
    adv_coef: torch.Tensor,
    adv_coef_3rd: torch.Tensor,
    slot_cells_matrix: torch.Tensor,
    slot_valid_matrix: torch.Tensor,
    slot_valid_matrix_all: bool | None,
    slot_count: int,
    apply_valid_masks: bool,
) -> torch.Tensor:
    """Evaluate Pattern-C slots with one row-slot gather and ordered reduction."""
    edge_count = int(adv_coef.shape[0])
    if slot_count == 0:
        return scalar_field.new_zeros(
            (edge_count, scalar_field.shape[1], scalar_field.shape[2])
        )
    scalar_batch = gather_rows(scalar_field, slot_cells_matrix.reshape(-1)).reshape(
        edge_count,
        slot_count,
        scalar_field.shape[1],
        scalar_field.shape[2],
    )
    slot_weight = adv_coef.unsqueeze(2) + uh_sign.unsqueeze(1) * adv_coef_3rd.unsqueeze(
        2
    )
    contrib = slot_weight.unsqueeze(3) * scalar_batch
    if apply_valid_masks and slot_valid_matrix_all is not True:
        contrib = contrib * slot_valid_matrix.view(edge_count, slot_count, 1, 1).to(
            dtype=scalar_field.dtype
        )
    return sum_fortran_order(contrib, dim=1)


def _compute_pattern_c_flux_rows(
    *,
    scalar_field: torch.Tensor,
    uh_sign: torch.Tensor,
    adv_coef: torch.Tensor,
    adv_coef_3rd: torch.Tensor,
    slot_cells_matrix: torch.Tensor,
    slot_valid_matrix: torch.Tensor,
    slot_valid_matrix_all: bool | None = None,
    slot_count: int,
    apply_valid_masks: bool,
) -> torch.Tensor:
    """Evaluate Pattern-C slots for one row set using Fortran slot order."""
    edge_count = int(adv_coef.shape[0])
    if slot_count > 1:
        return cast(
            torch.Tensor,
            _PatternCOrderedViewReduce.apply(
                scalar_field,
                uh_sign,
                adv_coef,
                adv_coef_3rd,
                slot_cells_matrix,
                slot_valid_matrix,
                slot_valid_matrix_all,
                apply_valid_masks,
            ),
        )
    horiz_flux: torch.Tensor | None = None
    batch_size = _slot_gather_batch(scalar_field, slot_count)
    if batch_size > 1:
        for batch_start, batch_stop in _slot_batches(slot_count, batch_size):
            scalar_batch = _gather_slot_batch(
                scalar_field,
                slot_cells_matrix,
                batch_start,
                batch_stop,
            )
            valid_batch = slot_valid_matrix[:, batch_start:batch_stop]
            for local_slot in range(batch_stop - batch_start):
                slot = batch_start + local_slot
                slot_scalars = scalar_batch[:, local_slot, :, :]
                slot_weight = adv_coef[:, slot].view(
                    edge_count, 1
                ) + uh_sign * adv_coef_3rd[:, slot].view(edge_count, 1)
                contrib = slot_weight.unsqueeze(2) * slot_scalars
                if apply_valid_masks and slot_valid_matrix_all is not True:
                    contrib = (
                        valid_batch[:, local_slot]
                        .view(edge_count, 1, 1)
                        .to(dtype=scalar_field.dtype)
                        * contrib
                    )
                horiz_flux = _accumulate_ordered_slot(horiz_flux, contrib)
        if horiz_flux is None:
            horiz_flux = scalar_field.new_zeros(
                (edge_count, scalar_field.shape[1], scalar_field.shape[2])
            )
        return horiz_flux

    for slot in range(slot_count):
        slot_cells = slot_cells_matrix[:, slot]
        slot_valid = slot_valid_matrix[:, slot] if apply_valid_masks else None
        if slot_valid is None:
            slot_mask = None
        else:
            slot_mask = slot_valid.view(edge_count, 1, 1).to(dtype=scalar_field.dtype)
        slot_scalars = gather_rows(scalar_field, slot_cells)
        slot_weight = adv_coef[:, slot].view(edge_count, 1) + uh_sign * adv_coef_3rd[
            :, slot
        ].view(edge_count, 1)
        contrib = slot_weight.unsqueeze(2) * slot_scalars
        if slot_mask is not None:
            contrib = slot_mask * contrib
        horiz_flux = _accumulate_ordered_slot(horiz_flux, contrib)
    if horiz_flux is None:
        horiz_flux = scalar_field.new_zeros(
            (edge_count, scalar_field.shape[1], scalar_field.shape[2])
        )
    return horiz_flux


def _pattern_c_connectivity_payload_cache_key(
    **kwargs: object,
) -> tuple[object, ...] | None:
    """Return the mutation-sensitive cache key for a Pattern C static payload."""
    tensors = (
        kwargs["n_adv_cells_for_edge"],
        kwargs["adv_cells_for_edge"],
        kwargs["adv_coefs"],
        kwargs["adv_coefs_3rd"],
    )
    if not _static_connectivity_cache_allowed(tensors):
        return None

    tensor_keys: list[tuple[object, ...]] = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            tensor_key = _pattern_c_tensor_cache_key(tensor)
            if tensor_key is None:
                return None
            tensor_keys.append(tensor_key)

    scalar_field = kwargs["scalar_field"]
    if not isinstance(scalar_field, torch.Tensor):
        return None

    return (
        str(scalar_field.device),
        str(scalar_field.dtype),
        kwargs["edge_start"],
        kwargs["edge_end"],
        kwargs["n_adv_cells_max"],
        *tensor_keys,
    )


def _pattern_c_connectivity_payload(**kwargs: object) -> dict[str, Any]:
    """Return cached Pattern C topology when topology is static."""
    key = _pattern_c_connectivity_payload_cache_key(**kwargs)
    if key is None:
        return _build_pattern_c_connectivity_payload(**kwargs)  # type: ignore[arg-type]
    cached = _PATTERN_C_CONNECTIVITY_CACHE.get(key)
    if cached is not None:
        return cached
    payload = _build_pattern_c_connectivity_payload(**kwargs)  # type: ignore[arg-type]
    if len(_PATTERN_C_CONNECTIVITY_CACHE) >= 8:
        _PATTERN_C_CONNECTIVITY_CACHE.pop(next(iter(_PATTERN_C_CONNECTIVITY_CACHE)))
    _PATTERN_C_CONNECTIVITY_CACHE[key] = payload
    return payload


def compute_pattern_c_flux(
    scalar_field: torch.Tensor,
    uh_avg: torch.Tensor,
    n_adv_cells_for_edge: torch.Tensor,
    adv_cells_for_edge: torch.Tensor,
    adv_coefs: torch.Tensor,
    adv_coefs_3rd: torch.Tensor,
    edge_start: int,
    edge_end: int,
    n_adv_cells_max: int,
    one: torch.Tensor,
    _connectivity_payload: Mapping[str, Any] | None = None,
) -> tuple[slice, torch.Tensor]:
    """
    Compute Pattern C horizontal scalar flux on edge range [edge_start, edge_end].

    Args:
        scalar_field: (nCells+1, nVertLevels, nScalars)
        uh_avg: (nEdges+1, nVertLevels)
        n_adv_cells_for_edge: (nEdges+1,)
        adv_cells_for_edge: (nEdges+1, nAdvCellsMax), Fortran 1-based indices
        adv_coefs: (nEdges+1, nAdvCellsMax)
        adv_coefs_3rd: (nEdges+1, nAdvCellsMax)
        edge_start: Fortran 1-based start edge (inclusive)
        edge_end: Fortran 1-based end edge (inclusive)
        n_adv_cells_max: maximum advection-cell stencil width
        one: scalar tensor with the target dtype/device

    Returns:
        (edge_slice, horiz_flux) where horiz_flux has shape
        (edge_count, nVertLevels, nScalars).
    """
    payload = (
        _connectivity_payload
        if _connectivity_payload is not None
        else _pattern_c_connectivity_payload(
            scalar_field=scalar_field,
            n_adv_cells_for_edge=n_adv_cells_for_edge,
            adv_cells_for_edge=adv_cells_for_edge,
            adv_coefs=adv_coefs,
            adv_coefs_3rd=adv_coefs_3rd,
            edge_start=edge_start,
            edge_end=edge_end,
            n_adv_cells_max=n_adv_cells_max,
        )
    )
    return compute_pattern_c_flux_from_payload(
        scalar_field=scalar_field,
        uh_avg=uh_avg,
        one=one,
        connectivity_payload=payload,
    )


def compute_pattern_c_flux_from_payload(
    *,
    scalar_field: torch.Tensor,
    uh_avg: torch.Tensor,
    one: torch.Tensor,
    connectivity_payload: Mapping[str, Any],
) -> tuple[slice, torch.Tensor]:
    """Compute Pattern-C flux using a payload-owned static advection operator."""
    payload = connectivity_payload
    edge_slice = payload["edge_slice"]
    return compute_pattern_c_flux_from_static_tensors(
        scalar_field=scalar_field,
        uh_avg=uh_avg,
        one=one,
        edge_start=int(edge_slice.start),
        edge_end=int(edge_slice.stop),
        edge_count=int(payload["edge_count"]),
        slot_count=int(payload["slot_count"]),
        adv_coef=payload["adv_coef"],
        adv_coef_3rd=payload["adv_coef_3rd"],
        slot_cells_matrix=payload["slot_cells_matrix"],
        slot_valid_matrix=payload["slot_valid_matrix"],
        slot_valid_matrix_all=payload.get("slot_valid_matrix_all"),
        use_full_width_patch_path=bool(payload.get("use_full_width_patch_path", False)),
        fallback_rows=payload["fallback_rows"],
    )


def compute_pattern_c_flux_from_static_tensors(
    *,
    scalar_field: torch.Tensor,
    uh_avg: torch.Tensor,
    one: torch.Tensor,
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
) -> tuple[slice, torch.Tensor]:
    """Compute Pattern-C from immutable tensors without mapping/object guards."""
    edge_slice = slice(edge_start, edge_end)
    uh_sign = fct_sign_pm(uh_avg[edge_slice, :]) * one
    if edge_count == 0:
        horiz_flux = torch.zeros(
            edge_count,
            scalar_field.shape[1],
            scalar_field.shape[2],
            dtype=scalar_field.dtype,
            device=scalar_field.device,
        )
        return edge_slice, horiz_flux

    if use_full_width_patch_path:
        horiz_flux = _compute_pattern_c_flux_rows(
            scalar_field=scalar_field,
            uh_sign=uh_sign,
            adv_coef=adv_coef,
            adv_coef_3rd=adv_coef_3rd,
            slot_cells_matrix=slot_cells_matrix,
            slot_valid_matrix=slot_valid_matrix,
            slot_valid_matrix_all=slot_valid_matrix_all,
            slot_count=slot_count,
            apply_valid_masks=False,
        )
        fallback_adv_coef = adv_coef.index_select(0, fallback_rows)
        fallback_adv_coef_3rd = adv_coef_3rd.index_select(0, fallback_rows)
        fallback_flux = _compute_pattern_c_flux_rows(
            scalar_field=scalar_field,
            uh_sign=uh_sign.index_select(0, fallback_rows),
            adv_coef=fallback_adv_coef,
            adv_coef_3rd=fallback_adv_coef_3rd,
            slot_cells_matrix=slot_cells_matrix.index_select(0, fallback_rows),
            slot_valid_matrix=slot_valid_matrix.index_select(0, fallback_rows),
            slot_valid_matrix_all=False,
            slot_count=slot_count,
            apply_valid_masks=True,
        )
        horiz_flux = _replace_rows(horiz_flux, fallback_rows, fallback_flux)
        return edge_slice, horiz_flux

    horiz_flux = _compute_pattern_c_flux_rows(
        scalar_field=scalar_field,
        uh_sign=uh_sign,
        adv_coef=adv_coef,
        adv_coef_3rd=adv_coef_3rd,
        slot_cells_matrix=slot_cells_matrix,
        slot_valid_matrix=slot_valid_matrix,
        slot_valid_matrix_all=slot_valid_matrix_all,
        slot_count=slot_count,
        apply_valid_masks=True,
    )
    return edge_slice, horiz_flux
