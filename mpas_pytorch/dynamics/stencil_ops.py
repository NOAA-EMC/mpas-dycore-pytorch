"""Reusable stencil operators for vectorized MPAS dynamics kernels.

These helpers provide gather-reduce and CSR-SpMM implementations for common
connectivity patterns used by diagnostic-solve and upcoming vectorized kernels.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

import torch

_STENCIL_SLOT_GATHER_BATCH = 4
_AUDIT_ALLOW_STATIC_METADATA_UNDER_COMPILE = False


def _torch_compiler_module() -> object | None:
    try:
        return torch.compiler
    except AttributeError:
        return None


def _torch_dynamo_module() -> object | None:
    try:
        return torch._dynamo
    except AttributeError:
        return None


def _compile_active() -> bool:
    compiler = _torch_compiler_module()
    if compiler is not None and hasattr(compiler, "is_compiling"):
        return bool(compiler.is_compiling())
    dynamo = _torch_dynamo_module()
    if dynamo is not None and hasattr(dynamo, "is_compiling"):
        return bool(dynamo.is_compiling())
    return False


def _tensor_carries_autodiff(tensor: torch.Tensor) -> bool:
    """Return whether a tensor participates in reverse-mode or forward-mode AD."""
    if tensor.requires_grad:
        return True
    try:
        from torch.autograd import forward_ad as fwAD

        _, tangent = fwAD.unpack_dual(tensor)
    except RuntimeError:
        return False
    return tangent is not None


def _static_connectivity_cache_allowed(tensors: Iterable[Any]) -> bool:
    """Allow static topology cache reuse unless compile or AD touches topology."""
    if _compile_active() and not _AUDIT_ALLOW_STATIC_METADATA_UNDER_COMPILE:
        return False
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor) and _tensor_carries_autodiff(tensor):
            return False
    return True


def _active_slot_count(
    counts: torch.Tensor, available: int, device: torch.device
) -> int:
    """Return the active max stencil width for CPU eager paths."""
    if counts.numel() > 0 and device.type == "cpu" and not _compile_active():
        return min(available, int(counts.max().detach().cpu()))
    return available


def _slot_batches(slot_count: int, batch_size: int = _STENCIL_SLOT_GATHER_BATCH):
    width = max(1, min(int(batch_size), max(int(slot_count), 1)))
    for start in range(0, int(slot_count), width):
        yield start, min(start + width, int(slot_count))


def _slot_gather_batch(field: torch.Tensor, slot_count: int) -> int:
    if field.device.type == "cuda" or _compile_active():
        slots = max(int(slot_count), 1)
        if slots <= 8:
            return slots
        return _STENCIL_SLOT_GATHER_BATCH
    return 1


def _gather_slot_batch(
    field: torch.Tensor,
    indices: torch.Tensor,
    start: int,
    stop: int,
) -> torch.Tensor:
    if stop - start == 1:
        return gather_rows(field, indices[:, start]).unsqueeze(1)
    return gather_rows(field, indices[:, start:stop])


def _static_mask_all_true(
    mask: torch.Tensor | None,
    *,
    allow_cuda_sync: bool = False,
) -> bool | None:
    """Return setup-time all-true metadata for a static mask.

    CUDA-resident masks synchronize on host reads.  The default therefore refuses
    non-CPU masks so generic hot reducers cannot accidentally serialize GPU
    execution.  Static payload builders may opt in once with ``allow_cuda_sync``
    because topology masks are immutable for the runner-owned payload.
    """
    if mask is None:
        return None
    if _compile_active() and not (
        allow_cuda_sync and _AUDIT_ALLOW_STATIC_METADATA_UNDER_COMPILE
    ):
        return None
    if mask.device.type != "cpu" and not allow_cuda_sync:
        return None
    if mask.numel() == 0:
        return True
    return bool(torch.all(mask.detach().cpu()))


def _static_slot_all_true(
    mask: torch.Tensor | None,
    *,
    allow_cuda_sync: bool = False,
) -> tuple[bool, ...] | None:
    """Return per-slot setup-time all-true metadata for a static mask."""
    if mask is None:
        return None
    if _compile_active() and not (
        allow_cuda_sync and _AUDIT_ALLOW_STATIC_METADATA_UNDER_COMPILE
    ):
        return None
    if mask.device.type != "cpu" and not allow_cuda_sync:
        return None
    if mask.shape[1] == 0:
        return ()
    host_mask = mask.detach().cpu()
    return tuple(bool(torch.all(host_mask[:, slot])) for slot in range(mask.shape[1]))


def sum_fortran_order(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Deterministic left-to-right reduction to mirror Fortran loop order."""
    if dim < 0:
        dim += x.ndim
    if dim < 0 or dim >= x.ndim:
        raise IndexError(f"sum dim out of range: dim={dim}, ndim={x.ndim}")
    n = int(x.shape[dim])
    if n == 0:
        out_shape = list(x.shape)
        del out_shape[dim]
        return torch.zeros(out_shape, dtype=x.dtype, device=x.device)
    if not torch.is_grad_enabled() and not _compile_active():
        # Preserve the exact Fortran association while avoiding one full output
        # allocation per slot in eager inference.  Clone the first slot so the
        # input remains immutable, then reuse that private accumulator.
        total = x.select(dim, 0).clone()
        for idx in range(1, n):
            total.add_(x.select(dim, idx))
        return total
    total = x.select(dim, 0)
    for idx in range(1, n):
        total = total + x.select(dim, idx)
    return total


# Backward-compatible alias for existing call sites/tests.
_sum_fortran_order = sum_fortran_order


def _accumulate_ordered_slot(
    total: torch.Tensor | None, contrib: torch.Tensor
) -> torch.Tensor:
    """Left-to-right slot accumulation with inference-only local in-place reuse."""
    if total is None:
        return contrib
    if not torch.is_grad_enabled() and not _compile_active():
        total.add_(contrib)
        return total
    return total + contrib


def _tensor_version_cache_key(tensor: torch.Tensor) -> tuple[object, ...] | None:
    """Return a mutation-sensitive cache key, or ``None`` when unsafe to cache."""
    try:
        version = int(tensor._version)
    except RuntimeError:
        return None
    return (
        int(tensor.data_ptr()),
        tuple(tensor.shape),
        str(tensor.device),
        str(tensor.dtype),
        version,
    )


class _IndexCopyRows(torch.autograd.Function):
    """Memory-efficient row replacement for autograd.

    Standard ``torch.index_copy`` saves the full output tensor in the
    backward graph (~5 MB per call on the coarse mesh). This custom Function saves
    only ``row_idx`` (tiny) and propagates gradients sparsely: the backward
    for ``base`` scatters the output gradient at non-replaced rows, and
    the backward for ``values`` gathers from the replaced rows.
    """

    @staticmethod
    def forward(
        ctx, base: torch.Tensor, row_idx: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        ctx.save_for_backward(row_idx)
        ctx.save_for_forward(row_idx)
        ctx.base_shape = base.shape
        return torch.index_copy(base, 0, row_idx, values)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (row_idx,) = ctx.saved_tensors
        grad_base = None
        if ctx.needs_input_grad[0]:
            grad_base = grad_output.clone()
            grad_base[row_idx] = 0
        grad_values = None
        if ctx.needs_input_grad[2]:
            grad_values = grad_output.index_select(0, row_idx)
        return grad_base, None, grad_values

    @staticmethod
    def jvp(
        ctx: Any,
        grad_base: torch.Tensor | None,
        grad_row_idx: torch.Tensor | None,
        grad_values: torch.Tensor | None,
    ) -> torch.Tensor:
        (row_idx,) = ctx.saved_tensors
        del grad_row_idx
        if grad_base is None:
            if grad_values is None:
                raise RuntimeError(
                    "index-copy JVP requires one differentiable tensor input"
                )
            out = torch.zeros(
                ctx.base_shape, dtype=grad_values.dtype, device=grad_values.device
            )
            return torch.index_copy(out, 0, row_idx, grad_values)
        out = grad_base.clone()
        if grad_values is not None:
            out = torch.index_copy(out, 0, row_idx, grad_values)
        else:
            out[row_idx] = 0
        return out


def _functional_replace_rows(
    base: torch.Tensor,
    row_idx: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """Functional dim-0 row replacement without custom autograd wrapping."""
    return torch.index_copy(base, 0, row_idx, values)


def _shares_storage(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    """Return True when tensors use the same backing storage."""
    try:
        lhs_ptr = lhs.untyped_storage().data_ptr()
        rhs_ptr = rhs.untyped_storage().data_ptr()
    except RuntimeError:
        return False
    return lhs_ptr == rhs_ptr


def _clone_if_storage_overlap(base: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Clone source values before in-place writes when they alias the target."""
    return values.clone() if _shares_storage(base, values) else values


def _replace_rows(
    base: torch.Tensor, row_idx: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    """Functional dim-0 row replacement with an inference fast path."""
    if int(row_idx.numel()) == 0 and not _tensor_carries_autodiff(values):
        return base
    if _compile_active():
        return _functional_replace_rows(base, row_idx, values)
    if not torch.is_grad_enabled():
        values = _clone_if_storage_overlap(base, values)
        base[row_idx] = values
        return base
    return cast(torch.Tensor, _IndexCopyRows.apply(base, row_idx, values))


def _replace_all_rows(base: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Replace every dim-0 row without constructing an all-row index tensor."""
    if not torch.is_grad_enabled() and not _compile_active():
        values = _clone_if_storage_overlap(base, values)
        base[...] = values
        return base
    return values


def _replace_row_slice(
    base: torch.Tensor,
    row_start: int,
    row_end: int,
    values: torch.Tensor,
) -> torch.Tensor:
    """Replace a contiguous dim-0 row slice without constructing row indices."""
    row_start = int(row_start)
    row_end = int(row_end)
    if row_start < 0 or row_end < row_start or row_end > base.shape[0]:
        raise ValueError(
            f"invalid row slice [{row_start}:{row_end}] for dim-0 size {base.shape[0]}"
        )
    if (row_end - row_start) != values.shape[0]:
        raise ValueError(
            f"row slice length {row_end - row_start} does not match values rows {values.shape[0]}"
        )
    if row_start == row_end and not _tensor_carries_autodiff(values):
        return base
    if row_start == 0 and row_end == base.shape[0]:
        return _replace_all_rows(base, values)
    if not torch.is_grad_enabled() and not _compile_active():
        values = _clone_if_storage_overlap(base, values)
        base[row_start:row_end, ...] = values
        return base
    # Functional concatenation writes the result once. Cloning ``base`` and
    # then copying an almost-full active patch writes most rows twice, which is
    # especially costly for MPAS's one-row-padded public fields. There is no
    # reduction or gradient accumulation here, so concatenation preserves the
    # exact replacement and its VJP/JVP.
    pieces = []
    if row_start:
        pieces.append(base[:row_start, ...])
    pieces.append(values)
    if row_end < base.shape[0]:
        pieces.append(base[row_end:, ...])
    return torch.cat(tuple(pieces), dim=0)


def _replace_row_slice_cols(
    base: torch.Tensor,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    values: torch.Tensor,
) -> torch.Tensor:
    """Replace a contiguous dim-0 row slice and dim-1 column slice."""
    row_start = int(row_start)
    row_end = int(row_end)
    col_start = int(col_start)
    col_end = int(col_end)
    if row_start < 0 or row_end < row_start or row_end > base.shape[0]:
        raise ValueError(
            f"invalid row slice [{row_start}:{row_end}] for dim-0 size {base.shape[0]}"
        )
    if col_start < 0 or col_end < col_start or col_end > base.shape[1]:
        raise ValueError(
            f"invalid column slice [{col_start}:{col_end}] for dim-1 size {base.shape[1]}"
        )
    if (row_end - row_start) != values.shape[0]:
        raise ValueError(
            f"row slice length {row_end - row_start} does not match values rows {values.shape[0]}"
        )
    if (col_end - col_start) != values.shape[1]:
        raise ValueError(
            f"column slice length {col_end - col_start} does not match values columns {values.shape[1]}"
        )
    if (row_start == row_end or col_start == col_end) and not _tensor_carries_autodiff(
        values
    ):
        return base
    if col_start == 0 and col_end == base.shape[1]:
        return _replace_row_slice(base, row_start, row_end, values)
    if not torch.is_grad_enabled() and not _compile_active():
        values = _clone_if_storage_overlap(base, values)
        base[row_start:row_end, col_start:col_end, ...] = values
        return base
    active_columns = []
    if col_start:
        active_columns.append(base[row_start:row_end, :col_start, ...])
    active_columns.append(values)
    if col_end < base.shape[1]:
        active_columns.append(base[row_start:row_end, col_end:, ...])
    active_rows = torch.cat(tuple(active_columns), dim=1)
    output_rows = []
    if row_start:
        output_rows.append(base[:row_start, ...])
    output_rows.append(active_rows)
    if row_end < base.shape[0]:
        output_rows.append(base[row_end:, ...])
    return torch.cat(tuple(output_rows), dim=0)


def _functional_replace_rows_cols(
    base: torch.Tensor,
    row_idx: torch.Tensor,
    col_start: int,
    col_end: int,
    values: torch.Tensor,
) -> torch.Tensor:
    """Functional dim-0 + dim-1 slice replacement without materializing cats."""
    if col_start == 0 and col_end == base.shape[1]:
        updated = values
    else:
        updated = base.index_select(0, row_idx).clone()
        updated[:, col_start:col_end, ...] = values
    return torch.index_copy(base, 0, row_idx, updated)


class _IndexCopyRowsCols(torch.autograd.Function):
    """Memory-efficient row+col slice replacement for autograd."""

    @staticmethod
    def forward(
        ctx,
        base: torch.Tensor,
        row_idx: torch.Tensor,
        col_start: int,
        col_end: int,
        values: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(row_idx)
        ctx.save_for_forward(row_idx)
        ctx.col_start = col_start
        ctx.col_end = col_end
        ctx.base_shape = base.shape
        return _functional_replace_rows_cols(base, row_idx, col_start, col_end, values)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (row_idx,) = ctx.saved_tensors
        col_start, col_end = ctx.col_start, ctx.col_end
        grad_base = None
        if ctx.needs_input_grad[0]:
            grad_base = grad_output.clone()
            grad_base[row_idx, col_start:col_end, ...] = 0
        grad_values = None
        if ctx.needs_input_grad[4]:
            grad_values = grad_output.index_select(0, row_idx)[
                :, col_start:col_end, ...
            ]
        return grad_base, None, None, None, grad_values

    @staticmethod
    def jvp(
        ctx: Any,
        grad_base: torch.Tensor | None,
        grad_row_idx: torch.Tensor | None,
        grad_col_start: int | None,
        grad_col_end: int | None,
        grad_values: torch.Tensor | None,
    ) -> torch.Tensor:
        (row_idx,) = ctx.saved_tensors
        del grad_row_idx, grad_col_start, grad_col_end
        col_start, col_end = ctx.col_start, ctx.col_end
        if grad_base is None:
            if grad_values is None:
                raise RuntimeError(
                    "IndexCopyRowsCols.jvp requires a tangent for base or values"
                )
            out = torch.zeros(
                ctx.base_shape, dtype=grad_values.dtype, device=grad_values.device
            )
        else:
            out = grad_base.clone()
        if grad_values is not None:
            out = _functional_replace_rows_cols(
                out, row_idx, col_start, col_end, grad_values
            )
        else:
            out[row_idx, col_start:col_end, ...] = 0
        return out


def _replace_rows_cols(
    base: torch.Tensor,
    row_idx: torch.Tensor,
    col_start: int,
    col_end: int,
    values: torch.Tensor,
) -> torch.Tensor:
    """Functional dim-0 + dim-1 slice replacement with an inference fast path."""
    if (
        int(row_idx.numel()) == 0 or col_start == col_end
    ) and not _tensor_carries_autodiff(values):
        return base
    if _compile_active():
        return _functional_replace_rows_cols(base, row_idx, col_start, col_end, values)
    if not torch.is_grad_enabled():
        values = _clone_if_storage_overlap(base, values)
        base[row_idx, col_start:col_end, ...] = values
        return base
    return cast(
        torch.Tensor,
        _IndexCopyRowsCols.apply(base, row_idx, col_start, col_end, values),
    )


def _zero_rows(base: torch.Tensor, row_idx: torch.Tensor) -> torch.Tensor:
    """Replace dim-0 rows by zeros without gathering those rows first."""
    zeros = base.new_zeros((int(row_idx.numel()), *base.shape[1:]))
    return _replace_rows(base, row_idx.reshape(-1), zeros)


def _replace_all_rows_cols(
    base: torch.Tensor,
    col_start: int,
    col_end: int,
    values: torch.Tensor,
) -> torch.Tensor:
    """Replace one column slice for every row without constructing row indices."""
    if col_start == 0 and col_end == base.shape[1]:
        if not torch.is_grad_enabled() and not _compile_active():
            values = _clone_if_storage_overlap(base, values)
            base[...] = values
            return base
        return values
    if not torch.is_grad_enabled() and not _compile_active():
        values = _clone_if_storage_overlap(base, values)
        base[:, col_start:col_end, ...] = values
        return base
    pieces = []
    if col_start:
        pieces.append(base[:, :col_start, ...])
    pieces.append(values)
    if col_end < base.shape[1]:
        pieces.append(base[:, col_end:, ...])
    return torch.cat(tuple(pieces), dim=1)


def _clone_for_grad(base: torch.Tensor) -> torch.Tensor:
    """Clone only when grad mode needs alias isolation across live graphs."""
    return base.clone() if torch.is_grad_enabled() else base


def gather_rows(field: torch.Tensor, row_idx: torch.Tensor) -> torch.Tensor:
    """Gather dim-0 rows for arbitrary-shaped row indices via ``index_select``."""
    if row_idx.ndim == 1:
        if row_idx.dtype != torch.int64:
            row_idx = row_idx.to(dtype=torch.int64)
        return field.index_select(0, row_idx)
    flat_idx = row_idx.reshape(-1)
    if flat_idx.dtype != torch.int64:
        flat_idx = flat_idx.to(dtype=torch.int64)
    gathered = field.index_select(0, flat_idx)
    return gathered.reshape(*row_idx.shape, *field.shape[1:])


def sum_pair_rows(field: torch.Tensor, pair_idx: torch.Tensor) -> torch.Tensor:
    """Gather two row slots once and return ``slot0 + slot1`` in Fortran order."""
    if pair_idx.ndim != 2 or pair_idx.shape[1] != 2:
        raise ValueError(
            f"pair_idx must have shape (n, 2), got {tuple(pair_idx.shape)}"
        )
    pair = gather_rows(field, pair_idx)
    return pair[:, 0, ...] + pair[:, 1, ...]


def average_pair_rows(field: torch.Tensor, pair_idx: torch.Tensor) -> torch.Tensor:
    """Gather two row slots once and return their ordered arithmetic mean."""
    return 0.5 * sum_pair_rows(field, pair_idx)


def build_sparse_csr_stencil(
    indices0: torch.Tensor,
    weights: torch.Tensor,
    valid_mask: torch.Tensor,
    n_cols: int,
) -> torch.Tensor:
    """Build a CSR sparse matrix from dense stencil indices/weights.

    Args:
        indices0: 0-based column indices, shape (n_rows, n_stencil)
        weights: per-stencil weights, same shape as indices0
        valid_mask: True where stencil entry is active, same shape
        n_cols: number of source columns in the sparse matrix

    Returns:
        torch.sparse_csr_tensor of shape (n_rows, n_cols)
    """
    if indices0.ndim != 2:
        raise ValueError(f"indices0 must be rank-2, got shape {tuple(indices0.shape)}")
    if weights.shape != indices0.shape:
        raise ValueError("weights shape must match indices0")
    if valid_mask.shape != indices0.shape:
        raise ValueError("valid_mask shape must match indices0")

    rows = indices0.shape[0]
    counts = valid_mask.sum(dim=1)
    crow = torch.zeros(rows + 1, dtype=torch.int64, device=indices0.device)
    # GPU NOTE: integer CSR row-pointer prefix sum (exact for int64).
    crow[1:] = torch.cumsum(counts, dim=0)
    col = indices0[valid_mask]
    values = weights[valid_mask]
    return torch.sparse_csr_tensor(
        crow,
        col,
        values,
        size=(rows, n_cols),
        dtype=weights.dtype,
        device=indices0.device,
    )


def apply_sparse_stencil(matrix_csr: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
    """Apply a CSR stencil matrix to a field with shape (n_src, n_levels)."""
    if field.ndim != 2:
        raise ValueError(f"field must be rank-2, got shape {tuple(field.shape)}")
    if matrix_csr.shape[1] != field.shape[0]:
        raise ValueError(
            f"matrix columns ({matrix_csr.shape[1]}) must match field rows ({field.shape[0]})"
        )
    return cast(torch.Tensor, torch.sparse.mm(matrix_csr, field))


def edge_to_cell_interp(
    cell_pairs_idx: torch.Tensor,
    cell_pairs_valid: torch.Tensor,
    cell_field: torch.Tensor,
) -> torch.Tensor:
    """Pattern A: interpolate cell-centered field to edges via 0.5*(c1+c2)."""
    c1 = cell_pairs_idx[:, 0]
    c2 = cell_pairs_idx[:, 1]
    vals = 0.5 * (cell_field[c1, :] + cell_field[c2, :])
    mask = cell_pairs_valid[:, 0] & cell_pairs_valid[:, 1]
    return torch.where(mask[:, None], vals, torch.zeros_like(vals))


def cell_divergence_gather(
    edge_field: torch.Tensor,
    edge_metric: torch.Tensor,
    cell_edges_idx: torch.Tensor,
    cell_edges_sign: torch.Tensor,
    cell_edges_valid: torch.Tensor,
    inv_area_cell: torch.Tensor,
    cell_edges_valid_all_by_slot: tuple[bool, ...] | None = None,
) -> torch.Tensor:
    """Pattern B gather-reduce: cell divergence-like sum over edge stencil."""
    n_cells = cell_edges_idx.shape[0]
    if cell_edges_valid_all_by_slot is None:
        cell_edges_valid_all_by_slot = _static_slot_all_true(cell_edges_valid)
    total: torch.Tensor | None = None
    for batch_start, batch_stop in _slot_batches(
        cell_edges_idx.shape[1],
        _slot_gather_batch(edge_field, cell_edges_idx.shape[1]),
    ):
        edge_batch = _gather_slot_batch(
            edge_field, cell_edges_idx, batch_start, batch_stop
        )
        metric_batch = _gather_slot_batch(
            edge_metric, cell_edges_idx, batch_start, batch_stop
        )
        for local_slot in range(batch_stop - batch_start):
            slot = batch_start + local_slot
            weight = cell_edges_sign[:, slot] * metric_batch[:, local_slot]
            valid_all = (
                cell_edges_valid_all_by_slot is not None
                and cell_edges_valid_all_by_slot[slot]
            )
            if not valid_all:
                weight = torch.where(
                    cell_edges_valid[:, slot], weight, torch.zeros_like(weight)
                )
            contrib = weight.unsqueeze(1) * edge_batch[:, local_slot, :]
            total = _accumulate_ordered_slot(total, contrib)
    if total is None:
        total = edge_field.new_zeros((n_cells, edge_field.shape[1]))
    return total * inv_area_cell[:, None]


def _sum_weighted_stencil_slots(
    *,
    field: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    valid_all_by_slot: tuple[bool, ...] | None = None,
    slot_factor: torch.Tensor | None = None,
) -> torch.Tensor:
    """Accumulate a fixed-width ordered stencil without materializing all slots."""
    if field.ndim != 2:
        raise ValueError(f"field must be rank-2, got shape {tuple(field.shape)}")
    if indices.ndim != 2:
        raise ValueError(f"indices must be rank-2, got shape {tuple(indices.shape)}")
    if weights is not None and weights.shape != indices.shape:
        raise ValueError("weights shape must match indices")
    if valid_mask is not None and valid_mask.shape != indices.shape:
        raise ValueError("valid_mask shape must match indices")
    if valid_all_by_slot is None:
        valid_all_by_slot = _static_slot_all_true(valid_mask)
    if (
        field.device.type == "cuda"
        and not torch.is_grad_enabled()
        and not _compile_active()
    ):
        return _sum_weighted_stencil_view_reduce(
            field=field,
            indices=indices,
            weights=weights,
            valid_mask=valid_mask,
            valid_all_by_slot=valid_all_by_slot,
            slot_factor=slot_factor,
        )

    n_rows = indices.shape[0]
    total: torch.Tensor | None = None
    for batch_start, batch_stop in _slot_batches(
        indices.shape[1],
        _slot_gather_batch(field, indices.shape[1]),
    ):
        values_batch = _gather_slot_batch(field, indices, batch_start, batch_stop)
        for local_slot in range(batch_stop - batch_start):
            slot = batch_start + local_slot
            contrib = values_batch[:, local_slot, :]
            if weights is not None:
                contrib = weights[:, slot].unsqueeze(1) * contrib
            valid_all = valid_all_by_slot is not None and valid_all_by_slot[slot]
            if valid_mask is not None and not valid_all:
                contrib = torch.where(
                    valid_mask[:, slot].unsqueeze(1),
                    contrib,
                    torch.zeros_like(contrib),
                )
            if slot_factor is not None:
                contrib = contrib * slot_factor
            total = _accumulate_ordered_slot(total, contrib)
    if total is None:
        return field.new_zeros((n_rows, field.shape[1]))
    return total


def _sum_weighted_stencil_view_reduce(
    *,
    field: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    valid_all_by_slot: tuple[bool, ...] | None = None,
    slot_factor: torch.Tensor | None = None,
) -> torch.Tensor:
    """No-grad row-owned stencil reduction without atomics.

    The row/slot relation is already grouped by row, so the flattened gather can
    be viewed as ``(n_rows, n_slots, n_levels)`` and reduced in upstream slot
    order.  This avoids atomic row accumulation on CUDA and remains deterministic.
    """
    n_rows = indices.shape[0]
    n_slots = indices.shape[1]
    if n_slots == 0:
        return field.new_zeros((n_rows, field.shape[1]))
    contrib = gather_rows(field, indices.reshape(-1)).reshape(
        n_rows, n_slots, field.shape[1]
    )
    if weights is not None:
        contrib = weights.unsqueeze(2) * contrib
    valid_all = (
        valid_all_by_slot is not None
        and len(valid_all_by_slot) == n_slots
        and all(valid_all_by_slot)
    )
    if valid_mask is not None and not valid_all:
        contrib = contrib.masked_fill(~valid_mask.unsqueeze(2), 0.0)
    if slot_factor is not None:
        contrib = contrib * slot_factor.unsqueeze(1)
    return sum_fortran_order(contrib, dim=1)


def _sum_weighted_stencil_components_view_reduce(
    *,
    field: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    valid_all_by_slot: tuple[bool, ...] | None = None,
) -> torch.Tensor:
    """No-grad row-owned component reduction with one materialized slot view."""
    n_rows = indices.shape[0]
    n_slots = indices.shape[1]
    n_components = weights.shape[2]
    if n_slots == 0:
        return field.new_zeros((n_rows, field.shape[1], n_components))
    values = gather_rows(field, indices.reshape(-1)).reshape(
        n_rows, n_slots, field.shape[1]
    )
    contrib = values.unsqueeze(3) * weights.unsqueeze(2)
    valid_all = (
        valid_all_by_slot is not None
        and len(valid_all_by_slot) == n_slots
        and all(valid_all_by_slot)
    )
    if valid_mask is not None and not valid_all:
        contrib = contrib.masked_fill(~valid_mask.view(n_rows, n_slots, 1, 1), 0.0)
    return sum_fortran_order(contrib, dim=1)


def _sum_weighted_stencil_components(
    *,
    field: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    valid_all_by_slot: tuple[bool, ...] | None = None,
) -> torch.Tensor:
    """Accumulate ordered stencil components with one source gather per slot."""
    if field.ndim != 2:
        raise ValueError(f"field must be rank-2, got shape {tuple(field.shape)}")
    if indices.ndim != 2:
        raise ValueError(f"indices must be rank-2, got shape {tuple(indices.shape)}")
    if weights.ndim != 3 or weights.shape[:2] != indices.shape:
        raise ValueError("weights must have shape (*indices.shape, n_components)")
    if valid_mask is not None and valid_mask.shape != indices.shape:
        raise ValueError("valid_mask shape must match indices")
    if valid_all_by_slot is None:
        valid_all_by_slot = _static_slot_all_true(valid_mask)
    if (
        field.device.type == "cuda" or _compile_active()
    ) and not torch.is_grad_enabled():
        return _sum_weighted_stencil_components_view_reduce(
            field=field,
            indices=indices,
            weights=weights,
            valid_mask=valid_mask,
            valid_all_by_slot=valid_all_by_slot,
        )

    n_rows = indices.shape[0]
    n_components = weights.shape[2]
    total: torch.Tensor | None = None
    for batch_start, batch_stop in _slot_batches(
        indices.shape[1],
        _slot_gather_batch(field, indices.shape[1]),
    ):
        values_batch = _gather_slot_batch(field, indices, batch_start, batch_stop)
        for local_slot in range(batch_stop - batch_start):
            slot = batch_start + local_slot
            values = values_batch[:, local_slot, :]
            contrib = values.unsqueeze(2) * weights[:, slot, :].unsqueeze(1)
            valid_all = valid_all_by_slot is not None and valid_all_by_slot[slot]
            if valid_mask is not None and not valid_all:
                contrib = torch.where(
                    valid_mask[:, slot].view(n_rows, 1, 1),
                    contrib,
                    torch.zeros_like(contrib),
                )
            total = _accumulate_ordered_slot(total, contrib)
    if total is None:
        return field.new_zeros((n_rows, field.shape[1], n_components))
    return total


def vertex_vorticity_gather(
    edge_field: torch.Tensor,
    edge_metric: torch.Tensor,
    vertex_edges_idx: torch.Tensor,
    vertex_edges_sign: torch.Tensor,
    vertex_edges_valid: torch.Tensor,
    inv_area_triangle: torch.Tensor,
) -> torch.Tensor:
    """Pattern D gather-reduce: vertex circulation-like sum over edge stencil."""
    weights = vertex_edges_sign * edge_metric[vertex_edges_idx]
    return (
        _sum_weighted_stencil_slots(
            field=edge_field,
            indices=vertex_edges_idx,
            weights=weights,
            valid_mask=vertex_edges_valid,
        )
        * inv_area_triangle[:, None]
    )


def edge_tangential_velocity_gather(
    edge_field: torch.Tensor,
    edge_edges_idx: torch.Tensor,
    edge_edges_weights: torch.Tensor,
    edge_edges_valid: torch.Tensor,
) -> torch.Tensor:
    """Pattern E gather-reduce: tangential reconstruction from edge-neighbor stencil."""
    return _sum_weighted_stencil_slots(
        field=edge_field,
        indices=edge_edges_idx,
        weights=edge_edges_weights,
        valid_mask=edge_edges_valid,
    )


def build_divergence_csr(
    cell_edges_idx: torch.Tensor,
    cell_edges_sign: torch.Tensor,
    cell_edges_valid: torch.Tensor,
    dv_edge: torch.Tensor,
    inv_area_cell: torch.Tensor,
    n_edges: int,
) -> torch.Tensor:
    """Build CSR operator for divergence-like Pattern B."""
    weights = cell_edges_sign * dv_edge[cell_edges_idx] * inv_area_cell[:, None]
    return build_sparse_csr_stencil(
        cell_edges_idx, weights, cell_edges_valid, n_cols=n_edges
    )


def build_vorticity_csr(
    vertex_edges_idx: torch.Tensor,
    vertex_edges_sign: torch.Tensor,
    vertex_edges_valid: torch.Tensor,
    dc_edge: torch.Tensor,
    inv_area_triangle: torch.Tensor,
    n_edges: int,
) -> torch.Tensor:
    """Build CSR operator for vorticity-like Pattern D."""
    weights = vertex_edges_sign * dc_edge[vertex_edges_idx] * inv_area_triangle[:, None]
    return build_sparse_csr_stencil(
        vertex_edges_idx, weights, vertex_edges_valid, n_cols=n_edges
    )


def build_tangential_velocity_csr(
    edge_edges_idx: torch.Tensor,
    weights_on_edge: torch.Tensor,
    edge_edges_valid: torch.Tensor,
    n_edges: int,
) -> torch.Tensor:
    """Build CSR operator for tangential reconstruction Pattern E."""
    return build_sparse_csr_stencil(
        edge_edges_idx, weights_on_edge, edge_edges_valid, n_cols=n_edges
    )
