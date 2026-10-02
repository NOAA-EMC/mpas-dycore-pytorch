"""Gradient-checkpointing helpers for tensor and state-dictionary call sites."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

import torch
from torch.autograd import forward_ad
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from . import differentiable_mode, is_grad_mode


def _forward_ad_active() -> bool:
    """Return True only while inside an active forward-ad dual_level context."""
    level = getattr(forward_ad, "_current_level", -1)
    if isinstance(level, int):
        return level != -1
    return False


def _non_tensor_equal(lhs: Any, rhs: Any) -> bool:
    """Best-effort recursive equality for non-tensor checkpoint payloads."""
    if type(lhs) is not type(rhs):
        return False
    if isinstance(lhs, torch.Tensor):
        # Defensive path: tensors should not appear in non-tensor outputs, but if
        # they do, compare deterministically instead of always failing.
        return torch.equal(lhs, rhs)
    if isinstance(lhs, Mapping):
        if lhs.keys() != rhs.keys():
            return False
        return all(_non_tensor_equal(lhs[k], rhs[k]) for k in lhs)
    if isinstance(lhs, tuple):
        if len(lhs) != len(rhs):
            return False
        return all(_non_tensor_equal(a, b) for a, b in zip(lhs, rhs))
    if isinstance(lhs, list):
        if len(lhs) != len(rhs):
            return False
        return all(_non_tensor_equal(a, b) for a, b in zip(lhs, rhs))
    if isinstance(lhs, set):
        return bool(lhs == rhs)
    if isinstance(lhs, (str, bytes, int, float, bool, type(None))):
        return bool(lhs == rhs)
    try:
        cmp = lhs == rhs
    except (TypeError, ValueError, RuntimeError):
        return False
    if isinstance(cmp, bool):
        return cmp
    return False


def _tensor_metadata_equal(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    return (
        lhs.shape == rhs.shape and lhs.dtype == rhs.dtype and lhs.device == rhs.device
    )


def _is_checkpoint_tensor_input(value: Any) -> bool:
    """Return True when a tensor must cross the checkpoint boundary.

    Tensors that do not participate in autograd can be captured in the closure
    instead of being threaded through ``torch.utils.checkpoint``. This keeps the
    checkpoint input arity focused on differentiable state while preserving
    deterministic access to static mesh/state tensors during recomputation.
    """
    return isinstance(value, torch.Tensor) and bool(value.requires_grad)


def checkpoint_if_enabled(
    fn: Callable[..., Any],
    *args: Any,
    enabled: bool = False,
    use_reentrant: bool = False,
    context_fn: Any = None,
    **kwargs: Any,
) -> Any:
    """Call function normally or through torch.utils.checkpoint."""
    if not enabled:
        return fn(*args, **kwargs)
    # PyTorch checkpoint internals use custom autograd.Function nodes without
    # forward-mode jvp support. In forward-AD contexts, bypass checkpointing to
    # keep JVP/TLAD paths functional.
    if _forward_ad_active():
        return fn(*args, **kwargs)

    preserve_dual_mode = is_grad_mode()
    restore_dual_mode = preserve_dual_mode and not use_reentrant

    def _wrapped(*inner_args: Any) -> Any:
        if restore_dual_mode and not is_grad_mode():
            with differentiable_mode():
                return fn(*inner_args, **kwargs)
        return fn(*inner_args, **kwargs)

    ckpt_kwargs: dict[str, Any] = {"use_reentrant": use_reentrant}
    if context_fn is not None and not use_reentrant:
        ckpt_kwargs["context_fn"] = context_fn
    return torch_checkpoint(_wrapped, *args, **ckpt_kwargs)


def checkpoint_state_dict(
    fn: Callable[[dict[str, Any]], dict[str, Any]],
    state_dict: Mapping[str, Any],
    *,
    enabled: bool = False,
    use_reentrant: bool = False,
    context_fn: Any = None,
) -> dict[str, Any]:
    """Checkpoint a function that accepts/returns a dict with tensor leaves.

    `torch.utils.checkpoint` tracks only tensor arguments/returns. This helper
    packs dict tensor leaves into positional tensor args and reconstructs the
    dict output, while preserving non-tensor entries and original dictionary
    order. Order can determine floating-point accumulation in the wrapped call.

    `fn` must be deterministic. During backward recomputation we require all
    non-tensor outputs to match the first forward invocation.
    """
    state_copy = dict(state_dict)
    if not enabled:
        result = fn(state_copy)
        if not isinstance(result, dict):
            raise TypeError(
                "checkpoint_state_dict expects the wrapped function to return a dict."
            )
        return result
    if _forward_ad_active():
        result = fn(state_copy)
        if not isinstance(result, dict):
            raise TypeError(
                "checkpoint_state_dict expects the wrapped function to return a dict."
            )
        return result

    preserve_dual_mode = is_grad_mode()
    tensor_keys = [k for k, v in state_copy.items() if _is_checkpoint_tensor_input(v)]
    if not tensor_keys:
        result = fn(state_copy)
        if not isinstance(result, dict):
            raise TypeError(
                "checkpoint_state_dict expects the wrapped function to return a dict."
            )
        return result

    tensor_inputs = tuple(state_copy[k] for k in tensor_keys)
    static_items = {k: v for k, v in state_copy.items() if k not in tensor_keys}

    captured_tensor_keys: list[str] = []
    captured_output_keys: list[str] = []
    captured_static_tensor_keys: list[str] = []
    captured_static_tensors: dict[str, Any] = {}
    captured_non_tensor: dict[str, Any] = {}
    call_count = 0
    restore_dual_mode = preserve_dual_mode and not use_reentrant

    def _wrapped(*flat_tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        nonlocal call_count
        call_count += 1
        dynamic_items = dict(zip(tensor_keys, flat_tensors, strict=True))
        rebuilt = {
            key: dynamic_items[key] if key in dynamic_items else static_items[key]
            for key in state_copy
        }
        if restore_dual_mode and not is_grad_mode():
            with differentiable_mode():
                result = fn(rebuilt)
        else:
            result = fn(rebuilt)
        if not isinstance(result, dict):
            raise TypeError(
                "checkpoint_state_dict expects the wrapped function to return a dict."
            )

        if use_reentrant:
            out_tensor_keys = [
                k for k, v in result.items() if isinstance(v, torch.Tensor)
            ]
            static_tensor_out: dict[str, Any] = {}
        else:
            out_tensor_keys = [
                k
                for k, v in result.items()
                if isinstance(v, torch.Tensor) and v.requires_grad
            ]
            static_tensor_out = {
                k: v
                for k, v in result.items()
                if isinstance(v, torch.Tensor) and not v.requires_grad
            }
        if not out_tensor_keys:
            raise RuntimeError(
                "checkpoint_state_dict requires at least one tensor output."
            )

        non_tensor_out = {
            k: v for k, v in result.items() if not isinstance(v, torch.Tensor)
        }
        if not captured_tensor_keys:
            captured_output_keys.extend(result)
            captured_tensor_keys.extend(out_tensor_keys)
            captured_static_tensor_keys.extend(static_tensor_out)
            captured_static_tensors.update(static_tensor_out)
            captured_non_tensor.update(deepcopy(non_tensor_out))
        else:
            if out_tensor_keys != captured_tensor_keys:
                raise RuntimeError(
                    "checkpoint_state_dict output tensor keys changed between invocations."
                )
            if list(static_tensor_out) != captured_static_tensor_keys:
                raise RuntimeError(
                    "checkpoint_state_dict static tensor output keys changed between invocations."
                )
            if list(result) != captured_output_keys:
                raise RuntimeError(
                    "checkpoint_state_dict output key order changed between invocations."
                )
            for key, value in static_tensor_out.items():
                if not _tensor_metadata_equal(value, captured_static_tensors[key]):
                    raise RuntimeError(
                        "checkpoint_state_dict static tensor output metadata changed between invocations: "
                        f"{key!r}"
                    )
            if not _non_tensor_equal(non_tensor_out, captured_non_tensor):
                raise RuntimeError(
                    "checkpoint_state_dict requires deterministic non-tensor outputs; "
                    f"mismatch detected on invocation {call_count}."
                )
        return tuple(result[k] for k in out_tensor_keys)

    ckpt_kwargs: dict[str, Any] = {"use_reentrant": use_reentrant}
    if context_fn is not None and not use_reentrant:
        ckpt_kwargs["context_fn"] = context_fn
    tensor_outputs = torch_checkpoint(_wrapped, *tensor_inputs, **ckpt_kwargs)
    if isinstance(tensor_outputs, torch.Tensor):
        tensor_outputs = (tensor_outputs,)
    else:
        tensor_outputs = tuple(tensor_outputs)
    if len(tensor_outputs) != len(captured_tensor_keys):
        raise RuntimeError(
            "checkpoint_state_dict output tensor arity changed between invocations."
        )

    restored = dict(captured_non_tensor)
    restored.update(captured_static_tensors)
    restored.update({k: t for k, t in zip(captured_tensor_keys, tensor_outputs)})
    return {key: restored[key] for key in captured_output_keys}


__all__ = ["checkpoint_if_enabled", "checkpoint_state_dict"]
