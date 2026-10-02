"""Validation helpers for MPAS PyTorch port."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch


def _iter_named_tensors(
    value: Any, default_name: str
) -> Iterable[tuple[str, torch.Tensor]]:
    """Yield (name, tensor) pairs from supported input types."""
    if value is None:
        return

    if isinstance(value, torch.Tensor):
        yield default_name, value
        return

    if isinstance(value, Mapping):
        for name, child in value.items():
            if isinstance(child, torch.Tensor):
                yield str(name), child
            elif isinstance(child, Mapping):
                for sub_name, sub_tensor in _iter_named_tensors(child, str(name)):
                    yield sub_name, sub_tensor
        return

    if isinstance(value, (list, tuple)):
        for idx, child in enumerate(value):
            child_name = f"{default_name}[{idx}]"
            if isinstance(child, torch.Tensor):
                yield child_name, child
            elif isinstance(child, Mapping):
                for sub_name, sub_tensor in _iter_named_tensors(child, child_name):
                    yield sub_name, sub_tensor


def require_consistent_device(*values: Any, context: str | None = None) -> torch.device:
    """
    Ensure all input tensors reside on one device.

    Accepts tensors directly and/or container values (dict/list/tuple) containing
    tensors. ``None`` values are ignored so optional arguments are easy to pass.

    Legacy compatibility:
      - ``require_consistent_device(context, mapping)`` is supported.

    Returns:
        The common torch.device. If no tensors are provided, returns CPU.

    Raises:
        ValueError: If tensors are split across multiple devices.
    """
    candidate_values = list(values)

    # Backward-compatible form: require_consistent_device("ctx", mapping)
    if (
        context is None
        and len(candidate_values) >= 2
        and isinstance(candidate_values[0], str)
    ):
        context = candidate_values[0]
        candidate_values = candidate_values[1:]

    named_tensors: list[tuple[str, torch.Tensor]] = []
    for idx, value in enumerate(candidate_values):
        named_tensors.extend(_iter_named_tensors(value, f"arg{idx}"))

    if not named_tensors:
        return torch.device("cpu")

    base_device = named_tensors[0][1].device
    mismatched = [
        (name, tensor.device)
        for name, tensor in named_tensors
        if tensor.device != base_device
    ]

    if mismatched:
        details = ", ".join(f"{name}={device}" for name, device in mismatched)
        prefix = f"{context}: " if context else ""
        raise ValueError(
            f"{prefix}expected all tensors on one device (baseline {base_device}), "
            f"found mismatches: {details}"
        )

    return base_device
