"""Finite-difference gradient verification helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class GradientCheckResult:
    passed: bool
    max_rel_error: float
    per_input: dict[str, float]
    max_abs_error: float = 0.0
    per_input_abs: dict[str, float] | None = None


def _flatten_sample_indices(
    tensor: torch.Tensor, num_random: int, seed: int
) -> torch.Tensor:
    total = tensor.numel()
    if total == 0:
        return torch.empty((0,), dtype=torch.int64, device=tensor.device)

    fixed = {0, total - 1, total // 2}
    rng = torch.Generator(device=tensor.device)
    rng.manual_seed(seed)
    if total <= len(fixed):
        return torch.tensor(sorted(fixed), dtype=torch.int64, device=tensor.device)

    candidate_count = min(total, num_random + len(fixed))
    random_idx = torch.randperm(total, generator=rng, device=tensor.device)[
        :candidate_count
    ].tolist()
    all_idx = sorted(set(random_idx).union(fixed))
    return torch.tensor(all_idx, dtype=torch.int64, device=tensor.device)


def _select_output(result: Any, output_key: str | None) -> torch.Tensor:
    if isinstance(result, torch.Tensor):
        return result
    if not isinstance(result, dict):
        raise TypeError("kernel_fn must return Tensor or dict[str, Tensor]")
    if output_key is None:
        raise ValueError("output_key is required when kernel_fn returns a dict")
    if output_key not in result:
        raise KeyError(f"output_key '{output_key}' missing from kernel outputs")
    output = result[output_key]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"kernel output '{output_key}' is not a tensor")
    return output


def _run_scalar_loss(
    kernel_fn: Callable[..., Any],
    inputs: dict[str, Any],
    output_key: str | None,
) -> torch.Tensor:
    output = _select_output(kernel_fn(**inputs), output_key)
    return output.sum()


def verify_kernel_gradients(
    kernel_fn: Callable[..., Any],
    inputs: dict[str, Any],
    output_key: str | None,
    eps: float = 1.0e-6,
    atol: float = 1.0e-5,
    rtol: float = 1.0e-3,
    random_samples: int = 50,
    seed: int = 42,
) -> GradientCheckResult:
    """Check autograd against finite differences on sampled tensor entries."""
    active: dict[str, Any] = {}
    for name, value in inputs.items():
        if not isinstance(value, torch.Tensor):
            continue
        if not torch.is_floating_point(value):
            continue
        if value.numel() == 0:
            continue
        active[name] = value.detach().clone().requires_grad_(True)

    if not active:
        return GradientCheckResult(
            passed=True,
            max_rel_error=0.0,
            max_abs_error=0.0,
            per_input={},
            per_input_abs={},
        )

    loss = _run_scalar_loss(kernel_fn, {**inputs, **active}, output_key)
    grads = torch.autograd.grad(
        loss, tuple(active.values()), retain_graph=False, allow_unused=False
    )
    grad_map = {name: grad.detach() for name, grad in zip(active.keys(), grads)}

    per_input: dict[str, float] = {}
    per_input_abs: dict[str, float] = {}
    max_rel = 0.0
    max_abs = 0.0
    max_scaled = 0.0
    for i, (name, tensor) in enumerate(active.items()):
        grad = grad_map[name]
        grad_flat = grad.reshape(-1)
        sample_idx = _flatten_sample_indices(tensor, random_samples, seed + i)
        local_max = 0.0
        local_max_abs = 0.0

        for idx in sample_idx.tolist():
            plus_inputs = {**inputs, **active}
            minus_inputs = {**inputs, **active}

            plus = tensor.detach().clone()
            minus = tensor.detach().clone()
            plus.view(-1)[idx] += eps
            minus.view(-1)[idx] -= eps

            plus_inputs[name] = plus
            minus_inputs[name] = minus

            f_plus = _run_scalar_loss(kernel_fn, plus_inputs, output_key)
            f_minus = _run_scalar_loss(kernel_fn, minus_inputs, output_key)
            fd = (f_plus - f_minus) / (2.0 * eps)
            ad = grad_flat[idx]
            abs_err = torch.abs(fd - ad)
            denom = torch.maximum(torch.abs(fd), torch.abs(ad)) + 1.0e-12
            rel = abs_err / denom
            scaled = abs_err / (
                atol + rtol * torch.maximum(torch.abs(fd), torch.abs(ad))
            )
            local_max = max(local_max, float(rel.item()))
            local_max_abs = max(local_max_abs, float(abs_err.item()))
            max_scaled = max(max_scaled, float(scaled.item()))

        per_input[name] = local_max
        per_input_abs[name] = local_max_abs
        max_rel = max(max_rel, local_max)
        max_abs = max(max_abs, local_max_abs)

    passed = max_scaled <= 1.0
    return GradientCheckResult(
        passed=passed,
        max_rel_error=max_rel,
        max_abs_error=max_abs,
        per_input=per_input,
        per_input_abs=per_input_abs,
    )


__all__ = ["GradientCheckResult", "verify_kernel_gradients"]
