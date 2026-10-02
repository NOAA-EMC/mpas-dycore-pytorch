"""PyTorch-native scan primitives for vertical tridiagonal recurrences."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Iterator, MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch

TRIDIAG_SCAN_EXPERIMENT_SHAPES: dict[str, tuple[int, int]] = {
    "120km": (40962, 55),
    "x1.40962": (40962, 55),
    "40962x55": (40962, 55),
    "060km": (163842, 55),
    "60km": (163842, 55),
    "x1.163842": (163842, 55),
    "163842x55": (163842, 55),
}
TRIDIAG_SCAN_POLICIES = frozenset(
    {"auto", "cpu-compile", "force", "off", "cuda-experiment"}
)
TRIDIAG_SCAN_DEFAULT_POLICY = "off"
_TRIDIAG_SCAN_POLICY_OVERRIDE: str | None = None
_TRIDIAG_SCAN_ALLOWED_SHAPES_OVERRIDE: frozenset[str] | None = None
_TRIDIAG_SCAN_PLAN_CACHE_OVERRIDE: (
    MutableMapping[tuple[int, str, str], VerticalScanPlan] | None
) = None
_TRIDIAG_SCAN_DEFAULT_PLAN_CACHE: dict[tuple[int, str, str], VerticalScanPlan] = {}


@dataclass(frozen=True)
class VerticalScanStep:
    """Precomputed metadata for one iterative-doubling vertical scan step."""

    step: int


@dataclass(frozen=True)
class VerticalScanPlan:
    """Runner-owned column metadata for a fixed vertical scan shape."""

    n_levels: int
    device: str
    dtype: str
    columns: torch.Tensor
    steps: tuple[VerticalScanStep, ...]

    def snapshot(self) -> dict[str, Any]:
        """Return JSON-safe metadata for audit artifacts."""
        return {
            "n_levels": int(self.n_levels),
            "device": self.device,
            "dtype": self.dtype,
            "step_count": len(self.steps),
            "steps": [int(step.step) for step in self.steps],
        }


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


def compile_active() -> bool:
    compiler = _torch_compiler_module()
    if compiler is not None and hasattr(compiler, "is_compiling"):
        return bool(compiler.is_compiling())
    dynamo = _torch_dynamo_module()
    if dynamo is not None and hasattr(dynamo, "is_compiling"):
        return bool(dynamo.is_compiling())
    return False


def tridiag_scan_shape_key(tensor: object) -> str | None:
    """Return the row/level key used for guarded scan experiments."""
    shape = getattr(tensor, "shape", None)
    if shape is None or len(shape) < 2:
        return None
    try:
        rows = int(shape[0])
        levels = int(shape[1])
    except (TypeError, ValueError):
        return None
    return f"{rows}x{levels}"


def _canonical_tridiag_scan_shape_token(token: str) -> str | None:
    value = token.strip().lower()
    if not value:
        return None
    if value in TRIDIAG_SCAN_EXPERIMENT_SHAPES:
        rows, levels = TRIDIAG_SCAN_EXPERIMENT_SHAPES[value]
        return f"{rows}x{levels}"
    match = re.fullmatch(r"(\d+)\s*[x,]\s*(\d+)", value)
    if match:
        return f"{int(match.group(1))}x{int(match.group(2))}"
    return None


def canonical_tridiag_scan_shape_keys(
    shapes: str | Iterable[object] | None,
) -> frozenset[str]:
    """Return canonical ``rowsxlevels`` keys from aliases or exact tokens."""
    if shapes is None:
        return frozenset()
    if isinstance(shapes, str):
        tokens = re.split(r"[\s,;]+", shapes)
    else:
        tokens = [str(shape) for shape in shapes]
    keys: set[str] = set()
    for token in tokens:
        key = _canonical_tridiag_scan_shape_token(token)
        if key is not None:
            keys.add(key)
    return frozenset(keys)


def tridiag_scan_allowed_shape_keys() -> frozenset[str]:
    """Return shape keys explicitly admitted for CUDA scan experiments."""
    override = _TRIDIAG_SCAN_ALLOWED_SHAPES_OVERRIDE
    if override is not None:
        return override
    return canonical_tridiag_scan_shape_keys(
        os.environ.get("MPAS_TRIDIAG_SCAN_ALLOWED_SHAPES", "")
    )


def tridiag_scan_shape_allowed(tensor: object) -> bool:
    """Return whether ``tensor`` matches the CUDA scan experiment allowlist."""
    key = tridiag_scan_shape_key(tensor)
    return key is not None and key in tridiag_scan_allowed_shape_keys()


def vertical_scan_plan_key(
    reference: torch.Tensor,
    n_levels: int | None = None,
) -> tuple[int, str, str]:
    """Return the cache key for scan metadata matching ``reference``."""
    levels = int(reference.shape[1] if n_levels is None else n_levels)
    return (levels, str(reference.device), str(reference.dtype))


def build_vertical_scan_plan(
    n_levels: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype | str,
) -> VerticalScanPlan:
    """Build reusable column metadata for iterative-doubling scans."""
    levels = int(n_levels)
    if levels < 0:
        raise ValueError("n_levels must be non-negative")
    target_device = torch.device(device)
    dtype_key = str(dtype)
    columns = torch.arange(levels, device=target_device, dtype=torch.int64)
    steps: list[VerticalScanStep] = []
    step = 1
    while step < levels:
        steps.append(VerticalScanStep(step=step))
        step *= 2
    return VerticalScanPlan(
        n_levels=levels,
        device=str(target_device),
        dtype=dtype_key,
        columns=columns,
        steps=tuple(steps),
    )


def vertical_scan_plan_for(
    reference: torch.Tensor,
    n_levels: int | None = None,
    *,
    cache: MutableMapping[tuple[int, str, str], VerticalScanPlan] | None = None,
) -> VerticalScanPlan:
    """Return cached scan metadata for ``reference``.

    Runners should pass a cache owned by their static-payload store.  The module
    fallback is only for focused direct tests and standalone helper probes.
    """
    selected_cache = cache
    if selected_cache is None:
        selected_cache = _TRIDIAG_SCAN_PLAN_CACHE_OVERRIDE
    if selected_cache is None:
        selected_cache = _TRIDIAG_SCAN_DEFAULT_PLAN_CACHE
    key = vertical_scan_plan_key(reference, n_levels)
    plan = selected_cache.get(key)
    if plan is None:
        plan = build_vertical_scan_plan(
            key[0],
            device=reference.device,
            dtype=reference.dtype,
        )
        selected_cache[key] = plan
    return plan


def ensure_vertical_scan_plan(
    reference: torch.Tensor,
    n_levels: int | None = None,
    *,
    cache: MutableMapping[tuple[int, str, str], VerticalScanPlan] | None = None,
) -> VerticalScanPlan:
    """Prebuild scan metadata before entering a compiled numeric region."""
    return vertical_scan_plan_for(reference, n_levels, cache=cache)


def replace_vertical_interval(
    base: torch.Tensor,
    patch: torch.Tensor,
    start: int,
) -> torch.Tensor:
    """Return ``base`` with one contiguous vertical interval replaced.

    This is the cat-free, mutation-free boundary operation used around scan
    recurrences.  Static column selection avoids the invalid vector/scalar C++
    ternary that CPU Inductor can emit for functionalized slice updates or
    broadcast ``where`` at production row counts.  Inductor lowers the stack of
    views to one output-owned copy kernel.
    """
    if base.ndim != 2 or patch.ndim != 2 or base.shape[0] != patch.shape[0]:
        raise ValueError("base and patch must be rank-2 tensors with equal row counts")
    interval_start = int(start)
    interval_end = interval_start + int(patch.shape[1])
    total_columns = int(base.shape[1])
    if interval_start < 0 or interval_end > total_columns:
        raise ValueError("vertical replacement interval is outside the base tensor")
    columns = [
        patch[:, column - interval_start]
        if interval_start <= column < interval_end
        else base[:, column]
        for column in range(total_columns)
    ]
    return torch.stack(columns, dim=1)


@contextmanager
def tridiag_scan_plan_cache(
    cache: MutableMapping[tuple[int, str, str], VerticalScanPlan] | None,
) -> Iterator[None]:
    """Temporarily route scan metadata lookup through a runner-owned cache."""
    global _TRIDIAG_SCAN_PLAN_CACHE_OVERRIDE
    old_cache = _TRIDIAG_SCAN_PLAN_CACHE_OVERRIDE
    if cache is not None:
        _TRIDIAG_SCAN_PLAN_CACHE_OVERRIDE = cache
    try:
        yield
    finally:
        _TRIDIAG_SCAN_PLAN_CACHE_OVERRIDE = old_cache


def normalize_tridiag_scan_policy(policy: object | None) -> str | None:
    """Normalize a user/config scan policy without changing the default."""
    if policy is None:
        return None
    value = str(policy).strip().lower()
    if not value or value in {"default", "unset"}:
        return None
    if value == "auto":
        return "auto"
    aliases = {
        "1": "force",
        "true": "force",
        "yes": "force",
        "on": "force",
        "scan": "force",
        "0": "off",
        "false": "off",
        "no": "off",
        "none": "off",
        "sequential": "off",
        "thomas": "off",
        "gpu": "cuda-experiment",
        "cuda": "cuda-experiment",
        "gpu-experiment": "cuda-experiment",
    }
    return aliases.get(value, value)


@contextmanager
def tridiag_scan_policy(
    policy: object | None,
    *,
    allowed_shapes: str | Iterable[object] | None = None,
) -> Iterator[None]:
    """Temporarily apply an explicit scan policy for one runner-owned call.

    This keeps production runners from depending on process-global shell state.
    The old environment variables remain a fallback for focused standalone
    probes and backward compatibility.
    """
    global _TRIDIAG_SCAN_POLICY_OVERRIDE, _TRIDIAG_SCAN_ALLOWED_SHAPES_OVERRIDE
    old_policy = _TRIDIAG_SCAN_POLICY_OVERRIDE
    old_shapes = _TRIDIAG_SCAN_ALLOWED_SHAPES_OVERRIDE
    _TRIDIAG_SCAN_POLICY_OVERRIDE = normalize_tridiag_scan_policy(policy)
    _TRIDIAG_SCAN_ALLOWED_SHAPES_OVERRIDE = (
        canonical_tridiag_scan_shape_keys(allowed_shapes)
        if allowed_shapes is not None
        else None
    )
    try:
        yield
    finally:
        _TRIDIAG_SCAN_POLICY_OVERRIDE = old_policy
        _TRIDIAG_SCAN_ALLOWED_SHAPES_OVERRIDE = old_shapes


def tridiag_scan_preferred(tensor: torch.Tensor) -> bool:
    """Return whether scan recurrences should replace Thomas sweeps.

    The scan path is the PyTorch-native parallel-prefix implementation for the
    vertical recurrence. The default production policy keeps the
    Fortran-ordered Thomas sweep. ``auto`` explicitly uses scan for compiled CPU
    and CUDA execution; ``cuda-experiment`` admits CUDA scan only for an explicit
    shape allowlist; ``off`` keeps the sequential Thomas path as the
    reference/fallback.
    """
    policy = _TRIDIAG_SCAN_POLICY_OVERRIDE
    if policy is None:
        policy = normalize_tridiag_scan_policy(
            os.environ.get("MPAS_TRIDIAG_SCAN_POLICY", TRIDIAG_SCAN_DEFAULT_POLICY)
        )
    if policy is None or policy == "auto":
        return tensor.device.type == "cuda" or compile_active()
    if policy == "force":
        return True
    if policy == "off":
        return False
    if policy == "cuda-experiment":
        return tensor.device.type == "cuda" and tridiag_scan_shape_allowed(tensor)
    if policy != "cpu-compile":
        raise ValueError(
            "unsupported tridiagonal scan policy "
            f"{policy!r}; expected one of {sorted(TRIDIAG_SCAN_POLICIES)}"
        )
    return compile_active() and tensor.device.type != "cuda"


def affine_forward_scan(
    linear: torch.Tensor,
    bias: torch.Tensor,
    initial: torch.Tensor,
) -> torch.Tensor:
    """Return ``y_i`` for ``y_i = linear_i * y_{i-1} + bias_i``.

    The recurrence is evaluated by iterative doubling over the vertical axis.
    Each stage replaces only its active tail through a functional scatter with
    unique column destinations. This avoids arithmetic and mask traffic on the
    unchanged prefix while keeping logarithmic-depth PyTorch-native structure.
    """
    if linear.shape != bias.shape:
        raise ValueError("linear and bias must have identical shapes")
    if linear.ndim != 2:
        raise ValueError("expected rank-2 tensors shaped (nRows, nLevels)")
    if initial.shape != linear.shape[:1]:
        raise ValueError("initial must have shape (nRows,)")

    n_levels = linear.shape[1]
    if n_levels == 0:
        return linear.new_empty((linear.shape[0], 0))

    plan = vertical_scan_plan_for(linear, n_levels)
    p_prefix = linear
    q_prefix = bias
    for scan_step in plan.steps:
        step = scan_step.step
        p_prev = p_prefix
        q_prev = q_prefix
        # Destinations are unique within every row, so this is deterministic and
        # has the ordinary gather-style adjoint rather than an atomic reduction.
        active_columns = plan.columns[step:].view(1, -1).expand(linear.shape[0], -1)
        p_tail = p_prev[:, step:]
        q_tail = q_prev[:, step:]
        p_prefix = torch.scatter(
            p_prev,
            1,
            active_columns,
            p_tail * p_prev[:, :-step],
        )
        q_prefix = torch.scatter(
            q_prev,
            1,
            active_columns,
            p_tail * q_prev[:, :-step] + q_tail,
        )

    return p_prefix * initial.unsqueeze(1) + q_prefix


def affine_reverse_scan(
    linear: torch.Tensor,
    bias: torch.Tensor,
    terminal: torch.Tensor,
) -> torch.Tensor:
    """Return ``y_i`` for ``y_i = linear_i * y_{i+1} + bias_i``."""
    if linear.shape != bias.shape:
        raise ValueError("linear and bias must have identical shapes")
    if linear.ndim != 2:
        raise ValueError("expected rank-2 tensors shaped (nRows, nLevels)")
    if terminal.shape != linear.shape[:1]:
        raise ValueError("terminal must have shape (nRows,)")

    n_levels = linear.shape[1]
    if n_levels == 0:
        return linear.new_empty((linear.shape[0], 0))

    plan = vertical_scan_plan_for(linear, n_levels)
    p_suffix = linear
    q_suffix = bias
    for scan_step in plan.steps:
        step = scan_step.step
        p_prev = p_suffix
        q_prev = q_suffix
        active_columns = plan.columns[:-step].view(1, -1).expand(linear.shape[0], -1)
        p_head = p_prev[:, :-step]
        q_head = q_prev[:, :-step]
        p_suffix = torch.scatter(
            p_prev,
            1,
            active_columns,
            p_head * p_prev[:, step:],
        )
        q_suffix = torch.scatter(
            q_prev,
            1,
            active_columns,
            p_head * q_prev[:, step:] + q_head,
        )

    return p_suffix * terminal.unsqueeze(1) + q_suffix


def mobius_prefix_scan_zero(
    a_num: torch.Tensor,
    b_num: torch.Tensor,
    c_den: torch.Tensor,
    d_den: torch.Tensor,
) -> torch.Tensor:
    """Prefix-scan linear-fractional transforms applied to zero.

    Each level transform is ``f(x) = (a_num*x + b_num) / (c_den*x + d_den)``.
    The returned value at slot ``i`` is ``f_i(...f_1(0))``.
    """
    if not (a_num.shape == b_num.shape == c_den.shape == d_den.shape):
        raise ValueError("all Mobius coefficient tensors must have identical shapes")
    if a_num.ndim != 2:
        raise ValueError("expected rank-2 tensors shaped (nRows, nLevels)")

    n_levels = a_num.shape[1]
    if n_levels == 0:
        return a_num.new_empty((a_num.shape[0], 0))

    plan = vertical_scan_plan_for(a_num, n_levels)
    a_prefix = a_num
    b_prefix = b_num
    c_prefix = c_den
    d_prefix = d_den
    for scan_step in plan.steps:
        step = scan_step.step
        a_prev = a_prefix
        b_prev = b_prefix
        c_prev = c_prefix
        d_prev = d_prefix
        active_columns = plan.columns[step:].view(1, -1).expand(a_num.shape[0], -1)
        a_tail = a_prev[:, step:]
        b_tail = b_prev[:, step:]
        c_tail = c_prev[:, step:]
        d_tail = d_prev[:, step:]
        a_shift = a_prev[:, :-step]
        b_shift = b_prev[:, :-step]
        c_shift = c_prev[:, :-step]
        d_shift = d_prev[:, :-step]

        a_prefix = torch.scatter(
            a_prev,
            1,
            active_columns,
            a_tail * a_shift + b_tail * c_shift,
        )
        b_prefix = torch.scatter(
            b_prev,
            1,
            active_columns,
            a_tail * b_shift + b_tail * d_shift,
        )
        c_prefix = torch.scatter(
            c_prev,
            1,
            active_columns,
            c_tail * a_shift + d_tail * c_shift,
        )
        d_prefix = torch.scatter(
            d_prev,
            1,
            active_columns,
            c_tail * b_shift + d_tail * d_shift,
        )

    return b_prefix / d_prefix


__all__ = [
    "TRIDIAG_SCAN_DEFAULT_POLICY",
    "TRIDIAG_SCAN_EXPERIMENT_SHAPES",
    "TRIDIAG_SCAN_POLICIES",
    "VerticalScanPlan",
    "VerticalScanStep",
    "affine_forward_scan",
    "affine_reverse_scan",
    "build_vertical_scan_plan",
    "canonical_tridiag_scan_shape_keys",
    "compile_active",
    "ensure_vertical_scan_plan",
    "mobius_prefix_scan_zero",
    "normalize_tridiag_scan_policy",
    "replace_vertical_interval",
    "tridiag_scan_allowed_shape_keys",
    "tridiag_scan_plan_cache",
    "tridiag_scan_policy",
    "tridiag_scan_preferred",
    "tridiag_scan_shape_allowed",
    "tridiag_scan_shape_key",
    "vertical_scan_plan_for",
    "vertical_scan_plan_key",
]
