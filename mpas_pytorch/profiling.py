"""Opt-in profiling ranges for MPAS forward runners and hot sections."""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import torch

_PROFILE_SECTION_STACK: ContextVar[tuple[str, ...]] = ContextVar(
    "_PROFILE_SECTION_STACK",
    default=(),
)


def profiling_enabled(config: Mapping[str, Any] | None = None) -> bool:
    """Return whether section-level profiler ranges are requested."""
    if config is None:
        return False
    return bool(
        config.get("config_profile_sections", config.get("profile_sections", False))
    )


def profile_section_stack() -> tuple[str, ...]:
    """Return the active nested MPAS profiling-section names for this context."""
    return _PROFILE_SECTION_STACK.get()


def current_profile_section() -> str | None:
    """Return the innermost active MPAS profiling-section name, if any."""
    stack = profile_section_stack()
    return stack[-1] if stack else None


@contextmanager
def _push_profile_section(name: str) -> Iterator[None]:
    stack = profile_section_stack()
    token = _PROFILE_SECTION_STACK.set((*stack, str(name)))
    try:
        yield
    finally:
        _PROFILE_SECTION_STACK.reset(token)


class SectionTimer:
    """Low-overhead section accumulator using synchronized wall timing."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        use_cuda: bool = False,
        synchronize_cuda: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.use_cuda = bool(use_cuda)
        self.synchronize_cuda = bool(synchronize_cuda)
        self._totals_seconds: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def _sync(self) -> None:
        if self.use_cuda and self.synchronize_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            elapsed = time.perf_counter() - start
            self._totals_seconds[name] = self._totals_seconds.get(name, 0.0) + elapsed
            self._counts[name] = self._counts.get(name, 0) + 1

    def summary(self, *, top_n: int = 40) -> dict[str, Any]:
        ranked = sorted(
            self._totals_seconds.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        return {
            "status": "ok",
            "use_cuda": self.use_cuda,
            "synchronize_cuda": self.synchronize_cuda,
            "section_count": len(ranked),
            "top_sections": [
                {
                    "name": name,
                    "total_seconds": total,
                    "total_ms": total * 1000.0,
                    "count": self._counts[name],
                    "mean_ms": (total * 1000.0) / max(self._counts[name], 1),
                }
                for name, total in ranked[:top_n]
            ],
            "section_totals_seconds": {name: total for name, total in ranked},
            "section_counts": {name: self._counts[name] for name, _total in ranked},
        }


def section_timer(config: Mapping[str, Any] | None = None) -> SectionTimer | None:
    """Return the optional low-overhead section timer attached to ``config``."""
    if config is None:
        return None
    timer = config.get("_mpas_section_timer")
    return timer if isinstance(timer, SectionTimer) else None


@contextmanager
def _cuda_nvtx_range(name: str, *, enabled: bool) -> Iterator[None]:
    """Emit an NVTX range for Nsight without making it a timing dependency."""

    if not enabled or not torch.cuda.is_available():
        yield
        return
    try:
        torch.cuda.nvtx.range_push(name)
    except RuntimeError:
        # CPU-only PyTorch builds can be exercised behind mocked CUDA runners.
        # Real GPU profiling fails later on missing required Nsight ranges.
        yield
        return
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


@contextmanager
def profile_section(
    name: str,
    config: Mapping[str, Any] | None = None,
    *,
    enabled: bool | None = None,
) -> Iterator[None]:
    """Emit a torch-profiler range when profiling is explicitly enabled."""
    with _push_profile_section(name):
        record_active = profiling_enabled(config) if enabled is None else bool(enabled)
        nvtx_active = bool(
            config is not None and config.get("_mpas_emit_nvtx_ranges", False)
        )
        timer = section_timer(config)
        if timer is None:
            if not record_active and not nvtx_active:
                yield
                return
            record_context = (
                torch.profiler.record_function(name) if record_active else nullcontext()
            )
            with (
                record_context,
                _cuda_nvtx_range(
                    name,
                    enabled=nvtx_active,
                ),
            ):
                yield
            return

        with timer.section(name):
            if not record_active and not nvtx_active:
                yield
                return
            record_context = (
                torch.profiler.record_function(name) if record_active else nullcontext()
            )
            with (
                record_context,
                _cuda_nvtx_range(
                    name,
                    enabled=nvtx_active,
                ),
            ):
                yield
        return


def profiled_call(
    name: str,
    config: Mapping[str, Any] | None,
    fn: Any,
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Call ``fn`` under an optional profiling range."""
    with profile_section(name, config):
        return fn(*args, **kwargs)


@contextmanager
def profile_trace(
    output_path: str | Path | None,
    *,
    enabled: bool = False,
    use_cuda: bool | None = None,
    with_stack: bool = False,
) -> Iterator[Any | None]:
    """Optionally capture a torch-profiler trace and export it on exit."""
    if not enabled or output_path is None:
        yield None
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if use_cuda is None:
        use_cuda = torch.cuda.is_available()
    if use_cuda and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        profile_memory=True,
        with_stack=bool(with_stack),
    ) as prof:
        yield prof
    prof.export_chrome_trace(str(path))
