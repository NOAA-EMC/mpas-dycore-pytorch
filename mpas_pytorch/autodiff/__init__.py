"""Autodiff mode controls for dual inference/gradient execution."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import torch

_GRAD_MODE_DEPTH: ContextVar[int] = ContextVar("_GRAD_MODE_DEPTH", default=0)


def _compile_active() -> bool:
    try:
        compiler: Any | None = torch.compiler
    except AttributeError:
        compiler = None
    if compiler is not None and hasattr(compiler, "is_compiling"):
        return bool(compiler.is_compiling())
    try:
        dynamo: Any | None = torch._dynamo
    except AttributeError:
        dynamo = None
    if dynamo is not None and hasattr(dynamo, "is_compiling"):
        return bool(dynamo.is_compiling())
    return False


def is_grad_mode() -> bool:
    """Return True only while inside differentiable_mode() and grad is enabled.

    If user code enters `torch.no_grad()` inside differentiable_mode(),
    this intentionally returns False.
    """
    if _compile_active():
        return False
    return _GRAD_MODE_DEPTH.get() > 0 and torch.is_grad_enabled()


@contextmanager
def differentiable_mode() -> Iterator[None]:
    """Enable explicit gradient mode for MPAS kernels.

    This mode is intentionally Python-context based and therefore incompatible
    with `torch.compile` graph capture. Depth tracking uses a ContextVar so
    nested calls remain isolated per thread/task.
    """
    token = _GRAD_MODE_DEPTH.set(_GRAD_MODE_DEPTH.get() + 1)
    try:
        with torch.enable_grad():
            yield
    finally:
        _GRAD_MODE_DEPTH.reset(token)


__all__ = ["differentiable_mode", "is_grad_mode"]
