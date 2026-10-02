"""Hard FCT primitives with derivatives of the actual forward program.

The optional ``tau``/``eps`` arguments remain source-compatible with older
callers, but do not alter the supported derivative. At smooth points PyTorch
therefore differentiates the exact hard primal. Validation classifies and
excludes directions that cross one of the operators' switching surfaces.

The private smooth helpers are retained only to construct explicit negative
controls for the former straight-through behavior. Production functions never
call them.
"""

from __future__ import annotations

import torch

_SOFTPLUS_LINEAR_SWITCH = 20.0
_DEFAULT_EPS = 1.0e-6
_USE_MODULE_DEFAULT = object()


def get_default_eps() -> float:
    """Return the legacy experimental-smoother epsilon.

    Hard production derivatives do not consult this value. It remains only for
    source compatibility and explicit negative-control experiments.
    """
    return _DEFAULT_EPS


def set_default_eps(eps: float) -> None:
    """Set the legacy experimental-smoother epsilon.

    This setting cannot alter a production FCT primal or derivative.
    """
    global _DEFAULT_EPS
    if eps <= 0.0:
        raise ValueError("eps must be > 0")
    _DEFAULT_EPS = float(eps)


def _coerce_tau(x: torch.Tensor, tau: float | torch.Tensor) -> torch.Tensor:
    tau_t = torch.as_tensor(tau, dtype=x.dtype, device=x.device)
    if tau_t.numel() != 1:
        raise ValueError("tau must be a scalar")
    if not bool(torch.isfinite(tau_t).detach().cpu().item()):
        raise ValueError("tau must be finite")
    if float(tau_t.detach().cpu().item()) <= 0.0:
        raise ValueError("tau must be > 0")
    return tau_t


def _stable_softplus(x: torch.Tensor, tau: float | torch.Tensor) -> torch.Tensor:
    """Experimental smooth helper used only by negative controls."""
    tau_t = _coerce_tau(x, tau)
    z = x / tau_t
    middle = tau_t * (
        torch.maximum(z, torch.zeros_like(z)) + torch.log1p(torch.exp(-torch.abs(z)))
    )
    return torch.where(
        z > _SOFTPLUS_LINEAR_SWITCH,
        x,
        torch.where(z < -_SOFTPLUS_LINEAR_SWITCH, torch.zeros_like(x), middle),
    )


def _soft_clamp_01_impl(x: torch.Tensor, tau: float | torch.Tensor) -> torch.Tensor:
    """Experimental smooth-clamp helper used only by negative controls."""
    return _stable_softplus(x, tau) - _stable_softplus(x - 1.0, tau)


def _ordered_neighbor_bounds(
    field: torch.Tensor,
    indices: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Canonical low-scratch hard-extrema schedule."""
    n_rows, n_slots = indices.shape
    tail = (1,) * (field.ndim - 1)
    large = torch.finfo(field.dtype).max
    shape = (n_rows, *field.shape[1:])
    if n_slots == 0:
        return field.new_full(shape, -large), field.new_full(shape, large)

    first_valid = valid[:, 0]
    first_mask = first_valid.view(n_rows, *tail)
    first_value = field[indices[:, 0]]
    maximum = first_value.masked_fill(~first_mask, -large)
    minimum = first_value.masked_fill(~first_mask, large)
    seen = first_valid
    for slot in range(1, n_slots):
        slot_valid = valid[:, slot]
        combine = (slot_valid & seen).view(n_rows, *tail)
        replace = (slot_valid & ~seen).view(n_rows, *tail)
        slot_value = field[indices[:, slot]]
        maximum = torch.where(
            combine,
            torch.maximum(maximum, slot_value),
            torch.where(replace, slot_value, maximum),
        )
        minimum = torch.where(
            combine,
            torch.minimum(minimum, slot_value),
            torch.where(replace, slot_value, minimum),
        )
        seen = seen | slot_valid
    return maximum, minimum


def fct_neighbor_bounds(
    field: torch.Tensor,
    indices: torch.Tensor,
    valid: torch.Tensor,
    slot_tile_width: int = 0,
    eps: float | torch.Tensor | object = _USE_MODULE_DEFAULT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce neighbor extrema with the hard primal and its exact derivative.

    Exact ties are switching surfaces and are handled by the release
    differentiability preflight. The ordered schedule uses one slot of scratch,
    so ``slot_tile_width`` no longer selects a second numerical path.
    """
    if field.ndim < 2 or indices.ndim != 2 or valid.shape != indices.shape:
        raise ValueError(
            "neighbor bounds require field (source_rows, ...), indices (rows, slots), "
            "and valid (rows, slots)"
        )
    del eps, slot_tile_width
    return _ordered_neighbor_bounds(field, indices, valid)


def fct_clamp_01(x: torch.Tensor, tau: float | torch.Tensor = 0.01) -> torch.Tensor:
    """Return ``clamp(x, 0, 1)`` with the hard operator's derivative."""
    del tau
    return torch.clamp(x, min=0.0, max=1.0)


def fct_max_zero(x: torch.Tensor, tau: float | torch.Tensor = 0.01) -> torch.Tensor:
    """Return ``max(x, 0)`` with the hard operator's derivative."""
    del tau
    return torch.clamp_min(x, 0.0)


def fct_abs(
    x: torch.Tensor,
    eps: float | torch.Tensor | object = _USE_MODULE_DEFAULT,
) -> torch.Tensor:
    """Return ``abs(x)`` with the hard operator's derivative."""
    del eps
    return torch.abs(x)


def fct_sign_pm(
    x: torch.Tensor,
    eps: float | torch.Tensor | object = _USE_MODULE_DEFAULT,
) -> torch.Tensor:
    """Return the hard sign selector in ``{-1, +1}``.

    Away from zero its derivative is exactly zero. Zero is a switching surface
    that the release differentiability preflight reports.
    """
    del eps
    one = torch.ones((), dtype=x.dtype, device=x.device)
    return torch.where(x >= 0.0, one, -one)


def fct_sqrt_pos(
    x: torch.Tensor,
    eps: float | torch.Tensor | object = _USE_MODULE_DEFAULT,
) -> torch.Tensor:
    """Return ``sqrt(x)`` with the hard operator's derivative."""
    del eps
    return torch.sqrt(x)


def fct_min(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float | torch.Tensor | object = _USE_MODULE_DEFAULT,
) -> torch.Tensor:
    """Return ``minimum(x, y)`` with the hard operator's derivative."""
    del eps
    return torch.minimum(x, y)


def fct_max(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float | torch.Tensor | object = _USE_MODULE_DEFAULT,
) -> torch.Tensor:
    """Return ``maximum(x, y)`` with the hard operator's derivative."""
    del eps
    return torch.maximum(x, y)


__all__ = [
    "_soft_clamp_01_impl",
    "_stable_softplus",
    "fct_abs",
    "fct_clamp_01",
    "fct_max",
    "fct_max_zero",
    "fct_min",
    "fct_neighbor_bounds",
    "fct_sign_pm",
    "fct_sqrt_pos",
    "get_default_eps",
    "set_default_eps",
]
