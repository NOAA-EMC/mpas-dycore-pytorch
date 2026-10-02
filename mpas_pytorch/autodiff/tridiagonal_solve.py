"""Custom autograd for acoustic-step-style tridiagonal forward/backward sweeps."""

from __future__ import annotations

from typing import Any, cast

import torch

from mpas_pytorch.autodiff.tridiagonal_scan import (
    affine_forward_scan,
    affine_reverse_scan,
    replace_vertical_interval,
    tridiag_scan_preferred,
)


def _tridiag_forward_elimination_scan(
    a_tri: torch.Tensor,
    alpha_tri: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    """Scan form of ``u_k = (rhs_k - a_k*u_{k-1}) * alpha_k``."""
    n_levels = a_tri.shape[1]
    if n_levels == 0:
        return rhs
    if n_levels == 1:
        return rhs
    p = -a_tri[:, 1:n_levels] * alpha_tri[:, 1:n_levels]
    q = rhs[:, 1:n_levels] * alpha_tri[:, 1:n_levels]
    interior = affine_forward_scan(p, q, rhs[:, 0])
    return replace_vertical_interval(rhs, interior, 1)


def _tridiag_back_substitution_scan(
    gamma_tri: torch.Tensor,
    eliminated: torch.Tensor,
) -> torch.Tensor:
    """Scan form of ``out_k = eliminated_k - gamma_k*out_{k+1}``."""
    n_levels = gamma_tri.shape[1]
    if n_levels == 0:
        return eliminated
    interior = affine_reverse_scan(
        -gamma_tri[:, :n_levels],
        eliminated[:, :n_levels],
        eliminated[:, n_levels],
    )
    return replace_vertical_interval(eliminated, interior, 0)


def _tridiag_solve_forward(
    a_tri: torch.Tensor,
    alpha_tri: torch.Tensor,
    gamma_tri: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    """PyTorch elimination + back-substitution shared by all forward modes."""
    n_levels = a_tri.shape[1]
    out = rhs.clone()
    for k in range(1, n_levels):
        out[:, k] = (out[:, k] - a_tri[:, k] * out[:, k - 1]) * alpha_tri[:, k]
    for k in range(n_levels - 1, -1, -1):
        out[:, k] = out[:, k] - gamma_tri[:, k] * out[:, k + 1]
    return out


def _tridiag_solve_forward_scan(
    a_tri: torch.Tensor,
    alpha_tri: torch.Tensor,
    gamma_tri: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    """Parallel-prefix form of acoustic-step tridiagonal solve sweeps."""
    eliminated = _tridiag_forward_elimination_scan(a_tri, alpha_tri, rhs)
    return _tridiag_back_substitution_scan(gamma_tri, eliminated)


def _tridiag_solve_backward_scan(
    a_tri: torch.Tensor,
    alpha_tri: torch.Tensor,
    gamma_tri: torch.Tensor,
    rhs: torch.Tensor,
    out: torch.Tensor,
    grad_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scan form of the reverse-mode solve sweeps."""
    _n_cells, n_levels = a_tri.shape
    grad_a = torch.zeros_like(a_tri)
    grad_alpha = torch.zeros_like(alpha_tri)
    grad_gamma = torch.zeros_like(gamma_tri)
    grad_rhs = torch.zeros_like(rhs)
    if n_levels == 0:
        return grad_a, grad_alpha, grad_gamma, grad_rhs

    if n_levels > 0:
        adj_tail = affine_forward_scan(
            -gamma_tri[:, :n_levels],
            grad_out[:, 1 : n_levels + 1],
            grad_out[:, 0],
        )
        adj_from_back_sub = grad_out.clone()
        adj_from_back_sub[:, 1 : n_levels + 1] = adj_tail
        grad_gamma[:, :n_levels] = (
            -adj_from_back_sub[:, :n_levels] * out[:, 1 : n_levels + 1]
        )
    else:
        adj_from_back_sub = grad_out

    u = _tridiag_forward_elimination_scan(a_tri, alpha_tri, rhs)
    if n_levels <= 1:
        grad_rhs[:, 0] = adj_from_back_sub[:, 0]
        if rhs.shape[1] > 1:
            grad_rhs[:, 1] = adj_from_back_sub[:, 1]
        return grad_a, grad_alpha, grad_gamma, grad_rhs

    p = -a_tri[:, 1:n_levels] * alpha_tri[:, 1:n_levels]
    q = adj_from_back_sub[:, 0 : n_levels - 1]
    adj_lower = affine_reverse_scan(
        p,
        q,
        adj_from_back_sub[:, n_levels - 1],
    )
    adj_u = adj_from_back_sub[:, :n_levels].clone()
    adj_u[:, : n_levels - 1] = adj_lower

    t = rhs[:, 1:n_levels] - a_tri[:, 1:n_levels] * u[:, 0 : n_levels - 1]
    adj_t = adj_u[:, 1:n_levels] * alpha_tri[:, 1:n_levels]
    grad_alpha[:, 1:n_levels] = adj_u[:, 1:n_levels] * t
    grad_rhs[:, 1:n_levels] = adj_t
    grad_a[:, 1:n_levels] = -adj_t * u[:, 0 : n_levels - 1]
    grad_rhs[:, 0] = adj_u[:, 0]
    grad_rhs[:, n_levels] = adj_from_back_sub[:, n_levels]
    return grad_a, grad_alpha, grad_gamma, grad_rhs


def _tridiag_solve_jvp_scan(
    a_tri: torch.Tensor,
    alpha_tri: torch.Tensor,
    gamma_tri: torch.Tensor,
    rhs: torch.Tensor,
    out: torch.Tensor,
    grad_a: torch.Tensor | None,
    grad_alpha: torch.Tensor | None,
    grad_gamma: torch.Tensor | None,
    grad_rhs: torch.Tensor | None,
) -> torch.Tensor:
    """Scan form of the forward-mode solve recurrence."""
    _n_cells, n_levels = a_tri.shape
    da = torch.zeros_like(a_tri) if grad_a is None else grad_a
    d_alpha = torch.zeros_like(alpha_tri) if grad_alpha is None else grad_alpha
    d_gamma = torch.zeros_like(gamma_tri) if grad_gamma is None else grad_gamma
    d_rhs = torch.zeros_like(rhs) if grad_rhs is None else grad_rhs
    if n_levels == 0:
        return d_rhs

    u = _tridiag_forward_elimination_scan(a_tri, alpha_tri, rhs)
    d_u = torch.zeros_like(rhs)
    d_u[:, 0] = d_rhs[:, 0]
    d_u[:, n_levels] = d_rhs[:, n_levels]
    if n_levels > 1:
        t = rhs[:, 1:n_levels] - a_tri[:, 1:n_levels] * u[:, 0 : n_levels - 1]
        p = -a_tri[:, 1:n_levels] * alpha_tri[:, 1:n_levels]
        q = (
            d_rhs[:, 1:n_levels] - da[:, 1:n_levels] * u[:, 0 : n_levels - 1]
        ) * alpha_tri[:, 1:n_levels] + t * d_alpha[:, 1:n_levels]
        d_u[:, 1:n_levels] = affine_forward_scan(p, q, d_u[:, 0])

    d_out_interior = affine_reverse_scan(
        -gamma_tri[:, :n_levels],
        d_u[:, :n_levels] - d_gamma[:, :n_levels] * out[:, 1 : n_levels + 1],
        d_u[:, n_levels],
    )
    d_out = d_u.clone()
    d_out[:, :n_levels] = d_out_interior
    return d_out


class TridiagonalSolve(torch.autograd.Function):
    """Solve tridiagonal system using acoustic-step sweep ordering.

    Notes:
    - Input `rhs` is expected to have shape `(nCells, nVertLevels + 1)`.
    - The extra last level is the boundary value used during back substitution.
    - Compatible with gradient checkpointing in later phases when
      `torch.utils.checkpoint(..., use_reentrant=False)` is used.

    Second-order extensibility (double backward) options:
    1) Keep this custom Function and make backward-path ops fully differentiable
       (remove blockers and avoid non-traceable backward logic).
       Pro: minimal interface churn. Con: subtle to get complete coverage.
    2) Convert backward implementation to purely functional tensor expressions,
       avoiding in-place updates so autograd can build grad-of-grad graph.
       Pro: straightforward gradgradcheck workflow. Con: potential perf overhead.
    3) Add explicit custom backward Function(s) for this solver path
       (e.g., TridiagonalSolveBackward) with hand-derived formulas.
       Pro: strongest control of stability/performance. Con: most complex to maintain.
    """

    @staticmethod
    def forward(
        ctx: Any,
        a_tri: torch.Tensor,
        alpha_tri: torch.Tensor,
        gamma_tri: torch.Tensor,
        rhs: torch.Tensor,
    ) -> torch.Tensor:
        if a_tri.shape != alpha_tri.shape or a_tri.shape != gamma_tri.shape:
            raise ValueError("a_tri, alpha_tri, gamma_tri must have identical shapes")
        if a_tri.ndim != 2:
            raise ValueError(
                "expected tridiagonal tensors with shape (nCells, nVertLevels)"
            )
        if (
            rhs.ndim != 2
            or rhs.shape[0] != a_tri.shape[0]
            or rhs.shape[1] != a_tri.shape[1] + 1
        ):
            raise ValueError("rhs must have shape (nCells, nVertLevels + 1)")

        use_scan = tridiag_scan_preferred(a_tri)
        if use_scan:
            out = _tridiag_solve_forward_scan(a_tri, alpha_tri, gamma_tri, rhs)
        else:
            out = _tridiag_solve_forward(a_tri, alpha_tri, gamma_tri, rhs)

        # Save tensors needed for explicit reverse-mode through both sweeps.
        ctx.use_scan = use_scan
        ctx.save_for_backward(a_tri, alpha_tri, gamma_tri, rhs, out)
        ctx.save_for_forward(a_tri, alpha_tri, gamma_tri, rhs, out)
        return out

    @staticmethod
    def backward(
        ctx: Any,
        grad_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reverse sweeps for acoustic-step solve:
        1) u_k = (rhs_k - a_k * u_{k-1}) * alpha_k
        2) out_k = u_k - gamma_k * out_{k+1}

        We first backprop through step (2), then through step (1). The
        `grad_gamma` term intentionally uses `out_{k+1}` from the final solution
        (saved as `out`), which is the exact dependency in step (2). Correctness
        is covered by `tests/test_tridiagonal_solve.py` gradcheck tests.
        """
        a_tri, alpha_tri, gamma_tri, rhs, out = ctx.saved_tensors
        _n_cells, n_levels = a_tri.shape
        if ctx.use_scan:
            return _tridiag_solve_backward_scan(
                a_tri,
                alpha_tri,
                gamma_tri,
                rhs,
                out,
                grad_out,
            )

        grad_a = torch.zeros_like(a_tri)
        grad_alpha = torch.zeros_like(alpha_tri)
        grad_gamma = torch.zeros_like(gamma_tri)
        grad_rhs = torch.zeros_like(rhs)

        # Backward through back-substitution:
        # out_k = u_k - gamma_k * out_{k+1}
        adj_out = grad_out.clone()
        adj_u = torch.zeros_like(rhs)
        for k in range(n_levels):
            adj_u[:, k] = adj_u[:, k] + adj_out[:, k]
            grad_gamma[:, k] = grad_gamma[:, k] - adj_out[:, k] * out[:, k + 1]
            adj_out[:, k + 1] = adj_out[:, k + 1] - gamma_tri[:, k] * adj_out[:, k]
        adj_u[:, n_levels] = adj_u[:, n_levels] + adj_out[:, n_levels]

        # Recompute forward-elimination intermediate u.
        u = rhs.clone()
        for k in range(1, n_levels):
            u[:, k] = (u[:, k] - a_tri[:, k] * u[:, k - 1]) * alpha_tri[:, k]

        # Backward through forward-elimination:
        # u_k = (rhs_k - a_k * u_{k-1}) * alpha_k
        for k in range(n_levels - 1, 0, -1):
            t = rhs[:, k] - a_tri[:, k] * u[:, k - 1]
            grad_alpha[:, k] = grad_alpha[:, k] + adj_u[:, k] * t
            adj_t = adj_u[:, k] * alpha_tri[:, k]
            grad_rhs[:, k] = grad_rhs[:, k] + adj_t
            grad_a[:, k] = grad_a[:, k] - adj_t * u[:, k - 1]
            adj_u[:, k - 1] = adj_u[:, k - 1] - adj_t * a_tri[:, k]

        # Boundary equations: u_0 = rhs_0, u_K = rhs_K
        grad_rhs[:, 0] = grad_rhs[:, 0] + adj_u[:, 0]
        grad_rhs[:, n_levels] = grad_rhs[:, n_levels] + adj_u[:, n_levels]

        return grad_a, grad_alpha, grad_gamma, grad_rhs

    @staticmethod
    def jvp(
        ctx: Any,
        grad_a: torch.Tensor | None,
        grad_alpha: torch.Tensor | None,
        grad_gamma: torch.Tensor | None,
        grad_rhs: torch.Tensor | None,
    ) -> torch.Tensor:
        """Forward-mode recurrence through elimination + back-substitution.

        Equations:
          u_k = (rhs_k - a_k * u_{k-1}) * alpha_k
          out_k = u_k - gamma_k * out_{k+1}
          d(u_k) = (d(rhs_k) - d(a_k)*u_{k-1} - a_k*d(u_{k-1})) * alpha_k
                   + (rhs_k - a_k*u_{k-1}) * d(alpha_k)
          d(out_k) = d(u_k) - d(gamma_k)*out_{k+1} - gamma_k*d(out_{k+1})
        """
        a_tri, alpha_tri, gamma_tri, rhs, out = ctx.saved_tensors
        _n_cells, n_levels = a_tri.shape
        if ctx.use_scan:
            return _tridiag_solve_jvp_scan(
                a_tri,
                alpha_tri,
                gamma_tri,
                rhs,
                out,
                grad_a,
                grad_alpha,
                grad_gamma,
                grad_rhs,
            )

        da = torch.zeros_like(a_tri) if grad_a is None else grad_a
        d_alpha = torch.zeros_like(alpha_tri) if grad_alpha is None else grad_alpha
        d_gamma = torch.zeros_like(gamma_tri) if grad_gamma is None else grad_gamma
        d_rhs = torch.zeros_like(rhs) if grad_rhs is None else grad_rhs

        # Recompute forward elimination intermediate `u`.
        u = rhs.clone()
        for k in range(1, n_levels):
            u[:, k] = (u[:, k] - a_tri[:, k] * u[:, k - 1]) * alpha_tri[:, k]

        # Forward-mode through elimination.
        d_u = torch.zeros_like(rhs)
        d_u[:, 0] = d_rhs[:, 0]
        d_u[:, n_levels] = d_rhs[:, n_levels]
        for k in range(1, n_levels):
            t = rhs[:, k] - a_tri[:, k] * u[:, k - 1]
            d_t = d_rhs[:, k] - da[:, k] * u[:, k - 1] - a_tri[:, k] * d_u[:, k - 1]
            d_u[:, k] = d_t * alpha_tri[:, k] + t * d_alpha[:, k]

        # Forward-mode through back-substitution.
        d_out = d_u.clone()
        for k in range(n_levels - 1, -1, -1):
            d_out[:, k] = (
                d_out[:, k]
                - d_gamma[:, k] * out[:, k + 1]
                - gamma_tri[:, k] * d_out[:, k + 1]
            )
        return d_out


def tridiagonal_solve(
    a_tri: torch.Tensor,
    alpha_tri: torch.Tensor,
    gamma_tri: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    """Convenience wrapper around TridiagonalSolve.apply."""
    return cast(torch.Tensor, TridiagonalSolve.apply(a_tri, alpha_tri, gamma_tri, rhs))


def tridiagonal_solve_scan(
    a_tri: torch.Tensor,
    alpha_tri: torch.Tensor,
    gamma_tri: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    """Public scan implementation for focused validation and profiling."""
    return _tridiag_solve_forward_scan(a_tri, alpha_tri, gamma_tri, rhs)


__all__ = ["TridiagonalSolve", "tridiagonal_solve", "tridiagonal_solve_scan"]
