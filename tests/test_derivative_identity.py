from __future__ import annotations

import torch
import torch.autograd.forward_ad as fw_ad
from mpas_pytorch.autodiff.tridiagonal_solve import TridiagonalSolve


def _identity_residual(
    jvp: torch.Tensor,
    weight: torch.Tensor,
    direction: torch.Tensor,
    vjp: torch.Tensor,
) -> tuple[float, float]:
    left_terms = (jvp.reshape(-1) * weight.reshape(-1)).contiguous()
    right_terms = (direction.reshape(-1) * vjp.reshape(-1)).contiguous()
    lhs = torch.sum(left_terms, dtype=torch.float64)
    rhs = torch.sum(right_terms, dtype=torch.float64)
    denominator = max(
        float(
            (
                torch.sum(torch.abs(left_terms), dtype=torch.float64)
                + torch.sum(torch.abs(right_terms), dtype=torch.float64)
            ).item()
        ),
        torch.finfo(torch.float64).tiny,
    )
    n = max(left_terms.numel(), right_terms.numel())
    eps = torch.finfo(torch.float64).eps
    gamma = n * eps / (1.0 - n * eps)
    return abs(float((lhs - rhs).item())) / denominator, 8.0 * gamma


def test_production_kernel_jvp_vjp_identity_and_failing_control() -> None:
    a = torch.tensor([[0.0, 0.08, -0.05, 0.03]], dtype=torch.float64)
    alpha = torch.tensor([[1.0, 0.95, 1.03, 0.98]], dtype=torch.float64)
    gamma = torch.tensor([[0.04, -0.02, 0.06, 0.01]], dtype=torch.float64)
    primal = torch.tensor([[1.0, -0.5, 0.75, 0.2, -0.1]], dtype=torch.float64)
    direction = torch.randn(
        primal.shape,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(7301),
    )
    weight = torch.randn(
        primal.shape,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(9109),
    )

    with fw_ad.dual_level():
        dual_rhs = fw_ad.make_dual(primal, direction)
        dual_output = TridiagonalSolve.apply(a, alpha, gamma, dual_rhs)
        _, jvp = fw_ad.unpack_dual(dual_output)
    assert jvp is not None

    reverse_rhs = primal.clone().requires_grad_(True)
    reverse_output = TridiagonalSolve.apply(a, alpha, gamma, reverse_rhs)
    vjp = torch.autograd.grad(
        torch.sum(reverse_output * weight, dtype=torch.float64), reverse_rhs
    )[0]

    residual, bound = _identity_residual(jvp, weight, direction, vjp)
    assert residual <= bound

    control_residual, _ = _identity_residual(
        jvp, weight, direction, torch.zeros_like(vjp)
    )
    assert control_residual > 1.0e-6
