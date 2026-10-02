"""Run a self-contained differentiable production-kernel smoke test."""

from __future__ import annotations

import torch
from mpas_pytorch.autodiff.tridiagonal_solve import tridiagonal_solve


def main() -> None:
    torch.set_default_dtype(torch.float64)
    a = torch.tensor([[0.0, 0.08, -0.05, 0.03]])
    alpha = torch.tensor([[1.0, 0.95, 1.03, 0.98]])
    gamma = torch.tensor([[0.04, -0.02, 0.06, 0.01]])
    rhs = torch.tensor([[1.0, -0.5, 0.75, 0.2, -0.1]], requires_grad=True)
    solution = tridiagonal_solve(a, alpha, gamma, rhs)
    gradient = torch.autograd.grad(solution.square().sum(), rhs)[0]
    assert torch.isfinite(solution).all()
    assert torch.isfinite(gradient).all()
    print({"solution_shape": list(solution.shape), "gradient_finite": True})


if __name__ == "__main__":
    main()
