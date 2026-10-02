"""
Pre-allocated workspace buffers for dycore kernels.

The workspace owns scratch tensors that are repeatedly reused across RK stages
to reduce allocator churn and GPU memory fragmentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from mpas_pytorch.constants import DTYPE


@dataclass(frozen=True)
class WorkspaceDims:
    """Canonical dimensions used to pre-shape workspace tensors."""

    nCells: int
    nEdges: int
    nVertices: int
    nVertLevels: int
    maxEdges: int
    maxEdges2: int


class DycoreWorkspace:
    """
    Container for pre-allocated dycore scratch tensors.

    The structure is organized by kernel namespace (`dynamics_tendency`, ...).
    """

    def __init__(
        self,
        dims: WorkspaceDims,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = DTYPE,
    ) -> None:
        self.dims = dims
        self.device = torch.device(device)
        self.dtype = dtype

        nCells = dims.nCells
        nEdges = dims.nEdges
        nVertices = dims.nVertices
        nVertLevels = dims.nVertLevels

        # dynamics-tendency: largest kernel scratch/output buffers.
        self.dynamics_tendency = SimpleNamespace(
            tend_u=torch.zeros(
                (nEdges + 1, nVertLevels), dtype=dtype, device=self.device
            ),
            tend_theta=torch.zeros(
                (nCells + 1, nVertLevels), dtype=dtype, device=self.device
            ),
            tend_w=torch.zeros(
                (nCells + 1, nVertLevels + 1), dtype=dtype, device=self.device
            ),
            h_divergence=torch.zeros(
                (nCells + 1, nVertLevels), dtype=dtype, device=self.device
            ),
            rthdynten=torch.zeros(
                (nCells + 1, nVertLevels), dtype=dtype, device=self.device
            ),
            kdiff=torch.zeros(
                (nCells + 1, nVertLevels), dtype=dtype, device=self.device
            ),
            tend_u_euler=torch.zeros(
                (nEdges + 1, nVertLevels), dtype=dtype, device=self.device
            ),
            tend_w_euler=torch.zeros(
                (nCells + 1, nVertLevels + 1), dtype=dtype, device=self.device
            ),
            tend_theta_euler=torch.zeros(
                (nCells + 1, nVertLevels), dtype=dtype, device=self.device
            ),
            tend_rho=torch.zeros(
                (nCells + 1, nVertLevels), dtype=dtype, device=self.device
            ),
            delsq_u=torch.zeros(
                (nEdges + 1, nVertLevels), dtype=dtype, device=self.device
            ),
            delsq_divergence=torch.zeros(
                (nCells + 1, nVertLevels), dtype=dtype, device=self.device
            ),
            delsq_vorticity=torch.zeros(
                (nVertices + 1, nVertLevels), dtype=dtype, device=self.device
            ),
            delsq_theta=torch.zeros(
                (nCells + 1, nVertLevels), dtype=dtype, device=self.device
            ),
            delsq_w=torch.zeros(
                (nCells + 1, nVertLevels + 1), dtype=dtype, device=self.device
            ),
            dpdz=torch.zeros(
                (nCells + 1, nVertLevels), dtype=dtype, device=self.device
            ),
        )

        # diagnostic-solve: solve-diagnostics outputs and scratch buffers.
        self.diagnostic_solve = SimpleNamespace(
            h_edge=torch.zeros(
                (nEdges + 1, nVertLevels), dtype=dtype, device=self.device
            ),
            divergence=torch.zeros(
                (nCells + 1, nVertLevels), dtype=dtype, device=self.device
            ),
            vorticity=torch.zeros(
                (nVertices + 1, nVertLevels), dtype=dtype, device=self.device
            ),
            ke=torch.zeros((nCells + 1, nVertLevels), dtype=dtype, device=self.device),
            pv_edge=torch.zeros(
                (nEdges + 1, nVertLevels), dtype=dtype, device=self.device
            ),
            v=torch.zeros((nEdges + 1, nVertLevels), dtype=dtype, device=self.device),
            pv_cell=torch.zeros(
                (nCells + 1, nVertLevels), dtype=dtype, device=self.device
            ),
        )

    def to(self, device: torch.device | str) -> DycoreWorkspace:
        """Return a new workspace with all tensor fields moved to ``device``."""
        target_device = torch.device(device)
        new_ws = DycoreWorkspace.__new__(DycoreWorkspace)
        new_ws.dims = self.dims
        new_ws.device = target_device
        new_ws.dtype = self.dtype

        for kernel_name in ("dynamics_tendency", "diagnostic_solve"):
            src_ns = getattr(self, kernel_name)
            dst_ns = SimpleNamespace()
            for field, value in vars(src_ns).items():
                if isinstance(value, torch.Tensor):
                    setattr(dst_ns, field, value.to(target_device, copy=True))
                else:
                    setattr(dst_ns, field, value)
            setattr(new_ws, kernel_name, dst_ns)

        return new_ws

    def bytes_by_kernel(self) -> dict[str, int]:
        """Return workspace memory usage in bytes by kernel namespace."""
        usage: dict[str, int] = {}
        for kernel_name in ("dynamics_tendency", "diagnostic_solve"):
            kernel = getattr(self, kernel_name, None)
            total = 0
            if kernel is not None:
                for value in vars(kernel).values():
                    if isinstance(value, torch.Tensor):
                        total += value.numel() * value.element_size()
            usage[kernel_name] = total
        return usage

    def total_bytes(self) -> int:
        """Total bytes allocated across all workspace kernels."""
        return sum(self.bytes_by_kernel().values())

    def memory_report(self) -> dict[str, float]:
        """
        Memory usage report in MiB.

        Returns:
            Dict with `total_mib` and per-kernel `*_mib` entries.
        """
        by_kernel = self.bytes_by_kernel()
        report: dict[str, float] = {
            "total_mib": float(self.total_bytes()) / (1024.0 * 1024.0),
        }
        for name, bytes_used in by_kernel.items():
            report[f"{name}_mib"] = float(bytes_used) / (1024.0 * 1024.0)
        return report


def workspace_dims_from_state(state_dict: dict) -> WorkspaceDims:
    """Build workspace dimensions from runtime state dictionary."""

    def _as_int(v: object) -> int:
        if isinstance(v, torch.Tensor):
            return int(v.item())
        if isinstance(v, (str, bytes, bytearray, int, float)):
            return int(v)
        raise TypeError(f"workspace dimension is not integer-like: {type(v).__name__}")

    return WorkspaceDims(
        nCells=_as_int(state_dict["nCells"]),
        nEdges=_as_int(state_dict["nEdges"]),
        nVertices=_as_int(state_dict["nVertices"]),
        nVertLevels=_as_int(state_dict["nVertLevels"]),
        maxEdges=_as_int(state_dict.get("maxEdges", 10)),
        maxEdges2=_as_int(state_dict.get("maxEdges2", 20)),
    )
