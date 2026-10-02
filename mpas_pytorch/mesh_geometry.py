"""Mesh-geometry synthesis for cold-boot initialization.

This module ports the geometry pieces of the MPAS vector bootstrap:
- ``mpas_initialize_vectors`` for local verticals, edge normals, and tangent planes
- ``mpas_init_reconstruct`` / RBF reconstruction coefficient setup

The functions operate on active, unpadded slices and return active tensors.
Loader code is responsible for append-padding them back to the runtime layout.
"""

from __future__ import annotations

import torch

from mpas_pytorch.constants import DTYPE


def _unit(vec: torch.Tensor) -> torch.Tensor:
    # Match mpas_unit_vec_in_r3 exactly.  This setup-time path feeds the
    # reconstruction coefficients, so a generalized reduction here can move the
    # T0 vector fields even when the reconstruction stencil itself is bitwise.
    norm = torch.sqrt(
        vec[..., 0:1].square() + vec[..., 1:2].square() + vec[..., 2:3].square()
    )
    return vec / norm


def _dot_r3_fortran_order(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Three-component dot product with MPAS's source association."""
    return (
        left[..., 0] * right[..., 0]
        + left[..., 1] * right[..., 1]
        + left[..., 2] * right[..., 2]
    )


def _cross_r3(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Cross product written in the same component order as MPAS Fortran."""
    return torch.stack(
        (
            left[..., 1] * right[..., 2] - left[..., 2] * right[..., 1],
            left[..., 2] * right[..., 0] - left[..., 0] * right[..., 2],
            left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0],
        ),
        dim=-1,
    )


def _solve_mpas_legs_batched(matrix: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Batched MPAS ``legs`` solve with identical scaled-pivot operation order.

    MPAS solves each tiny reconstruction system with its own Gaussian-elimination
    routine rather than LAPACK.  Vectorizing only across independent cells keeps
    the original elimination and back-substitution order inside every system.
    """
    if matrix.ndim != 3 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError(
            f"matrix must have shape (batch, n, n), got {tuple(matrix.shape)}"
        )
    if rhs.shape != matrix.shape[:2]:
        raise ValueError(
            f"rhs must have shape {tuple(matrix.shape[:2])}, got {tuple(rhs.shape)}"
        )

    a = matrix.clone()
    b = rhs.clone()
    batch_size, n, _ = a.shape
    batch = torch.arange(batch_size, dtype=torch.int64, device=a.device)
    indices = (
        torch.arange(n, dtype=torch.int64, device=a.device)
        .expand(batch_size, n)
        .clone()
    )
    row_scale = torch.amax(torch.abs(a), dim=2)

    for column in range(n - 1):
        candidate_rows = indices[:, column:]
        ratios = torch.abs(
            a[batch[:, None], candidate_rows, column]
        ) / row_scale.gather(1, candidate_rows)
        pivot_position = column + torch.argmax(ratios, dim=1)
        old_column_row = indices[:, column].clone()
        pivot_row = indices[batch, pivot_position].clone()
        indices[:, column] = pivot_row
        indices[batch, pivot_position] = old_column_row

        for position in range(column + 1, n):
            row = indices[:, position]
            pivot = indices[:, column]
            multiplier = a[batch, row, column] / a[batch, pivot, column]
            a[batch, row, column] = multiplier
            for trailing_column in range(column + 1, n):
                a[batch, row, trailing_column] = (
                    a[batch, row, trailing_column]
                    - multiplier * a[batch, pivot, trailing_column]
                )

    for column in range(n - 1):
        pivot = indices[:, column]
        pivot_rhs = b[batch, pivot]
        for position in range(column + 1, n):
            row = indices[:, position]
            b[batch, row] = b[batch, row] - a[batch, row, column] * pivot_rhs

    solution = torch.empty_like(b)
    last_row = indices[:, n - 1]
    solution[:, n - 1] = b[batch, last_row] / a[batch, last_row, n - 1]
    for row_number in range(n - 2, -1, -1):
        row = indices[:, row_number]
        value = b[batch, row]
        for column in range(row_number + 1, n):
            value = value - a[batch, row, column] * solution[:, column]
        solution[:, row_number] = value / a[batch, row, row_number]
    return solution


def compute_local_vertical_unit_vectors(
    n_cells: int,
    x_cell: torch.Tensor,
    y_cell: torch.Tensor,
    z_cell: torch.Tensor,
    *,
    on_a_sphere: bool = True,
) -> torch.Tensor:
    """Compute local vertical unit vectors at cell centers."""
    cell_xyz = torch.stack(
        [x_cell[:n_cells], y_cell[:n_cells], z_cell[:n_cells]],
        dim=-1,
    ).to(dtype=DTYPE)
    if on_a_sphere:
        return _unit(cell_xyz)

    out = torch.zeros_like(cell_xyz)
    out[:, 2] = 1.0
    return out


def compute_edge_normal_vectors(
    n_cells: int,
    n_edges: int,
    cells_on_edge: torch.Tensor,
    x_cell: torch.Tensor,
    y_cell: torch.Tensor,
    z_cell: torch.Tensor,
    x_edge: torch.Tensor,
    y_edge: torch.Tensor,
    z_edge: torch.Tensor,
    *,
    on_a_sphere: bool = True,
) -> torch.Tensor:
    """Compute unit edge-normal vectors that point from cell1 toward cell2."""
    del on_a_sphere  # Only the spherical/non-periodic path is implemented here.

    coe = cells_on_edge[:n_edges, :2].to(dtype=torch.int64)
    cell1 = coe[:, 0]
    cell2 = coe[:, 1]

    cell1_boundary = cell1 == (n_cells + 1)
    cell2_boundary = cell2 == (n_cells + 1)

    safe_cell1 = torch.where(cell1_boundary, torch.ones_like(cell1), cell1) - 1
    safe_cell2 = torch.where(cell2_boundary, torch.ones_like(cell2), cell2) - 1

    cell1_xyz = torch.stack(
        [x_cell[safe_cell1], y_cell[safe_cell1], z_cell[safe_cell1]],
        dim=-1,
    ).to(dtype=DTYPE)
    cell2_xyz = torch.stack(
        [x_cell[safe_cell2], y_cell[safe_cell2], z_cell[safe_cell2]],
        dim=-1,
    ).to(dtype=DTYPE)
    edge_xyz = torch.stack(
        [x_edge[:n_edges], y_edge[:n_edges], z_edge[:n_edges]],
        dim=-1,
    ).to(dtype=DTYPE)

    normals = torch.where(
        cell1_boundary.unsqueeze(-1),
        cell2_xyz - edge_xyz,
        torch.where(
            cell2_boundary.unsqueeze(-1),
            edge_xyz - cell1_xyz,
            cell2_xyz - cell1_xyz,
        ),
    )
    return _unit(normals)


def compute_cell_tangent_plane(
    n_cells: int,
    edges_on_cell: torch.Tensor,
    x_cell: torch.Tensor,
    y_cell: torch.Tensor,
    z_cell: torch.Tensor,
    edge_normal_vectors: torch.Tensor,
    *,
    on_a_sphere: bool = True,
) -> torch.Tensor:
    """Compute the 2-vector tangent-plane basis at each cell center."""
    r_hat = compute_local_vertical_unit_vectors(
        n_cells,
        x_cell,
        y_cell,
        z_cell,
        on_a_sphere=on_a_sphere,
    )
    first_edge = edges_on_cell[:n_cells, 0].to(dtype=torch.int64) - 1
    edge_normal = edge_normal_vectors[first_edge]

    normal_dot_r = _dot_r3_fortran_order(edge_normal, r_hat).unsqueeze(-1)
    x_hat = _unit(edge_normal - normal_dot_r * r_hat)
    y_hat = _unit(_cross_r3(r_hat, x_hat))
    return torch.stack([x_hat, y_hat], dim=1)


def compute_coeffs_reconstruct(
    n_cells: int,
    n_edges: int,
    max_edges: int,
    edges_on_cell: torch.Tensor,
    n_edges_on_cell: torch.Tensor,
    x_cell: torch.Tensor,
    y_cell: torch.Tensor,
    z_cell: torch.Tensor,
    x_edge: torch.Tensor,
    y_edge: torch.Tensor,
    z_edge: torch.Tensor,
    edge_normal_vectors: torch.Tensor,
    cell_tangent_plane: torch.Tensor,
) -> torch.Tensor:
    """Compute RBF vector-reconstruction coefficients from mesh geometry."""
    device = x_cell.device
    coeffs = torch.zeros((n_cells, max_edges, 3), dtype=DTYPE, device=device)

    counts = n_edges_on_cell[:n_cells].to(dtype=torch.int64)
    unique_counts = torch.unique(counts)

    x_cell = x_cell.to(dtype=DTYPE)
    y_cell = y_cell.to(dtype=DTYPE)
    z_cell = z_cell.to(dtype=DTYPE)
    x_edge = x_edge.to(dtype=DTYPE)
    y_edge = y_edge.to(dtype=DTYPE)
    z_edge = z_edge.to(dtype=DTYPE)

    for point_count_tensor in unique_counts:
        point_count = int(point_count_tensor.item())
        if point_count <= 0:
            continue

        cell_mask = counts == point_count
        cell_ids = torch.nonzero(cell_mask, as_tuple=False).squeeze(-1)
        if cell_ids.numel() == 0:
            continue

        edge_ids = edges_on_cell[cell_ids, :point_count].to(dtype=torch.int64) - 1
        cell_centers = torch.stack(
            [x_cell[cell_ids], y_cell[cell_ids], z_cell[cell_ids]],
            dim=-1,
        )
        edge_locations = torch.stack(
            [x_edge[edge_ids], y_edge[edge_ids], z_edge[edge_ids]],
            dim=-1,
        )
        edge_normals = edge_normal_vectors[edge_ids]

        displacement = cell_centers[:, None, :] - edge_locations
        radii = torch.sqrt(
            displacement[..., 0].square()
            + displacement[..., 1].square()
            + displacement[..., 2].square()
        )
        alpha = torch.zeros(cell_ids.numel(), dtype=DTYPE, device=device)
        for slot in range(point_count):
            alpha = alpha + radii[:, slot]
        alpha = alpha / point_count
        plane_basis = cell_tangent_plane[cell_ids]

        planar_sources = torch.stack(
            tuple(
                _dot_r3_fortran_order(edge_locations, plane_basis[:, axis, None, :])
                for axis in range(2)
            ),
            dim=-1,
        )
        planar_units = torch.stack(
            tuple(
                _dot_r3_fortran_order(edge_normals, plane_basis[:, axis, None, :])
                for axis in range(2)
            ),
            dim=-1,
        )
        planar_dest = torch.stack(
            tuple(
                _dot_r3_fortran_order(cell_centers, plane_basis[:, axis, :])
                for axis in range(2)
            ),
            dim=-1,
        )

        matrix_size = point_count + 2
        matrix = torch.zeros(
            (cell_ids.numel(), matrix_size, matrix_size),
            dtype=DTYPE,
            device=device,
        )
        rhs = torch.zeros(
            (cell_ids.numel(), matrix_size, 2),
            dtype=DTYPE,
            device=device,
        )

        diff_ij = planar_sources[:, :, None, :] - planar_sources[:, None, :, :]
        r_squared = (diff_ij[..., 0].square() + diff_ij[..., 1].square()) / alpha[
            :, None, None
        ].square()
        rbf = 1.0 / torch.sqrt(1.0 + r_squared)
        uv_left = planar_units[:, :, None, :]
        uv_right = planar_units[:, None, :, :]
        uv_dot = uv_left[..., 0] * uv_right[..., 0] + uv_left[..., 1] * uv_right[..., 1]
        matrix[:, :point_count, :point_count] = rbf * uv_dot

        matrix[:, :point_count, point_count : point_count + 2] = planar_units
        matrix[:, point_count : point_count + 2, :point_count] = planar_units.transpose(
            1, 2
        )

        diff_dest = planar_dest[:, None, :] - planar_sources
        dest_r_squared = (
            diff_dest[..., 0].square() + diff_dest[..., 1].square()
        ) / alpha[:, None].square()
        dest_rbf = 1.0 / torch.sqrt(1.0 + dest_r_squared)
        rhs[:, :point_count, :] = dest_rbf.unsqueeze(-1) * planar_units
        rhs[:, point_count, 0] = 1.0
        rhs[:, point_count + 1, 1] = 1.0

        coeffs_plane_1 = _solve_mpas_legs_batched(matrix, rhs[:, :, 0])
        coeffs_plane_2 = _solve_mpas_legs_batched(matrix, rhs[:, :, 1])
        coeffs[cell_ids, :point_count, :] = (
            plane_basis[:, 0, None, :] * coeffs_plane_1[:, :point_count, None]
            + plane_basis[:, 1, None, :] * coeffs_plane_2[:, :point_count, None]
        )

    return coeffs
