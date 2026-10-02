"""
Vertical implicit coefficients (vertical-implicit-coefficients).

Computes implicit vertical stiffness coefficients for the acoustic solver.
This is called before the acoustic timestep loop to set up the tridiagonal
system for the implicit vertical solve.


Subroutine: atm_compute_vert_imp_coefs_work
"""

import torch

from mpas_pytorch.autodiff.tridiagonal_factor import (
    tridiagonal_factorize,
    tridiagonal_factorize_scan,
)
from mpas_pytorch.autodiff.tridiagonal_scan import tridiag_scan_preferred
from mpas_pytorch.constants import DTYPE
from mpas_pytorch.dynamics.stencil_ops import _replace_row_slice
from mpas_pytorch.utils.checks import require_consistent_device

PUBLIC_VERT_IMP_COEF_KEYS = (
    "cofrz",
    "cofwr",
    "cofwz",
    "coftz",
    "cofwt",
    "a_tri",
    "alpha_tri",
    "gamma_tri",
)
ACTIVE_VERT_IMP_COEF_KEYS = (
    "_vertical_implicit_coefficients_cofwr_active",
    "_vertical_implicit_coefficients_cofwz_active",
    "_vertical_implicit_coefficients_coftz_active",
    "_vertical_implicit_coefficients_cofwt_active",
    "_vertical_implicit_coefficients_a_tri_active",
    "_vertical_implicit_coefficients_alpha_tri_active",
    "_vertical_implicit_coefficients_gamma_tri_active",
)
_ACTIVE_TO_PUBLIC = {
    "_vertical_implicit_coefficients_cofwr_active": "cofwr",
    "_vertical_implicit_coefficients_cofwz_active": "cofwz",
    "_vertical_implicit_coefficients_coftz_active": "coftz",
    "_vertical_implicit_coefficients_cofwt_active": "cofwt",
    "_vertical_implicit_coefficients_a_tri_active": "a_tri",
    "_vertical_implicit_coefficients_alpha_tri_active": "alpha_tri",
    "_vertical_implicit_coefficients_gamma_tri_active": "gamma_tri",
}


def _pack_active_rows(
    rows: torch.Tensor,
    *,
    n_cells: int,
    n_cols: int,
    cell_start: int,
    cell_end: int,
    device: torch.device,
) -> torch.Tensor:
    full = torch.zeros(n_cells + 1, n_cols, dtype=rows.dtype, device=device)
    if cell_end > cell_start:
        full = _replace_row_slice(full, cell_start, cell_end, rows)
    return full


def _leading_zero_columns(
    tail: torch.Tensor,
    *,
    rows: int,
    n_cols: int,
    fill_first: float = 0.0,
) -> torch.Tensor:
    if n_cols <= 0:
        return tail.new_zeros((rows, 0))
    first = tail.new_full((rows, 1), fill_first)
    if n_cols == 1:
        return first
    return torch.cat((first, tail), dim=1)


def _boundary_zero_columns(
    interior: torch.Tensor,
    *,
    rows: int,
    n_cols: int,
) -> torch.Tensor:
    if n_cols <= 0:
        return interior.new_zeros((rows, 0))
    if n_cols == 1:
        return interior.new_zeros((rows, 1))
    left = interior.new_zeros((rows, 1))
    right = interior.new_zeros((rows, 1))
    if n_cols == 2:
        return torch.cat((left, right), dim=1)
    return torch.cat((left, interior, right), dim=1)


def _tridiag_factor_active_thomas(
    a_tri: torch.Tensor,
    b_tri: torch.Tensor,
    c_tri: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Functional Thomas-order factorization for vertical-implicit-coefficients compact coefficients."""
    if a_tri.shape != b_tri.shape or a_tri.shape != c_tri.shape:
        raise ValueError("a_tri, b_tri, c_tri must have identical shapes")
    if a_tri.ndim != 2:
        raise ValueError("expected shape (nCells, nVertLevels)")
    n_cells, n_levels = a_tri.shape
    if n_levels == 0:
        empty = a_tri.new_zeros((n_cells, 0))
        return empty, empty
    alpha_cols = [a_tri.new_zeros((n_cells,))]
    gamma_cols = [a_tri.new_zeros((n_cells,))]
    for k in range(1, n_levels):
        alpha_k = 1.0 / (b_tri[:, k] - a_tri[:, k] * gamma_cols[-1])
        gamma_k = c_tri[:, k] * alpha_k
        alpha_cols.append(alpha_k)
        gamma_cols.append(gamma_k)
    return torch.stack(alpha_cols, dim=1), torch.stack(gamma_cols, dim=1)


def _tridiag_factor_active(
    a_tri: torch.Tensor,
    b_tri: torch.Tensor,
    c_tri: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Factor vertical-implicit-coefficients active coefficients with Thomas/reference or scan policy."""
    if tridiag_scan_preferred(a_tri):
        if torch.is_grad_enabled():
            return tridiagonal_factorize(a_tri, b_tri, c_tri)
        return tridiagonal_factorize_scan(a_tri, b_tri, c_tri)
    return _tridiag_factor_active_thomas(a_tri, b_tri, c_tri)


def pack_vert_imp_coefs_public(
    coefs: dict,
    *,
    nCells: int,
    nVertLevels: int,
    device: torch.device | None = None,
) -> dict:
    """Pack vertical-implicit-coefficients compact active coefficient patches into public padded fields."""
    missing = [
        key
        for key in (
            "cofrz",
            "_vertical_implicit_coefficients_active_cell_start",
            "_vertical_implicit_coefficients_active_cell_end",
            *ACTIVE_VERT_IMP_COEF_KEYS,
        )
        if key not in coefs
    ]
    if missing:
        if all(key in coefs for key in PUBLIC_VERT_IMP_COEF_KEYS):
            return {key: coefs[key] for key in PUBLIC_VERT_IMP_COEF_KEYS}
        raise KeyError(
            f"Cannot pack vertical-implicit-coefficients public coefficients; missing {missing}"
        )

    cell_start = int(coefs["_vertical_implicit_coefficients_active_cell_start"])
    cell_end = int(coefs["_vertical_implicit_coefficients_active_cell_end"])
    if device is None:
        device = coefs["_vertical_implicit_coefficients_cofwr_active"].device

    packed = {"cofrz": coefs["cofrz"]}
    for active_key, public_key in _ACTIVE_TO_PUBLIC.items():
        rows = coefs[active_key]
        expected_rows = max(cell_end - cell_start, 0)
        if rows.shape[0] != expected_rows:
            raise ValueError(
                f"{active_key} row count mismatch: got {rows.shape[0]}, "
                f"expected {expected_rows}"
            )
        n_cols = nVertLevels + 1 if public_key == "coftz" else nVertLevels
        packed[public_key] = _pack_active_rows(
            rows,
            n_cells=nCells,
            n_cols=n_cols,
            cell_start=cell_start,
            cell_end=cell_end,
            device=device,
        )
    return packed


def atm_compute_vert_imp_coefs_work(
    nCells: int,
    nVertLevels: int,
    moist_start: int,
    moist_end: int,
    dts: float,
    epssm: float,
    cellSolveStart: int,
    cellSolveEnd: int,
    zz: torch.Tensor,  # (nCells+1, nVertLevels)
    cqw: torch.Tensor,  # (nCells+1, nVertLevels)
    p: torch.Tensor,  # (nCells+1, nVertLevels)
    t: torch.Tensor,  # (nCells+1, nVertLevels)
    rb: torch.Tensor,  # (nCells+1, nVertLevels)
    rtb: torch.Tensor,  # (nCells+1, nVertLevels)
    pb: torch.Tensor,  # (nCells+1, nVertLevels)
    rt: torch.Tensor,  # (nCells+1, nVertLevels)
    rdzw: torch.Tensor,  # (nVertLevels,)
    fzm: torch.Tensor,  # (nVertLevels,)
    fzp: torch.Tensor,  # (nVertLevels,)
    rdzu: torch.Tensor,  # (nVertLevels,)
    qtot: torch.Tensor,  # (nCells+1, nVertLevels)
    gravity: float,
    rgas: float,
    cp: float,
    pack_public: bool = True,
) -> dict:
    """
    Compute vertical implicit coefficients for acoustic solver.

    This routine sets up the tridiagonal coefficient matrices used in the
    implicit vertical solve of the acoustic timestep. The tridiagonal system
    couples vertical velocity (w) with thermodynamic variables.

    Index Convention:
        Fortran uses 1-based indexing: k=1..nVertLevels, iCell=1..nCells
        PyTorch uses 0-based indexing: k=0..nVertLevels-1, iCell=0..nCells-1

        Fortran loop `do k = 2, nVertLevels` maps to Python `for k in range(1, nVertLevels)`
        Fortran `cellSolveStart=1, cellSolveEnd=nCells` maps to Python `range(0, nCells)`

    Args:
        nCells: Number of cells in the mesh
        nVertLevels: Number of vertical levels
        moist_start: Start index for moisture scalars (unused, moisture is in qtot)
        moist_end: End index for moisture scalars (unused)
        dts: Small (acoustic) timestep in seconds
        epssm: Off-centering parameter for implicit solve (typically 0.1)
        cellSolveStart: First cell to process (1-based Fortran index)
        cellSolveEnd: Last cell to process (1-based Fortran index)
        zz: Jacobian of coordinate transformation (nCells+1, nVertLevels)
        cqw: Moisture coefficient for w equation (nCells+1, nVertLevels)
        p: Pressure perturbation (nCells+1, nVertLevels)
        t: Temperature (nCells+1, nVertLevels)
        rb: Base state density (nCells+1, nVertLevels)
        rtb: Base state rho*theta (nCells+1, nVertLevels)
        pb: Base state pressure (nCells+1, nVertLevels)
        rt: Perturbation rho*theta (nCells+1, nVertLevels)
        rdzw: Inverse dz at w-levels (nVertLevels,)
        fzm: Interpolation weight for k level (nVertLevels,)
        fzp: Interpolation weight for k-1 level (nVertLevels,)
        rdzu: Inverse dz at u-levels (nVertLevels,)
        qtot: Total moisture mixing ratio (nCells+1, nVertLevels)
        gravity: Gravitational acceleration (m/s^2)
        rgas: Gas constant for dry air (J/(kg*K))
        cp: Specific heat at constant pressure (J/(kg*K))

    Returns:
        Dictionary containing:
            cofrz: Coefficient for rho-z coupling (nVertLevels,)
            cofwr: w-rho coefficient (nCells+1, nVertLevels)
            cofwz: w-z coefficient (nCells+1, nVertLevels)
            coftz: theta-z coefficient (nCells+1, nVertLevels+1)
            cofwt: w-theta coefficient (nCells+1, nVertLevels)
            a_tri: Tridiagonal lower diagonal (nCells+1, nVertLevels)
            alpha_tri: Tridiagonal factorization coefficient (nCells+1, nVertLevels)
            gamma_tri: Tridiagonal factorization coefficient (nCells+1, nVertLevels)

    Note:
        Default execution keeps the Thomas k-ordering exactly. PyTorch-native
        scan recurrences remain opt-in until the drift-proof/budget chain
        authorizes them for production; the Thomas path is the reference
        comparison path.
    """
    device = require_consistent_device(
        "atm_compute_vert_imp_coefs_work",
        zz,
        cqw,
        p,
        t,
        rb,
        rtb,
        pb,
        rt,
        rdzw,
        fzm,
        fzp,
        rdzu,
        qtot,
    )
    # =========================================================================
    # Step 1: Compute scalar coefficients

    # =========================================================================
    dtseps = 0.5 * dts * (1.0 + epssm)
    rcv = rgas / (cp - rgas)
    c2 = cp * rcv
    # Some orchestration paths carry nVertLevels+1 metric vectors. vertical-implicit-coefficients uses
    # Fortran levels k=1..nVertLevels, so trim to this working range.
    rdzw_k = rdzw[:nVertLevels]
    fzm_k = fzm[:nVertLevels]
    fzp_k = fzp[:nVertLevels]
    rdzu_k = rdzu[:nVertLevels]

    # =========================================================================
    # Step 2: Compute cofrz (vertical level only)
    # do k=1,nVertLevels
    # This is a simple 1D coefficient that doesn't depend on cell
    # =========================================================================
    cofrz = dtseps * rdzw_k  # Shape: (nVertLevels,)

    # =========================================================================
    # Step 3: Convert Fortran 1-based indices to Python 0-based
    # Upstream bounds: cellSolveStart=1, cellSolveEnd=nCells
    # Python: iCell in range(0, nCells)
    # =========================================================================
    cell_start = cellSolveStart - 1  # 0
    cell_end = cellSolveEnd  # nCells (exclusive upper bound)
    n_solve = max(cell_end - cell_start, 0)

    # =========================================================================
    # Step 4: Cell-parallel coefficient assembly and batched tridiagonal factorization

    # Default execution keeps k-ordering exact. Scan recurrences are opt-in until
    # the drift-proof/budget chain authorizes them for production.
    # =========================================================================
    cofwr_s = torch.zeros(n_solve, nVertLevels, dtype=DTYPE, device=device)
    cofwz_s = torch.zeros(n_solve, nVertLevels, dtype=DTYPE, device=device)
    coftz_s = torch.zeros(n_solve, nVertLevels + 1, dtype=DTYPE, device=device)
    cofwt_s = torch.zeros(n_solve, nVertLevels, dtype=DTYPE, device=device)
    a_tri_s = torch.zeros(n_solve, nVertLevels, dtype=DTYPE, device=device)
    alpha_tri_s = torch.zeros(n_solve, nVertLevels, dtype=DTYPE, device=device)
    gamma_tri_s = torch.zeros(n_solve, nVertLevels, dtype=DTYPE, device=device)

    if n_solve > 0:
        solve_slice = slice(cell_start, cell_end)
        zz_s = zz[solve_slice, :nVertLevels]
        cqw_s = cqw[solve_slice, :nVertLevels]
        p_s = p[solve_slice, :nVertLevels]
        t_s = t[solve_slice, :nVertLevels]
        rb_s = rb[solve_slice, :nVertLevels]
        rtb_s = rtb[solve_slice, :nVertLevels]
        pb_s = pb[solve_slice, :nVertLevels]
        rt_s = rt[solve_slice, :nVertLevels]
        qtot_s = qtot[solve_slice, :nVertLevels]

        cofwr_interior = (
            0.5
            * dtseps
            * gravity
            * (fzm_k[1:] * zz_s[:, 1:] + fzp_k[1:] * zz_s[:, :-1])
        )
        cofwr_s = _leading_zero_columns(
            cofwr_interior,
            rows=n_solve,
            n_cols=nVertLevels,
        )

        z_interp = fzm_k[1:] * zz_s[:, 1:] + fzp_k[1:] * zz_s[:, :-1]
        p_interp = fzm_k[1:] * p_s[:, 1:] + fzp_k[1:] * p_s[:, :-1]
        t_interp = fzm_k[1:] * t_s[:, 1:] + fzp_k[1:] * t_s[:, :-1]
        cofwz_interior = (
            dtseps * c2 * z_interp * rdzu_k[1:nVertLevels] * cqw_s[:, 1:] * p_interp
        )
        cofwz_s = _leading_zero_columns(
            cofwz_interior,
            rows=n_solve,
            n_cols=nVertLevels,
        )
        coftz_s = _boundary_zero_columns(
            dtseps * t_interp,
            rows=n_solve,
            n_cols=nVertLevels + 1,
        )

        cofwt_s = (
            0.5
            * dtseps
            * rcv
            * zz_s
            * gravity
            * rb_s
            / (1.0 + qtot_s)
            * p_s
            / ((rtb_s + rt_s) * pb_s)
        )

        b_tri_s = torch.ones(n_solve, nVertLevels, dtype=DTYPE, device=device)
        c_tri_s = torch.zeros(n_solve, nVertLevels, dtype=DTYPE, device=device)

        if nVertLevels > 1:
            # keep explicit indexed expression text for guardrail traceability:
            # a_tri_s[:, 1:nVertLevels], b_tri_s[:, 1:nVertLevels], c_tri_s[:, 1:nVertLevels]
            a_rest = (
                -cofwz_s[:, 1:nVertLevels]
                * coftz_s[:, 0 : nVertLevels - 1]
                * rdzw_k[0 : nVertLevels - 1]
                * zz_s[:, 0 : nVertLevels - 1]
                + cofwr_s[:, 1:nVertLevels] * cofrz[0 : nVertLevels - 1]
                - cofwt_s[:, 0 : nVertLevels - 1]
                * coftz_s[:, 0 : nVertLevels - 1]
                * rdzw_k[0 : nVertLevels - 1]
            )
            b_rest = (
                1.0
                + cofwz_s[:, 1:nVertLevels]
                * (
                    coftz_s[:, 1:nVertLevels]
                    * rdzw_k[1:nVertLevels]
                    * zz_s[:, 1:nVertLevels]
                    + coftz_s[:, 1:nVertLevels]
                    * rdzw_k[0 : nVertLevels - 1]
                    * zz_s[:, 0 : nVertLevels - 1]
                )
                - coftz_s[:, 1:nVertLevels]
                * (
                    cofwt_s[:, 1:nVertLevels] * rdzw_k[1:nVertLevels]
                    - cofwt_s[:, 0 : nVertLevels - 1] * rdzw_k[0 : nVertLevels - 1]
                )
                + cofwr_s[:, 1:nVertLevels]
                * (cofrz[1:nVertLevels] - cofrz[0 : nVertLevels - 1])
            )
            c_rest = (
                -cofwz_s[:, 1:nVertLevels]
                * coftz_s[:, 2 : nVertLevels + 1]
                * rdzw_k[1:nVertLevels]
                * zz_s[:, 1:nVertLevels]
                - cofwr_s[:, 1:nVertLevels] * cofrz[1:nVertLevels]
                + cofwt_s[:, 1:nVertLevels]
                * coftz_s[:, 2 : nVertLevels + 1]
                * rdzw_k[1:nVertLevels]
            )
            a_tri_s = _leading_zero_columns(
                a_rest,
                rows=n_solve,
                n_cols=nVertLevels,
            )
            b_tri_s = _leading_zero_columns(
                b_rest,
                rows=n_solve,
                n_cols=nVertLevels,
                fill_first=1.0,
            )
            c_tri_s = _leading_zero_columns(
                c_rest,
                rows=n_solve,
                n_cols=nVertLevels,
            )

        alpha_tri_s, gamma_tri_s = _tridiag_factor_active(
            a_tri_s,
            b_tri_s,
            c_tri_s,
        )

    compact_result = {
        "cofrz": cofrz,
        "_vertical_implicit_coefficients_active_cell_start": cell_start,
        "_vertical_implicit_coefficients_active_cell_end": cell_end,
        "_vertical_implicit_coefficients_cofwr_active": cofwr_s,
        "_vertical_implicit_coefficients_cofwz_active": cofwz_s,
        "_vertical_implicit_coefficients_coftz_active": coftz_s,
        "_vertical_implicit_coefficients_cofwt_active": cofwt_s,
        "_vertical_implicit_coefficients_a_tri_active": a_tri_s,
        "_vertical_implicit_coefficients_alpha_tri_active": alpha_tri_s,
        "_vertical_implicit_coefficients_gamma_tri_active": gamma_tri_s,
    }
    if pack_public:
        compact_result.update(
            pack_vert_imp_coefs_public(
                compact_result,
                nCells=nCells,
                nVertLevels=nVertLevels,
                device=device,
            )
        )
    return compact_result
