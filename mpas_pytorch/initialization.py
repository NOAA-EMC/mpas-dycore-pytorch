"""
Python-side initialization pipeline for required diagnostics.

This module provides functions to compute derived fields that are required
by the RK3 time integration but are not present in raw initial conditions.

The MPAS dycore expects many arrays that Fortran computes during initialization:
- Edge sign arrays (edgesOnCell_sign, edgesOnVertex_sign)
- Mesh scaling arrays (meshScalingDel2, meshScalingDel4)
- Base state arrays (rho_base, rtheta_base, exner_base)
- Zero tendency arrays (rt_diabatic_tend)

All functions are pure (no side effects) and use torch.float64 for consistency
with the machine-precision requirement of the MPAS-PyTorch port.

Index Convention:
    These functions work with 1-based Fortran-style arrays with +1 padding:
    - Arrays have shape (N+1, ...) where index 0 is padding
    - Connectivity arrays contain 1-based indices
    - Sentinel value 0 indicates "no neighbor" (boundary)

Strict Mode:
    Fields are categorized into:
    - CHECKPOINT_REQUIRED_FIELDS: Must come from checkpoint for machine precision
    - SYNTHETICALLY_EXACT_FIELDS: Can be safely computed from mesh topology

    In strict mode (default), missing CHECKPOINT_REQUIRED_FIELDS raise ValueError.
    In non-strict mode, synthetic approximations are used with RuntimeWarning.
"""

import warnings
from typing import Any

import torch

from mpas_pytorch.constants import DTYPE, PhysicalConstants
from mpas_pytorch.exceptions import MpasConfigError
from mpas_pytorch.utils.sentinel import is_sentinel_batch

# =============================================================================
# Field Categorization for Strict Mode
# =============================================================================

# Fields that must come from the supplied model inputs for full precision.
# Non-strict mode synthesizes approximations that violate the accuracy contract.
# Production runs MUST use strict_initialization=True.

# Using synthetic formulas for these violates the machine-precision contract:
# - rho_base: must come from checkpoint/raw IC
# - rtheta_base, exner_base: if absent, they may be derived exactly from raw
#   checkpoint base-state primitives (rho_base, theta_base, zz)
# - dry-adiabatic synthesis remains non-exact and is non-strict only
# - zgrid: Geometric height; using zz (Jacobian-weighted) is only approximate
# - meshScalingDel2, meshScalingDel4: Mesh-dependent scaling that should come
#   from upstream initialization to ensure consistency
CHECKPOINT_REQUIRED_FIELDS: frozenset[str] = frozenset(
    {
        "rho_base",
        "zgrid",
        "meshScalingDel2",
        "meshScalingDel4",
    }
)

# Fields that can be safely computed from mesh topology without loss of precision.
# These computations are exact consequences of the supplied topology:
# - edgesOnCell_sign, edgesOnVertex_sign: Orientation signs derived from
#   mesh connectivity (cellsOnEdge, verticesOnEdge)
# - rt_diabatic_tend: Zero for adiabatic runs (exact)
SYNTHETICALLY_EXACT_FIELDS: frozenset[str] = frozenset(
    {
        "edgesOnCell_sign",
        "edgesOnVertex_sign",
        "rt_diabatic_tend",
    }
)


def compute_edge_signs(mesh_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Compute edge sign arrays for cells and vertices.

    The edge sign indicates the orientation of an edge relative to a cell or vertex.
    For cells: +1 if the cell is cellsOnEdge[edge, 0], -1 if cellsOnEdge[edge, 1]
    For vertices: +1 if the vertex is verticesOnEdge[edge, 0], -1 if verticesOnEdge[edge, 1]

    These arrays are used in divergence and vorticity computations to ensure
    proper sign conventions when summing edge contributions.

    Args:
        mesh_dict: Dictionary containing mesh arrays with 1-based indexing:
            - nCells: int, number of cells
            - nEdges: int, number of edges
            - nVertices: int, number of vertices
            - maxEdges: int, maximum edges per cell
            - vertexDegree: int, edges per vertex (typically 3)
            - cellsOnEdge: (nEdges+1, 2) int tensor, cells adjacent to each edge
            - edgesOnCell: (nCells+1, maxEdges) int tensor, edges of each cell
            - nEdgesOnCell: (nCells+1,) int tensor, number of edges per cell
            - edgesOnVertex: (nVertices+1, vertexDegree) int tensor, edges at each vertex
            - verticesOnEdge: (nEdges+1, 2) int tensor, vertices of each edge

    Returns:
        Dictionary containing:
            - edgesOnCell_sign: (nCells+1, maxEdges) float64 tensor, +1/-1 signs
            - edgesOnVertex_sign: (nVertices+1, vertexDegree) float64 tensor, +1/-1 signs
    """
    nCells = int(mesh_dict["nCells"])
    nEdges = int(mesh_dict["nEdges"])
    nVertices = int(mesh_dict["nVertices"])
    maxEdges = int(mesh_dict["maxEdges"])
    vertexDegree = int(mesh_dict["vertexDegree"])

    cellsOnEdge = mesh_dict["cellsOnEdge"]  # (nEdges+1, 2)
    edgesOnCell = mesh_dict["edgesOnCell"]  # (nCells+1, maxEdges)
    nEdgesOnCell = mesh_dict["nEdgesOnCell"]  # (nCells+1,)
    edgesOnVertex = mesh_dict["edgesOnVertex"]  # (nVertices+1, vertexDegree)
    verticesOnEdge = mesh_dict["verticesOnEdge"]  # (nEdges+1, 2)

    # Infer device from input tensors
    device = (
        cellsOnEdge.device
        if isinstance(cellsOnEdge, torch.Tensor)
        else torch.device("cpu")
    )

    # =========================================================================
    # Compute edgesOnCell_sign
    # For each cell, for each edge on that cell:
    #   sign = +1 if cell is cellsOnEdge[edge, 0]
    #   sign = -1 if cell is cellsOnEdge[edge, 1]

    # Indexing convention: Arrays use 0-based first-dimension indexing with
    # trailing padding (real data at 0..N-1, padding at index N). Cross-
    # reference VALUES (cellsOnEdge values, edgesOnCell values) are 1-based
    # Fortran cell/edge numbers. We convert the 0-based array index to a
    # 1-based cell number for comparison with cellsOnEdge values.
    # =========================================================================
    edgesOnCell_sign = torch.zeros(nCells + 1, maxEdges, dtype=DTYPE, device=device)
    cell_ids0 = torch.arange(nCells, dtype=torch.int64, device=device)
    cell_ids_1based = cell_ids0 + 1
    edge_slots = torch.arange(maxEdges, dtype=torch.int64, device=device)
    n_edges_this_cell = nEdgesOnCell[:nCells]
    slot_valid = edge_slots[None, :] < n_edges_this_cell[:, None]

    edge_id_1based = edgesOnCell[:nCells, :]
    edge_id_0based = edge_id_1based - 1
    sentinel_mask = is_sentinel_batch(edge_id_0based, max_valid=nEdges - 1)
    valid = slot_valid & (~sentinel_mask)
    safe_edges = torch.where(valid, edge_id_0based, torch.zeros_like(edge_id_0based))

    # Compare 1-based Fortran cell numbers for sign assignment.
    cell1_1based = cellsOnEdge[safe_edges, 0]
    sign = torch.where(
        cell1_1based == cell_ids_1based[:, None],
        torch.tensor(1.0, dtype=DTYPE, device=device),
        torch.tensor(-1.0, dtype=DTYPE, device=device),
    )
    edgesOnCell_sign[:nCells, :] = torch.where(
        valid,
        sign,
        torch.zeros_like(sign),
    )

    # =========================================================================
    # Compute edgesOnVertex_sign
    # For each vertex, for each edge on that vertex:
    #   sign = +1 if vertex is verticesOnEdge[edge, 0]
    #   sign = -1 if vertex is verticesOnEdge[edge, 1]

    # Same 0-based array / 1-based value convention as edgesOnCell_sign.
    # =========================================================================
    edgesOnVertex_sign = torch.zeros(
        nVertices + 1, vertexDegree, dtype=DTYPE, device=device
    )
    vertex_ids0 = torch.arange(nVertices, dtype=torch.int64, device=device)
    vertex_ids_1based = vertex_ids0 + 1
    edge_id_1based = edgesOnVertex[:nVertices, :]
    edge_id_0based = edge_id_1based - 1
    sentinel_mask = is_sentinel_batch(edge_id_0based, max_valid=nEdges - 1)
    valid = ~sentinel_mask
    safe_edges = torch.where(valid, edge_id_0based, torch.zeros_like(edge_id_0based))

    # Fortran convention: sign = +1 when vertex == verticesOnEdge(2, edge).
    vertex2_1based = verticesOnEdge[safe_edges, 1]
    sign = torch.where(
        vertex2_1based == vertex_ids_1based[:, None],
        torch.tensor(1.0, dtype=DTYPE, device=device),
        torch.tensor(-1.0, dtype=DTYPE, device=device),
    )
    edgesOnVertex_sign[:nVertices, :] = torch.where(
        valid,
        sign,
        torch.zeros_like(sign),
    )

    return {
        "edgesOnCell_sign": edgesOnCell_sign,
        "edgesOnVertex_sign": edgesOnVertex_sign,
    }


def compute_mesh_scaling(
    mesh_dict: dict[str, Any], config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Compute mesh scaling arrays for Del2 and Del4 hyperviscosity operators.

    For global uniform-resolution meshes, these are typically 1.0 everywhere.
    For variable-resolution meshes, scaling can be based on meshDensity or dcEdge.

    The scaling modifies the effective diffusion coefficient so that
    the damping timescale is approximately uniform across the mesh.

    Args:
        mesh_dict: Dictionary containing mesh arrays:
            - nEdges: int, number of edges
            - meshDensity: (nCells+1,) float64 tensor (optional)
            - dcEdge: (nEdges+1,) float64 tensor (optional)
        config: Optional configuration dictionary:
            - mesh_scaling_type: 'uniform', 'density', or 'dcEdge'
              Default is 'uniform' (all ones)

    Returns:
        Dictionary containing:
            - meshScalingDel2: (nEdges+1,) float64 tensor for Del2 operator
            - meshScalingDel4: (nEdges+1,) float64 tensor for Del4 operator
    """
    nEdges = int(mesh_dict["nEdges"])
    nCells = int(
        mesh_dict.get(
            "nCells",
            mesh_dict["meshDensity"].shape[0] - 1 if "meshDensity" in mesh_dict else 0,
        )
    )

    # Infer device from input
    device = torch.device("cpu")
    for key in ["dcEdge", "meshDensity", "cellsOnEdge"]:
        if key in mesh_dict and isinstance(mesh_dict[key], torch.Tensor):
            device = mesh_dict[key].device
            break

    h_scale_with_mesh = True
    scaling_type = None
    if config is not None:
        from mpas_pytorch.config_defaults import get_config_value_safe

        scaling_type = config.get("mesh_scaling_type")
        h_scale_with_mesh = bool(
            get_config_value_safe(config, "config_h_ScaleWithMesh", True)
        )

    if scaling_type is None:
        if h_scale_with_mesh and "meshDensity" in mesh_dict:
            scaling_type = "density"
        else:
            scaling_type = "uniform"

    if scaling_type == "uniform":
        # Uniform scaling - no mesh adaptation
        meshScalingDel2 = torch.ones(nEdges + 1, dtype=DTYPE, device=device)
        meshScalingDel4 = torch.ones(nEdges + 1, dtype=DTYPE, device=device)

    elif scaling_type == "density" and "meshDensity" in mesh_dict:
        # Scale based on meshDensity (for variable-resolution meshes)
        # Higher density = finer mesh = less diffusion needed
        # This is a simplified approximation
        meshDensity = mesh_dict["meshDensity"]
        cellsOnEdge = mesh_dict["cellsOnEdge"]
        meshScalingDel2 = torch.ones(nEdges + 1, dtype=DTYPE, device=device)
        meshScalingDel4 = torch.ones(nEdges + 1, dtype=DTYPE, device=device)
        padded_edges = cellsOnEdge.shape[0] >= nEdges + 1 and bool(
            torch.all(cellsOnEdge[0, :] == 0).item()
        )
        edge_slice = slice(1, nEdges + 1) if padded_edges else slice(0, nEdges)
        cell1_1based = cellsOnEdge[edge_slice, 0]
        cell2_1based = cellsOnEdge[edge_slice, 1]
        # Convert 1-based connectivity values to 0-based for batched sentinel checks.
        sent_cell1 = is_sentinel_batch(cell1_1based - 1, max_valid=nCells - 1)
        sent_cell2 = is_sentinel_batch(cell2_1based - 1, max_valid=nCells - 1)
        valid = ~(sent_cell1 | sent_cell2)
        safe_cell1 = torch.where(
            valid, cell1_1based - 1, torch.zeros_like(cell1_1based)
        )
        safe_cell2 = torch.where(
            valid, cell2_1based - 1, torch.zeros_like(cell2_1based)
        )

        density_edge = 0.5 * (meshDensity[safe_cell1] + meshDensity[safe_cell2])
        valid_positive = valid & (density_edge > 0)
        del2_vals = 1.0 / density_edge**0.25
        del4_vals = 1.0 / density_edge**0.75
        meshScalingDel2[edge_slice] = torch.where(
            valid_positive,
            del2_vals,
            meshScalingDel2[edge_slice],
        )
        meshScalingDel4[edge_slice] = torch.where(
            valid_positive,
            del4_vals,
            meshScalingDel4[edge_slice],
        )

    elif scaling_type == "dcEdge" and "dcEdge" in mesh_dict:
        # Scale based on edge length
        dcEdge = mesh_dict["dcEdge"]
        dc_max = dcEdge[:nEdges].max()

        meshScalingDel2 = torch.ones(nEdges + 1, dtype=DTYPE, device=device)
        meshScalingDel4 = torch.ones(nEdges + 1, dtype=DTYPE, device=device)
        ratio = dcEdge[:nEdges] / dc_max
        meshScalingDel2[:nEdges] = ratio
        meshScalingDel4[:nEdges] = ratio * ratio

    else:
        # Default to uniform if unknown type or missing data
        meshScalingDel2 = torch.ones(nEdges + 1, dtype=DTYPE, device=device)
        meshScalingDel4 = torch.ones(nEdges + 1, dtype=DTYPE, device=device)

    # For global meshes these stay at 1.0; regional runs may override them later.
    meshScalingRegionalCell = torch.ones(nCells + 1, dtype=DTYPE, device=device)
    meshScalingRegionalEdge = torch.ones(nEdges + 1, dtype=DTYPE, device=device)

    return {
        "meshScalingDel2": meshScalingDel2,
        "meshScalingDel4": meshScalingDel4,
        "meshScalingRegionalCell": meshScalingRegionalCell,
        "meshScalingRegionalEdge": meshScalingRegionalEdge,
    }


def compute_damping_coefs(
    zgrid: torch.Tensor,
    mesh_density: torch.Tensor,
    n_cells: int,
    n_vert_levels: int,
    config_zd: float = 22000.0,
    config_xnutr: float = 0.2,
) -> dict[str, Any]:
    """Compute MPAS Rayleigh damping coefficients in the raw input layout.

    The returned tensor contains active rows only, without a padded cell row.
    """
    device = zgrid.device
    pii = torch.acos(torch.tensor(-1.0, dtype=DTYPE, device=device))
    dx_scale_power = 1.0

    zt = zgrid[:n_cells, n_vert_levels]
    z_mid = 0.5 * (
        zgrid[:n_cells, :n_vert_levels] + zgrid[:n_cells, 1 : n_vert_levels + 1]
    )

    mask = z_mid > config_zd
    denom = zt.unsqueeze(1) - config_zd
    arg = 0.5 * pii * (z_mid - config_zd) / denom
    raw_dss = config_xnutr * torch.sin(arg) ** 2.0
    density_scale = mesh_density[:n_cells].unsqueeze(1) ** (0.25 * dx_scale_power)
    raw_dss = raw_dss / density_scale
    dss = torch.where(mask, raw_dss, torch.zeros_like(raw_dss))

    return {"dss": dss}


def resolve_len_disp(
    config_len_disp: float,
    nominal_min_dc: float | None,
) -> float:
    """Resolve the horizontal dispersion length scale with Fortran semantics."""
    if config_len_disp > 0.0:
        return config_len_disp

    if nominal_min_dc is not None and nominal_min_dc > 0.0:
        return nominal_min_dc

    raise MpasConfigError(
        "Both config_len_disp and nominalMinDc are <= 0 or unavailable. "
        "Specify config_len_disp in the &nhyd_model namelist group, "
        "or use an input file that provides a valid nominalMinDc value."
    )


def compute_base_state(
    mesh_dict: dict[str, Any], config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Compute hydrostatic base state arrays.

    The base state provides a reference atmosphere for the perturbation formulation
    of the equations. It consists of:
    - rho_base: Base state density at cell centers
    - rtheta_base: Base state rho*theta at cell centers
    - exner_base: Base state Exner function at w-levels

    For a dry adiabatic atmosphere with constant potential temperature:
        theta = theta_0 = constant
        exner = 1 - (g / (cp * theta_0)) * z
        rho = p0 / (Rd * theta_0) * exner^(cv/Rd)

    Args:
        mesh_dict: Dictionary containing:
            - nCells: int, number of cells
            - nVertLevels: int, number of vertical levels
            - zgrid: (nCells+1, nVertLevels+1) float64 tensor, geometric height at w-levels
              OR zz: (nCells+1, nVertLevels+1) float64 tensor, Jacobian*height
        config: Optional configuration dictionary:
            - theta_base: Base state potential temperature (K), default 288.0

    Returns:
        Dictionary containing:
            - rho_base: (nCells+1, nVertLevels) float64 tensor
            - rtheta_base: (nCells+1, nVertLevels) float64 tensor
            - exner_base: (nCells+1, nVertLevels+1) float64 tensor
    """
    nCells = int(mesh_dict["nCells"])
    nVertLevels = int(mesh_dict["nVertLevels"])

    # Infer device from input
    device = torch.device("cpu")
    for key in ["zgrid", "zz"]:
        if key in mesh_dict and isinstance(mesh_dict[key], torch.Tensor):
            device = mesh_dict[key].device
            break

    # Get base state potential temperature from config or use default
    theta_0 = 288.0
    if config is not None:
        theta_0 = config.get("theta_base", 288.0)

    # Physical constants
    g = float(PhysicalConstants.G)
    cp = float(PhysicalConstants.CP)
    cv = float(PhysicalConstants.CV)
    rd = float(PhysicalConstants.RD)
    p0 = float(PhysicalConstants.P0)

    # Exponent for pressure-density relationship
    cv_rd = cv / rd

    # Allocate output arrays
    rho_base = torch.zeros(nCells + 1, nVertLevels, dtype=DTYPE, device=device)
    rtheta_base = torch.zeros(nCells + 1, nVertLevels, dtype=DTYPE, device=device)
    exner_base = torch.zeros(nCells + 1, nVertLevels + 1, dtype=DTYPE, device=device)

    # Get vertical coordinate
    if "zgrid" in mesh_dict:
        zgrid = mesh_dict["zgrid"]
    elif "zz" in mesh_dict:
        # zz is Jacobian*height, but for base state we can use it approximately
        zgrid = mesh_dict["zz"]
    else:
        # No vertical coordinate available - return zeros (caller should provide zgrid)
        return {
            "rho_base": rho_base,
            "rtheta_base": rtheta_base,
            "exner_base": exner_base,
        }

    z_active = zgrid[1 : nCells + 1, : nVertLevels + 1]
    # Dry adiabatic atmosphere: exner = 1 - g*z / (cp*theta_0)
    exner_active = 1.0 - (g * z_active) / (cp * theta_0)
    # Ensure exner stays positive (model top constraint)
    exner_active = torch.clamp(exner_active, min=0.001)
    exner_base[1 : nCells + 1, : nVertLevels + 1] = exner_active

    # Density and rho*theta at cell centers (layer midpoints)
    exner_mid = 0.5 * (
        exner_active[:, :nVertLevels] + exner_active[:, 1 : nVertLevels + 1]
    )
    rho_active = (p0 / (rd * theta_0)) * (exner_mid**cv_rd)
    rho_base[1 : nCells + 1, :nVertLevels] = rho_active
    rtheta_base[1 : nCells + 1, :nVertLevels] = rho_active * theta_0

    return {
        "rho_base": rho_base,
        "rtheta_base": rtheta_base,
        "exner_base": exner_base,
    }


def compute_base_state_from_checkpoint_primitives(
    state_dict: dict[str, Any],
) -> dict[str, Any]:
    """Derive exact base-state diagnostics from raw checkpoint primitives."""
    nCells = int(state_dict["nCells"])
    nVertLevels = int(state_dict["nVertLevels"])
    rho_base = state_dict["rho_base"]
    theta_base = state_dict["theta_base"]
    zz = state_dict["zz"]

    if zz.shape[0] < nCells + 1 or zz.shape[1] < nVertLevels:
        raise MpasConfigError(
            f"Cannot derive base state from checkpoint primitives: zz shape {tuple(zz.shape)} "
            f"incompatible with nCells={nCells}, nVertLevels={nVertLevels}"
        )

    # State arrays in this codebase use append-padding: valid rows are 0..N-1 and
    # the trailing row N is the garbage/padding row.
    active_rho_base = rho_base[:nCells, :nVertLevels]
    active_theta_base = theta_base[:nCells, :nVertLevels]
    active_zz = zz[:nCells, :nVertLevels]

    rtheta_base = torch.zeros_like(rho_base, dtype=DTYPE, device=active_rho_base.device)
    exner_base = torch.zeros(
        nCells + 1, nVertLevels, dtype=DTYPE, device=active_rho_base.device
    )

    active_rtheta_base = active_theta_base * active_rho_base
    rcv = float(PhysicalConstants.CV) / float(PhysicalConstants.RD)
    exner_active = (
        active_zz
        * (float(PhysicalConstants.RD) / float(PhysicalConstants.P0))
        * active_rtheta_base
    ) ** rcv

    rtheta_base[:nCells, :nVertLevels] = active_rtheta_base
    exner_base[:nCells, :nVertLevels] = exner_active
    return {
        "rtheta_base": rtheta_base,
        "exner_base": exner_base,
    }


def initialize_zero_tendencies(
    nCells: int, nVertLevels: int, device: torch.device | None = None
) -> dict[str, Any]:
    """
    Create zero-initialized tendency arrays.

    For adiabatic runs, diabatic tendencies are zero. This function creates
    the rt_diabatic_tend array initialized to zeros.

    Args:
        nCells: Number of cells
        nVertLevels: Number of vertical levels
        device: Device for tensor creation (default: CPU)

    Returns:
        Dictionary containing:
            - rt_diabatic_tend: (nCells+1, nVertLevels) float64 tensor of zeros
    """
    if device is None:
        device = torch.device("cpu")

    rt_diabatic_tend = torch.zeros(nCells + 1, nVertLevels, dtype=DTYPE, device=device)

    return {
        "rt_diabatic_tend": rt_diabatic_tend,
    }


def initialize_from_state(
    state_dict: dict[str, Any],
    mesh_dict: dict[str, Any],
    config: dict[str, Any] | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """
    Main entry point to compute all missing initialization fields.

    This function examines the state dictionary and computes any missing
    derived fields that are required by the RK3 time integration:
    - Edge sign arrays (edgesOnCell_sign, edgesOnVertex_sign)
    - Mesh scaling arrays (meshScalingDel2, meshScalingDel4)
    - Base state arrays (rho_base, rtheta_base, exner_base)
    - Zero tendency arrays (rt_diabatic_tend)

    Fields that already exist in state_dict are NOT overwritten.

    Strict Mode:
        Fields are categorized into CHECKPOINT_REQUIRED_FIELDS (must come from
        checkpoint for machine precision) and SYNTHETICALLY_EXACT_FIELDS (can
        be safely computed from mesh topology).

        In strict mode (default=True), missing CHECKPOINT_REQUIRED_FIELDS raise
        ValueError. In non-strict mode, synthetic approximations are used with
        RuntimeWarning emitted for each approximate computation.

    Args:
        state_dict: Dictionary containing state arrays from checkpoint
        mesh_dict: Dictionary containing mesh connectivity arrays
        config: Optional configuration dictionary for initialization options.
            The 'strict_initialization' key can override the strict parameter.
        strict: If True (default), raise ValueError when CHECKPOINT_REQUIRED_FIELDS
            are missing. If False, use synthetic approximations with warnings.
            Can be overridden by config['strict_initialization'].

    Returns:
        Updated state_dict with all required initialization fields added.
        The original state_dict is modified in place AND returned.

    Raises:
        ValueError: In strict mode, if CHECKPOINT_REQUIRED_FIELDS are missing
            from state_dict and cannot be loaded from checkpoint.

    Note:
        This function calls solve_diagnostics if diagnostics (divergence,
        vorticity, ke, pv_edge) are missing and all required inputs are present.
        However, that is typically handled separately by the RK step itself.
    """
    # Resolve strict mode from config if provided
    if config is not None and "strict_initialization" in config:
        strict = config["strict_initialization"]

    # Warn when non-strict mode is used
    if not strict:
        warnings.warn(
            "Non-strict initialization uses synthetic approximations that "
            "VIOLATE machine-precision accuracy (atol=1e-14, rtol=1e-12). "
            "Use strict_initialization=True for production runs. "
            "Non-strict mode is for development/debugging ONLY.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Get dimensions
    nCells = int(mesh_dict.get("nCells", state_dict.get("nCells", 0)))
    nVertLevels = int(mesh_dict.get("nVertLevels", state_dict.get("nVertLevels", 0)))

    if nCells == 0 or nVertLevels == 0:
        raise MpasConfigError(
            "Cannot initialize: nCells or nVertLevels is 0 or not found in mesh_dict or state_dict"
        )

    # =========================================================================
    # 1. Edge signs (always compute if missing)
    # =========================================================================
    if "edgesOnCell_sign" not in state_dict or "edgesOnVertex_sign" not in state_dict:
        # Check required connectivity arrays — prefer state_dict (IC ordering)
        # over mesh_dict (may use different edge ordering).
        required_for_signs = [
            "nCells",
            "nEdges",
            "nVertices",
            "maxEdges",
            "vertexDegree",
            "cellsOnEdge",
            "edgesOnCell",
            "nEdgesOnCell",
            "edgesOnVertex",
            "verticesOnEdge",
        ]
        # Use state_dict connectivity if available (preserves IC edge ordering);
        # fall back to mesh_dict if state_dict is missing connectivity.
        signs_source = (
            state_dict
            if all(k in state_dict for k in required_for_signs)
            else mesh_dict
        )
        missing = [k for k in required_for_signs if k not in signs_source]

        if not missing:
            sign_result = compute_edge_signs(signs_source)
            if "edgesOnCell_sign" not in state_dict:
                state_dict["edgesOnCell_sign"] = sign_result["edgesOnCell_sign"]
            if "edgesOnVertex_sign" not in state_dict:
                state_dict["edgesOnVertex_sign"] = sign_result["edgesOnVertex_sign"]

    # =========================================================================
    # 2. Mesh scaling (CHECKPOINT_REQUIRED in strict mode)
    # =========================================================================
    # First, copy mesh scaling from mesh_dict if missing from state_dict.
    # This handles the case where mesh fields were removed from IC state
    # (e.g., due to edge ordering conflicts) but are available in mesh_dict.
    for _sk in ["meshScalingDel2", "meshScalingDel4"]:
        if _sk not in state_dict and _sk in mesh_dict:
            state_dict[_sk] = mesh_dict[_sk]

    missing_scaling = []
    if "meshScalingDel2" not in state_dict:
        missing_scaling.append("meshScalingDel2")
    if "meshScalingDel4" not in state_dict:
        missing_scaling.append("meshScalingDel4")

    if missing_scaling:
        if strict:
            raise MpasConfigError(
                f"Strict mode: missing CHECKPOINT_REQUIRED_FIELDS {missing_scaling}. "
                f"These fields must come from checkpoint for machine precision. "
                f"Set strict=False or config['strict_initialization']=False to use "
                f"synthetic approximations (uniform scaling), but this violates the "
                f"machine-precision contract."
            )
        else:
            # Non-strict mode: use synthetic approximation with warning
            if "nEdges" in mesh_dict:
                warnings.warn(
                    f"Non-strict mode: synthesizing {missing_scaling} with uniform "
                    f"scaling (all ones). This may not match upstream initialization "
                    f"and violates machine-precision requirements.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                scaling_result = compute_mesh_scaling(mesh_dict, config)
                if "meshScalingDel2" not in state_dict:
                    state_dict["meshScalingDel2"] = scaling_result["meshScalingDel2"]
                if "meshScalingDel4" not in state_dict:
                    state_dict["meshScalingDel4"] = scaling_result["meshScalingDel4"]

    # Copy zgrid from mesh_dict if missing from state_dict (CHECKPOINT_REQUIRED).
    # zgrid may have been removed from IC state due to edge ordering conflicts
    # but is available in the mesh checkpoint.
    if "zgrid" not in state_dict and "zgrid" in mesh_dict:
        state_dict["zgrid"] = mesh_dict["zgrid"]

    # =========================================================================
    # 3. Base state (CHECKPOINT_REQUIRED in strict mode)
    # =========================================================================
    base_state_keys = ["rho_base", "rtheta_base", "exner_base"]
    # Copy base state from mesh_dict if missing from state_dict (same pattern
    # as mesh scaling above — fields may have been removed from IC state but
    # are available from the mesh/init checkpoint).
    for k in base_state_keys:
        if k not in state_dict and k in mesh_dict:
            state_dict[k] = mesh_dict[k]

    derivable_base_keys = {"rtheta_base", "exner_base"}
    missing_base_state = [k for k in base_state_keys if k not in state_dict]
    if any(k in missing_base_state for k in derivable_base_keys):  # noqa: SIM102
        if all(k in state_dict for k in ("rho_base", "theta_base", "zz")):
            derived_base = compute_base_state_from_checkpoint_primitives(state_dict)
            for key in derivable_base_keys:
                if key not in state_dict:
                    state_dict[key] = derived_base[key]
            missing_base_state = [k for k in base_state_keys if k not in state_dict]

    if missing_base_state:
        if strict:
            raise MpasConfigError(
                f"Strict mode: missing CHECKPOINT_REQUIRED_FIELDS {missing_base_state}. "
                f"Base state arrays must come from checkpoint for machine precision. "
                f"Set strict=False or config['strict_initialization']=False to use "
                f"synthetic dry adiabatic approximation, but this violates the "
                f"machine-precision contract."
            )
        else:
            # Non-strict mode: use synthetic approximation with warning
            # Check if we have vertical coordinate
            has_zgrid = "zgrid" in mesh_dict or "zz" in mesh_dict
            has_zgrid = has_zgrid or "zgrid" in state_dict or "zz" in state_dict

            if has_zgrid:
                # Determine if using zgrid or zz (approximate)
                using_zz = "zgrid" not in mesh_dict and "zgrid" not in state_dict
                coord_warning = ""
                if using_zz:
                    coord_warning = " Using zz (Jacobian-weighted) as proxy for zgrid introduces additional error."

                warnings.warn(
                    f"Non-strict mode: synthesizing {missing_base_state} with "
                    f"simplified dry adiabatic formulas.{coord_warning} "
                    f"This may not match upstream initialization and violates "
                    f"machine-precision requirements.",
                    RuntimeWarning,
                    stacklevel=2,
                )

                # Merge zgrid from state_dict if needed
                merged_mesh = dict(mesh_dict)
                if "zgrid" not in merged_mesh and "zgrid" in state_dict:
                    merged_mesh["zgrid"] = state_dict["zgrid"]
                if "zz" not in merged_mesh and "zz" in state_dict:
                    merged_mesh["zz"] = state_dict["zz"]
                if "nCells" not in merged_mesh:
                    merged_mesh["nCells"] = nCells
                if "nVertLevels" not in merged_mesh:
                    merged_mesh["nVertLevels"] = nVertLevels

                base_result = compute_base_state(merged_mesh, config)
                for key in base_state_keys:
                    if key not in state_dict:
                        state_dict[key] = base_result[key]

    # =========================================================================
    # 4. Zero tendencies (always create if missing)
    # =========================================================================
    if "rt_diabatic_tend" not in state_dict:
        # Infer device from existing tensors
        device = torch.device("cpu")
        for val in state_dict.values():
            if isinstance(val, torch.Tensor):
                device = val.device
                break

        tend_result = initialize_zero_tendencies(nCells, nVertLevels, device)
        state_dict["rt_diabatic_tend"] = tend_result["rt_diabatic_tend"]

    return state_dict
