"""Standalone initial-condition loader for the MPAS-A PyTorch dycore."""

from __future__ import annotations

import copy
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mpas_pytorch.config_defaults import (
    CONFIG_ALIASES,
    FORTRAN_DEFAULTS,
    get_config_value_safe,
    validate_config,
)
from mpas_pytorch.dynamics.vector_reconstruction import (
    CELL_PADDING_OFFSET_KEY,
    COEFFS_RECONSTRUCT_ALL_ZERO_KEY,
)
from mpas_pytorch.exceptions import (
    MpasConfigError,
    MpasMeshMismatchError,
    MpasMissingFieldError,
    MpasStrictModeError,
    MpasUnsupportedCapabilityError,
)
from mpas_pytorch.field_provenance import ALLOWED_FROM_FILE, validate_no_injection
from mpas_pytorch.initialization import (
    compute_damping_coefs,
    compute_edge_signs,
    compute_mesh_scaling,
)
from mpas_pytorch.mesh_geometry import (
    compute_cell_tangent_plane,
    compute_coeffs_reconstruct,
    compute_edge_normal_vectors,
    compute_local_vertical_unit_vectors,
)
from mpas_pytorch.namelist_metadata import (
    active_unsupported_namelist_keys,
    known_unwired_namelist_keys,
)
from mpas_pytorch.release_capabilities import validate_runtime_request
from mpas_pytorch.utils import load_checkpoint, parse_namelist

_SUPPLEMENTAL_FIELDS: tuple[str, ...] = (
    "nAdvCellsForEdge",
    "advCellsForEdge",
    "adv_coefs",
    "adv_coefs_3rd",
    "fzm",
    "fzp",
    "rdzw",
    "latCell",
    "lonCell",
    "u_init",
    "v_init",
    "qv_init",
)

_MESH_FROM_INIT_FIELDS: tuple[str, ...] = (
    "fzm",
    "fzp",
    "zz",
    "rdzw",
    "dss",
)

_EDGE_ORDER_FIELDS: tuple[str, ...] = ("u", "rho_edge")

_DERIVED_INIT_FIELDS: tuple[str, ...] = ("sh2o",)

MOISTURE_SCALAR_FIELDS: tuple[str, ...] = ("qv", "qc", "qr", "qi", "qs", "qg")

_DIMENSION_KEYS: tuple[str, ...] = (
    "nCells",
    "nEdges",
    "nVertLevels",
    "nVertices",
    "maxEdges",
    "maxEdges2",
    "vertexDegree",
)

_INTEGER_FIELD_KEYS: tuple[str, ...] = (
    "indexToCellID",
    "indexToEdgeID",
    "indexToVertexID",
    "bdyMaskVertex",
    "landmask",
    "ivgtyp",
    "isltyp",
)

PHYSICS_NAMELIST_KEYS: frozenset[str] = frozenset(
    {
        "config_physics_suite",
        "config_microp_scheme",
        "config_convection_scheme",
        "config_pbl_scheme",
        "config_gwdo_scheme",
        "config_radt_cld_scheme",
        "config_radt_lw_scheme",
        "config_radt_sw_scheme",
        "config_sfclayer_scheme",
        "config_lsm_scheme",
        "config_radtlw_interval",
        "config_radtsw_interval",
        "config_conv_interval",
        "config_pbl_interval",
        "config_start_time",
        "config_run_duration",
    }
)

MESOSCALE_REFERENCE_SCHEMES: dict[str, str] = {
    "config_microp_scheme": "mp_wsm6",
    "config_convection_scheme": "cu_ntiedtke",
    "config_pbl_scheme": "bl_ysu",
    "config_gwdo_scheme": "off",
    "config_radt_cld_scheme": "cld_fraction",
    "config_radt_lw_scheme": "rrtmg_lw",
    "config_radt_sw_scheme": "rrtmg_sw",
    "config_sfclayer_scheme": "sf_monin_obukhov_rev",
    "config_lsm_scheme": "sf_noah",
}

# Required raw prognostics/base-state fields for `load_initial_state`.
# This is intentionally narrower than `field_provenance.ALLOWED_FROM_FILE`:
# `ALLOWED_FROM_FILE` is a permissive provenance allowlist, while this set is
# the minimal always-required subset for successful cold-boot/runtime startup.
# If a future registry change adds a new mandatory init prognostic, update this
# set alongside the provenance tables.
_REQUIRED_INIT_FIELDS: frozenset[str] = frozenset(
    {
        "u",
        "w",
        "theta",
        "rho",
        "rho_base",
        "theta_base",
        "zz",
    }
)

_REQUIRED_MESH_FIELDS: frozenset[str] = frozenset(
    {
        "cellsOnCell",
        "edgesOnCell",
        "verticesOnCell",
        "cellsOnEdge",
        "verticesOnEdge",
        "edgesOnVertex",
        "cellsOnVertex",
        "edgesOnEdge",
        "nEdgesOnCell",
        "nEdgesOnEdge",
        "dcEdge",
        "dvEdge",
        "weightsOnEdge",
        "kiteAreasOnVertex",
        "areaCell",
        "areaTriangle",
        "latCell",
        "lonCell",
        "xCell",
        "yCell",
        "zCell",
        "xEdge",
        "yEdge",
        "zEdge",
    }
)

# The dycore consumes these raw arrays; unrelated physics fields may be unwritten
# in a stock dycore-only initialization and are deliberately outside this check.
_UNUSED_PHYSICS_FIELDS = frozenset(
    {
        "smois",
        "tslb",
        "skintemp",
        "snow",
        "snowc",
        "snowh",
        "snoalb",
        "tmn",
        "vegfra",
        "xland",
        "xice",
        "greenfrac",
        "var2d",
        "con",
        "oa1",
        "oa2",
        "oa3",
        "oa4",
        "ol1",
        "ol2",
        "ol3",
        "ol4",
    }
)
_VALIDATED_RAW_FIELDS = (
    (ALLOWED_FROM_FILE - _UNUSED_PHYSICS_FIELDS)
    | frozenset(MOISTURE_SCALAR_FIELDS)
    | frozenset(_SUPPLEMENTAL_FIELDS)
)


def _synthesize_cell_metric_fields(
    state: dict[str, Any],
    mesh: dict[str, Any],
    *,
    coef_3rd_order: float,
    force_zb3_cell: bool = False,
) -> None:
    """Populate cell-based zb fields from edge-based inputs when absent.

    MPAS builds ``zb_cell`` directly from the edge-side ``zb`` coefficients, then
    applies ``config_coef_3rd_order`` to the copied ``zb3`` values to form
    ``zb3_cell``. New standalone ``init.nc`` files do not carry the cell-based
    arrays, so reproduce the same deterministic mesh transform here.
    """
    needs_zb_cell = "zb_cell" not in state and "zb" in state
    needs_zb3_cell = "zb3_cell" not in state and "zb3" in state
    if not needs_zb_cell and not needs_zb3_cell and not force_zb3_cell:
        return

    if "edgesOnCell" not in mesh or "cellsOnEdge" not in mesh:
        return

    n_cells = _to_int(mesh["nCells"])
    n_edges = _to_int(mesh["nEdges"])
    max_edges = _to_int(mesh["maxEdges"])

    edges_on_cell = mesh["edgesOnCell"][:n_cells, :max_edges].to(dtype=torch.int64)
    valid = (edges_on_cell > 0) & (edges_on_cell <= n_edges)
    edge_rows = torch.where(valid, edges_on_cell - 1, torch.zeros_like(edges_on_cell))

    cells_on_edge = mesh["cellsOnEdge"][:n_edges, :2].to(dtype=torch.int64)
    gather_cells = cells_on_edge.index_select(0, edge_rows.reshape(-1)).reshape(
        n_cells, max_edges, 2
    )
    cell_ids = torch.arange(
        1, n_cells + 1, device=edge_rows.device, dtype=torch.int64
    ).unsqueeze(1)
    side = (gather_cells[..., 1] == cell_ids).to(dtype=torch.int64)

    def _gather_edge_metric(name: str, scale: float) -> torch.Tensor:
        source = state[name][:n_edges]
        levels = int(source.shape[-1])
        flat_rows = edge_rows.reshape(-1)
        gathered = source.index_select(0, flat_rows).reshape(
            n_cells, max_edges, 2, levels
        )
        picked = gathered.gather(
            2,
            side.unsqueeze(-1).unsqueeze(-1).expand(n_cells, max_edges, 1, levels),
        ).squeeze(2)
        picked = picked * valid.unsqueeze(-1).to(dtype=source.dtype)
        pad = torch.zeros(
            (1, max_edges, levels), dtype=source.dtype, device=source.device
        )
        return torch.cat([picked * scale, pad], dim=0)

    if needs_zb_cell:
        zb_cell = _gather_edge_metric("zb", 1.0)
        state["zb_cell"] = zb_cell
        mesh.setdefault("zb_cell", zb_cell)
    if needs_zb3_cell or force_zb3_cell:
        zb3_cell = _gather_edge_metric("zb3", float(coef_3rd_order))
        state["zb3_cell"] = zb3_cell
        mesh["zb3_cell"] = zb3_cell


def _synthesize_exact_mesh_fields(mesh: dict[str, Any]) -> None:
    """Populate exact geometry-derived mesh fields when absent."""
    n_cells = _to_int(mesh["nCells"])
    n_edges = _to_int(mesh["nEdges"])
    n_vertices = _to_int(mesh["nVertices"])
    max_edges = _to_int(mesh["maxEdges"])
    vertex_degree = _to_int(mesh["vertexDegree"])

    def _inverse_from(name: str, target: str, count: int) -> None:
        if target in mesh or name not in mesh:
            return
        source = mesh[name][:count]
        inv = torch.zeros((count + 1,), dtype=source.dtype, device=source.device)
        valid = source != 0
        inv[:count] = torch.where(valid, 1.0 / source, torch.zeros_like(source))
        mesh[target] = inv

    _inverse_from("areaCell", "invAreaCell", n_cells)
    _inverse_from("areaTriangle", "invAreaTriangle", n_vertices)
    _inverse_from("dvEdge", "invDvEdge", n_edges)
    _inverse_from("dcEdge", "invDcEdge", n_edges)

    if "kiteForCell" in mesh:
        return
    if (
        "verticesOnCell" not in mesh
        or "cellsOnVertex" not in mesh
        or "nEdgesOnCell" not in mesh
    ):
        return

    vertices_on_cell = mesh["verticesOnCell"][:n_cells, :max_edges].to(
        dtype=torch.int64
    )
    cells_on_vertex = mesh["cellsOnVertex"][:n_vertices, :vertex_degree].to(
        dtype=torch.int64
    )
    n_edges_on_cell = mesh["nEdgesOnCell"][:n_cells].to(dtype=torch.int64)

    slot_ids = torch.arange(
        max_edges, device=vertices_on_cell.device, dtype=torch.int64
    )
    slot_valid = slot_ids.unsqueeze(0) < n_edges_on_cell.unsqueeze(1)
    vertex_valid = (vertices_on_cell > 0) & (vertices_on_cell <= n_vertices)
    gather_valid = slot_valid & vertex_valid
    vertex_rows = torch.where(
        gather_valid, vertices_on_cell - 1, torch.zeros_like(vertices_on_cell)
    )
    gathered = cells_on_vertex.index_select(0, vertex_rows.reshape(-1)).reshape(
        n_cells, max_edges, vertex_degree
    )
    cell_ids = torch.arange(
        1, n_cells + 1, device=vertices_on_cell.device, dtype=torch.int64
    )
    matches = gathered == cell_ids.view(n_cells, 1, 1)

    if torch.any(gather_valid & (~matches.any(dim=2))):
        raise MpasConfigError(
            "Cannot synthesize kiteForCell: cellsOnVertex/verticesOnCell mismatch"
        )

    kite = torch.zeros(
        (n_cells + 1, max_edges), dtype=torch.int32, device=vertices_on_cell.device
    )
    resolved = torch.argmax(matches.to(dtype=torch.int64), dim=2) + 1
    values = torch.where(
        slot_valid,
        torch.where(
            vertex_valid,
            resolved.to(dtype=torch.int32),
            torch.ones_like(resolved, dtype=torch.int32),
        ),
        torch.zeros_like(resolved, dtype=torch.int32),
    )
    kite[:n_cells, :] = values
    mesh["kiteForCell"] = kite


def _synthesize_advective_stencil_fields(
    state: dict[str, Any],
    mesh: dict[str, Any],
    *,
    coef_3rd_order: float,
    force_adv_coefs_3rd: bool = False,
) -> None:
    """Rebuild compressed scalar-transport stencils from ``deriv_two`` exactly."""
    needs = [
        "advCellsForEdge",
        "nAdvCellsForEdge",
        "adv_coefs",
        "adv_coefs_3rd",
    ]
    if all(key in mesh or key in state for key in needs) and not force_adv_coefs_3rd:
        _trim_advective_stencil_fields(state, mesh)
        return
    if "deriv_two" not in state:
        return
    required_mesh = ["cellsOnCell", "cellsOnEdge", "nEdgesOnCell", "dcEdge", "dvEdge"]
    if any(key not in mesh for key in required_mesh):
        return

    n_cells = _to_int(mesh["nCells"])
    n_edges = _to_int(mesh["nEdges"])
    max_edges = _to_int(mesh["maxEdges"])
    max_adv = int(state["deriv_two"].shape[-1])

    dtype = state["deriv_two"].dtype
    device = state["deriv_two"].device
    int_dtype = mesh["cellsOnEdge"].dtype

    cells_on_edge = mesh["cellsOnEdge"][:n_edges, :2].detach().cpu().numpy()
    cells_on_cell = mesh["cellsOnCell"][:n_cells, :max_edges].detach().cpu().numpy()
    n_edges_on_cell = mesh["nEdgesOnCell"][:n_cells].detach().cpu().numpy()
    deriv_two = state["deriv_two"][:n_edges].detach().cpu().numpy()
    dc_edge = mesh["dcEdge"][:n_edges].detach().cpu().numpy()
    dv_edge = mesh["dvEdge"][:n_edges].detach().cpu().numpy()

    adv_cells_np = np.zeros((n_edges + 1, max_adv), dtype=cells_on_edge.dtype)
    n_adv_cells_np = np.zeros((n_edges + 1,), dtype=cells_on_edge.dtype)
    adv_coefs_np = np.zeros((n_edges + 1, max_adv), dtype=deriv_two.dtype)
    adv_coefs_3rd_np = np.zeros((n_edges + 1, max_adv), dtype=deriv_two.dtype)

    def _neighbor_cells(cell_id: int) -> tuple[int, list[int]]:
        if 1 <= cell_id <= n_cells:
            row = cell_id - 1
            count = int(n_edges_on_cell[row])
            return count, [int(value) for value in cells_on_cell[row, :count]]
        return 0, []

    for edge_row in range(n_edges):
        cell1, cell2 = (int(value) for value in cells_on_edge[edge_row, :2])
        if cell1 > n_cells and cell2 > n_cells:
            continue

        cell_list = [cell1, cell2]
        positions = {cell1: 0, cell2: 1}

        count1, neighbors1 = _neighbor_cells(cell1)
        for neighbor in neighbors1:
            if neighbor != cell2 and neighbor not in positions:
                positions[neighbor] = len(cell_list)
                cell_list.append(neighbor)

        count2, neighbors2 = _neighbor_cells(cell2)
        for neighbor in neighbors2:
            if neighbor not in positions:
                positions[neighbor] = len(cell_list)
                cell_list.append(neighbor)

        n = len(cell_list)
        if n > max_adv:
            raise MpasConfigError(
                f"advCellsForEdge overflow at edge {edge_row + 1}: n={n}, maxAdv={max_adv}"
            )

        n_adv_cells_np[edge_row] = n
        adv_cells_np[edge_row, :n] = cell_list
        edge_adv = adv_coefs_np[edge_row]
        edge_adv_3rd = adv_coefs_3rd_np[edge_row]

        j_in = positions[cell1]
        edge_adv[j_in] = edge_adv[j_in] + deriv_two[edge_row, 0, 0]
        edge_adv_3rd[j_in] = edge_adv_3rd[j_in] + deriv_two[edge_row, 0, 0]
        for i_cell in range(count1):
            j_in = positions[neighbors1[i_cell]]
            coeff = deriv_two[edge_row, 0, i_cell + 1]
            edge_adv[j_in] = edge_adv[j_in] + coeff
            edge_adv_3rd[j_in] = edge_adv_3rd[j_in] + coeff

        j_in = positions[cell2]
        edge_adv[j_in] = edge_adv[j_in] + deriv_two[edge_row, 1, 0]
        edge_adv_3rd[j_in] = edge_adv_3rd[j_in] - deriv_two[edge_row, 1, 0]
        for i_cell in range(count2):
            j_in = positions[neighbors2[i_cell]]
            coeff = deriv_two[edge_row, 1, i_cell + 1]
            edge_adv[j_in] = edge_adv[j_in] + coeff
            edge_adv_3rd[j_in] = edge_adv_3rd[j_in] - coeff

        # Preserve the source statement's multiply-then-divide association:
        # ``-(dcEdge**2) * coefficient / 12``.  Precomputing
        # ``-(dcEdge**2)/12`` changes the rounded static operator and that
        # difference feeds every horizontal transport call.
        dc_squared = dc_edge[edge_row] ** 2
        edge_adv[:n] = -(dc_squared) * edge_adv[:n] / 12.0
        edge_adv_3rd[:n] = -(dc_squared) * edge_adv_3rd[:n] / 12.0

        edge_adv[positions[cell1]] = edge_adv[positions[cell1]] + 0.5
        edge_adv[positions[cell2]] = edge_adv[positions[cell2]] + 0.5

        edge_adv[:n] = edge_adv[:n] * dv_edge[edge_row]
        edge_adv_3rd[:n] = edge_adv_3rd[:n] * dv_edge[edge_row]

    adv_coefs_3rd_np[:n_edges] = adv_coefs_3rd_np[:n_edges] * float(coef_3rd_order)

    adv_cells = torch.as_tensor(adv_cells_np, dtype=int_dtype, device=device)
    n_adv_cells = torch.as_tensor(n_adv_cells_np, dtype=int_dtype, device=device)
    adv_coefs = torch.as_tensor(adv_coefs_np, dtype=dtype, device=device)
    adv_coefs_3rd = torch.as_tensor(adv_coefs_3rd_np, dtype=dtype, device=device)

    for target in (state, mesh):
        target.setdefault("advCellsForEdge", adv_cells)
        target.setdefault("nAdvCellsForEdge", n_adv_cells)
        target.setdefault("adv_coefs", adv_coefs)
        if force_adv_coefs_3rd:
            target["adv_coefs_3rd"] = adv_coefs_3rd
        else:
            target.setdefault("adv_coefs_3rd", adv_coefs_3rd)
    _trim_advective_stencil_fields(state, mesh)


def _trim_advective_stencil_fields(
    state: dict[str, Any],
    mesh: dict[str, Any],
) -> None:
    """Trim trailing inactive Pattern-C advection stencil slots."""
    n_adv_cells = state.get("nAdvCellsForEdge", mesh.get("nAdvCellsForEdge"))
    if not isinstance(n_adv_cells, torch.Tensor) or n_adv_cells.numel() == 0:
        return

    widths = [
        value.shape[1]
        for target in (state, mesh)
        for key in ("advCellsForEdge", "adv_coefs", "adv_coefs_3rd")
        if isinstance((value := target.get(key)), torch.Tensor) and value.ndim >= 2
    ]
    if not widths:
        return

    current_width = min(widths)
    active_width = int(torch.max(n_adv_cells).item())
    target_width = max(1, min(current_width, active_width))
    if active_width > current_width:
        raise MpasConfigError(
            "nAdvCellsForEdge exceeds available advective stencil width: "
            f"active={active_width}, width={current_width}"
        )

    for target in (state, mesh):
        for key in ("advCellsForEdge", "adv_coefs", "adv_coefs_3rd"):
            value = target.get(key)
            if (
                isinstance(value, torch.Tensor)
                and value.ndim >= 2
                and value.shape[1] > target_width
            ):
                target[key] = value[:, :target_width].contiguous()
        target["nAdvCellsMax"] = target_width


def _set_dimension_value(target: dict[str, Any], key: str, value: int) -> None:
    current = target.get(key)
    if isinstance(current, torch.Tensor):
        target[key] = torch.tensor(value, dtype=current.dtype, device=current.device)
    else:
        target[key] = int(value)


def _trim_edge_neighbor_stencil_fields(
    state: dict[str, Any],
    mesh: dict[str, Any],
) -> None:
    """Trim trailing inactive edge-neighbor stencil slots."""
    n_edges_on_edge = state.get("nEdgesOnEdge", mesh.get("nEdgesOnEdge"))
    if not isinstance(n_edges_on_edge, torch.Tensor) or n_edges_on_edge.numel() == 0:
        return

    widths = [
        value.shape[1]
        for target in (state, mesh)
        for key in ("edgesOnEdge", "weightsOnEdge")
        if isinstance((value := target.get(key)), torch.Tensor) and value.ndim >= 2
    ]
    if not widths:
        return

    current_width = min(widths)
    active_width = int(torch.max(n_edges_on_edge).item())
    target_width = max(1, min(current_width, active_width))
    if active_width > current_width:
        raise MpasConfigError(
            "nEdgesOnEdge exceeds available edge-neighbor stencil width: "
            f"active={active_width}, width={current_width}"
        )

    for target in (state, mesh):
        for key in ("edgesOnEdge", "weightsOnEdge"):
            value = target.get(key)
            if (
                isinstance(value, torch.Tensor)
                and value.ndim >= 2
                and value.shape[1] > target_width
            ):
                target[key] = value[:, :target_width].contiguous()
        _set_dimension_value(target, "maxEdges2", target_width)


_CELL_STENCIL_SLOT_FIELDS: tuple[str, ...] = (
    "cellsOnCell",
    "edgesOnCell",
    "verticesOnCell",
    "kiteForCell",
    "edgesOnCell_sign",
    "defc_a",
    "defc_b",
    "zb_cell",
    "zb3_cell",
)


def _trim_coeffs_reconstruct_slots(
    coeffs: torch.Tensor,
    *,
    n_cells: int,
    target_width: int,
) -> torch.Tensor:
    """Trim the maxEdges axis for supported coeffs_reconstruct layouts."""
    if coeffs.ndim != 3:
        return coeffs
    s0, s1, s2 = coeffs.shape
    if s2 == 3 and s0 in (n_cells, n_cells + 1) and s1 > target_width:
        return coeffs[:, :target_width, :].contiguous()
    if s0 == 3 and s2 in (n_cells, n_cells + 1) and s1 > target_width:
        return coeffs[:, :target_width, :].contiguous()
    if s1 == 3 and s0 in (n_cells, n_cells + 1) and s2 > target_width:
        return coeffs[:, :, :target_width].contiguous()
    return coeffs


def _trim_cell_stencil_fields(
    state: dict[str, Any],
    mesh: dict[str, Any],
) -> None:
    """Trim trailing inactive cell-stencil slots and keep maxEdges consistent."""
    n_edges_on_cell = state.get("nEdgesOnCell", mesh.get("nEdgesOnCell"))
    if not isinstance(n_edges_on_cell, torch.Tensor) or n_edges_on_cell.numel() == 0:
        return

    n_cells = _to_int(
        mesh.get("nCells", state.get("nCells", n_edges_on_cell.shape[0] - 1))
    )
    widths = [
        value.shape[1]
        for target in (state, mesh)
        for key in _CELL_STENCIL_SLOT_FIELDS
        if isinstance((value := target.get(key)), torch.Tensor)
        and value.ndim >= 2
        and value.shape[0] in (n_cells, n_cells + 1)
    ]
    if not widths:
        return

    current_width = min(widths)
    active_width = int(torch.max(n_edges_on_cell[:n_cells]).item())
    target_width = max(1, min(current_width, active_width))
    if active_width > current_width:
        raise MpasConfigError(
            "nEdgesOnCell exceeds available cell-stencil width: "
            f"active={active_width}, width={current_width}"
        )

    for target in (state, mesh):
        for key in _CELL_STENCIL_SLOT_FIELDS:
            value = target.get(key)
            if (
                isinstance(value, torch.Tensor)
                and value.ndim >= 2
                and value.shape[0] in (n_cells, n_cells + 1)
                and value.shape[1] > target_width
            ):
                target[key] = value[:, :target_width, ...].contiguous()
        coeffs = target.get("coeffs_reconstruct")
        if isinstance(coeffs, torch.Tensor):
            target["coeffs_reconstruct"] = _trim_coeffs_reconstruct_slots(
                coeffs,
                n_cells=n_cells,
                target_width=target_width,
            )
        _set_dimension_value(target, "maxEdges", target_width)


def apply_coef_3rd_order_override_fields(
    state: dict[str, Any],
    mesh: dict[str, Any],
    *,
    coef_3rd_order: float,
) -> None:
    """Refresh fields whose values depend on ``coef_3rd_order``.

    Upstream MPAS ``atm_couple_coef_3rd_order`` does not rebuild the
    transport stencils. It scales the already-built ``adv_coefs_3rd`` and
    ``zb3_cell`` arrays by the configured coefficient. Preserve that behavior
    here so runtime overrides preserve the same model semantics.

    The helper writes back to both ``state`` and ``mesh`` because runtime callers
    may source these fields from either mapping. Keeping both views synchronized
    avoids a class of CLI / reload edge cases where one side still exposes the
    pre-override coefficient-dependent tensors.
    """
    # This helper also runs from the public forecast boundary, which may use
    # torch.inference_mode(). Its outputs become dynamics-tendency static-plan inputs and
    # must retain version counters for mutation-sensitive cache keys.
    with torch.inference_mode(False):
        _apply_coef_3rd_order_override_fields_impl(
            state, mesh, coef_3rd_order=coef_3rd_order
        )


def _apply_coef_3rd_order_override_fields_impl(
    state: dict[str, Any],
    mesh: dict[str, Any],
    *,
    coef_3rd_order: float,
) -> None:
    has_state_coef = "coef_3rd_order" in state
    has_mesh_coef = "coef_3rd_order" in mesh
    current_coef = float(
        state.get("coef_3rd_order", mesh.get("coef_3rd_order", coef_3rd_order))
    )
    coef = float(coef_3rd_order)
    if abs(current_coef - coef) < 1.0e-15:
        if not has_state_coef:
            state["coef_3rd_order"] = coef
        if not has_mesh_coef:
            mesh["coef_3rd_order"] = coef
        return

    if "_coef_3rd_base_adv_coefs_3rd" not in state:
        current_adv = state.get("adv_coefs_3rd", mesh.get("adv_coefs_3rd"))
        if isinstance(current_adv, torch.Tensor):
            if current_coef != 0.0:
                state["_coef_3rd_base_adv_coefs_3rd"] = current_adv / current_coef
            else:
                state["_coef_3rd_base_adv_coefs_3rd"] = current_adv.clone()
    base_adv = state.get("_coef_3rd_base_adv_coefs_3rd")
    if isinstance(base_adv, torch.Tensor):
        adv_coefs_3rd = base_adv * coef
        state["adv_coefs_3rd"] = adv_coefs_3rd
        if "adv_coefs_3rd" in mesh:
            mesh["adv_coefs_3rd"] = adv_coefs_3rd
    else:
        _synthesize_advective_stencil_fields(
            state,
            mesh,
            coef_3rd_order=coef,
            force_adv_coefs_3rd=True,
        )

    if "_coef_3rd_base_zb3_cell" not in state:
        current_zb3_cell = state.get("zb3_cell", mesh.get("zb3_cell"))
        if isinstance(current_zb3_cell, torch.Tensor):
            if current_coef != 0.0:
                state["_coef_3rd_base_zb3_cell"] = current_zb3_cell / current_coef
            else:
                state["_coef_3rd_base_zb3_cell"] = current_zb3_cell.clone()
    base_zb3_cell = state.get("_coef_3rd_base_zb3_cell")
    if isinstance(base_zb3_cell, torch.Tensor):
        zb3_cell = base_zb3_cell * coef
        state["zb3_cell"] = zb3_cell
        mesh["zb3_cell"] = zb3_cell
    else:
        _synthesize_cell_metric_fields(
            state,
            mesh,
            coef_3rd_order=coef,
            force_zb3_cell=True,
        )
    state["coef_3rd_order"] = coef
    mesh["coef_3rd_order"] = coef


def _synthesize_damping_coefs(
    state: dict[str, Any],
    mesh: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
) -> None:
    """Compute dss when loaded data only carries the pre-init zero field."""
    n_cells = _to_int(mesh.get("nCells", state.get("nCells", 0)))
    n_vert_levels = _to_int(mesh.get("nVertLevels", state.get("nVertLevels", 0)))
    if n_cells == 0 or n_vert_levels == 0:
        return

    existing = state.get("dss", mesh.get("dss"))
    if isinstance(existing, torch.Tensor):
        active = existing[:n_cells, :n_vert_levels]
        if torch.any(active != 0):
            return

    zgrid = state.get("zgrid", mesh.get("zgrid"))
    mesh_density = state.get("meshDensity", mesh.get("meshDensity"))
    if not isinstance(zgrid, torch.Tensor) or not isinstance(
        mesh_density, torch.Tensor
    ):
        return

    cfg = config or {}
    config_zd = float(get_config_value_safe(cfg, "config_zd", 22000.0))
    config_xnutr = float(get_config_value_safe(cfg, "config_xnutr", 0.2))
    computed = compute_damping_coefs(
        zgrid=zgrid[:n_cells, : n_vert_levels + 1],
        mesh_density=mesh_density[:n_cells],
        n_cells=n_cells,
        n_vert_levels=n_vert_levels,
        config_zd=config_zd,
        config_xnutr=config_xnutr,
    )["dss"]

    if isinstance(existing, torch.Tensor):
        padded = existing.clone()
    else:
        padded = torch.zeros(
            (n_cells + 1, n_vert_levels),
            dtype=computed.dtype,
            device=computed.device,
        )
    padded[:n_cells, :n_vert_levels] = computed
    state["dss"] = padded
    mesh["dss"] = padded


def _validate_coeffs_reconstruct(
    coeffs: torch.Tensor,
    n_cells: int,
) -> None:
    """Reject all-zero reconstruction coefficients over the active cells."""
    if not isinstance(coeffs, torch.Tensor):
        return
    if not torch.any(coeffs[:n_cells]):
        raise MpasConfigError(
            "coeffs_reconstruct is all zero for the active cell range. "
            "This typically indicates a static mesh file was loaded without "
            "the corresponding init.nc reconstruction coefficients."
        )


def _active_tensor_missing_or_zero(value: Any, active_count: int) -> bool:
    if not isinstance(value, torch.Tensor):
        return True
    if value.ndim == 0 or value.shape[0] < active_count:
        return True
    return bool(torch.count_nonzero(value[:active_count]).item() == 0)


def _pad_active_tensor(active: torch.Tensor, active_count: int) -> torch.Tensor:
    pad_shape = list(active.shape)
    pad_shape[0] = 1
    pad = torch.zeros(pad_shape, dtype=active.dtype, device=active.device)
    return torch.cat([active[:active_count], pad], dim=0)


def _synthesize_vector_geometry_fields(
    state: dict[str, Any],
    mesh: dict[str, Any],
) -> None:
    n_cells = _to_int(mesh["nCells"])
    n_edges = _to_int(mesh["nEdges"])
    max_edges = _to_int(mesh["maxEdges"])

    # mpas_rbf_interp_initialize always rebuilds these three geometry fields,
    # followed by mpas_init_reconstruct rebuilding coeffs_reconstruct.  Keeping
    # nonzero init.nc copies created a split state: coefficients used fresh
    # geometry while later dynamics observed stale geometry.
    need_local_vertical = True
    need_edge_normals = True
    need_tangent_plane = True
    need_coeffs = True

    if not (
        need_local_vertical or need_edge_normals or need_tangent_plane or need_coeffs
    ):
        return

    local_vertical_active = compute_local_vertical_unit_vectors(
        n_cells=n_cells,
        x_cell=mesh["xCell"],
        y_cell=mesh["yCell"],
        z_cell=mesh["zCell"],
    )
    edge_normals_active = compute_edge_normal_vectors(
        n_cells=n_cells,
        n_edges=n_edges,
        cells_on_edge=mesh["cellsOnEdge"],
        x_cell=mesh["xCell"],
        y_cell=mesh["yCell"],
        z_cell=mesh["zCell"],
        x_edge=mesh["xEdge"],
        y_edge=mesh["yEdge"],
        z_edge=mesh["zEdge"],
    )
    tangent_plane_active = compute_cell_tangent_plane(
        n_cells=n_cells,
        edges_on_cell=mesh["edgesOnCell"],
        x_cell=mesh["xCell"],
        y_cell=mesh["yCell"],
        z_cell=mesh["zCell"],
        edge_normal_vectors=edge_normals_active,
    )

    if need_local_vertical:
        padded = _pad_active_tensor(local_vertical_active, n_cells)
        mesh["localVerticalUnitVectors"] = padded
        state["localVerticalUnitVectors"] = padded
    if need_edge_normals:
        padded = _pad_active_tensor(edge_normals_active, n_edges)
        mesh["edgeNormalVectors"] = padded
        state["edgeNormalVectors"] = padded
    if need_tangent_plane:
        padded = _pad_active_tensor(tangent_plane_active, n_cells)
        mesh["cellTangentPlane"] = padded
        state["cellTangentPlane"] = padded
    if need_coeffs:
        coeffs_active = compute_coeffs_reconstruct(
            n_cells=n_cells,
            n_edges=n_edges,
            max_edges=max_edges,
            edges_on_cell=mesh["edgesOnCell"],
            n_edges_on_cell=mesh["nEdgesOnCell"],
            x_cell=mesh["xCell"],
            y_cell=mesh["yCell"],
            z_cell=mesh["zCell"],
            x_edge=mesh["xEdge"],
            y_edge=mesh["yEdge"],
            z_edge=mesh["zEdge"],
            edge_normal_vectors=edge_normals_active,
            cell_tangent_plane=tangent_plane_active,
        )
        padded = _pad_active_tensor(coeffs_active, n_cells)
        mesh["coeffs_reconstruct"] = padded
        state["coeffs_reconstruct"] = padded
        mesh[COEFFS_RECONSTRUCT_ALL_ZERO_KEY] = False
        state[COEFFS_RECONSTRUCT_ALL_ZERO_KEY] = False


def _to_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def _strip_time_dim(_key: str, value: Any) -> Any:
    if not isinstance(value, torch.Tensor):
        return value
    if value.ndim >= 2 and value.shape[0] == 1:
        return value[0]
    return value


def _infer_dimensions(mesh: dict[str, Any], ic_raw: dict[str, Any]) -> dict[str, Any]:
    if "nCells" not in mesh:
        mesh["nCells"] = torch.tensor(
            int(mesh["cellsOnCell"].shape[0]), dtype=torch.int32
        )
    if "nEdges" not in mesh:
        mesh["nEdges"] = torch.tensor(
            int(mesh["cellsOnEdge"].shape[0]), dtype=torch.int32
        )
    if "nVertices" not in mesh:
        mesh["nVertices"] = torch.tensor(
            int(mesh["cellsOnVertex"].shape[0]), dtype=torch.int32
        )
    if "maxEdges" not in mesh:
        mesh["maxEdges"] = torch.tensor(
            int(mesh["cellsOnCell"].shape[1]), dtype=torch.int32
        )
    if "vertexDegree" not in mesh:
        mesh["vertexDegree"] = torch.tensor(
            int(mesh["cellsOnVertex"].shape[1]), dtype=torch.int32
        )
    if "maxEdges2" not in mesh and "weightsOnEdge" in mesh:
        mesh["maxEdges2"] = torch.tensor(
            int(mesh["weightsOnEdge"].shape[1]), dtype=torch.int32
        )
    if "nVertLevels" not in mesh:
        zz = mesh.get("zz")
        if isinstance(zz, torch.Tensor) and zz.ndim >= 2:
            mesh["nVertLevels"] = torch.tensor(int(zz.shape[-1]), dtype=torch.int32)
        else:
            inferred = _extract_dim(ic_raw, "nVertLevels")
            if inferred is None:
                raise MpasMissingFieldError(fields=("nVertLevels",), source="unknown")
            mesh["nVertLevels"] = torch.tensor(int(inferred), dtype=torch.int32)
    return mesh


def _extract_dim(payload: dict[str, Any], dim_name: str) -> int | None:
    if dim_name in payload:
        value = payload[dim_name]
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            return int(value.item())
        if not isinstance(value, torch.Tensor):
            return int(value)

    candidates: tuple[tuple[str, int | None], ...]
    if dim_name == "nCells":
        candidates = (
            ("cellsOnCell", 0),
            ("edgesOnCell", 0),
            ("verticesOnCell", 0),
            ("areaCell", 0),
            ("theta", 0),
            ("rho", 0),
            ("qv", 0),
            ("scalars", 0),
            ("rho_base", 0),
            ("theta_base", 0),
            ("zz", 0),
        )
    elif dim_name == "nEdges":
        candidates = (
            ("cellsOnEdge", 0),
            ("verticesOnEdge", 0),
            ("dcEdge", 0),
            ("dvEdge", 0),
            ("weightsOnEdge", 0),
            ("u", 0),
            ("fEdge", 0),
        )
    elif dim_name == "nVertices":
        candidates = (
            ("cellsOnVertex", 0),
            ("edgesOnVertex", 0),
            ("areaTriangle", 0),
            ("fVertex", 0),
        )
    elif dim_name == "nVertLevels":
        candidates = (
            ("theta", -1),
            ("rho", -1),
            ("qv", -1),
            ("scalars", 1),
            ("rho_base", -1),
            ("theta_base", -1),
            ("zz", -1),
        )
    else:
        return None

    for key, axis in candidates:
        value = payload.get(key)
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            size = int(value.shape[axis])
            if key == "zz" and dim_name == "nVertLevels" and size > 0:
                return size - 1
            return size
    return None


def _check_mesh_dim_consistency(
    init_raw: dict[str, Any], mesh_raw: dict[str, Any]
) -> None:
    for dim_name in ("nCells", "nEdges", "nVertices", "nVertLevels"):
        init_value = _extract_dim(init_raw, dim_name)
        grid_value = _extract_dim(mesh_raw, dim_name)
        if (
            init_value is not None
            and grid_value is not None
            and init_value != grid_value
        ):
            raise MpasMeshMismatchError(
                dim_name=dim_name,
                init_value=int(init_value),
                grid_value=int(grid_value),
            )


def _check_required_load_fields(
    init_raw: dict[str, Any],
    runtime_mesh: dict[str, Any],
) -> None:
    missing_init = set(_REQUIRED_INIT_FIELDS).difference(init_raw)
    if "qv" not in init_raw and "scalars" not in init_raw:
        missing_init.add("qv|scalars")
    missing_mesh = set(_REQUIRED_MESH_FIELDS).difference(runtime_mesh)

    if missing_init and not missing_mesh:
        raise MpasMissingFieldError(
            fields=tuple(sorted(missing_init)), source="init.nc"
        )
    if missing_mesh and not missing_init:
        raise MpasMissingFieldError(
            fields=tuple(sorted(missing_mesh)), source="grid.nc"
        )
    if missing_init or missing_mesh:
        raise MpasMissingFieldError(
            fields=tuple(sorted(missing_init | missing_mesh)),
            source="unknown",
        )


def _check_required_init_fields(init_raw: dict[str, Any]) -> None:
    missing_init = set(_REQUIRED_INIT_FIELDS).difference(init_raw)
    if "qv" not in init_raw and "scalars" not in init_raw:
        missing_init.add("qv|scalars")
    if missing_init:
        raise MpasMissingFieldError(
            fields=tuple(sorted(missing_init)), source="init.nc"
        )


def _pad_indexed_tensors(
    payload: dict[str, Any],
    *,
    n_cells: int,
    n_edges: int,
    n_vertices: int,
) -> dict[str, Any]:
    padded: dict[str, Any] = {}
    valid_sizes = {n_cells, n_edges, n_vertices}
    for key, value in payload.items():
        if not isinstance(value, torch.Tensor) or value.ndim == 0:
            padded[key] = value
            continue
        if int(value.shape[0]) not in valid_sizes:
            padded[key] = value
            continue
        pad_shape = list(value.shape)
        pad_shape[0] = 1
        pad_row = torch.zeros(pad_shape, dtype=value.dtype, device=value.device)
        padded[key] = torch.cat([value, pad_row], dim=0)
    return padded


def _mesh_support_compatible(
    raw_mesh: dict[str, Any],
    support_mesh: dict[str, Any],
) -> bool:
    """Return True only when support mesh matches the raw grid closely enough.

    A stale ``mesh_complete.nc`` can carry a different connectivity/order than
    the raw ``grid.nc``. In that case the raw grid must remain authoritative.
    """
    exact_keys = (
        "nCells",
        "nEdges",
        "nVertices",
        "maxEdges",
        "vertexDegree",
        "cellsOnEdge",
        "edgesOnCell",
        "cellsOnCell",
        "verticesOnCell",
        "verticesOnEdge",
        "edgesOnVertex",
        "nEdgesOnCell",
        "nEdgesOnEdge",
    )
    float_keys = (
        "dcEdge",
        "dvEdge",
        "latCell",
        "lonCell",
        "latEdge",
        "lonEdge",
        "latVertex",
        "lonVertex",
        "xCell",
        "yCell",
        "zCell",
        "xEdge",
        "yEdge",
        "zEdge",
        "xVertex",
        "yVertex",
        "zVertex",
    )

    for key in exact_keys:
        raw = raw_mesh.get(key)
        support = support_mesh.get(key)
        if (
            isinstance(raw, torch.Tensor)
            and isinstance(support, torch.Tensor)
            and (raw.shape != support.shape or not torch.equal(raw, support))
        ):
            return False
    for key in float_keys:
        raw = raw_mesh.get(key)
        support = support_mesh.get(key)
        if (
            isinstance(raw, torch.Tensor)
            and isinstance(support, torch.Tensor)
            and (
                raw.shape != support.shape
                or not torch.allclose(
                    raw.to(torch.float64),
                    support.to(torch.float64),
                    atol=1.0e-12,
                    rtol=0.0,
                )
            )
        ):
            return False
    return True


def _build_edge_permutation(
    raw_mesh: dict[str, Any],
    runtime_mesh: dict[str, Any],
) -> torch.Tensor:
    n_edges = _to_int(runtime_mesh["nEdges"])
    raw_cells_on_edge = raw_mesh["cellsOnEdge"][:n_edges, :2]
    runtime_cells_on_edge = runtime_mesh["cellsOnEdge"][:n_edges, :2]
    raw_lookup = {
        tuple(sorted((int(a), int(b)))): idx
        for idx, (a, b) in enumerate(raw_cells_on_edge.tolist())
    }
    perm = [
        raw_lookup[tuple(sorted((int(a), int(b))))]
        for a, b in runtime_cells_on_edge.tolist()
    ]
    return torch.tensor(perm, dtype=torch.int64)


def _reorder_edge_tensor(
    tensor: torch.Tensor, perm: torch.Tensor, n_edges: int
) -> torch.Tensor:
    if tensor.ndim == 0:
        return tensor
    if tensor.shape[0] == n_edges + 1:
        return torch.cat([tensor[:-1].index_select(0, perm), tensor[-1:]], dim=0)
    if tensor.shape[0] == n_edges:
        return tensor.index_select(0, perm)
    return tensor


def _copy_missing_fields(target: dict[str, Any], source_paths: list[Path]) -> None:
    sources = [
        load_checkpoint(str(path), validated_fields=_VALIDATED_RAW_FIELDS)
        for path in source_paths
        if path.exists()
    ]
    for key in _SUPPLEMENTAL_FIELDS:
        if key in target:
            continue
        for source in sources:
            if key in source:
                target[key] = _strip_time_dim(key, source[key])
                break


def _convert_ic_fields(state: dict[str, Any]) -> None:
    if "scalars" not in state:
        present = [
            field
            for field in MOISTURE_SCALAR_FIELDS
            if isinstance(state.get(field), torch.Tensor)
        ]
        if present:
            template = state[present[0]]
            scalar_fields = []
            for field in MOISTURE_SCALAR_FIELDS:
                value = state.get(field)
                if isinstance(value, torch.Tensor):
                    scalar_fields.append(value)
                else:
                    scalar_fields.append(torch.zeros_like(template))
            state["scalars"] = torch.stack(scalar_fields, dim=-1)
    if "qv" not in state and "scalars" in state:
        state["qv"] = state["scalars"][..., 0]
    if "num_scalars" not in state and "scalars" in state:
        state["num_scalars"] = int(state["scalars"].shape[-1])


def load_initial_state(
    init_path: str | Path,
    grid_path: str | Path,
    mesh_support_path: str | Path | None = None,
    *,
    device: torch.device | str = "cpu",
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load raw IC and mesh files into runtime-ready state and mesh dictionaries."""
    config = validate_config(config)
    validate_runtime_request(
        device=torch.device(device).type, dtype="float64", physics_mode="dycore"
    )
    init_path = Path(init_path)
    grid_path = Path(grid_path)
    if not init_path.exists():
        raise FileNotFoundError(f"IC file not found: {init_path}")
    if not grid_path.exists():
        raise FileNotFoundError(f"Grid file not found: {grid_path}")

    init_raw = load_checkpoint(str(init_path), validated_fields=_VALIDATED_RAW_FIELDS)
    raw_mesh = load_checkpoint(str(grid_path), validated_fields=_VALIDATED_RAW_FIELDS)
    _check_required_init_fields(init_raw)
    _check_mesh_dim_consistency(init_raw, raw_mesh)
    runtime_mesh = copy.deepcopy(raw_mesh)
    raw_mesh = _infer_dimensions(raw_mesh, init_raw)

    support_path: Path | None = None
    if mesh_support_path is not None:
        support_path = Path(mesh_support_path)
        if not support_path.exists():
            raise FileNotFoundError(f"Mesh support file not found: {support_path}")
        support_mesh = load_checkpoint(
            str(support_path), validated_fields=_VALIDATED_RAW_FIELDS
        )
        support_mesh = _infer_dimensions(support_mesh, init_raw)
        if _mesh_support_compatible(raw_mesh, support_mesh):
            runtime_mesh = copy.deepcopy(support_mesh)
            for key, value in raw_mesh.items():
                runtime_mesh.setdefault(key, value)
        else:
            warnings.warn(
                f"Ignoring incompatible mesh support file {support_path}; raw grid will be used instead.",
                UserWarning,
                stacklevel=2,
            )
            support_path = None
    if support_path is None:
        for key, value in raw_mesh.items():
            if key in init_raw:
                runtime_mesh[key] = _strip_time_dim(key, init_raw[key])

    for key in _MESH_FROM_INIT_FIELDS:
        if key not in runtime_mesh and key in init_raw:
            runtime_mesh[key] = _strip_time_dim(key, init_raw[key])

    _check_required_load_fields(init_raw, runtime_mesh)
    runtime_mesh = _infer_dimensions(runtime_mesh, init_raw)

    n_cells = _to_int(runtime_mesh["nCells"])
    n_edges = _to_int(runtime_mesh["nEdges"])
    n_vertices = _to_int(runtime_mesh["nVertices"])

    state = {key: _strip_time_dim(key, value) for key, value in init_raw.items()}
    for key in _DERIVED_INIT_FIELDS:
        state.pop(key, None)
    if (
        support_path is not None
        and "cellsOnEdge" in raw_mesh
        and "cellsOnEdge" in runtime_mesh
    ):
        perm = _build_edge_permutation(raw_mesh, runtime_mesh)
        for key in _EDGE_ORDER_FIELDS:
            value = state.get(key)
            if isinstance(value, torch.Tensor):
                state[key] = _reorder_edge_tensor(value, perm, n_edges)

    source_paths = [init_path, grid_path]
    if support_path is not None:
        source_paths.append(support_path)
    _copy_missing_fields(state, source_paths)
    _convert_ic_fields(state)

    padded_state = _pad_indexed_tensors(
        state,
        n_cells=n_cells,
        n_edges=n_edges,
        n_vertices=n_vertices,
    )
    padded_state[CELL_PADDING_OFFSET_KEY] = 0
    padded_mesh = _pad_indexed_tensors(
        runtime_mesh,
        n_cells=n_cells,
        n_edges=n_edges,
        n_vertices=n_vertices,
    )
    padded_mesh[CELL_PADDING_OFFSET_KEY] = 0

    if "bdyMaskCell" not in padded_state:
        padded_state["bdyMaskCell"] = torch.zeros(n_cells + 1, dtype=torch.int32)
    if "bdyMaskEdge" not in padded_state:
        padded_state["bdyMaskEdge"] = torch.zeros(n_edges + 1, dtype=torch.int32)
    _synthesize_exact_mesh_fields(padded_mesh)

    combined = dict(padded_mesh)
    combined.update(padded_state)
    if "edgesOnCell_sign" not in combined or "edgesOnVertex_sign" not in combined:
        signs = compute_edge_signs(combined)
        padded_state["edgesOnCell_sign"] = signs["edgesOnCell_sign"]
        padded_state["edgesOnVertex_sign"] = signs["edgesOnVertex_sign"]
        padded_mesh["edgesOnCell_sign"] = signs["edgesOnCell_sign"]
        padded_mesh["edgesOnVertex_sign"] = signs["edgesOnVertex_sign"]

    _trim_cell_stencil_fields(padded_state, padded_mesh)

    scaling = compute_mesh_scaling(padded_mesh, config=config)
    padded_state.setdefault("meshScalingDel2", scaling["meshScalingDel2"])
    padded_state.setdefault("meshScalingDel4", scaling["meshScalingDel4"])
    padded_state.setdefault(
        "meshScalingRegionalCell", scaling["meshScalingRegionalCell"]
    )
    padded_state.setdefault(
        "meshScalingRegionalEdge", scaling["meshScalingRegionalEdge"]
    )
    padded_mesh.setdefault("meshScalingDel2", scaling["meshScalingDel2"])
    padded_mesh.setdefault("meshScalingDel4", scaling["meshScalingDel4"])
    padded_mesh.setdefault(
        "meshScalingRegionalCell", scaling["meshScalingRegionalCell"]
    )
    padded_mesh.setdefault(
        "meshScalingRegionalEdge", scaling["meshScalingRegionalEdge"]
    )

    for key in _INTEGER_FIELD_KEYS:
        if key in padded_mesh and isinstance(padded_mesh[key], torch.Tensor):
            padded_mesh[key] = padded_mesh[key].to(dtype=torch.int64)
        if key in padded_state and isinstance(padded_state[key], torch.Tensor):
            padded_state[key] = padded_state[key].to(dtype=torch.int64)

    for key in _DIMENSION_KEYS:
        if key in padded_mesh:
            padded_state.setdefault(key, padded_mesh[key])

    num_scalars = int(padded_state.get("num_scalars", 1))
    padded_state["moist_start"] = int(padded_state.get("moist_start", 1))
    padded_state["moist_end"] = int(padded_state.get("moist_end", num_scalars))
    padded_state["coef_3rd_order"] = float(padded_state.get("coef_3rd_order", 0.25))
    padded_state["c_s"] = float(
        config.get(
            "config_smagorinsky_coef",
            config.get("c_s", padded_state.get("c_s", 0.125)),
        )
    )

    _synthesize_advective_stencil_fields(
        padded_state,
        padded_mesh,
        coef_3rd_order=float(padded_state["coef_3rd_order"]),
    )
    _trim_edge_neighbor_stencil_fields(padded_state, padded_mesh)
    _synthesize_cell_metric_fields(
        padded_state,
        padded_mesh,
        coef_3rd_order=float(padded_state["coef_3rd_order"]),
    )
    _synthesize_damping_coefs(padded_state, padded_mesh, config=config)
    _synthesize_vector_geometry_fields(padded_state, padded_mesh)

    for key, value in padded_mesh.items():
        padded_state[key] = value

    if "coeffs_reconstruct" in padded_state:
        _validate_coeffs_reconstruct(padded_state["coeffs_reconstruct"], n_cells)
        padded_state[COEFFS_RECONSTRUCT_ALL_ZERO_KEY] = False
        padded_mesh[COEFFS_RECONSTRUCT_ALL_ZERO_KEY] = False

    required_state = ["u", "w", "theta", "rho", "scalars"]
    missing = [key for key in required_state if key not in padded_state]
    if missing:
        raise MpasMissingFieldError(fields=tuple(sorted(missing)), source="init.nc")
    validate_no_injection(padded_state)

    target_device = torch.device(device)
    padded_mesh = {
        key: value.to(device=target_device)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in padded_mesh.items()
    }
    padded_state = {
        key: value.to(device=target_device)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in padded_state.items()
    }
    return padded_state, padded_mesh


def load_config_from_namelist(
    namelist_path: str | Path,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Load and validate an MPAS namelist for direct dycore execution."""
    namelist_path = Path(namelist_path)
    if not namelist_path.exists():
        raise FileNotFoundError(f"Namelist file not found: {namelist_path}")

    raw = parse_namelist(namelist_path)
    allowed_keys = (
        set(FORTRAN_DEFAULTS)
        | set(CONFIG_ALIASES)
        | set(CONFIG_ALIASES.values())
        | set(PHYSICS_NAMELIST_KEYS)
        | set(known_unwired_namelist_keys())
    )
    unknown_keys = sorted(key for key in raw if key not in allowed_keys)
    if unknown_keys:
        message = (
            f"Unknown namelist keys in {namelist_path}: {unknown_keys}. "
            "These keys are not wired into mpas-pytorch and will be ignored."
        )
        if strict:
            raise MpasStrictModeError(message)
        warnings.warn(message, UserWarning, stacklevel=2)
    unsupported_active = active_unsupported_namelist_keys(raw)
    if unsupported_active:
        message = (
            f"Unsupported active MPAS namelist settings in {namelist_path}: {unsupported_active}. "
            "The current PyTorch package is dycore-only for these controls."
        )
        raise MpasUnsupportedCapabilityError(
            message,
            capability="namelist",
            requested=tuple(unsupported_active),
        )
    filtered = {
        key: value
        for key, value in raw.items()
        if key
        in set(FORTRAN_DEFAULTS) | set(CONFIG_ALIASES) | set(CONFIG_ALIASES.values())
    }
    config = validate_config(filtered)
    config.update(_resolve_physics_namelist_config(raw))
    substeps = int(
        config.get("config_number_of_sub_steps", config.get("nAcousticSteps", 2))
    )
    split_steps = int(config.get("config_dynamics_split_steps", 3))
    config["config_number_of_sub_steps"] = substeps
    config["config_dynamics_split_steps"] = split_steps
    config["config_split_dynamics_transport"] = bool(
        config.get("config_split_dynamics_transport", True)
    )
    config["config_time_integration_order"] = int(
        config.get("config_time_integration_order", 2)
    )
    config["config_dt"] = float(config.get("config_dt", 720.0))
    config["nAcousticSteps"] = int(config.get("nAcousticSteps", substeps))
    config["strict"] = True
    config["strict_initialization"] = True
    config["_mpas_config_validated"] = True
    return config


def _resolve_physics_namelist_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve explicit physics switches and resolve suite-selected schemes."""

    out = {key: raw[key] for key in PHYSICS_NAMELIST_KEYS if key in raw}
    suite = str(raw.get("config_physics_suite", "")).strip().strip("'\"").lower()
    if suite == "mesoscale_reference":
        for key, suite_value in MESOSCALE_REFERENCE_SCHEMES.items():
            value = out.get(key)
            if value is None or str(value).strip().strip("'\"").lower() == "suite":
                out[key] = suite_value
    return out
