"""I/O for the package's dycore restart contract."""

from __future__ import annotations

import hashlib
import json
from numbers import Number
from pathlib import Path
from typing import Any

import netCDF4 as nc
import numpy as np
import torch

from mpas_pytorch.exceptions import (
    MpasConfigError,
    MpasDeviceError,
    MpasMeshMismatchError,
    MpasMissingFieldError,
)
from mpas_pytorch.io_schema import (
    PACKAGE_VERSION,
    RESTART_SCHEMA,
    RESTART_SCHEMA_VERSION,
    UPSTREAM_COMMIT,
    UPSTREAM_VERSION,
    configuration_digest,
    format_mpas_time,
    initial_time_for_valid,
    mesh_digest,
    parse_mpas_time,
    tensor_device_and_dtype,
    validate_runtime_metadata,
    validate_scientific_array,
)
from mpas_pytorch.state import (
    _boundary_prognostic_digest,
    has_complete_step_boundary,
    mark_complete_step_boundary,
)
from mpas_pytorch.utils.atomic_output import atomic_output

_REGISTRY_RESTART_FIELDS: tuple[str, ...] = (
    "u",
    "w",
    "rho_zz",
    "theta_m",
    "scalars",
    "pressure_p",
    "pressure_base",
    "rho_base",
    "theta_base",
    "rho",
    "theta",
    "exner",
    "exner_base",
    "rtheta_base",
    "ru",
    "rw",
    "rho_p",
    "rtheta_p",
    "u_init",
    "v_init",
    "t_init",
    "qv_init",
)

_OPTIONAL_RESTART_SNAPSHOT_FIELDS: tuple[str, ...] = (
    "v",
    "vorticity",
    "divergence",
    "ke",
    "pv_edge",
    "rho_edge",
    "ur_cell",
    "vr_cell",
    "uReconstructX",
    "uReconstructY",
    "uReconstructZ",
    "uReconstructZonal",
    "uReconstructMeridional",
)

_OPTIONAL_RESTART_SUPPORT_FIELDS: tuple[str, ...] = (
    "fEdge",
    "rdzu",
    "deriv_two",
    "defc_a",
    "defc_b",
    "zgrid",
    "zxu",
    "cf1",
    "cf2",
    "cf3",
)

_RESTART_CARRY_FIELDS: tuple[str, ...] = (
    "ru_p",
    "rw_p",
    "rho_pp",
    "rtheta_pp",
    "rtheta_pp_old",
)
_RESTART_METADATA_FIELDS: tuple[str, ...] = (
    "_edge_ordering_mismatch",
    "_mesh_connectivity_preferred",
)

_SUBSTEP_LOCAL_FIELDS: tuple[str, ...] = _RESTART_CARRY_FIELDS

_EDGE_LEVEL_FIELDS = frozenset({"u", "ru", "ru_p", "v", "rho_edge"})
_CELL_LEVEL_FIELDS = frozenset(
    {
        "rho_zz",
        "theta_m",
        "pressure_p",
        "pressure_base",
        "rho_base",
        "theta_base",
        "rho",
        "theta",
        "exner",
        "exner_base",
        "rtheta_base",
        "rho_p",
        "rtheta_p",
        "rho_pp",
        "rtheta_pp",
        "rtheta_pp_old",
        "t_init",
        "divergence",
        "ke",
        "ur_cell",
        "vr_cell",
        "uReconstructX",
        "uReconstructY",
        "uReconstructZ",
        "uReconstructZonal",
        "uReconstructMeridional",
    }
)
_CELL_W_FIELDS = frozenset({"w", "rw", "rw_p"})
_VERTEX_LEVEL_FIELDS = frozenset({"vorticity"})
_EDGE_LEVEL_NO_TIME_FIELDS = frozenset({"pv_edge"})
_LEVEL_PROFILE_FIELDS = frozenset({"u_init", "v_init", "qv_init"})
_EDGE_SCALAR_FIELDS = frozenset({"fEdge"})
_EDGE_LEVEL_NO_TIME_2D_FIELDS = frozenset({"zxu"})
_CELL_LEVEL_NO_TIME_FIELDS = frozenset({"defc_a", "defc_b"})
_CELL_W_NO_TIME_FIELDS = frozenset({"zgrid"})
_EDGE_EXTRA_3D_FIELDS = frozenset({"deriv_two"})
_LEVEL_PROFILE_FIELDS = _LEVEL_PROFILE_FIELDS | frozenset({"rdzu"})
_SCALAR_NO_TIME_FIELDS = frozenset({"cf1", "cf2", "cf3"})


def _n_cells(mesh: dict[str, Any], state: dict[str, Any]) -> int:
    return int(mesh.get("nCells", state.get("nCells", 0)))


def _n_edges(mesh: dict[str, Any], state: dict[str, Any]) -> int:
    return int(mesh.get("nEdges", state.get("nEdges", 0)))


def _n_vertices(mesh: dict[str, Any], state: dict[str, Any]) -> int:
    return int(mesh.get("nVertices", state.get("nVertices", 0)))


def _n_vert_levels(mesh: dict[str, Any], state: dict[str, Any]) -> int:
    return int(mesh.get("nVertLevels", state.get("nVertLevels", 0)))


def _require_restart_fields(state: dict[str, Any], mesh: dict[str, Any]) -> None:
    missing = [
        key for key in _REGISTRY_RESTART_FIELDS if key not in state and key not in mesh
    ]
    if missing:
        raise MpasMissingFieldError(
            fields=tuple(sorted(missing)),
            source="restart state",
            message=f"Missing required restart fields from restart state: {tuple(sorted(missing))}",
        )


def _ensure_not_mid_rk3(state: dict[str, Any]) -> None:
    has_boundary_snapshot = all(
        isinstance(state.get(key), torch.Tensor)
        for key in _OPTIONAL_RESTART_SNAPSHOT_FIELDS
    ) and all(isinstance(state.get(key), torch.Tensor) for key in ("ruAvg", "wwAvg"))
    # Post-step boundary states may legitimately carry non-zero RK bookkeeping
    # tensors while also carrying the fully re-derived diagnostic snapshot. In
    # that case we serialize the boundary snapshot and the reader resumes from
    # it directly. Reject only the ambiguous case: non-zero substep tensors
    # without the matching boundary snapshot, which indicates a true mid-RK3
    # savepoint.
    for key in _SUBSTEP_LOCAL_FIELDS:
        value = state.get(key)
        if not isinstance(value, torch.Tensor):
            continue
        if torch.count_nonzero(value).item() != 0 and not has_boundary_snapshot:
            raise MpasConfigError(
                f"write_restart cannot serialize mid-RK3 substep state: field {key!r} is non-zero."
            )


def _encode_char64(text: str) -> np.ndarray:
    if len(text) > 64:
        raise MpasConfigError(f"Restart time string exceeds 64 chars: {text!r}")
    try:
        payload = text.ljust(64).encode("ascii")
    except UnicodeEncodeError as exc:
        raise MpasConfigError(f"Restart time string must be ASCII: {text!r}") from exc
    return np.frombuffer(payload, dtype="S1").reshape(1, 64)


def _as_numpy_f64(tensor: torch.Tensor) -> np.ndarray:
    return validate_scientific_array("restart field", tensor)


def _as_numpy_preserve_dtype(tensor: torch.Tensor) -> np.ndarray:
    cpu_tensor = tensor.detach().to(device="cpu")
    if cpu_tensor.dtype == torch.bool:
        cpu_tensor = cpu_tensor.to(dtype=torch.int8)
    return cpu_tensor.numpy()


def _is_restart_scalar_value(value: Any) -> bool:
    return isinstance(value, (bool, Number, np.generic))


def _as_numpy_scalar_f64(value: Any) -> np.ndarray:
    if isinstance(value, bool):
        value = int(value)
    return np.asarray(value, dtype=np.float64)


def _active_slice(
    name: str,
    tensor: torch.Tensor,
    *,
    nCells: int,
    nEdges: int,
    nVertices: int,
    nVertLevels: int,
) -> torch.Tensor:
    if name in _EDGE_LEVEL_FIELDS:
        return tensor[:nEdges, :nVertLevels]
    if name in _CELL_LEVEL_FIELDS:
        return tensor[:nCells, :nVertLevels]
    if name in _CELL_W_FIELDS:
        return tensor[:nCells, : nVertLevels + 1]
    if name in _VERTEX_LEVEL_FIELDS:
        return tensor[:nVertices, :nVertLevels]
    if name in _EDGE_LEVEL_NO_TIME_FIELDS:
        return tensor[:nEdges, :nVertLevels]
    if name in _EDGE_SCALAR_FIELDS:
        return tensor[:nEdges]
    if name in _EDGE_LEVEL_NO_TIME_2D_FIELDS:
        return tensor[:nEdges, :nVertLevels]
    if name in _CELL_LEVEL_NO_TIME_FIELDS:
        return tensor[:nCells, :]
    if name in _CELL_W_NO_TIME_FIELDS:
        return tensor[:nCells, : nVertLevels + 1]
    if name in _EDGE_EXTRA_3D_FIELDS:
        return tensor[:nEdges, ...]
    if name in _LEVEL_PROFILE_FIELDS:
        return tensor[:nVertLevels]
    if name == "scalars":
        return tensor[:nCells, :nVertLevels, :]
    return tensor


def _dims_for_field(name: str, *, nScalars: int) -> tuple[str, ...]:
    if name in _EDGE_LEVEL_FIELDS:
        return ("Time", "nEdges", "nVertLevels")
    if name in _CELL_LEVEL_FIELDS:
        return ("Time", "nCells", "nVertLevels")
    if name in _CELL_W_FIELDS:
        return ("Time", "nCells", "nVertLevelsP1")
    if name in _VERTEX_LEVEL_FIELDS:
        return ("Time", "nVertices", "nVertLevels")
    if name in _EDGE_LEVEL_NO_TIME_FIELDS:
        return ("Time", "nEdges", "nVertLevels")
    if name in _EDGE_SCALAR_FIELDS:
        return ("nEdges",)
    if name in _EDGE_LEVEL_NO_TIME_2D_FIELDS:
        return ("nEdges", "nVertLevels")
    if name in _CELL_LEVEL_NO_TIME_FIELDS:
        return ("nCells", "maxEdges")
    if name in _CELL_W_NO_TIME_FIELDS:
        return ("nCells", "nVertLevelsP1")
    if name in _EDGE_EXTRA_3D_FIELDS:
        return ("nEdges", "nDerivComp", "nDerivStencil")
    if name in _LEVEL_PROFILE_FIELDS:
        return ("nVertLevels",)
    if name in _SCALAR_NO_TIME_FIELDS:
        return ()
    if name == "scalars":
        return ("Time", "nCells", "nVertLevels", "nScalars")
    raise KeyError(name)


def _extra_restart_tensor_fields(state: dict[str, Any]) -> tuple[str, ...]:
    known = (
        set(_REGISTRY_RESTART_FIELDS)
        | set(_OPTIONAL_RESTART_SNAPSHOT_FIELDS)
        | set(_OPTIONAL_RESTART_SUPPORT_FIELDS)
        | set(_SUBSTEP_LOCAL_FIELDS)
        | {"xtime", "initial_time"}
        | set(_RESTART_METADATA_FIELDS)
    )
    return tuple(
        key
        for key, value in state.items()
        if key not in known
        and (not key.startswith("_") or key in _RESTART_METADATA_FIELDS)
        and isinstance(value, torch.Tensor)
    )


def _extra_restart_scalar_fields(state: dict[str, Any]) -> tuple[str, ...]:
    known = (
        set(_REGISTRY_RESTART_FIELDS)
        | set(_OPTIONAL_RESTART_SNAPSHOT_FIELDS)
        | set(_OPTIONAL_RESTART_SUPPORT_FIELDS)
        | set(_SUBSTEP_LOCAL_FIELDS)
        | {"xtime", "initial_time"}
        | set(_RESTART_METADATA_FIELDS)
    )
    return tuple(
        key
        for key, value in state.items()
        if key not in known
        and (not key.startswith("_") or key in _RESTART_METADATA_FIELDS)
        and _is_restart_scalar_value(value)
    )


def _dims_for_extra_field(
    ds: nc.Dataset, name: str, tensor: torch.Tensor
) -> tuple[str, ...]:
    dims: list[str] = []
    for axis, size in enumerate(tensor.shape):
        dim_name = f"{name}__dim{axis}"
        if dim_name not in ds.dimensions:
            ds.createDimension(dim_name, int(size))
        dims.append(dim_name)
    return tuple(dims)


def _restart_payload_digest(ds: nc.Dataset) -> str:
    """Bind the entire serialized payload and metadata, excluding this digest."""
    digest = hashlib.sha256()
    for name in sorted(ds.ncattrs()):
        if name == "restart_payload_sha256":
            continue
        value = ds.getncattr(name)
        if isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, np.generic):
            value = value.item()
        digest.update(json.dumps([name, value], allow_nan=False).encode())
    for name, dimension in sorted(ds.dimensions.items()):
        digest.update(
            json.dumps([name, len(dimension), dimension.isunlimited()]).encode()
        )
    for name, variable in sorted(ds.variables.items()):
        values = variable[:]
        if np.ma.isMaskedArray(values) and np.ma.getmaskarray(values).any():
            raise MpasConfigError(f"restart field {name!r} has missing data")
        array = np.ascontiguousarray(values)
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise MpasConfigError(f"restart field {name!r} has nonfinite data")
        digest.update(
            json.dumps(
                [name, variable.dimensions, str(array.dtype), array.shape]
            ).encode()
        )
        digest.update(array.tobytes())
        for attr in sorted(variable.ncattrs()):
            digest.update(repr((attr, variable.getncattr(attr))).encode())
    return digest.hexdigest()


def write_restart(
    state: dict[str, Any],
    mesh: dict[str, Any],
    path: str | Path,
    xtime: str,
    *,
    initial_time: str | None = None,
    config: dict[str, Any] | None = None,
    dt: float | None = None,
    model_time_seconds: float = 0.0,
    source_provenance: str = "caller-provided in-memory state",
    clobber: bool = False,
) -> None:
    """Emit a package dycore restart NetCDF at ``path``."""
    restart_path = Path(path)
    if restart_path.exists() and not clobber:
        raise FileExistsError(f"Restart file already exists: {restart_path}")

    _require_restart_fields(state, mesh)
    _ensure_not_mid_rk3(state)
    complete_boundary = has_complete_step_boundary(state)
    if "_restart_skip_rederivation" in state and not complete_boundary:
        raise MpasConfigError(
            "write_restart cannot save an edited timestep boundary with stale diagnostics; "
            "advance it through run_timestep before saving the resulting complete boundary."
        )
    if complete_boundary and not has_complete_step_boundary(
        state, mesh=mesh, config=config, dt=dt
    ):
        raise MpasConfigError("complete boundary continuation identity mismatch")
    if complete_boundary:
        missing = [
            name
            for name in _OPTIONAL_RESTART_SNAPSHOT_FIELDS + ("ruAvg", "wwAvg")
            if not isinstance(state.get(name), torch.Tensor)
        ]
        if missing:
            raise MpasMissingFieldError(
                fields=missing, source="complete restart boundary"
            )

    nCells = _n_cells(mesh, state)
    nEdges = _n_edges(mesh, state)
    nVertices = _n_vertices(mesh, state)
    nVertLevels = _n_vert_levels(mesh, state)
    scalars = state["scalars"]
    nScalars = int(scalars.shape[-1])

    boundary_identity = state.get("_restart_skip_rederivation", {})
    try:
        timestep = float(
            dt
            if dt is not None
            else (config or {}).get("config_dt", boundary_identity.get("dt") or 0.0)
        )
    except (TypeError, ValueError) as exc:
        raise MpasConfigError("restart timestep is malformed") from exc
    if (
        not np.isfinite(timestep)
        or timestep < 0.0
        or (complete_boundary and timestep == 0.0)
    ):
        raise MpasConfigError(
            "restart timestep must be finite and positive for continuation"
        )
    if not np.isfinite(model_time_seconds) or model_time_seconds < 0.0:
        raise MpasConfigError("model_time_seconds must be finite and nonnegative")
    if dt is not None and (not np.isfinite(dt) or dt <= 0.0):
        raise MpasConfigError("restart timestep must be finite and positive")
    try:
        xtime = format_mpas_time(parse_mpas_time(xtime))
        initial_time = format_mpas_time(
            parse_mpas_time(
                initial_time
                if initial_time is not None
                else initial_time_for_valid(xtime, model_time_seconds)
            )
        )
    except ValueError as exc:
        raise MpasConfigError(str(exc)) from exc

    with atomic_output(restart_path, clobber=clobber) as temporary:
        with nc.Dataset(str(temporary), "w", format="NETCDF4") as ds:
            device_name, dtype_name = tensor_device_and_dtype(state)
            ds.restart_schema = RESTART_SCHEMA
            ds.restart_schema_version = RESTART_SCHEMA_VERSION
            ds.package_version = PACKAGE_VERSION
            ds.upstream_version = UPSTREAM_VERSION
            ds.upstream_commit = UPSTREAM_COMMIT
            ds.runtime_dtype = dtype_name
            ds.runtime_device = device_name
            ds.model_time_seconds = float(model_time_seconds)
            ds.timestep_seconds = timestep
            ds.configuration_sha256 = configuration_digest(config)
            if complete_boundary:
                ds.continuation_configuration_sha256 = boundary_identity["config"]
            ds.mesh_sha256 = mesh_digest(mesh)
            ds.source_provenance = source_provenance
            ds.complete_step_boundary = int(complete_boundary)
            if complete_boundary:
                ds.boundary_prognostics_sha256 = _boundary_prognostic_digest(state)
            ds.createDimension("Time", 1)
            ds.createDimension("nCells", nCells)
            ds.createDimension("nEdges", nEdges)
            ds.createDimension("nVertices", nVertices)
            ds.createDimension("nVertLevels", nVertLevels)
            ds.createDimension("nVertLevelsP1", nVertLevels + 1)
            ds.createDimension("nScalars", nScalars)
            ds.createDimension("StrLen", 64)

            deriv_two = state.get("deriv_two", mesh.get("deriv_two"))
            if isinstance(deriv_two, torch.Tensor):
                active_deriv = _active_slice(
                    "deriv_two",
                    deriv_two,
                    nCells=nCells,
                    nEdges=nEdges,
                    nVertices=nVertices,
                    nVertLevels=nVertLevels,
                )
                ds.createDimension("nDerivComp", int(active_deriv.shape[1]))
                ds.createDimension("nDerivStencil", int(active_deriv.shape[2]))
            if "maxEdges" not in ds.dimensions:
                max_edges = int(mesh.get("maxEdges", state.get("maxEdges", 0)))
                if max_edges > 0:
                    ds.createDimension("maxEdges", max_edges)

            explicit_fields = _REGISTRY_RESTART_FIELDS + tuple(
                field
                for field in (
                    _OPTIONAL_RESTART_SNAPSHOT_FIELDS
                    + _OPTIONAL_RESTART_SUPPORT_FIELDS
                    + _RESTART_CARRY_FIELDS
                    + _RESTART_METADATA_FIELDS
                )
                if field in state or field in mesh
            )
            extra_fields = _extra_restart_tensor_fields(state)
            extra_scalar_fields = _extra_restart_scalar_fields(state)

            for name in explicit_fields + extra_fields + extra_scalar_fields:
                value = state.get(name, mesh.get(name))
                if name in _RESTART_METADATA_FIELDS and _is_restart_scalar_value(value):
                    scalar_array = np.asarray(bool(value), dtype=np.int8)
                    var = ds.createVariable(name, scalar_array.dtype, ())
                    var[...] = scalar_array
                    continue
                if (
                    name in explicit_fields
                    and name in _SCALAR_NO_TIME_FIELDS
                    and _is_restart_scalar_value(value)
                ):
                    scalar_array = _as_numpy_scalar_f64(value)
                    dims = _dims_for_field(name, nScalars=nScalars)
                    var = ds.createVariable(name, scalar_array.dtype, dims)
                    var[...] = scalar_array
                    continue
                if name in extra_scalar_fields and _is_restart_scalar_value(value):
                    scalar_array = np.asarray(value)
                    if scalar_array.dtype.kind == "b":
                        scalar_array = scalar_array.astype(np.int8)
                    var = ds.createVariable(name, scalar_array.dtype, ())
                    var[...] = scalar_array
                    continue
                tensor = value
                if not isinstance(tensor, torch.Tensor):
                    if name in _REGISTRY_RESTART_FIELDS:
                        raise MpasMissingFieldError(
                            fields=(name,),
                            source="restart state",
                            message=f"Missing required restart fields from restart state: {(name,)}",
                        )
                    continue
                if name in explicit_fields:
                    active_tensor = _active_slice(
                        name,
                        tensor,
                        nCells=nCells,
                        nEdges=nEdges,
                        nVertices=nVertices,
                        nVertLevels=nVertLevels,
                    )
                    dims = _dims_for_field(name, nScalars=nScalars)
                else:
                    active_tensor = tensor
                    dims = _dims_for_extra_field(ds, name, tensor)
                values = (
                    _as_numpy_f64(active_tensor)
                    if name in explicit_fields
                    else _as_numpy_preserve_dtype(active_tensor)
                )
                var_dtype = values.dtype
                if values.dtype.kind in "fciu":
                    validate_scientific_array(
                        name, values, float64=values.dtype.kind in "fc"
                    )
                expected_shape = tuple(
                    len(ds.dimensions[key])
                    for key in (dims[1:] if dims and dims[0] == "Time" else dims)
                )
                if values.shape != expected_shape:
                    raise MpasConfigError(
                        f"restart field {name!r} has malformed shape {values.shape}; expected {expected_shape}"
                    )
                var = ds.createVariable(name, var_dtype, dims)
                if dims and dims[0] == "Time":
                    var[0, ...] = values
                else:
                    var[...] = values

            xtime_var = ds.createVariable("xtime", "S1", ("Time", "StrLen"))
            xtime_var[:, :] = _encode_char64(xtime)
            initial_time_var = ds.createVariable("initial_time", "S1", ("StrLen",))
            initial_time_var[:] = _encode_char64(initial_time)[0]
            ds.restart_payload_sha256 = _restart_payload_digest(ds)
        read_restart(temporary, mesh=mesh, config=config, dt=dt)


def _decode_char_var(var: nc.Variable) -> str:
    chars = nc.chartostring(var[:])
    if np.asarray(chars).ndim == 0:
        return str(np.asarray(chars).item()).rstrip()
    return str(np.asarray(chars)[0]).rstrip()


def _pad_loaded_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    nCells: int,
    nEdges: int,
    nVertices: int,
    nVertLevels: int,
) -> torch.Tensor:
    if name in _EDGE_LEVEL_FIELDS:
        padded = torch.zeros(
            (nEdges + 1, nVertLevels), dtype=tensor.dtype, device=tensor.device
        )
        padded[:nEdges, :nVertLevels] = tensor
        return padded
    if name in _CELL_LEVEL_FIELDS:
        padded = torch.zeros(
            (nCells + 1, nVertLevels), dtype=tensor.dtype, device=tensor.device
        )
        padded[:nCells, :nVertLevels] = tensor
        return padded
    if name in _CELL_W_FIELDS:
        padded = torch.zeros(
            (nCells + 1, nVertLevels + 1), dtype=tensor.dtype, device=tensor.device
        )
        padded[:nCells, : nVertLevels + 1] = tensor
        return padded
    if name in _VERTEX_LEVEL_FIELDS:
        padded = torch.zeros(
            (nVertices + 1, nVertLevels), dtype=tensor.dtype, device=tensor.device
        )
        padded[:nVertices, :nVertLevels] = tensor
        return padded
    if name in _EDGE_LEVEL_NO_TIME_FIELDS:
        padded = torch.zeros(
            (nEdges + 1, nVertLevels), dtype=tensor.dtype, device=tensor.device
        )
        padded[:nEdges, :nVertLevels] = tensor
        return padded
    if name in _EDGE_SCALAR_FIELDS:
        padded = torch.zeros((nEdges + 1,), dtype=tensor.dtype, device=tensor.device)
        padded[:nEdges] = tensor
        return padded
    if name in _EDGE_LEVEL_NO_TIME_2D_FIELDS:
        padded = torch.zeros(
            (nEdges + 1, nVertLevels), dtype=tensor.dtype, device=tensor.device
        )
        padded[:nEdges, :nVertLevels] = tensor
        return padded
    if name in _CELL_LEVEL_NO_TIME_FIELDS:
        padded = torch.zeros(
            (nCells + 1, tensor.shape[1]), dtype=tensor.dtype, device=tensor.device
        )
        padded[:nCells, :] = tensor
        return padded
    if name in _CELL_W_NO_TIME_FIELDS:
        padded = torch.zeros(
            (nCells + 1, nVertLevels + 1), dtype=tensor.dtype, device=tensor.device
        )
        padded[:nCells, : nVertLevels + 1] = tensor
        return padded
    if name in _EDGE_EXTRA_3D_FIELDS:
        padded = torch.zeros(
            (nEdges + 1, tensor.shape[1], tensor.shape[2]),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        padded[:nEdges, :, :] = tensor
        return padded
    if name in _LEVEL_PROFILE_FIELDS:
        return tensor
    if name == "scalars":
        padded = torch.zeros(
            (nCells + 1, nVertLevels, tensor.shape[-1]),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        padded[:nCells, :nVertLevels, :] = tensor
        return padded
    return tensor


def read_restart(
    path: str | Path,
    *,
    mesh: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    dt: float | None = None,
    device: torch.device | str = "cpu",
) -> tuple[dict[str, Any], str]:
    """Load a restart NetCDF into a state dict."""
    restart_path = Path(path)
    if not restart_path.exists():
        raise FileNotFoundError(f"Restart file not found: {restart_path}")

    try:
        target_device = torch.device(device)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise MpasDeviceError(
            "Restart loading requires a CPU device", requested=str(device)
        ) from exc
    if target_device.type != "cpu":
        raise MpasDeviceError(
            "Restart loading supports CPU float64 only", requested=str(device)
        )
    with nc.Dataset(str(restart_path), "r") as ds:
        schema_version = getattr(ds, "restart_schema_version", None)
        if schema_version is None:
            raise MpasConfigError("restart schema version is missing")
        if (
            not isinstance(schema_version, (int, np.integer))
            or schema_version != RESTART_SCHEMA_VERSION
        ):
            raise MpasConfigError(
                f"unsupported restart schema version {schema_version!r}; "
                f"expected {RESTART_SCHEMA_VERSION}"
            )
        if getattr(ds, "restart_schema", None) != RESTART_SCHEMA:
            raise MpasConfigError("restart schema name is missing or malformed")
        for attr in (
            "package_version",
            "upstream_version",
            "upstream_commit",
            "runtime_dtype",
            "runtime_device",
            "model_time_seconds",
            "timestep_seconds",
            "configuration_sha256",
            "mesh_sha256",
            "source_provenance",
        ):
            if attr not in ds.ncattrs():
                raise MpasConfigError(f"restart attribute {attr!r} is missing")
        complete_boundary = getattr(ds, "complete_step_boundary", None)
        if not isinstance(
            complete_boundary, (int, np.integer)
        ) or complete_boundary not in (0, 1):
            raise MpasConfigError("restart complete-step boundary flag is malformed")
        digest_attributes = ["configuration_sha256", "mesh_sha256"]
        if complete_boundary:
            digest_attributes.extend(
                ("continuation_configuration_sha256", "boundary_prognostics_sha256")
            )
        for attr in digest_attributes:
            value = getattr(ds, attr, None)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise MpasConfigError(
                    f"restart digest {attr!r} is missing or malformed"
                )
        validate_runtime_metadata(ds, complete_boundary=bool(complete_boundary))
        if config is not None and ds.configuration_sha256 != configuration_digest(
            config
        ):
            raise MpasConfigError("restart configuration digest mismatch")
        if dt is not None and float(ds.timestep_seconds) != float(dt):
            raise MpasConfigError("restart timestep mismatch")
        missing = [
            name for name in _REGISTRY_RESTART_FIELDS if name not in ds.variables
        ]
        if missing:
            raise MpasMissingFieldError(
                fields=tuple(sorted(missing)),
                source=str(restart_path),
                message=f"Missing required restart fields from {restart_path}: {tuple(sorted(missing))}",
            )
        snapshot_fields = _OPTIONAL_RESTART_SNAPSHOT_FIELDS + ("ruAvg", "wwAvg")
        if getattr(ds, "complete_step_boundary", 0):
            missing = [name for name in snapshot_fields if name not in ds.variables]
            if missing:
                raise MpasMissingFieldError(
                    fields=missing, source="complete restart boundary"
                )
        if "xtime" not in ds.variables:
            raise MpasMissingFieldError(
                fields=("xtime",),
                source=str(restart_path),
                message=f"Missing required restart fields from {restart_path}: {('xtime',)}",
            )
        if "initial_time" not in ds.variables:
            raise MpasMissingFieldError(
                fields=("initial_time",),
                source=str(restart_path),
                message=(
                    f"Missing required restart fields from {restart_path}: "
                    f"{('initial_time',)}"
                ),
            )
        try:
            xtime = _decode_char_var(ds.variables["xtime"])
            initial_time = _decode_char_var(ds.variables["initial_time"])
            valid_datetime = parse_mpas_time(xtime)
            initial_datetime = parse_mpas_time(initial_time)
            expected_seconds = (valid_datetime - initial_datetime).total_seconds()
            if expected_seconds != float(ds.model_time_seconds):
                raise MpasConfigError(
                    "restart time metadata is malformed: xtime, initial_time and "
                    "model_time_seconds are inconsistent"
                )
            if float(ds.model_time_seconds) < 0.0:
                raise MpasConfigError(
                    "restart time metadata is malformed: model_time_seconds is negative"
                )
            if float(ds.timestep_seconds) < 0.0:
                raise MpasConfigError(
                    "restart time metadata is malformed: timestep_seconds is negative"
                )
        except (TypeError, ValueError) as exc:
            raise MpasConfigError(f"restart time metadata is malformed: {exc}") from exc

        try:
            actual_payload_digest = _restart_payload_digest(ds)
        except (ValueError, TypeError) as exc:
            raise MpasConfigError(
                f"restart payload metadata is malformed: {exc}"
            ) from exc
        if getattr(ds, "restart_payload_sha256", None) != actual_payload_digest:
            raise MpasConfigError("restart boundary payload integrity digest mismatch")

        for name in (
            "Time",
            "nCells",
            "nEdges",
            "nVertices",
            "nVertLevels",
            "nVertLevelsP1",
            "nScalars",
            "StrLen",
        ):
            if name not in ds.dimensions or len(ds.dimensions[name]) <= 0:
                raise MpasConfigError(f"restart dimension {name!r} is missing or empty")
        if len(ds.dimensions["StrLen"]) != 64:
            raise MpasConfigError("restart time string dimension is malformed")
        if ds.variables["xtime"].dimensions != ("Time", "StrLen") or ds.variables[
            "initial_time"
        ].dimensions != ("StrLen",):
            raise MpasConfigError("restart time string dimensions are malformed")
        nCells = len(ds.dimensions["nCells"])
        nEdges = len(ds.dimensions["nEdges"])
        nVertices = (
            len(ds.dimensions["nVertices"]) if "nVertices" in ds.dimensions else 0
        )
        nVertLevels = len(ds.dimensions["nVertLevels"])
        nScalars = len(ds.dimensions["nScalars"]) if "nScalars" in ds.dimensions else 1
        if min(nCells, nEdges, nVertices, nVertLevels, nScalars) <= 0:
            raise MpasConfigError("restart dimensions must be positive")
        if (
            len(ds.dimensions.get("Time", ())) != 1
            or len(ds.dimensions.get("nVertLevelsP1", ())) != nVertLevels + 1
        ):
            raise MpasConfigError(
                "restart time or vertical face dimension is malformed"
            )

        state: dict[str, Any] = {
            "nCells": nCells,
            "nEdges": nEdges,
            "nVertLevels": nVertLevels,
            "nVertices": nVertices,
            "num_scalars": nScalars,
            "_restart_model_time_seconds": float(ds.model_time_seconds),
        }
        if mesh is not None:
            for dim_name, restart_value, mesh_value in (
                ("nCells", nCells, int(mesh.get("nCells", nCells))),
                ("nEdges", nEdges, int(mesh.get("nEdges", nEdges))),
                ("nVertLevels", nVertLevels, int(mesh.get("nVertLevels", nVertLevels))),
            ):
                if restart_value != mesh_value:
                    raise MpasMeshMismatchError(
                        dim_name=dim_name,
                        init_value=restart_value,
                        grid_value=mesh_value,
                    )
            expected_mesh_digest = mesh_digest(mesh)
            if ds.mesh_sha256 != expected_mesh_digest:
                raise MpasMeshMismatchError("restart mesh digest mismatch")
            for key, value in mesh.items():
                state.setdefault(key, value)
        for name, variable in ds.variables.items():
            if name in {"xtime", "initial_time"}:
                continue
            values = variable[:]
            if (
                name
                in _REGISTRY_RESTART_FIELDS
                + _OPTIONAL_RESTART_SNAPSHOT_FIELDS
                + _OPTIONAL_RESTART_SUPPORT_FIELDS
                + _RESTART_CARRY_FIELDS
                and variable.dimensions != _dims_for_field(name, nScalars=nScalars)
            ):
                raise MpasConfigError(
                    f"restart field {name!r} dimensions are malformed"
                )
            if np.dtype(variable.dtype).kind in "fciu":
                validate_scientific_array(
                    name, values, float64=np.dtype(variable.dtype).kind in "fc"
                )
            array = np.asarray(values)
            if array.dtype.kind in {"S", "U"}:
                continue
            tensor = torch.as_tensor(array, device=target_device)
            if tensor.ndim == 0:
                if name in _RESTART_METADATA_FIELDS:
                    state[name] = bool(tensor.item())
                    continue
                if array.dtype.kind in {"f"}:
                    state[name] = float(tensor.item())
                else:
                    state[name] = int(tensor.item())
                continue
            if (
                tensor.ndim > 0
                and name not in _LEVEL_PROFILE_FIELDS
                and variable.dimensions
                and variable.dimensions[0] == "Time"
                and tensor.shape[0] == 1
                and name != "scalars"
            ):
                tensor = tensor[0, ...]
            if name == "scalars" and tensor.ndim == 4 and tensor.shape[0] == 1:
                tensor = tensor[0, ...]
            padded = _pad_loaded_tensor(
                name,
                tensor,
                nCells=nCells,
                nEdges=nEdges,
                nVertices=nVertices,
                nVertLevels=nVertLevels,
            )
            state[name] = padded

        rho_zz = state.get("rho_zz")
        if (
            isinstance(rho_zz, torch.Tensor)
            and rho_zz.ndim == 2
            and rho_zz.shape[0] > nCells
        ):
            rho_zz = rho_zz.clone()
            rho_zz[nCells, :] = 1.0
            state["rho_zz"] = rho_zz
        xtime = _decode_char_var(ds.variables["xtime"])
        state["initial_time"] = _decode_char_var(ds.variables["initial_time"])
        snapshot_fields = _OPTIONAL_RESTART_SNAPSHOT_FIELDS + ("ruAvg", "wwAvg")
        if complete_boundary:
            missing = [name for name in snapshot_fields if name not in state]
            if missing:
                raise MpasMissingFieldError(
                    fields=missing, source="complete restart boundary"
                )
            if "complete_step_boundary" in ds.ncattrs():
                expected = getattr(ds, "boundary_prognostics_sha256", None)
                if expected != _boundary_prognostic_digest(state):
                    raise MpasConfigError("restart boundary prognostic digest mismatch")
            mark_complete_step_boundary(state)
            identity = state["_restart_skip_rederivation"]
            identity["mesh"] = ds.mesh_sha256
            identity["config"] = ds.continuation_configuration_sha256
            identity["dt"] = float(ds.timestep_seconds)
    return state, xtime


__all__ = ["read_restart", "write_restart"]
