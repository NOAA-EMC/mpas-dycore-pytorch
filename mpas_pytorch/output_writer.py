"""Writer for the package's partial dycore history contract."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from mpas_pytorch.constants import PhysicalConstants
from mpas_pytorch.exceptions import MpasConfigError, MpasMissingFieldError
from mpas_pytorch.io_schema import (
    HISTORY_FIELD_METADATA,
    HISTORY_SCHEMA,
    PACKAGE_VERSION,
    UPSTREAM_COMMIT,
    UPSTREAM_VERSION,
    configuration_digest,
    format_mpas_time,
    initial_time_for_valid,
    parse_mpas_time,
    tensor_device_and_dtype,
    validate_runtime_metadata,
    validate_scientific_array,
)
from mpas_pytorch.utils.atomic_output import atomic_output

RVORD = PhysicalConstants.RVORD
MOISTURE_FIELDS = ("qv", "qc", "qr", "qi", "qs", "qg")
SCALAR_METADATA = {
    name: (("Time", "nCells", "nVertLevels"), "kg kg-1", f"{name} mixing ratio")
    for name in MOISTURE_FIELDS[1:]
}


def history_scalar_fields(state: dict[str, Any]) -> tuple[str, ...]:
    scalars = state.get("scalars")
    if isinstance(scalars, torch.Tensor):
        count = scalars.shape[-1]
        if count > len(MOISTURE_FIELDS):
            raise MpasConfigError(
                "Named history scalar expansion supports qv/qc/qr/qi/qs/qg only"
            )
        return MOISTURE_FIELDS[:count]
    return tuple(name for name in MOISTURE_FIELDS if name in state)


def _field_metadata(name: str) -> tuple[tuple[str, ...], str, str] | None:
    return HISTORY_FIELD_METADATA.get(name, SCALAR_METADATA.get(name))


def validate_history(filepath: str | Path) -> None:
    """Validate the complete public diagnostic history schema."""

    import netCDF4 as nc

    with nc.Dataset(str(filepath), "r") as ds:
        required_attrs = {
            "history_schema": HISTORY_SCHEMA,
            "package_version": PACKAGE_VERSION,
            "upstream_version": UPSTREAM_VERSION,
            "upstream_commit": UPSTREAM_COMMIT,
        }
        for name, expected in required_attrs.items():
            if getattr(ds, name, None) != expected:
                raise MpasConfigError(f"history {name} mismatch or missing")
        for name in (
            "configuration_sha256",
            "source_provenance",
            "runtime_dtype",
            "runtime_device",
            "timestep_seconds",
        ):
            if name not in ds.ncattrs():
                raise MpasConfigError(f"history attribute {name!r} is missing")
        validate_runtime_metadata(ds)
        for name in ("nCells", "nEdges", "nVertices", "nVertLevels"):
            if name not in ds.dimensions or len(ds.dimensions[name]) <= 0:
                raise MpasConfigError(f"history dimension {name!r} is missing or empty")
        if (
            len(ds.dimensions.get("nVertLevelsP1", ()))
            != len(ds.dimensions["nVertLevels"]) + 1
        ):
            raise MpasConfigError("history vertical face dimension is malformed")
        if "time" not in ds.variables or "xtime" not in ds.variables:
            raise MpasConfigError("history time/xtime coordinate is missing")
        time = ds.variables["time"]
        if (
            time.dimensions != ("Time",)
            or not getattr(time, "units", "")
            or getattr(time, "standard_name", None) != "time"
        ):
            raise MpasConfigError("history time coordinate is malformed")
        if getattr(time, "calendar", None) != "gregorian":
            raise MpasConfigError("history time calendar is malformed")
        if ds.variables["xtime"].dimensions != ("Time", "StrLen"):
            raise MpasConfigError("history xtime dimensions are malformed")
        try:
            values = validate_scientific_array("history time", time[:])
            if values.size != 1 or not np.isfinite(values).all() or values[0] < 0.0:
                raise ValueError("time value must be one finite nonnegative number")
            decoded = cast(
                Any,
                nc.num2date(
                    values[0],
                    units=time.units,
                    calendar=time.calendar,
                    only_use_cftime_datetimes=False,
                ),
            )
            decoded_time = datetime(
                decoded.year,
                decoded.month,
                decoded.day,
                decoded.hour,
                decoded.minute,
                decoded.second,
                tzinfo=UTC,
            )
            xtime = str(
                np.asarray(nc.chartostring(ds.variables["xtime"][:]))[0]
            ).strip()
            if format_mpas_time(decoded_time) != format_mpas_time(
                parse_mpas_time(xtime)
            ):
                raise ValueError("numeric time and xtime disagree")
        except (IndexError, TypeError, ValueError, OverflowError) as exc:
            raise MpasConfigError(
                f"history time coordinate is malformed: {exc}"
            ) from exc
        data_fields = set(ds.variables) - {"time", "xtime"}
        if not data_fields:
            raise MpasConfigError("history contains no advertised diagnostic fields")
        for name, variable in ds.variables.items():
            if name in {"time", "xtime"}:
                continue
            metadata = _field_metadata(name)
            if metadata is None:
                raise MpasConfigError(f"history field {name!r} is not advertised")
            validate_scientific_array(name, variable[:])
            dimensions, units, long_name = metadata
            if variable.dimensions != dimensions:
                raise MpasConfigError(
                    f"history field {name!r} dimensions are malformed"
                )
            if getattr(variable, "units", None) != units:
                raise MpasConfigError(
                    f"history field {name!r} units are missing or wrong"
                )
            if getattr(variable, "long_name", None) != long_name:
                raise MpasConfigError(
                    f"history field {name!r} long_name is missing or wrong"
                )


def read_history(filepath: str | Path, *, validate: bool = True) -> dict[str, Any]:
    """Read one MPAS-PyTorch diagnostic history file.

    The returned mapping has three keys: ``fields`` contains NumPy arrays,
    ``dimensions`` contains dimension sizes (``None`` for unlimited
    dimensions), and ``attributes`` contains global attributes.  No source
    checkout or implicit search path is consulted.
    """

    import netCDF4 as nc

    history_path = Path(filepath)
    if validate:
        validate_history(history_path)
    with nc.Dataset(str(history_path), "r") as ds:
        fields = {
            name: np.asarray(variable[:]) for name, variable in ds.variables.items()
        }
        dimensions = {
            name: None if dimension.isunlimited() else len(dimension)
            for name, dimension in ds.dimensions.items()
        }
        attributes = {name: ds.getncattr(name) for name in ds.ncattrs()}
    return {
        "fields": fields,
        "dimensions": dimensions,
        "attributes": attributes,
    }


def _strip_padding(tensor: torch.Tensor, n_valid: int) -> torch.Tensor:
    if tensor.shape[0] == n_valid + 1:
        return tensor[:n_valid]
    return tensor[:n_valid]


def _extract_qv(
    combined: dict[str, Any], n_cells: int, n_vert_levels: int
) -> torch.Tensor:
    if "scalars" in combined:
        scalars = _strip_padding(combined["scalars"], n_cells)
        return scalars[:, :n_vert_levels, 0]
    if "qv" in combined:
        return _strip_padding(combined["qv"], n_cells)[:, :n_vert_levels]
    raise MpasMissingFieldError(fields=("scalars|qv",), source="history state")


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return validate_scientific_array("history field", tensor)


def _extract_history_fields(
    state: dict[str, Any],
    mesh: dict[str, Any],
    requested_fields: list[str] | None = None,
) -> dict[str, np.ndarray]:
    combined = dict(mesh)
    combined.update(state)

    n_cells = int(combined["nCells"])
    n_edges = int(combined["nEdges"])
    n_vertices = int(combined["nVertices"])
    n_vert_levels = int(combined["nVertLevels"])

    if requested_fields is not None:
        expanded: list[str] = []
        for name in requested_fields:
            expanded.extend(
                history_scalar_fields(combined) if name == "scalars" else (name,)
            )
        requested_fields = list(dict.fromkeys(expanded))
        unknown = [name for name in requested_fields if _field_metadata(name) is None]
        if unknown:
            raise MpasConfigError(f"Unsupported history fields: {unknown}")
    requested = set(requested_fields) if requested_fields is not None else None

    def wants(name: str) -> bool:
        return requested is None or name in requested

    required_inputs = {
        "u": ("u",),
        "w": ("w",),
        "theta": ("theta_m",),
        "rho": ("rho_zz", "zz"),
        "zz": ("zz",),
    }
    missing_inputs = sorted(
        {
            key
            for name, keys in required_inputs.items()
            if wants(name)
            for key in keys
            if key not in combined
        }
    )
    if missing_inputs:
        raise MpasMissingFieldError(fields=missing_inputs, source="history state")

    fields: dict[str, np.ndarray] = {}
    if wants("u"):
        fields["u"] = _as_numpy(
            _strip_padding(combined["u"], n_edges)[:, :n_vert_levels]
        )
    if wants("w"):
        fields["w"] = _as_numpy(
            _strip_padding(combined["w"], n_cells)[:, : n_vert_levels + 1]
        )
    if wants("theta"):
        qv = _extract_qv(combined, n_cells, n_vert_levels)
        theta_m = _strip_padding(combined["theta_m"], n_cells)[:, :n_vert_levels]
        fields["theta"] = _as_numpy(theta_m / (1.0 + RVORD * qv))
    if wants("rho"):
        rho_zz = _strip_padding(combined["rho_zz"], n_cells)[:, :n_vert_levels]
        zz = _strip_padding(combined["zz"], n_cells)[:, :n_vert_levels]
        fields["rho"] = _as_numpy(rho_zz * zz)
    if wants("zz"):
        fields["zz"] = _as_numpy(
            _strip_padding(combined["zz"], n_cells)[:, :n_vert_levels]
        )
    if wants("qv"):
        fields["qv"] = _as_numpy(_extract_qv(combined, n_cells, n_vert_levels))

    if wants("pressure") and "pressure_p" in combined and "pressure_base" in combined:
        pressure = (
            _strip_padding(combined["pressure_p"], n_cells)[:, :n_vert_levels]
            + _strip_padding(combined["pressure_base"], n_cells)[:, :n_vert_levels]
        )
        fields["pressure"] = _as_numpy(pressure)
    if wants("divergence") and "divergence" in combined:
        fields["divergence"] = _as_numpy(
            _strip_padding(combined["divergence"], n_cells)[:, :n_vert_levels]
        )
    if wants("ke") and "ke" in combined:
        fields["ke"] = _as_numpy(
            _strip_padding(combined["ke"], n_cells)[:, :n_vert_levels]
        )
    if wants("vorticity") and "vorticity" in combined:
        fields["vorticity"] = _as_numpy(
            _strip_padding(combined["vorticity"], n_vertices)[:, :n_vert_levels]
        )
    for field_name, alias in (
        ("uReconstructZonal", "ur_cell"),
        ("uReconstructMeridional", "vr_cell"),
    ):
        if not wants(field_name):
            continue
        source = combined.get(field_name)
        if source is None:
            source = combined.get(alias)
        if source is not None:
            fields[field_name] = _as_numpy(
                _strip_padding(source, n_cells)[:, :n_vert_levels]
            )
    for index, name in enumerate(MOISTURE_FIELDS[1:], start=1):
        if requested is None or name not in requested:
            continue
        if name in history_scalar_fields(combined):
            source = combined.get("scalars")
            value = (
                source[..., index]
                if isinstance(source, torch.Tensor)
                else combined[name]
            )
            fields[name] = _as_numpy(_strip_padding(value, n_cells)[:, :n_vert_levels])
    if requested_fields is not None:
        missing = [name for name in requested_fields if name not in fields]
        if missing:
            raise MpasMissingFieldError(fields=missing, source="history state")
    return fields


def write_history(
    filepath: str | Path,
    state: dict[str, Any],
    mesh: dict[str, Any],
    *,
    xtime: str = "",
    global_attrs: dict[str, Any] | None = None,
    fields: list[str] | None = None,
    config: dict[str, Any] | None = None,
    dt: float | None = None,
    model_time_seconds: float = 0.0,
    initial_time: str | None = None,
    source_provenance: str = "caller-provided in-memory state",
    clobber: bool = False,
) -> None:
    """Write a single package history file for the selected dycore fields."""
    import netCDF4 as nc

    filepath = Path(filepath)

    combined = dict(mesh)
    combined.update(state)
    n_cells = int(combined["nCells"])
    n_edges = int(combined["nEdges"])
    n_vertices = int(combined["nVertices"])
    n_vert_levels = int(combined["nVertLevels"])
    history_fields = _extract_history_fields(state, mesh, requested_fields=fields)

    try:
        valid_xtime = format_mpas_time(parse_mpas_time(xtime or "1970-01-01_00:00:00"))
        if not np.isfinite(model_time_seconds) or model_time_seconds < 0.0:
            raise MpasConfigError("model_time_seconds must be finite and nonnegative")
        reference_time = format_mpas_time(
            parse_mpas_time(
                initial_time or initial_time_for_valid(valid_xtime, model_time_seconds)
            )
        )
    except (ValueError, OverflowError) as exc:
        raise MpasConfigError(str(exc)) from exc
    timestep = float(dt if dt is not None else (config or {}).get("config_dt", 0.0))
    if not np.isfinite(timestep) or timestep < 0.0:
        raise MpasConfigError("history timestep must be finite and nonnegative")
    reserved = {
        "history_schema",
        "package_version",
        "upstream_version",
        "upstream_commit",
        "configuration_sha256",
        "source_provenance",
        "runtime_dtype",
        "runtime_device",
        "timestep_seconds",
    }
    conflict = set(global_attrs or {}) & reserved
    if conflict:
        raise MpasConfigError(f"Reserved global attributes: {sorted(conflict)}")
    if not history_fields:
        raise MpasConfigError("History must contain at least one diagnostic field")
    dimension_sizes = {
        "Time": 1,
        "nCells": n_cells,
        "nEdges": n_edges,
        "nVertices": n_vertices,
        "nVertLevels": n_vert_levels,
        "nVertLevelsP1": n_vert_levels + 1,
    }
    for name, values in history_fields.items():
        metadata = _field_metadata(name)
        assert metadata is not None
        expected_shape = tuple(dimension_sizes[key] for key in metadata[0][1:])
        if values.shape != expected_shape:
            raise MpasConfigError(
                f"history field {name!r} shape {values.shape} differs from {expected_shape}"
            )
    with atomic_output(filepath, clobber=clobber) as temporary:
        ds = nc.Dataset(str(temporary), "w", format="NETCDF4")
        try:
            ds.createDimension("Time", None)
            ds.createDimension("nCells", n_cells)
            ds.createDimension("nEdges", n_edges)
            ds.createDimension("nVertices", n_vertices)
            ds.createDimension("nVertLevels", n_vert_levels)
            ds.createDimension("nVertLevelsP1", n_vert_levels + 1)
            ds.createDimension("StrLen", 64)

            for name, values in history_fields.items():
                metadata = _field_metadata(name)
                assert metadata is not None
                dimensions, units, long_name = metadata
                var = ds.createVariable(name, "f8", dimensions)
                var.units = units
                var.long_name = long_name
                var[0, ...] = values

            time = ds.createVariable("time", "f8", ("Time",))
            time.units = f"seconds since {reference_time.replace('_', ' ')}"
            time.calendar = "gregorian"
            time.standard_name = "time"
            time.long_name = "model time"
            time[0] = float(model_time_seconds)
            xtime_var = ds.createVariable("xtime", "S1", ("Time", "StrLen"))
            encoded = np.full((1, 64), b" ", dtype="S1")
            raw_xtime = valid_xtime.encode("ascii")
            encoded[0, : len(raw_xtime)] = np.frombuffer(raw_xtime, dtype="S1")
            xtime_var[:, :] = encoded
            xtime_var.long_name = "model valid time"

            ds.model_name = "mpas"
            ds.core_name = "atmosphere"
            ds.source = "MPAS-Atmosphere PyTorch"
            ds.Conventions = "CF-1.10, MPAS"
            device, dtype = tensor_device_and_dtype(state)
            ds.history_schema = HISTORY_SCHEMA
            ds.package_version = PACKAGE_VERSION
            ds.upstream_version = UPSTREAM_VERSION
            ds.upstream_commit = UPSTREAM_COMMIT
            ds.runtime_dtype = dtype
            ds.runtime_device = device
            ds.timestep_seconds = float(
                dt if dt is not None else (config or {}).get("config_dt", 0.0)
            )
            ds.configuration_sha256 = configuration_digest(config)
            ds.source_provenance = source_provenance
            if global_attrs:
                for name, value in global_attrs.items():
                    if name in {
                        "history_schema",
                        "package_version",
                        "upstream_version",
                        "upstream_commit",
                        "configuration_sha256",
                    }:
                        raise MpasConfigError(f"global attribute {name!r} is reserved")
                    setattr(ds, name, value)
        finally:
            ds.close()

        validate_history(temporary)


def write_history_fields(
    filepath: str | Path,
    history_fields: dict[str, Any],
    mesh: dict[str, Any],
    *,
    xtime: str = "",
    global_attrs: dict[str, Any] | None = None,
    fields: list[str] | None = None,
) -> None:
    """Write a history file from already materialized full-physics fields."""

    import netCDF4 as nc

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    n_cells = int(mesh["nCells"])
    n_edges = int(mesh["nEdges"])
    n_vertices = int(mesh.get("nVertices", 0))
    n_vert_levels = int(mesh["nVertLevels"])
    selected = dict(history_fields)
    if fields is not None:
        selected = {name: value for name, value in selected.items() if name in fields}

    arrays = {name: _as_numpy(value) for name, value in selected.items()}
    n_soil_levels = _infer_soil_levels(arrays)

    ds = nc.Dataset(str(filepath), "w", format="NETCDF4")
    try:
        ds.createDimension("Time", None)
        ds.createDimension("nCells", n_cells)
        ds.createDimension("nEdges", n_edges)
        ds.createDimension("nVertices", n_vertices)
        ds.createDimension("nVertLevels", n_vert_levels)
        ds.createDimension("nVertLevelsP1", n_vert_levels + 1)
        if n_soil_levels is not None:
            ds.createDimension("nSoilLevels", n_soil_levels)

        for name, values in arrays.items():
            dims = _history_dims_for_array(
                name,
                values,
                n_cells=n_cells,
                n_edges=n_edges,
                n_vertices=n_vertices,
                n_vert_levels=n_vert_levels,
                n_soil_levels=n_soil_levels,
            )
            if dims is None:
                continue
            var = ds.createVariable(name, "f8", dims)
            var[0, ...] = _trim_history_values(
                values,
                dims=dims,
                n_cells=n_cells,
                n_edges=n_edges,
                n_vertices=n_vertices,
                n_vert_levels=n_vert_levels,
                n_soil_levels=n_soil_levels,
            )

        ds.model_name = "mpas"
        ds.core_name = "atmosphere"
        ds.source = "MPAS-Atmosphere PyTorch full physics"
        ds.Conventions = "MPAS"
        if xtime:
            ds.xtime = xtime
        if global_attrs:
            for name, value in global_attrs.items():
                setattr(ds, name, value)
    finally:
        ds.close()


def _infer_soil_levels(arrays: dict[str, np.ndarray]) -> int | None:
    for name in ("smois", "sh2o", "tslb"):
        value = arrays.get(name)
        if value is not None and value.ndim == 2:
            return int(value.shape[1])
    return None


def _history_dims_for_array(
    name: str,
    values: np.ndarray,
    *,
    n_cells: int,
    n_edges: int,
    n_vertices: int,
    n_vert_levels: int,
    n_soil_levels: int | None,
) -> tuple[str, ...] | None:
    if values.ndim == 1:
        if values.shape[0] >= n_edges and name == "u":
            return ("Time", "nEdges")
        if values.shape[0] >= n_vertices and name == "vorticity":
            return ("Time", "nVertices")
        if values.shape[0] >= n_cells:
            return ("Time", "nCells")
    if values.ndim == 2:
        if (
            values.shape[0] >= n_edges
            and values.shape[1] >= n_vert_levels
            and name == "u"
        ):
            return ("Time", "nEdges", "nVertLevels")
        if (
            values.shape[0] >= n_vertices
            and values.shape[1] >= n_vert_levels
            and name == "vorticity"
        ):
            return ("Time", "nVertices", "nVertLevels")
        if (
            values.shape[0] >= n_cells
            and values.shape[1] >= n_vert_levels + 1
            and name == "w"
        ):
            return ("Time", "nCells", "nVertLevelsP1")
        if (
            n_soil_levels is not None
            and values.shape[0] >= n_cells
            and values.shape[1] == n_soil_levels
            and name in {"smois", "sh2o", "tslb"}
        ):
            return ("Time", "nCells", "nSoilLevels")
        if values.shape[0] >= n_cells and values.shape[1] >= n_vert_levels:
            return ("Time", "nCells", "nVertLevels")
    return None


def _trim_history_values(
    values: np.ndarray,
    *,
    dims: tuple[str, ...],
    n_cells: int,
    n_edges: int,
    n_vertices: int,
    n_vert_levels: int,
    n_soil_levels: int | None,
) -> np.ndarray:
    if dims == ("Time", "nCells"):
        return values[:n_cells]
    if dims == ("Time", "nEdges"):
        return values[:n_edges]
    if dims == ("Time", "nVertices"):
        return values[:n_vertices]
    if dims == ("Time", "nEdges", "nVertLevels"):
        return values[:n_edges, :n_vert_levels]
    if dims == ("Time", "nVertices", "nVertLevels"):
        return values[:n_vertices, :n_vert_levels]
    if dims == ("Time", "nCells", "nVertLevelsP1"):
        return values[:n_cells, : n_vert_levels + 1]
    if dims == ("Time", "nCells", "nSoilLevels"):
        return values[:n_cells, :n_soil_levels]
    if dims == ("Time", "nCells", "nVertLevels"):
        return values[:n_cells, :n_vert_levels]
    return values
