"""Package-owned metadata and digests for public history/restart contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mpas_pytorch import __version__
from mpas_pytorch.exceptions import MpasConfigError

HISTORY_SCHEMA = "MPAS-PyTorch diagnostic history schema v1"
RESTART_SCHEMA = "MPAS-PyTorch restart schema v1"
RESTART_SCHEMA_VERSION = 1
PACKAGE_VERSION = __version__
UPSTREAM_VERSION = "v8.3.1"
UPSTREAM_COMMIT = "b9090a1434b8ceb2a92fc985f4d97e7947cd95a6"

HISTORY_FIELD_METADATA: dict[str, tuple[tuple[str, ...], str, str]] = {
    "u": (
        ("Time", "nEdges", "nVertLevels"),
        "m s-1",
        "Horizontal normal velocity at edges",
    ),
    "w": (
        ("Time", "nCells", "nVertLevelsP1"),
        "m s-1",
        "Vertical velocity at vertical cell faces",
    ),
    "theta": (("Time", "nCells", "nVertLevels"), "K", "Potential temperature"),
    "rho": (("Time", "nCells", "nVertLevels"), "kg m-3", "Dry air density"),
    "zz": (
        ("Time", "nCells", "nVertLevels"),
        "1",
        "d(zeta)/dz, vertical metric term",
    ),
    "qv": (
        ("Time", "nCells", "nVertLevels"),
        "kg kg-1",
        "Water vapor mixing ratio",
    ),
    "pressure": (("Time", "nCells", "nVertLevels"), "Pa", "Pressure"),
    "divergence": (
        ("Time", "nCells", "nVertLevels"),
        "s-1",
        "Horizontal velocity divergence at cell center",
    ),
    "vorticity": (
        ("Time", "nVertices", "nVertLevels"),
        "s-1",
        "Relative vorticity at vertices",
    ),
    "ke": (
        ("Time", "nCells", "nVertLevels"),
        "m2 s-2",
        "Kinetic energy at a cell center",
    ),
    "uReconstructZonal": (
        ("Time", "nCells", "nVertLevels"),
        "m s-1",
        "Zonal component of reconstructed horizontal velocity at cell centers",
    ),
    "uReconstructMeridional": (
        ("Time", "nCells", "nVertLevels"),
        "m s-1",
        "Meridional component of reconstructed horizontal velocity at cell centers",
    ),
}

_MPAS_TIME_FORMATS = (
    "%Y-%m-%d_%H:%M:%S",
    "%Y-%m-%d_%H.%M.%S",
    "%Y-%m-%d %H:%M:%S",
)


def parse_mpas_time(value: str) -> datetime:
    for time_format in _MPAS_TIME_FORMATS:
        try:
            return datetime.strptime(value, time_format).replace(tzinfo=UTC)
        except ValueError:
            continue
    raise ValueError(f"invalid MPAS time {value!r}; expected YYYY-MM-DD_HH:MM:SS")


def format_mpas_time(value: datetime) -> str:
    return value.strftime("%Y-%m-%d_%H:%M:%S")


def initial_time_for_valid(xtime: str, model_time_seconds: float) -> str:
    valid = parse_mpas_time(xtime)
    return format_mpas_time(valid - timedelta(seconds=float(model_time_seconds)))


def source_provenance(paths: Mapping[str, str | Path]) -> str:
    entries: list[dict[str, str]] = []
    for role, raw_path in sorted(paths.items()):
        path = Path(raw_path)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        entries.append({"role": role, "name": path.name, "sha256": digest.hexdigest()})
    return json.dumps(entries, sort_keys=True, separators=(",", ":"))


def configuration_digest(config: Mapping[str, Any] | None) -> str:
    payload = json.dumps(
        dict(config or {}), sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def continuation_configuration_digest(config: Mapping[str, Any] | None) -> str:
    """Exclude private caches and execution policy from scientific identity."""
    return configuration_digest(
        {
            key: value
            for key, value in (config or {}).items()
            if not key.startswith("_") and key != "gradient_checkpointing"
        }
    )


def mesh_digest(mesh: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    encoded = 0
    for name, value in sorted(mesh.items()):
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().contiguous().numpy()
        elif isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
        elif isinstance(value, (bool, int, float, str, np.generic)):
            digest.update(name.encode() + b"\0scalar\0")
            digest.update(
                json.dumps(value, sort_keys=True, default=str).encode("utf-8") + b"\0"
            )
            encoded += 1
            continue
        else:
            continue
        digest.update(name.encode() + b"\0")
        digest.update(str(array.dtype).encode() + b"\0")
        digest.update(str(array.shape).encode() + b"\0")
        digest.update(array.tobytes())
        digest.update(b"\0")
        encoded += 1
    if encoded == 0:
        raise ValueError("mesh digest requires at least one numerical identifier")
    return digest.hexdigest()


def tensor_device_and_dtype(state: Mapping[str, Any]) -> tuple[str, str]:
    for name in ("u", "w", "theta_m", "rho_zz", "scalars", *state):
        value = state.get(name)
        if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
            return str(value.device), str(value.dtype).removeprefix("torch.")
        if isinstance(value, np.ndarray) and value.dtype.kind == "f":
            return "cpu", str(value.dtype)
    return "unknown", "unknown"


def validate_scientific_array(
    name: str, values: Any, *, float64: bool = True
) -> np.ndarray:
    """Check actual storage before conversions can conceal precision or masks."""
    if isinstance(values, torch.Tensor):
        if values.device.type != "cpu":
            raise MpasConfigError(f"{name}: only CPU scientific arrays are supported")
        if float64 and values.dtype != torch.float64:
            raise MpasConfigError(
                f"{name}: scientific data must be float64, got {values.dtype}"
            )
        values = values.detach().numpy()
    if np.ma.isMaskedArray(values) and np.ma.getmaskarray(values).any():
        raise MpasConfigError(f"{name}: missing scientific data")
    array = np.asarray(values)
    if float64 and array.dtype != np.dtype("float64"):
        raise MpasConfigError(
            f"{name}: scientific data must be float64, got {array.dtype}"
        )
    if not np.isfinite(array).all():
        raise MpasConfigError(f"{name}: scientific data must be finite")
    return array


def validate_runtime_metadata(dataset: Any, *, complete_boundary: bool = False) -> None:
    for name, expected in (
        ("package_version", PACKAGE_VERSION),
        ("upstream_version", UPSTREAM_VERSION),
        ("upstream_commit", UPSTREAM_COMMIT),
        ("runtime_dtype", "float64"),
        ("runtime_device", "cpu"),
    ):
        if getattr(dataset, name, None) != expected:
            raise MpasConfigError(
                f"unsupported or missing {name}: expected {expected!r}"
            )
    try:
        timestep = float(dataset.timestep_seconds)
    except (AttributeError, TypeError, ValueError) as exc:
        raise MpasConfigError("timestep metadata is malformed") from exc
    if (
        not np.isfinite(timestep)
        or timestep < 0.0
        or (complete_boundary and timestep == 0.0)
    ):
        raise MpasConfigError(
            "timestep metadata must be finite and positive for continuation"
        )
