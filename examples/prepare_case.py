"""Prepare the public spherical Jablonowski-Williamson case 2.

Requires the upstream MPAS-Atmosphere v8.3.1 init_atmosphere_model executable.
The initializer supplies the scientific initial conditions; this script prepares
its input files and checks its output. Data is written only to an explicit directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import netCDF4 as nc
import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_raw_sphere(path: Path) -> dict[str, Any]:
    with nc.Dataset(path) as ds:
        ds.set_auto_maskandscale(False)
        radius = float(ds.sphere_radius)
        x = np.asarray(ds["xCell"][:], dtype=np.float64)
        dc = np.asarray(ds["dcEdge"][:], dtype=np.float64)
        if ds.on_a_sphere != "YES":
            raise ValueError(f"{path}: mesh is not spherical")
        if radius != 1.0 or not np.isfinite(x).all() or np.max(np.abs(x)) > 1.0:
            raise ValueError(f"{path}: expected unscaled unit-sphere coordinates")
        if not np.isfinite(dc).all() or np.max(dc) >= 1.0:
            raise ValueError(f"{path}: expected unscaled unit-sphere edge lengths")
        return {
            "nCells": len(ds.dimensions["nCells"]),
            "sphere_radius": radius,
            "max_abs_xCell": float(np.max(np.abs(x))),
            "max_dcEdge": float(np.max(dc)),
        }


def _complete_global_mesh_metadata(path: Path) -> None:
    """Supply global-only input metadata absent from the public raw grid."""

    with nc.Dataset(path, "r+") as ds:
        ds.setncattr("parent_id", "")
        for name, dim in (
            ("bdyMaskCell", "nCells"),
            ("bdyMaskEdge", "nEdges"),
            ("bdyMaskVertex", "nVertices"),
        ):
            if name in ds.variables:
                raise ValueError(f"unexpected pre-existing {name} in raw mesh")
            field = ds.createVariable(name, "i4", (dim,))
            field[:] = np.zeros(len(ds.dimensions[dim]), dtype=np.int32)
        if "xtime" in ds.variables or "Time" in ds.dimensions:
            raise ValueError("unexpected time coordinate in raw mesh")
    _assert_raw_sphere(path)


def _namelist(mesh: str, init_case: int) -> str:
    return f"""&nhyd_model
    config_init_case = {init_case}
    config_start_time = '2026-01-17_06:00:00'
    config_stop_time = '2026-01-17_06:00:00'
    config_theta_adv_order = 3
    config_coef_3rd_order = 0.25
/
&dimensions
    config_nvertlevels = 55
    config_nsoillevels = 4
    config_nfglevels = 100
    config_nfgsoillevels = 4
/
&preproc_stages
    config_static_interp = false
    config_native_gwd_static = false
    config_vertical_grid = false
    config_met_interp = false
    config_input_sst = false
/
&io
    config_pio_num_iotasks = 0
    config_pio_stride = 1
/
&decomposition
    config_block_decomp_file_prefix = '{mesh}.graph.info.part.'
/
"""


def _streams(mesh: str) -> str:
    return f"""<streams>
<immutable_stream name="input" type="input" precision="single"
                  filename_template="{mesh}.grid.nc"
                  input_interval="initial_only" />
<immutable_stream name="output" type="output"
                  filename_template="{mesh}.init.nc"
                  packages="initial_conds" output_interval="initial_only" />
</streams>
"""


def generate(
    mesh: str,
    init_case: int,
    output_dir: Path,
    *,
    source_grid: Path,
    source_graph: Path,
    initializer: Path | None = None,
    command_factory: Callable[[Path], list[str]] | None = None,
) -> dict[str, Any]:
    if init_case not in (1, 2):
        raise ValueError("only JW case 1 control or case 2 is supported")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite idealized IC: {output_dir}")
    source_geometry = _assert_raw_sphere(source_grid)
    if command_factory is None:
        if initializer is None or not initializer.is_file():
            raise FileNotFoundError(
                "supply an upstream init_atmosphere_model executable"
            )
        initializer = initializer.resolve()
        command_factory = lambda _directory: [str(initializer)]
    output_dir.mkdir(parents=True, exist_ok=False)
    for source in (source_grid, source_graph):
        shutil.copy2(source, output_dir / source.name)
    prepared_grid = output_dir / source_grid.name
    _complete_global_mesh_metadata(prepared_grid)
    (output_dir / "namelist.init_atmosphere").write_text(
        _namelist(mesh, init_case), encoding="utf-8"
    )
    (output_dir / "streams.init_atmosphere").write_text(
        _streams(mesh), encoding="utf-8"
    )
    command = command_factory(output_dir)
    completed = subprocess.run(
        command, cwd=output_dir, text=True, capture_output=True, check=False
    )
    (output_dir / "initializer.stdout").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "initializer.stderr").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(
            f"{mesh} case {init_case}: initializer exited {completed.returncode}; "
            f"see {output_dir / 'initializer.stderr'}"
        )
    init_path = output_dir / f"{mesh}.init.nc"
    if not init_path.is_file():
        raise FileNotFoundError(f"initializer did not produce {init_path}")
    log_path = output_dir / "log.init_atmosphere.0000.out"
    log = log_path.read_text(encoding="utf-8")
    if "Finished running the init_atmosphere core" not in log:
        raise ValueError("initializer completion was not recorded")
    counts = re.findall(r"(Warning|Error|Critical error) messages =\s*(\d+)", log)
    message_counts = {name: int(value) for name, value in counts}
    expected_warning = (
        f"WARNING: File {mesh}.grid.nc does not contain a seekable xtime variable. "
        "Forcing a read of the first time record."
    )
    if (
        message_counts.get("Warning") != 1
        or log.count("WARNING:") != 1
        or expected_warning not in log
        or any(
            message_counts.get(name, -1) != 0 for name in ("Error", "Critical error")
        )
    ):
        raise ValueError(f"initializer reported warnings/errors: {message_counts}")
    with nc.Dataset(init_path) as ds:
        ds.set_auto_maskandscale(False)
        radius = float(ds.sphere_radius)
        x = np.asarray(ds["xCell"][:], dtype=np.float64)
        if not (6.3e6 < radius < 6.5e6) or np.max(np.abs(x)) > radius * 1.001:
            raise ValueError("idealized output is not a single-scaled sphere")
        fields = {}
        for name in ("u", "w", "theta", "rho", "qv"):
            values = np.asarray(ds[name][:])
            if values.dtype != np.float64 or not np.isfinite(values).all():
                raise ValueError(f"{name}: idealized IC must be finite float64")
            fields[name] = {
                "shape": list(values.shape),
                "nonzero": int(np.count_nonzero(values)),
            }
        if fields["qv"]["nonzero"] != 0:
            raise ValueError("JW case 1/2 must retain exact dry qv structural zero")
    report = {
        "schema": "mpas-jw-case2-ic.v1",
        "script_sha256": sha256_file(Path(__file__)),
        "mesh": mesh,
        "init_case": init_case,
        "init_binary_sha256": sha256_file(initializer) if initializer else None,
        "raw_geometry": source_geometry,
        "source_grid_sha256": sha256_file(source_grid),
        "prepared_grid_sha256": sha256_file(prepared_grid),
        "prepared_mesh_additions": [
            "parent_id_empty",
            "three_zero_global_boundary_masks",
        ],
        "source_graph_sha256": sha256_file(source_graph),
        "namelist_sha256": sha256_file(output_dir / "namelist.init_atmosphere"),
        "streams_sha256": sha256_file(output_dir / "streams.init_atmosphere"),
        "init_path": str(init_path),
        "init_sha256": sha256_file(init_path),
        "log_sha256": sha256_file(log_path),
        "message_counts": message_counts,
        "expected_input_schema_warning": expected_warning,
        "output_sphere_radius": radius,
        "fields": fields,
        "passed": True,
    }
    (output_dir / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def fetch_mesh(directory: Path) -> tuple[Path, Path]:
    manifest = json.loads(Path(__file__).with_name("mesh_fetch.json").read_text())
    directory.mkdir(parents=True, exist_ok=False)
    archive = directory / "x1.10242.tar.gz"
    with (
        urllib.request.urlopen(manifest["url"], timeout=120) as response,
        archive.open("xb") as target,
    ):
        shutil.copyfileobj(response, target)
    if (
        archive.stat().st_size != manifest["archive_bytes"]
        or sha256_file(archive) != manifest["archive_sha256"]
    ):
        raise ValueError("downloaded mesh archive does not match the fetch manifest")
    with tarfile.open(archive) as source:
        member = source.getmember(manifest["grid_member"])
        if not member.isfile() or member.size != manifest["grid_bytes"]:
            raise ValueError("grid archive member is not the expected regular file")
        stream = source.extractfile(member)
        if stream is None:
            raise ValueError("grid archive member has no readable payload")
        grid = directory / manifest["grid_member"]
        with stream, grid.open("xb") as target:
            shutil.copyfileobj(stream, target)
    if sha256_file(grid) != manifest["grid_sha256"]:
        raise ValueError("grid checksum does not match the fetch manifest")
    geometry = _assert_raw_sphere(grid)
    if geometry["nCells"] != manifest["nCells"]:
        raise ValueError("downloaded mesh has the wrong cell count")
    graph = directory / "x1.10242.graph.info.part.1"
    graph.write_text("0\n" * manifest["nCells"], encoding="ascii")
    return grid, graph


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initializer", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Fresh directory for fetched mesh and initialized case",
    )
    args = parser.parse_args()
    destination = args.data_dir.resolve()
    if destination.exists():
        parser.error("--data-dir must be a fresh directory")
    if not args.initializer.is_file():
        parser.error("--initializer must name an existing executable")
    destination.mkdir(parents=True, exist_ok=False)
    grid, graph = fetch_mesh(destination / "mesh")
    generate(
        "x1.10242",
        2,
        destination / "case",
        source_grid=grid,
        source_graph=graph,
        initializer=args.initializer,
    )
    for name in ("namelist.atmosphere", "streams.atmosphere"):
        shutil.copyfile(Path(__file__).with_name(name), destination / "case" / name)
    print(destination / "case")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
