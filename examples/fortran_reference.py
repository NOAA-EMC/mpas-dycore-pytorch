"""Prepare and check short independent Fortran forecasts from the same inputs."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import netCDF4 as nc
import numpy as np
from forward_metrics import AXES, field_metrics

FIELDS = (
    "u",
    "w",
    "theta",
    "rho",
    "qv",
    "zz",
    "pressure",
    "divergence",
    "vorticity",
    "ke",
    "uReconstructZonal",
    "uReconstructMeridional",
)
INPUTS = ("x1.10242.init.nc", "x1.10242.grid.nc", "namelist.atmosphere")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


FIELD_DIMENSIONS = {
    name: (
        ("nCells", "nVertLevels")
        if name == "zz"
        else (
            "Time",
            "nEdges"
            if name == "u"
            else "nVertices"
            if name == "vorticity"
            else "nCells",
            "nVertLevelsP1" if name == "w" else "nVertLevels",
        )
    )
    for name in FIELDS
}


def case_dimensions(case: Path) -> dict[str, int]:
    with nc.Dataset(case / INPUTS[0]) as ds:
        dimensions = {
            name: len(ds.dimensions[name])
            for name in (
                "Time",
                "nCells",
                "nEdges",
                "nVertices",
                "nVertLevels",
                "nVertLevelsP1",
            )
        }
    if dimensions != {
        "Time": 1,
        "nCells": 10242,
        "nEdges": 30720,
        "nVertices": 20480,
        "nVertLevels": 55,
        "nVertLevelsP1": 56,
    }:
        raise ValueError("reference requires the complete x1.10242 55-level case")
    return dimensions


def read_fields(path: Path, dimensions: dict[str, int]) -> dict[str, np.ndarray]:
    result = {}
    with nc.Dataset(path) as ds:
        ds.set_auto_chartostring(False)
        if len(ds.dimensions["Time"]) != 1:
            raise ValueError("history requires Time=1")
        for key in FIELDS:
            var = ds[key]
            value = var[:]
            dims = FIELD_DIMENSIONS[key]
            if (
                var.dimensions != dims
                or value.shape != tuple(dimensions[d] for d in dims)
                or value.dtype != np.float64
                or np.ma.getmaskarray(value).any()
                or not np.isfinite(np.asarray(value)).all()
                or any(name in var.ncattrs() for name in ("scale_factor", "add_offset"))
            ):
                raise ValueError(f"{key}: malformed or missing Fortran history")
            result[key] = np.asarray(value[0] if dims[0] == "Time" else value).copy()
        if ds["xtime"].dimensions != ("Time", "StrLen"):
            raise ValueError("malformed Fortran timestamp dimensions")
        timestamp = nc.chartostring(ds["xtime"][:])[0].strip()
        if timestamp not in (
            "2026-01-17_06:00:00",
            "2026-01-17_06:12:00",
            "2026-01-17_06:24:00",
        ):
            raise AssertionError("Fortran timestamp outside the public schedule")
        if path.name != "history." + timestamp.replace(":", ".") + ".nc":
            raise AssertionError("Fortran timestamp disagrees with history filename")
    return result


def contained(root: Path, name: str, digest: str) -> Path:
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root.resolve()) or sha(path) != digest:
        raise AssertionError("reference input or output changed or escaped its root")
    return path


def verify_twins(case: Path, root: Path, binding: dict[str, Any]) -> dict[str, Path]:
    """Recompute the fixed ULP changes and compare every other raw variable."""
    from twin_inputs import perturb_axis

    root = root.resolve(strict=True)
    manifest = contained(root, binding["path"], binding["sha256"])
    report = json.loads(manifest.read_text())
    source = case / INPUTS[0]
    if report["schema"] != "mpas-fortran-long-twins.v1" or report[
        "source_sha256"
    ] != sha(source):
        raise AssertionError("twin source provenance differs")
    cases = report["cases"]
    if len(cases) != 5 or {item["axis"] for item in cases} != set(AXES):
        raise AssertionError("exactly five distinct raw twin inputs required")
    paths = {}
    with nc.Dataset(source) as raw:
        for key in (*AXES, "indexToCellID", "indexToEdgeID"):
            values = raw[key][:]
            if (
                np.ma.getmaskarray(values).any()
                or not np.isfinite(np.asarray(values)).all()
            ):
                raise AssertionError("twin source has missing or nonfinite raw inputs")
    with nc.Dataset(source) as original:
        original.set_auto_maskandscale(False)
        for item in cases:
            axis = item["axis"]
            # The generator records absolute paths; relocation retains the fixed basename.
            name = Path(item["path"]).name
            path = contained(
                root, str(manifest.parent.relative_to(root) / name), item["sha256"]
            )
            if path in paths.values():
                raise AssertionError("duplicate raw twin input")
            paths[axis] = path
            ids = np.asarray(
                original["indexToEdgeID" if axis == "u" else "indexToCellID"][:]
            )
            expected, counts = perturb_axis(
                np.asarray(original[axis][:]), ids, axis=axis
            )
            if any(item.get(key) != value for key, value in counts.items()):
                raise AssertionError("twin input counts or perturbation policy differ")
            with nc.Dataset(path) as twin:
                twin.set_auto_maskandscale(False)
                if (
                    set(original.variables) != set(twin.variables)
                    or set(original.dimensions) != set(twin.dimensions)
                    or original.ncattrs() != twin.ncattrs()
                    or any(
                        original.getncattr(k) != twin.getncattr(k)
                        for k in original.ncattrs()
                    )
                    or any(
                        len(original.dimensions[k]) != len(twin.dimensions[k])
                        for k in original.dimensions
                    )
                ):
                    raise AssertionError("twin raw schema or metadata changed")
                for key, var in original.variables.items():
                    other = twin[key]
                    wanted = expected if key == axis else np.asarray(var[:])
                    if (
                        var.dimensions != other.dimensions
                        or var.dtype != other.dtype
                        or var.ncattrs() != other.ncattrs()
                        or any(
                            not np.array_equal(
                                var.getncattr(k), other.getncattr(k), equal_nan=True
                            )
                            if np.asarray(var.getncattr(k)).dtype.kind in "fc"
                            else not np.array_equal(
                                var.getncattr(k), other.getncattr(k)
                            )
                            for k in var.ncattrs()
                        )
                        or wanted.tobytes() != np.asarray(other[:]).tobytes()
                    ):
                        raise AssertionError(
                            f"{axis}: raw twin variable {key} differs from its one-ULP contract"
                        )
    return paths


def prepare_references(case: Path, binary: Path, output: Path) -> Path:
    from twin_inputs import prepare

    binary = binary.resolve(strict=True)
    output = output.resolve()
    output.mkdir(exist_ok=False)
    dimensions = case_dimensions(case)
    from mpas_pytorch import load_config_from_namelist

    config = load_config_from_namelist(case / "namelist.atmosphere")
    if float(config["config_dt"]) != 720 or config["config_run_duration"] != "00:24:00":
        raise ValueError("reference requires two 720-second steps")
    inputs = {name: sha(case / name) for name in INPUTS}
    binary_hash = sha(binary)
    twins = prepare(case / INPUTS[0], output / "twins")
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        OMP_DYNAMIC="FALSE",
        VECLIB_MAXIMUM_THREADS="1",
    )
    rows = {}
    for axis in ("control", *AXES):
        run = output / axis
        run.mkdir()
        init = (
            case / INPUTS[0]
            if axis == "control"
            else Path(next(i["path"] for i in twins["cases"] if i["axis"] == axis))
        )
        for name in ("x1.10242.grid.nc", "x1.10242.graph.info", "namelist.atmosphere"):
            if (case / name).exists():
                (run / name).symlink_to((case / name).resolve())
        (run / INPUTS[0]).symlink_to(init.resolve())
        variables = (
            "".join(f'<var name="{k}"/>' for k in FIELDS if k != "qv")
            + '<var_array name="scalars"/><var name="xtime"/>'
        )
        (run / "streams.atmosphere").write_text(
            '<streams>\n<immutable_stream name="input" type="input" filename_template="x1.10242.init.nc" input_interval="initial_only"/>\n<immutable_stream name="restart" type="input;output" filename_template="restart.nc" input_interval="initial_only" output_interval="none"/>\n<stream name="output" type="output" filename_template="history.$Y-$M-$D_$h.$m.$s.nc" output_interval="00:12:00" clobber_mode="never_modify" precision="double">'
            + variables
            + "</stream>\n</streams>\n"
        )
        with (run / "stdout.log").open("w") as log:
            subprocess.run(
                [str(binary)],
                cwd=run,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=1200,
            )
        model_log = (run / "log.atmosphere.0000.out").read_text()
        if "Finished running the atmosphere core" not in model_log:
            raise AssertionError("Fortran did not complete")
        paths = sorted(run.glob("history.*.nc"))
        if len(paths) != 3:
            raise AssertionError("Fortran history must cover steps 0,1,2")
        for path in paths:
            read_fields(path, dimensions)
        rows[axis] = [
            {"path": str(p.relative_to(output)), "sha256": sha(p)} for p in paths
        ]
    if sha(binary) != binary_hash or inputs != {
        name: sha(case / name) for name in INPUTS
    }:
        raise AssertionError("reference inputs or executable changed")
    (output / "references.json").write_text(
        json.dumps(
            {
                "schema": "mpas-public-fortran-references.v2",
                "dimensions": dimensions,
                "twin_inputs": {
                    "path": "twins/twin_inputs.json",
                    "sha256": sha(output / "twins/twin_inputs.json"),
                },
                "generation": {
                    axis: {
                        "init_sha256": sha(case / INPUTS[0])
                        if axis == "control"
                        else next(
                            i["sha256"] for i in twins["cases"] if i["axis"] == axis
                        ),
                        "binary_sha256": binary_hash,
                        "streams_sha256": sha(output / axis / "streams.atmosphere"),
                        "log_sha256": sha(output / axis / "log.atmosphere.0000.out"),
                    }
                    for axis in ("control", *AXES)
                },
                "inputs": inputs,
                "binary_sha256": binary_hash,
                "binary_path": str(binary),
                "histories": rows,
            },
            indent=2,
        )
        + "\n"
    )
    return output


def check_references(
    case: Path, root: Path, histories: list[dict[str, Any]]
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    receipt = json.loads((root / "references.json").read_text())
    if receipt["schema"] != "mpas-public-fortran-references.v2" or receipt[
        "inputs"
    ] != {name: sha(case / name) for name in INPUTS}:
        raise AssertionError("reference inputs differ from forecast inputs")
    if set(receipt["histories"]) != {"control", *AXES} or len(histories) != 3:
        raise AssertionError("complete five-axis, three-time references required")
    dimensions = case_dimensions(case)
    if receipt["dimensions"] != dimensions:
        raise AssertionError("reference dimensions differ from forecast inputs")
    twins = verify_twins(case, root, receipt["twin_inputs"])
    if (
        set(receipt["generation"]) != {"control", *AXES}
        or not isinstance(receipt["binary_sha256"], str)
        or len(receipt["binary_sha256"]) != 64
    ):
        raise AssertionError("incomplete executable or generation provenance")
    if (
        sha(Path(receipt["binary_path"]).resolve(strict=True))
        != receipt["binary_sha256"]
    ):
        raise AssertionError("reference executable changed")
    data = {}
    seen = set()
    for axis, items in receipt["histories"].items():
        if len(items) != 3:
            raise AssertionError("incomplete reference times")
        generation = receipt["generation"][axis]
        if generation["binary_sha256"] != receipt["binary_sha256"] or generation[
            "init_sha256"
        ] != sha(case / INPUTS[0] if axis == "control" else twins[axis]):
            raise AssertionError(
                "reference generation input or executable provenance differs"
            )
        run = root / axis
        contained(
            root,
            str((run / "streams.atmosphere").relative_to(root)),
            generation["streams_sha256"],
        )
        log = contained(
            root,
            str((run / "log.atmosphere.0000.out").relative_to(root)),
            generation["log_sha256"],
        )
        if "Finished running the atmosphere core" not in log.read_text():
            raise AssertionError("reference generation did not complete")
        data[axis] = []
        for item in items:
            p = (root / item["path"]).resolve(strict=True)
            if not p.is_relative_to(root.resolve()) or sha(p) != item["sha256"]:
                raise AssertionError("reference history changed")
            if p in seen:
                raise AssertionError("duplicate reference history")
            seen.add(p)
            if (
                p.name
                != (
                    "history.2026-01-17_06.00.00.nc",
                    "history.2026-01-17_06.12.00.nc",
                    "history.2026-01-17_06.24.00.nc",
                )[len(data[axis])]
            ):
                raise AssertionError("Fortran reference schedule changed")
            data[axis].append(read_fields(p, dimensions))
    rows = []
    for step, history in enumerate(histories):
        time_values = np.asarray(history["fields"]["time"])
        if (
            time_values.shape != (1,)
            or time_values.dtype != np.float64
            or float(time_values[0]) != step * 720
        ):
            raise AssertionError("forecast times differ from reference schedule")
        for field in FIELDS:
            initial = data["control"][0][field]
            f0 = data["control"][step][field]
            values = history["fields"][field]
            if np.ma.getmaskarray(values).any() or np.shape(values) != (1, *f0.shape):
                raise AssertionError(
                    "forecast reference dimensions or missing values differ"
                )
            port = np.asarray(values)[0]
            twins = [data[axis][step][field] for axis in AXES]
            if field == "qv":
                with nc.Dataset(case / INPUTS[0]) as ds:
                    if not np.any(ds["qv"][:]) and any(
                        np.any(value) for value in (f0, port, *twins)
                    ):
                        raise AssertionError("dry structural scalar zeros changed")
            if (
                port.shape != f0.shape
                or port.dtype != np.float64
                or not np.isfinite(port).all()
            ):
                raise AssertionError(
                    "forecast reference coverage or finite values differ"
                )
            metric = field_metrics(f0, initial, port, twins, field=field)
            if not metric["passed"]:
                raise AssertionError(
                    f"Fortran twin contract failed: step {step} {metric}"
                )
            bad = f0.copy()
            index = np.unravel_index(np.argmax(abs(f0)), f0.shape)
            bad[index] += max(1.0, 2 * metric["bound_inf"] * metric["dinf"])
            if field_metrics(f0, initial, bad, twins, field=field)["passed"]:
                raise AssertionError("outward field negative control passed")
            rows.append(dict(step=step, **metric))
    return {
        "passed": True,
        "contract": "five-axis one-ULP, K=10, unchanged L2/Linf eps floors",
        "reference_sha256": sha(root / "references.json"),
        "rows": rows,
        "compared_values": sum(i["active_element_count"] for i in rows),
    }
