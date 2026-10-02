#!/usr/bin/env python3
"""Prepare the five preregistered one-ULP Fortran long-lead IC twins.

This only changes one raw init-file variable per copy. It does not validate a
trajectory or infer an acceptance envelope from the generated inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import netCDF4 as nc
import numpy as np

AXES = ("u", "w", "theta", "rho", "qv")
POSITIVE = frozenset(("theta", "rho"))
MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signs(global_ids: np.ndarray, nlevels: int) -> np.ndarray:
    """SplitMix64 parity of a global entity ID and 1-based vertical level."""

    ids = np.asarray(global_ids, dtype=np.uint64)
    levels = np.arange(1, nlevels + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        value = (ids[:, None] * np.uint64(0x9E3779B97F4A7C15)) ^ (
            levels[None, :] * np.uint64(0xBF58476D1CE4E5B9)
        )
        value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        value = (value ^ (value >> np.uint64(31))) & MASK64
    return np.where((value & np.uint64(1)) == 0, 1, -1)


def perturb_axis(
    values: np.ndarray, global_ids: np.ndarray, *, axis: str
) -> tuple[np.ndarray, dict[str, Any]]:
    """Move each active nonzero element exactly one float64 ULP."""

    source = np.asarray(values)
    if source.dtype != np.float64 or source.ndim != 3 or source.shape[0] != 1:
        raise ValueError(f"{axis}: expected float64 (Time=1, entity, level)")
    if not np.isfinite(source).all():
        raise ValueError(f"{axis}: source has nonfinite values")
    if axis in POSITIVE and not np.all(source > 0):
        raise ValueError(f"{axis}: source violates positive bound")
    ids = np.asarray(global_ids)
    if ids.ndim != 1 or ids.size != source.shape[1]:
        raise ValueError(f"{axis}: global entity ID shape mismatch")
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError(f"{axis}: global entity IDs must be integers")
    if np.any(ids <= 0) or np.unique(ids).size != ids.size:
        raise ValueError(f"{axis}: global entity IDs must be positive and unique")

    directions = _signs(ids, source.shape[2])[None, :, :]
    active = source != 0.0
    toward = np.where(directions > 0, np.inf, -np.inf)
    candidate = np.nextafter(source, toward)
    invalid = np.zeros(source.shape, dtype=bool)
    if axis in POSITIVE:
        invalid = active & (candidate <= 0.0)
    reversed_count = int(np.count_nonzero(invalid))
    if reversed_count:
        opposite = np.where(directions > 0, -np.inf, np.inf)
        candidate[invalid] = np.nextafter(source, opposite)[invalid]
    candidate[~active] = source[~active]
    if not np.isfinite(candidate).all():
        raise ValueError(f"{axis}: perturbation produced nonfinite values")
    if axis in POSITIVE and not np.all(candidate > 0):
        raise ValueError(f"{axis}: perturbation crossed positive bound")

    final_toward = np.where(directions > 0, np.inf, -np.inf)
    final_toward = np.where(invalid, -final_toward, final_toward)
    expected = np.nextafter(source, final_toward)
    if not np.array_equal(candidate[active], expected[active]):
        raise AssertionError(f"{axis}: a changed element is not one ULP away")
    if not np.array_equal(candidate[~active], source[~active]):
        raise AssertionError(f"{axis}: an exact zero changed")
    changed = int(np.count_nonzero(candidate != source))
    if changed != int(np.count_nonzero(active)):
        raise AssertionError(f"{axis}: active and changed counts disagree")
    return candidate, {
        "axis": axis,
        "shape": list(source.shape),
        "element_count": int(source.size),
        "modified": changed,
        "excluded_zero": int(np.count_nonzero(~active)),
        "reversed_for_bound": reversed_count,
        "direction_rule": "SplitMix64 parity of positive global entity ID and 1-based level",
        "one_ulp_verified": True,
        "structural_zeros_unchanged": True,
    }


def _read_fields(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    fields: dict[str, np.ndarray] = {}
    ids: dict[str, np.ndarray] = {}
    with nc.Dataset(path, "r") as ds:
        ds.set_auto_maskandscale(False)
        for axis in AXES:
            fields[axis] = np.asarray(ds.variables[axis][:]).copy()
        ids["edge"] = np.asarray(ds.variables["indexToEdgeID"][:]).copy()
        ids["cell"] = np.asarray(ds.variables["indexToCellID"][:]).copy()
    return fields, ids


def prepare(source: Path, output_dir: Path) -> dict[str, Any]:
    source = source.resolve(strict=True)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite twin directory: {output_dir}")
    originals, ids = _read_fields(source)
    output_dir.mkdir(parents=True, exist_ok=False)
    cases: list[dict[str, Any]] = []
    for axis in AXES:
        target = output_dir / f"init_twin_{axis}.nc"
        shutil.copy2(source, target)
        entity_ids = ids["edge" if axis == "u" else "cell"]
        expected, counts = perturb_axis(originals[axis], entity_ids, axis=axis)
        with nc.Dataset(target, "r+") as ds:
            ds.set_auto_maskandscale(False)
            ds.variables[axis][:] = expected
        observed, observed_ids = _read_fields(target)
        if any(not np.array_equal(observed_ids[key], ids[key]) for key in ids):
            raise AssertionError(f"{axis}: global IDs changed")
        for name in AXES:
            wanted = expected if name == axis else originals[name]
            if not np.array_equal(observed[name], wanted):
                raise AssertionError(f"{axis}: variable {name!r} changed unexpectedly")
        cases.append({**counts, "path": str(target), "sha256": sha256_file(target)})
    report = {
        "schema": "mpas-fortran-long-twins.v1",
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "script_sha256": sha256_file(Path(__file__)),
        "cases": cases,
        "passed": len(cases) == len(AXES)
        and all(case["one_ulp_verified"] for case in cases),
    }
    receipt = output_dir / "twin_inputs.json"
    receipt.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def self_test() -> None:
    ids = np.array([17, 31], dtype=np.int32)
    for axis in AXES:
        base = np.array([[[1.0, 0.0], [2.0, 3.0]]], dtype=np.float64)
        if axis in POSITIVE:
            base[0, 0, 1] = 4.0
        moved, counts = perturb_axis(base, ids, axis=axis)
        expected_count = 4 if axis in POSITIVE else 3
        assert counts["modified"] == expected_count
        assert counts["excluded_zero"] == 4 - expected_count
        if axis not in POSITIVE:
            assert moved[0, 0, 1] == 0.0
        assert np.count_nonzero(moved != base) == expected_count
    print("five-axis one-ULP and exact-zero self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.init is None or args.output_dir is None:
        parser.error("--init and --output-dir are required unless --self-test is set")
    report = prepare(args.init, args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
