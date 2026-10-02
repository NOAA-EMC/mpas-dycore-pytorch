"""Reject malformed scientific inputs before integration or reference comparison."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture
def references(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    import fortran_reference

    return fortran_reference


@pytest.mark.parametrize("invalid", ["masked", "nan", "inf", "-inf"])
def test_missing_required_ic_rejected_before_initialization(tmp_path, invalid):
    from mpas_pytorch import MpasConfigError, load_initial_state

    path = tmp_path / "init.nc"
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("nCells", 2)
        v = ds.createVariable("theta", "f8", ("nCells",))
        v[:] = 300.0
        v[0] = np.ma.masked if invalid == "masked" else float(invalid)
    with pytest.raises(MpasConfigError, match="theta contains missing or nonfinite"):
        load_initial_state(path, path)


def _history(path, refs, dims, *, masked=False, wrong_names=False):
    with nc.Dataset(path, "w") as ds:
        for name, size in {**dims, "StrLen": 64}.items():
            ds.createDimension(name, size)
        if wrong_names:
            ds.createDimension("wrong", dims["nCells"])
        for key, names in refs.FIELD_DIMENSIONS.items():
            if wrong_names and key == "theta":
                names = tuple("wrong" if name == "nCells" else name for name in names)
            v = ds.createVariable(key, "f8", names)
            v[:] = 1.0
            if masked and key == "theta":
                v[(0,) * v.ndim] = np.ma.masked
        ds.createVariable("xtime", "S1", ("Time", "StrLen"))[:] = np.frombuffer(
            b"2026-01-17_06:00:00".ljust(64, b" "), dtype="S1"
        ).reshape(1, 64)


@pytest.mark.parametrize("invalid", ["masked", "wrong_names", "wrong_extent"])
def test_reference_reader_rejects_missing_and_incomplete_fields(
    tmp_path, references, invalid
):
    dims = {
        "Time": 1,
        "nCells": 2,
        "nEdges": 3,
        "nVertices": 4,
        "nVertLevels": 2,
        "nVertLevelsP1": 3,
    }
    path = tmp_path / "history.2026-01-17_06.00.00.nc"
    _history(
        path,
        references,
        dims,
        masked=invalid == "masked",
        wrong_names=invalid == "wrong_names",
    )
    expected = dict(dims, nCells=10242) if invalid == "wrong_extent" else dims
    with pytest.raises(ValueError, match="malformed or missing"):
        references.read_fields(path, expected)


@pytest.fixture
def raw_twins(tmp_path, references, monkeypatch):
    # Import the maintained generator in both exported and canonical test layouts.
    try:
        from twin_inputs import prepare
    except ModuleNotFoundError:
        source = (
            Path(__file__).resolve().parents[3]
            / "orchestration/scripts/prepare_fortran_long_twins.py"
        )
        spec = importlib.util.spec_from_file_location("twin_inputs", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        monkeypatch.setitem(__import__("sys").modules, "twin_inputs", module)
        prepare = module.prepare
    path = tmp_path / "x1.10242.init.nc"
    with nc.Dataset(path, "w") as ds:
        for name, size in (
            ("Time", 1),
            ("nCells", 2),
            ("nEdges", 3),
            ("nVertLevels", 2),
        ):
            ds.createDimension(name, size)
        for axis in ("u", "w", "theta", "rho", "qv"):
            dims = ("Time", "nEdges" if axis == "u" else "nCells", "nVertLevels")
            ds.createVariable(axis, "f8", dims)[:] = 300.0 if axis == "theta" else 1.0
        ds.createVariable("indexToCellID", "i4", ("nCells",))[:] = [1, 2]
        ds.createVariable("indexToEdgeID", "i4", ("nEdges",))[:] = [1, 2, 3]
        ds.createVariable("unchanged", "f8", ("nCells",))[:] = [-0.0, 4.0]
    prepare(path, tmp_path / "twins")
    return tmp_path


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        "multi_ulp",
        "wrong_axis",
        "other_variable",
        "ids",
        "duplicate",
        "source",
        "missing",
    ],
)
def test_raw_twin_contract_is_recomputed(raw_twins, references, invalid):
    root = raw_twins
    manifest = root / "twins/twin_inputs.json"
    report = json.loads(manifest.read_text())
    item = report["cases"][0]
    target = Path(item["path"])
    if invalid in {"multi_ulp", "wrong_axis", "other_variable", "ids"}:
        with nc.Dataset(target, "a") as ds:
            key = {
                "multi_ulp": "u",
                "wrong_axis": "w",
                "other_variable": "unchanged",
                "ids": "indexToCellID",
            }[invalid]
            v = ds[key]
            index = (0,) * v.ndim
            v[index] = np.nextafter(float(v[index]), np.inf)
            if invalid == "ids":
                v[index] = 99
        item["sha256"] = references.sha(
            target
        )  # Self-issued hashes cannot hide the wrong perturbation.
    elif invalid == "duplicate":
        report["cases"][1] = dict(item)
    elif invalid == "source":
        report["source_sha256"] = "0" * 64
    elif invalid == "missing":
        target.rename(target.with_suffix(".preserved"))
    manifest.write_text(json.dumps(report))
    binding = {"path": "twins/twin_inputs.json", "sha256": references.sha(manifest)}
    if invalid is None:
        assert set(references.verify_twins(root, root, binding)) == {
            "u",
            "w",
            "theta",
            "rho",
            "qv",
        }
    else:
        with pytest.raises((AssertionError, FileNotFoundError)):
            references.verify_twins(root, root, binding)


@pytest.mark.parametrize("path_kind", ["relative", "symlink"])
def test_raw_twin_reference_directory_paths(
    raw_twins, references, monkeypatch, path_kind
):
    root = raw_twins
    manifest = root / "twins/twin_inputs.json"
    binding = {"path": "twins/twin_inputs.json", "sha256": references.sha(manifest)}
    if path_kind == "relative":
        monkeypatch.chdir(root.parent)
        supplied = Path(root.name)
    else:
        supplied = root.parent / "linked-references"
        supplied.symlink_to(root, target_is_directory=True)
    assert set(references.verify_twins(root, supplied, binding)) == {
        "u",
        "w",
        "theta",
        "rho",
        "qv",
    }


def test_full_checker_rejects_old_unproven_receipt_before_comparison(
    tmp_path, references
):
    for name in references.INPUTS:
        (tmp_path / name).write_bytes(b"input")
    receipt = {
        "schema": "mpas-public-fortran-references.v1",
        "inputs": {name: references.sha(tmp_path / name) for name in references.INPUTS},
    }
    (tmp_path / "references.json").write_text(json.dumps(receipt))
    with pytest.raises(AssertionError, match="reference inputs differ"):
        references.check_references(tmp_path, tmp_path, [{"fields": {}}] * 3)


def test_case_geometry_cannot_be_replaced_by_a_tiny_mesh(tmp_path, references):
    path = tmp_path / references.INPUTS[0]
    with nc.Dataset(path, "w") as ds:
        for name, size in {
            "Time": 1,
            "nCells": 2,
            "nEdges": 3,
            "nVertices": 4,
            "nVertLevels": 55,
            "nVertLevelsP1": 56,
        }.items():
            ds.createDimension(name, size)
    with pytest.raises(ValueError, match="complete x1.10242"):
        references.case_dimensions(tmp_path)
