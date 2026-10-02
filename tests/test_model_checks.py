"""Negative controls for the runnable model example's known-answer checks."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture()
def model_checks(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    path = EXAMPLES / "validate_model.py"
    spec = importlib.util.spec_from_file_location("public_model_checks", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _known_history(checks):
    fields = {}
    for name, reference in checks.REFERENCE.items():
        values = np.zeros((1, *reference["shape"]), dtype=np.float64)
        for (row, level), expected in zip(
            reference["indices"], reference["values"], strict=True
        ):
            values[0, row, level] = expected
        fields[name] = values
    return {"fields": fields}


def test_known_answer_checker_rejects_one_wrong_value(model_checks):
    history = _known_history(model_checks)
    assert model_checks.check_known_answers(history) == 0.0
    history["fields"]["u"][0, 0, 0] += 1.0
    with pytest.raises(AssertionError, match="Fortran known answer failed"):
        model_checks.check_known_answers(history)


def test_known_answer_checker_rejects_missing_or_nonfinite_field(model_checks):
    history = _known_history(model_checks)
    history["fields"]["pressure"][0, 0, 0] = np.nan
    with pytest.raises(AssertionError, match="nonfinite"):
        model_checks.check_known_answers(history)
    del history["fields"]["u"]
    with pytest.raises(KeyError):
        model_checks.check_known_answers(history)


def test_reference_table_has_all_twelve_fields_and_fixed_points(model_checks):
    assert len(model_checks.REFERENCE) == 12
    assert sum(len(item["indices"]) for item in model_checks.REFERENCE.values()) == 108
    assert model_checks.ATOL_FLOOR == 1e-15
    assert model_checks.C_STRICT == 16384
    for item in model_checks.REFERENCE.values():
        assert item["indices"] == [
            (row, level)
            for row in (0, item["shape"][0] // 2, item["shape"][0] - 1)
            for level in (0, item["shape"][1] // 2, item["shape"][1] - 1)
        ]


def test_field_metrics_use_fixed_independent_bounds(model_checks):
    from forward_metrics import EPS, K, field_metrics

    assert K == 10.0
    assert EPS == np.finfo(np.float64).eps
    source = np.array([[1.0, 2.0]], dtype=np.float64)
    twins = [source.copy() for _ in range(5)]
    twins[3][0, 0] += 1e-6
    result = field_metrics(source, source, source, twins, field="u")
    assert result["passed"]
    assert result["bound_inf"] == 10 * abs(twins[3][0, 0] - source[0, 0]) / 2.0
    assert result["active_element_count"] == 2
    wrong = source.copy()
    wrong[0, 0] += 2e-5
    assert not field_metrics(source, source, wrong, twins, field="u")["passed"]
    assert result["tinf_by_axis"]["u"] == 0.0
    assert result["tinf_by_axis"]["rho"] > 0.0


def test_field_metrics_reject_incomplete_nonfinite_and_wrong_shapes(model_checks):
    from forward_metrics import field_metrics

    source = np.ones((2, 2), dtype=np.float64)
    twins = [source.copy() for _ in range(5)]
    with pytest.raises(ValueError, match="exactly five"):
        field_metrics(source, source, source, twins[:-1], field="w")
    with pytest.raises(ValueError, match="shape"):
        field_metrics(source, source, source[0], twins, field="w")
    wrong = source.copy()
    wrong[0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        field_metrics(source, source, wrong, twins, field="w")
    with pytest.raises(ValueError, match="shape"):
        field_metrics(source, source, source.astype(np.float32), twins, field="w")
