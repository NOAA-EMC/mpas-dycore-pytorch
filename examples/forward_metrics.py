"""Field norms and independent one-ULP reference bounds for CPU float64."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

AXES = ("u", "w", "theta", "rho", "qv")
K = 10.0
EPS = float(np.finfo(np.float64).eps)


def norm2(values: np.ndarray) -> float:
    return float(np.linalg.norm(np.ascontiguousarray(values).ravel()))


def field_metrics(
    f0: np.ndarray,
    initial: np.ndarray,
    port: np.ndarray,
    twins: list[np.ndarray],
    *,
    field: str,
    floor: float = 1e-15,
) -> dict[str, Any]:
    if not math.isfinite(floor) or floor <= 0:
        raise ValueError("normalization floor must be finite and positive")
    if len(twins) != 5:
        raise ValueError(f"{field}: exactly five independent axes required")
    for value in (f0, initial, port, *twins):
        if value.shape != f0.shape or value.size == 0 or value.dtype != np.float64:
            raise ValueError(f"{field}: malformed float64 shape")
        if not np.isfinite(value).all():
            raise ValueError(f"{field}: nonfinite field")
    count = f0.size
    d2 = max(norm2(f0), norm2(initial), math.sqrt(count) * floor)
    dinf = max(float(np.max(np.abs(f0))), float(np.max(np.abs(initial))), floor)
    difference = port - f0
    e2 = norm2(difference) / d2
    einf = float(np.max(np.abs(difference))) / dinf
    twin2 = [norm2(twin - f0) / d2 for twin in twins]
    twin_inf = [float(np.max(np.abs(twin - f0))) / dinf for twin in twins]
    bound2 = max(K * max(twin2), 64.0 * EPS)
    bound_inf = max(K * max(twin_inf), 256.0 * EPS)
    return {
        "field": field,
        "active_element_count": count,
        "absolute_floor": floor,
        "d2": d2,
        "dinf": dinf,
        "e2": e2,
        "einf": einf,
        "t2_by_axis": dict(zip(AXES, twin2, strict=True)),
        "tinf_by_axis": dict(zip(AXES, twin_inf, strict=True)),
        "bound2": bound2,
        "bound_inf": bound_inf,
        "l2_passed": e2 <= bound2,
        "linf_passed": einf <= bound_inf,
        "passed": e2 <= bound2 and einf <= bound_inf,
    }
