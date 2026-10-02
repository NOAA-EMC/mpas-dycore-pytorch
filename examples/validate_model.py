"""Validate a prepared public JW case through the installed MPAS-PyTorch APIs.

Run prepare_case.py first, then supply its case directory and a fresh output
folder. A failed check raises an error and returns a nonzero exit status.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import mpas_pytorch as mpas
import numpy as np
import torch
from known_answers import ATOL_FLOOR, C_STRICT, REFERENCE
from torch.overrides import TorchFunctionMode

KEYS = ("u", "w", "theta_m", "rho_zz", "scalars")
BOUNDARY_KEYS = KEYS + (
    "divergence",
    "exner",
    "exner_base",
    "h_edge",
    "ke",
    "pressure_base",
    "pressure_p",
    "pv_edge",
    "rho_edge",
    "rho_p",
    "rtheta_base",
    "rtheta_p",
    "ru",
    "rw",
    "uReconstructMeridional",
    "uReconstructX",
    "uReconstructY",
    "uReconstructZ",
    "uReconstructZonal",
    "ur_cell",
    "v",
    "vorticity",
    "vr_cell",
)
FD_STEPS = tuple(2.0 ** (-4 - 4 * index) for index in range(11))


class WindBranches(TorchFunctionMode):
    """Record the actual hard choices made by each nonlinear wind run."""

    def __init__(self) -> None:
        self.records: list[tuple[str, tuple[int, ...], str]] = []
        self.static_topology_clamps = 0

    def __torch_function__(self, function, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = getattr(function, "__name__", "")

        # This builder clamps fixed integer mesh indices. Its cache can omit the
        # calls; topology is checked independently for every nonlinear forecast.
        if (
            name == "clamp"
            and isinstance(args[0], torch.Tensor)
            and not args[0].is_floating_point()
        ):
            frame = sys._getframe(1)
            while frame is not None and "mpas_pytorch" not in frame.f_code.co_filename:
                frame = frame.f_back
            if (
                frame is not None
                and Path(frame.f_code.co_filename).name == "moist_coefficients.py"
                and frame.f_code.co_name == "_build_moist_connectivity_payload"
            ):
                self.static_topology_clamps += 1
                return function(*args, **kwargs)

        def array(value):
            return (
                value.detach().cpu().numpy()
                if isinstance(value, torch.Tensor)
                else np.asarray(value)
            )

        masks = []
        if name in {"abs", "absolute"}:
            value = array(args[0])
            masks = [value < 0, value == 0]
        elif name in {"minimum", "maximum"}:
            left, right = np.broadcast_arrays(array(args[0]), array(args[1]))
            masks = [left < right, left == right]
        elif name == "where":
            masks = [array(args[0])]
        elif name in {"masked_fill", "masked_fill_"}:
            masks = [array(args[1])]
        elif name in {"clamp", "clamp_min", "clamp_max"}:
            value = array(args[0])
            lower = kwargs.get("min", args[1] if len(args) > 1 else None)
            upper = kwargs.get("max", args[2] if len(args) > 2 else None)
            if name == "clamp_max":
                upper, lower = (
                    kwargs.get("max", args[1] if len(args) > 1 else None),
                    None,
                )
            if lower is not None:
                operand, bound = np.broadcast_arrays(value, array(lower))
                masks.extend((operand < bound, operand == bound))
            if upper is not None:
                operand, bound = np.broadcast_arrays(value, array(upper))
                masks.extend((operand > bound, operand == bound))
        result = function(*args, **kwargs)
        if name in {"__bool__", "is_nonzero", "all", "any"}:
            masks = [array(result)]
        if masks:
            packed = np.packbits(
                np.concatenate(
                    [np.asarray(mask, dtype=bool).reshape(-1) for mask in masks]
                )
            ).tobytes()
            self.records.append(
                (name, tuple(np.shape(masks[0])), hashlib.sha256(packed).hexdigest())
            )
        return result


def wind_curve_passes(rows: list[dict[str, Any]], bound: float) -> bool:
    """Apply the fixed convergence/affine rule only to unchanged hard branches."""
    if len(rows) != len(FD_STEPS) or [row["h"] for row in rows] != list(FD_STEPS):
        return False
    errors = [row["relative_error"] for row in rows]
    if (
        not math.isfinite(bound)
        or bound <= 0
        or not all(math.isfinite(error) and error >= 0 for error in errors)
    ):
        return False
    for i in range(len(errors) - 2):
        if not rows[i]["branch_consistent"] or not rows[i + 1]["branch_consistent"]:
            continue
        accurate = errors[i] <= bound and errors[i + 1] <= bound
        nonlinear = False
        if errors[i] > 0 and errors[i + 1] > 0:
            order = math.log(errors[i] / errors[i + 1]) / math.log(
                FD_STEPS[i] / FD_STEPS[i + 1]
            )
            nonlinear = 1.5 <= order <= 2.5 and errors[i + 1] <= bound
        wall = any(
            rows[j]["branch_consistent"]
            and errors[j] > 2 * max(errors[i + 1], np.finfo(np.float64).tiny)
            for j in range(i + 2, len(errors))
        )
        if (accurate or nonlinear) and wall:
            return True
    return False


def clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: clone(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(clone(item) for item in value)
    return copy.deepcopy(value)


def check_known_answers(history: dict[str, Any]) -> float:
    worst = 0.0
    for field, expected in REFERENCE.items():
        values = np.asarray(history["fields"][field])[0]
        if values.shape != expected["shape"] or not np.isfinite(values).all():
            raise AssertionError(f"{field}: missing, nonfinite or wrong shape")
        for (entity, level), reference in zip(
            expected["indices"], expected["values"], strict=True
        ):
            budget = ATOL_FLOOR + C_STRICT * np.finfo(np.float64).eps * abs(reference)
            error = abs(float(values[entity, level]) - reference)
            worst = max(worst, error / budget)
            if error > budget:
                raise AssertionError(
                    f"Fortran known answer failed: {field}[{entity},{level}] "
                    f"error {error:.17g}, budget {budget:.17g}"
                )
    return worst


def equal_state(left: dict[str, Any], right: dict[str, Any]) -> int:
    names = set(BOUNDARY_KEYS)
    if "rt_diabatic_tend" in left:
        names.add("rt_diabatic_tend")
    if not names <= left.keys() or not names <= right.keys():
        raise AssertionError("complete scientific boundary tensors are missing")
    for key in sorted(names):
        a, b = left[key], right[key]
        if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            raise AssertionError(f"state mismatch: {key}")
        # torch.equal alone treats signed zeros as equal.
        if a.detach().numpy().tobytes() != b.detach().numpy().tobytes():
            raise AssertionError(f"state bit mismatch: {key}")
    return len(names)


def expect_error(function: Any, error: type[Exception]) -> None:
    try:
        function()
    except error:
        return
    raise AssertionError(f"required negative control did not raise {error.__name__}")


def prepare_wind_control(
    raw: dict[str, Any], mesh: dict[str, Any], config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Materialize one fixed interior control base through the public model API."""
    state = clone(raw)
    original = state["u"].clone()
    mantissa, exponent = torch.frexp(original)
    state["u"] = torch.ldexp(torch.round(mantissa * 65536.0), exponent - 16)
    state["u"][-1] = 0.0
    levels = int(mesh["nVertLevels"])
    k = torch.arange(levels + 1, dtype=torch.float64)
    state["w"] = torch.zeros_like(state["w"])
    state["w"][:-1] = 0.1 * torch.sin(torch.pi * k / levels)[None, :]
    state["w"][:, 0] = 0.0
    state["w"][:, -1] = 0.0
    metadata = {
        "base": "JW primary fields with fixed vertical motion and dyadic wind",
        "wind_significant_binary_bits": 16,
        "vertical_motion_m_per_s": "0.1*sin(pi*k/L); bottom/top/padding zero",
        "maximum_wind_rounding_m_per_s": float((state["u"] - original).abs().max()),
        "wind_sha256": hashlib.sha256(state["u"].numpy().tobytes()).hexdigest(),
        "forward_reference_case_modified": False,
    }
    prepared: dict[str, Any] = {}
    mpas.run_simulation(
        state,
        clone(mesh),
        config=config,
        dt=720.0,
        nsteps=1,
        on_initial=lambda initial: prepared.update(clone(initial)),
    )
    if not prepared:
        raise AssertionError(
            "wind control preparation did not return initialized state"
        )
    return prepared, metadata


def check_wind_input(
    wind: torch.Tensor, perturbed: torch.Tensor, h: float, sign: float
) -> bool | None:
    """Verify exact nominal dyadic perturbations without using the model or AD."""
    shift = int(-math.log2(h))
    if h != 2.0**-shift or sign not in {1.0, -1.0}:
        raise AssertionError("wind perturbation is outside the fixed dyadic grid")
    if shift > 36:
        return None  # These small points deliberately measure the roundoff wall.
    values = wind.detach().numpy()
    mantissa, exponent = np.frexp(values)
    coefficient = np.rint(mantissa * 65536.0).astype(np.int64)
    if not np.array_equal(
        np.ldexp(coefficient.astype(np.float64), exponent - 16), values
    ):
        raise AssertionError("wind base is not the preregistered dyadic control")
    exact_integer = coefficient * ((1 << shift) + int(sign))
    if not np.array_equal(
        exact_integer.astype(np.float64).astype(np.int64), exact_integer
    ):
        raise AssertionError("wind perturbation coefficient exceeds binary64 precision")
    expected = np.ldexp(exact_integer.astype(np.float64), exponent - 16 - shift)
    if not np.isfinite(expected).all() or not np.array_equal(
        expected, perturbed.detach().numpy()
    ):
        raise AssertionError(
            "stored wind perturbation differs from its exact nominal value"
        )
    return True


def topology_signature(mesh: dict[str, Any]) -> str:
    """Bind fixed integer connectivity and mesh counts independently of caches."""
    digest = hashlib.sha256()
    for key, value in sorted(mesh.items()):
        if (
            isinstance(value, torch.Tensor)
            and not value.is_floating_point()
            and not value.is_complex()
        ):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(key.encode())
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        elif key in {"nCells", "nEdges", "nVertices", "nVertLevels"}:
            digest.update(f"{key}:{value}".encode())
    return digest.hexdigest()


def assert_topology(mesh: dict[str, Any], expected: str) -> None:
    if topology_signature(mesh) != expected:
        raise AssertionError("nonlinear forecast topology changed")


def validate_derivatives(
    state: dict[str, Any], mesh: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(20260930)
    direction, seed = {}, {}
    for key in KEYS:
        direction[key] = torch.randn(
            state[key].shape, generator=generator, dtype=torch.float64
        )
        direction[key][-1] = 0.0
        direction[key] /= torch.linalg.vector_norm(direction[key])
    generator.manual_seed(20261001)
    for key in KEYS:
        seed[key] = torch.randn(
            state[key].shape, generator=generator, dtype=torch.float64
        )
        seed[key][-1] = 0.0
        seed[key] /= torch.linalg.vector_norm(seed[key])
    kwargs = {"config": config, "dt": 720.0, "nsteps": 1, "checkpoint_timesteps": False}

    def objective(output: dict[str, Any]) -> torch.Tensor:
        return sum((output[key] * seed[key]).sum() for key in KEYS)

    tangent = mpas.run_simulation.jvp(clone(state), clone(mesh), direction, **kwargs)
    gradient = mpas.run_simulation.vjp(
        clone(state), clone(mesh), objective, track_keys=KEYS, **kwargs
    )
    for label, result in (("JVP", tangent), ("VJP", gradient)):
        for key in KEYS:
            value = result.get(key)
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != state[key].shape
                or value.dtype != torch.float64
                or not bool(torch.isfinite(value).all())
            ):
                raise AssertionError(
                    f"{label} {key}: missing, malformed or nonfinite derivative"
                )
    lhs = sum(float((tangent[key] * seed[key]).sum()) for key in KEYS)
    rhs = sum(float((gradient[key] * direction[key]).sum()) for key in KEYS)
    bound = max(
        1024
        * math.sqrt(sum(value.numel() for value in direction.values()))
        * np.finfo(np.float64).eps,
        1e-10,
    )
    scale = max(abs(lhs), abs(rhs), np.finfo(np.float64).tiny)
    residual = abs(lhs - rhs) / scale
    if not all(math.isfinite(value) for value in (lhs, rhs, bound, scale, residual)):
        raise AssertionError("nonfinite JVP/VJP identity calculation")
    if residual > bound:
        raise AssertionError(f"JVP/VJP identity {residual:.17g} exceeds {bound:.17g}")
    if abs(lhs + rhs) / scale <= bound:
        raise AssertionError("sign-flipped adjoint negative control passed")
    expect_error(
        lambda: mpas.run_simulation.vjp(
            clone(state),
            clone(mesh),
            lambda output: objective(output).detach(),
            track_keys=KEYS,
            **kwargs,
        ),
        RuntimeError,
    )

    # Preregistered jet-amplitude control; initial wind fixes the projection.
    wind = state["u"].detach().clone()
    wind[-1] = 0.0
    wind_norm = torch.linalg.vector_norm(wind)
    if not bool(torch.isfinite(wind_norm)) or float(wind_norm) <= 0:
        raise AssertionError(
            "wind-amplitude example requires finite nonzero initial wind"
        )
    projection = wind / wind_norm
    calibration = wind + FD_STEPS[0] * wind
    position = int(torch.nonzero(wind.reshape(-1), as_tuple=False)[0, 0])
    calibration.reshape(-1)[position] = torch.nextafter(
        calibration.reshape(-1)[position], torch.tensor(math.inf, dtype=torch.float64)
    )
    expect_error(
        lambda: check_wind_input(wind, calibration, FD_STEPS[0], 1.0), AssertionError
    )
    derivative = float(
        (
            mpas.run_simulation.jvp(clone(state), clone(mesh), {"u": wind}, **kwargs)[
                "u"
            ]
            * projection
        ).sum()
    )
    if not math.isfinite(derivative) or abs(derivative) <= np.finfo(np.float64).tiny:
        raise AssertionError("nonfinite or zero wind derivative")
    topology = topology_signature(mesh)
    baseline_mesh = clone(mesh)
    assert_topology(baseline_mesh, topology)
    with WindBranches() as base_branches:
        baseline = mpas.run_simulation(clone(state), baseline_mesh, **kwargs)
    assert_topology(baseline_mesh, topology)
    del baseline_mesh
    baseline_wind = baseline["u"].detach().clone()
    del baseline
    errors, rows = [], []
    for h in FD_STEPS:
        values = []
        branches = []
        exact_inputs = []
        for sign in (1.0, -1.0):
            perturbed = clone(state)
            perturbed["u"] = perturbed["u"] + sign * h * wind
            exact_inputs.append(check_wind_input(wind, perturbed["u"], h, sign))
            forecast_mesh = clone(mesh)
            assert_topology(forecast_mesh, topology)
            with WindBranches() as trace:
                output = mpas.run_simulation(perturbed, forecast_mesh, **kwargs)
            assert_topology(forecast_mesh, topology)
            del forecast_mesh
            values.append(float(((output["u"] - baseline_wind) * projection).sum()))
            branches.append(trace.records == base_branches.records)
        if not all(math.isfinite(value) for value in values):
            raise AssertionError("nonfinite wind finite-difference output")
        fd = (values[0] - values[1]) / (2 * h)
        error = abs(fd - derivative) / max(abs(derivative), np.finfo(np.float64).tiny)
        if not math.isfinite(fd) or not math.isfinite(error):
            raise AssertionError("nonfinite wind finite-difference calculation")
        errors.append(error)
        rows.append(
            {
                "h": h,
                "finite_difference": fd,
                "jvp": derivative,
                "relative_error": error,
                "branch_consistent": all(branches),
                "plus_branch_consistent": branches[0],
                "minus_branch_consistent": branches[1],
                "input_perturbations_exact": all(exact_inputs)
                if h >= 2.0**-36
                else None,
            }
        )
    if not wind_curve_passes(rows, bound):
        raise AssertionError(f"fixed wind finite-difference curve failed: {rows}")
    for wrong in (-derivative, 0.0):
        wrong_rows = [
            {
                **row,
                "relative_error": abs(row["finite_difference"] - wrong)
                / max(abs(wrong), np.finfo(np.float64).tiny),
            }
            for row in rows
        ]
        if wind_curve_passes(wrong_rows, bound):
            raise AssertionError("incorrect wind derivative negative control passed")
    return {
        "identity_relative_error": residual,
        "bound": bound,
        "wind_fd_curve": rows,
        "wind_direction": "active initial u, unnormalized; padding zero",
        "wind_projection": "fixed initial u divided by its float64 vector norm",
        "wind_branch_call_count": len(base_branches.records),
        "fixed_topology_sha256": topology,
        "static_topology_clamps_in_baseline": base_branches.static_topology_clamps,
        "derivative_scope": "selected AD linearization; nonlinear FD covers wind only",
    }


def validate(
    case: Path,
    output: Path,
    *,
    fortran_executable: Path | None = None,
    reference_dir: Path | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("--output-dir must be fresh")
    output.mkdir(parents=True)
    config_path = case / "namelist.atmosphere"
    init_path, grid_path = case / "x1.10242.init.nc", case / "x1.10242.grid.nc"
    cli_output = output / "cli"
    command = [
        sys.executable,
        "-m",
        "mpas_pytorch.cli.run",
        "--config",
        str(config_path),
        "--init",
        str(init_path),
        "--grid",
        str(grid_path),
        "--nsteps",
        "2",
        "--output-interval",
        "00:12:00",
        "--output-dir",
        str(cli_output),
    ]
    subprocess.run(command, check=True)
    summary = json.loads((cli_output / "run_summary.json").read_text())
    histories = [mpas.read_history(path) for path in summary["history_files"]]
    times = [float(history["fields"]["time"][0]) for history in histories]
    if times != [0.0, 720.0, 1440.0]:
        raise AssertionError(f"history times differ: {times}")
    paired = None
    if fortran_executable is not None or reference_dir is not None:
        from fortran_reference import check_references, prepare_references

        if fortran_executable is not None:
            reference_dir = prepare_references(
                case, fortran_executable, output / "fortran"
            )
        assert reference_dir is not None
        paired = check_references(case, reference_dir, histories)
        worst = None
    else:
        worst = check_known_answers(histories[-1])
    corrupt = copy.deepcopy(histories[-1])
    corrupt["fields"]["u"][0, 0, 0] += 1.0
    if paired is None:
        expect_error(lambda: check_known_answers(corrupt), AssertionError)
    else:
        expect_error(
            lambda values=[*histories[:-1], corrupt]: check_references(
                case, reference_dir, values
            ),
            AssertionError,
        )

    config = mpas.load_config_from_namelist(config_path)
    state, mesh = mpas.load_initial_state(init_path, grid_path, config=config)
    two = mpas.run_simulation(
        clone(state), clone(mesh), config=config, dt=720.0, nsteps=2
    )
    mpas.write_history(
        output / "api.nc",
        two,
        mesh,
        xtime="2026-01-17_06:24:00",
        config=config,
        dt=720.0,
        model_time_seconds=1440.0,
    )
    api_history = mpas.read_history(output / "api.nc")
    for name in REFERENCE:
        if (
            api_history["fields"][name].tobytes()
            != histories[-1]["fields"][name].tobytes()
        ):
            raise AssertionError(f"CLI/API history bit mismatch: {name}")
    restarted = output / "restart.nc"
    mpas.write_restart(
        two,
        mesh,
        restarted,
        "2026-01-17_06:24:00",
        config=config,
        dt=720.0,
        model_time_seconds=1440.0,
    )
    loaded, xtime = mpas.read_restart(restarted, mesh=mesh, config=config, dt=720.0)
    if xtime != "2026-01-17_06:24:00":
        raise AssertionError("restart time mismatch")
    continuous = mpas.run_simulation(
        clone(state), clone(mesh), config=config, dt=720.0, nsteps=4
    )
    resumed = mpas.run_simulation(
        loaded, clone(mesh), config=config, dt=720.0, nsteps=2
    )
    compared = equal_state(continuous, resumed)
    edited = clone(two)
    edited["u"][0, 27] += 0.01
    expect_error(
        lambda: mpas.write_restart(
            edited,
            mesh,
            output / "stale_restart.nc",
            "2026-01-17_06:24:00",
            config=config,
            dt=720.0,
            model_time_seconds=1440.0,
        ),
        mpas.MpasConfigError,
    )
    if (output / "stale_restart.nc").exists():
        raise AssertionError("rejected stale restart left an output file")
    via_step = mpas.run_timestep(clone(edited), clone(mesh), config=config, dt=720.0)
    via_run = mpas.run_simulation(
        clone(edited), clone(mesh), config=config, dt=720.0, nsteps=1
    )
    equal_state(via_step, via_run)
    unedited = mpas.run_simulation(
        clone(two), clone(mesh), config=config, dt=720.0, nsteps=1
    )
    if torch.equal(via_run["u"], unedited["u"]):
        raise AssertionError("edited wind had no effect")
    wrong_mesh = clone(mesh)
    wrong_mesh["nCells"] += 1
    expect_error(
        lambda: mpas.read_restart(restarted, mesh=wrong_mesh, config=config, dt=720.0),
        mpas.MpasConfigError,
    )
    expect_error(
        lambda: mpas.read_restart(restarted, mesh=mesh, config=config, dt=360.0),
        mpas.MpasConfigError,
    )
    for name in mpas.__all__:
        item = getattr(mpas, name)
        if isinstance(item, type) and issubclass(item, mpas.MpasError):
            kwargs = (
                {"capability": "device", "requested": "unsupported"}
                if item is mpas.MpasUnsupportedCapabilityError
                else {}
            )
            if not isinstance(item("public error check", **kwargs), mpas.MpasError):
                raise TypeError(f"exception hierarchy differs: {name}")
    # Completed checks need no forecast arrays during the derivative calculation.
    corrupt.clear()
    edited.clear()
    wrong_mesh.clear()
    del (
        histories,
        api_history,
        two,
        loaded,
        continuous,
        resumed,
        via_step,
        via_run,
        unedited,
    )
    wind_state, wind_base = prepare_wind_control(state, mesh, config)
    derivatives = validate_derivatives(wind_state, mesh, config)
    derivatives["control_base"] = wind_base
    report = {
        "passed": True,
        "known_answer_max_budget_fraction": worst,
        "paired_fortran": paired,
        "history_times_seconds": times,
        "restart_tensor_count": compared,
        "derivatives": derivatives,
    }
    (output / "validation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    references = parser.add_mutually_exclusive_group()
    references.add_argument("--fortran-executable", type=Path)
    references.add_argument("--reference-dir", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    report = validate(
        args.case_dir.resolve(),
        args.output_dir.resolve(),
        fortran_executable=args.fortran_executable,
        reference_dir=args.reference_dir,
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
