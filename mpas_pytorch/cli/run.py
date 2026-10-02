"""Unified mpas-pytorch command-line runner."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import warnings
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import torch

from mpas_pytorch.atmosphere_model import run_simulation
from mpas_pytorch.exceptions import MpasConfigError, MpasError
from mpas_pytorch.ic_loader import load_config_from_namelist, load_initial_state
from mpas_pytorch.io_schema import HISTORY_FIELD_METADATA, source_provenance
from mpas_pytorch.output_writer import (
    MOISTURE_FIELDS,
    _extract_history_fields,
    history_scalar_fields,
    write_history,
)
from mpas_pytorch.release_capabilities import (
    get_release_capabilities,
    validate_runtime_request,
)
from mpas_pytorch.restart import read_restart, write_restart
from mpas_pytorch.state import clone_tree
from mpas_pytorch.utils import (
    INITIAL_ONLY,
    NEVER,
    parse_interval,
    parse_namelist,
    parse_streams,
)
from mpas_pytorch.utils.atomic_output import atomic_output

LOGGER = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capabilities",
        action="store_true",
        help="Print the machine-readable release capability contract and exit",
    )
    parser.add_argument("--config", help="Required: path to namelist.atmosphere")
    parser.add_argument(
        "--streams", default=None, help="Optional streams.atmosphere path"
    )
    parser.add_argument("--init", help="Required: path to init.nc")
    parser.add_argument("--grid", help="Required: separate compatible grid.nc")
    parser.add_argument(
        "--mesh-support", default=None, help="Optional mesh_complete.nc support file"
    )
    parser.add_argument(
        "--restart-in", default=None, help="Optional restart NetCDF path"
    )
    parser.add_argument(
        "--output-dir",
        default="./out",
        help="Directory for history/diag/restart output",
    )
    parser.add_argument("--device", default="cpu", help="Runtime device (cpu only)")
    parser.add_argument(
        "--dtype", default="float64", help="Runtime dtype (float64 only)"
    )
    parser.add_argument(
        "--run-duration", default=None, help="Duration string DD_HH:MM:SS or HH:MM:SS"
    )
    parser.add_argument(
        "--nsteps", type=int, default=None, help="Number of timesteps to run"
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    parser.add_argument("--quiet", action="store_true", help="Suppress non-error logs")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic torch algorithms",
    )
    parser.add_argument(
        "--output-interval",
        default="01:00:00",
        help="Fallback cadence when --streams is absent; initial and final output are always written",
    )
    return parser.parse_args(argv)


def _configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO))


def _require_run_inputs(args: argparse.Namespace) -> None:
    missing = [
        flag for flag in ("config", "init", "grid") if getattr(args, flag) is None
    ]
    if missing:
        rendered = ", ".join(f"--{name}" for name in missing)
        raise MpasConfigError(f"A model run requires {rendered}")


def _xtime_for_step(base_time: datetime, dt: float, step: int) -> str:
    return (base_time + timedelta(seconds=step * dt)).strftime("%Y-%m-%d_%H:%M:%S")


def _history_filename(xtime: str) -> str:
    return f"history.{xtime.replace(':', '.')}.nc"


def _resolve_nsteps(args: argparse.Namespace, *, dt: float) -> int:
    if not math.isfinite(dt) or dt <= 0.0:
        raise MpasConfigError("dt must be finite and positive")
    if args.nsteps is not None:
        if args.nsteps <= 0:
            raise MpasConfigError("--nsteps must be positive")
        if args.run_duration is not None:
            warnings.warn(
                "--nsteps overrides --run-duration; ignoring --run-duration",
                UserWarning,
                stacklevel=2,
            )
        return int(args.nsteps)
    if args.run_duration is not None:
        seconds = parse_interval(args.run_duration)
        if not isinstance(seconds, int):
            raise MpasConfigError("--run-duration requires a positive duration")
        quotient = seconds / dt
        if not quotient.is_integer() or quotient <= 0:
            raise MpasConfigError(
                "Run duration must be a positive integral multiple of dt"
            )
        return int(quotient)
    raise MpasConfigError("Supply --nsteps, --run-duration or config_run_duration")


def _load_runtime(
    *,
    config_path: str | Path,
    init_path: str | Path,
    grid_path: str | Path,
    mesh_support_path: str | None,
    restart_in: str | None,
    device: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str, float, str]:
    config = load_config_from_namelist(config_path)
    raw_namelist = parse_namelist(config_path)
    registry_defaults = get_release_capabilities()["core_surface"]["registry_defaults"][
        "values"
    ]
    base_xtime = str(
        raw_namelist.get("config_start_time", registry_defaults["config_start_time"])
    )
    state, mesh = load_initial_state(
        init_path,
        grid_path,
        mesh_support_path=mesh_support_path,
        device=device,
        config=config,
    )
    provenance_paths: dict[str, str | Path] = {
        "init": init_path,
        "grid": grid_path,
        "namelist": config_path,
    }
    model_time_offset = 0.0
    if restart_in:
        state, base_xtime = read_restart(
            restart_in,
            mesh=mesh,
            config=config,
            dt=float(config["config_dt"]),
            device=device,
        )
        model_time_offset = float(state.pop("_restart_model_time_seconds"))
        provenance_paths["restart"] = restart_in
    return (
        state,
        mesh,
        config,
        base_xtime,
        model_time_offset,
        source_provenance(provenance_paths),
    )


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        if args.capabilities:
            print(json.dumps(get_release_capabilities(), indent=2, sort_keys=True))
            return 0
        _configure_logging("ERROR" if args.quiet else args.log_level)
        validate_runtime_request(
            device=args.device,
            dtype=args.dtype,
            physics_mode="dycore",
        )
        _require_run_inputs(args)
        if args.deterministic:
            torch.use_deterministic_algorithms(True)

        state, mesh, config, start_xtime, model_time_offset, provenance = _load_runtime(
            config_path=args.config,
            init_path=args.init,
            grid_path=args.grid,
            mesh_support_path=args.mesh_support,
            restart_in=args.restart_in,
            device=args.device,
        )
        dt = float(config.get("config_dt", 720.0))
        if args.nsteps is None and args.run_duration is None:
            args.run_duration = parse_namelist(args.config).get("config_run_duration")
        nsteps = _resolve_nsteps(args, dt=dt)
        if str(config.get("config_physics_suite", "none")).lower() != "none":
            raise MpasConfigError(
                "This CLI preview requires config_physics_suite='none'"
            )
        output_dir = Path(args.output_dir).resolve()
        summary_path = output_dir / "run_summary.json"
        if summary_path.exists():
            raise MpasConfigError(
                "run_summary.json already exists; use a fresh output directory"
            )
        streams_cfg = parse_streams(args.streams) if args.streams else None
        fallback_interval_seconds = (
            parse_interval(args.output_interval) if streams_cfg is None else 3600
        )
        if not isinstance(fallback_interval_seconds, int):
            raise MpasConfigError("--output-interval requires a positive duration")
        base_time = datetime.strptime(start_xtime, "%Y-%m-%d_%H:%M:%S").replace(
            tzinfo=UTC
        )
        resumed = bool(args.restart_in)
        skipped_fields: dict[str, list[str]] = {}
        skipped_streams: dict[str, str] = {}
        stream_fields: dict[str, list[str] | None] = {}
        plans: dict[int, list[tuple[Any, Path]]] = {}
        targets: set[Path] = set()
        stock_unsupported = {
            "initial_time",
            "olrtoa",
            "rainc",
            "rainnc",
            "refl10cm_max",
            "refl10cm_1km",
            "refl10cm_1km_max",
            "precipw",
            "u10",
            "v10",
            "q2",
            "t2m",
            "th2m",
            "mslp",
            "t_isobaric",
            "t_iso_levels",
            "z_isobaric",
            "z_iso_levels",
            "meanT_500_300",
            "cape",
            "cin",
            "lcl",
            "lfc",
            "srh_0_1km",
            "srh_0_3km",
            "uzonal_surface",
            "uzonal_1km",
            "uzonal_6km",
            "umeridional_surface",
            "umeridional_1km",
            "umeridional_6km",
            "temperature_surface",
            "dewpoint_surface",
            "updraft_helicity_max",
            "w_velocity_max",
            "wind_speed_level1_max",
            "t_oml",
            "t_oml_initial",
            "t_oml_200m_initial",
            "h_oml",
            "h_oml_initial",
            "hu_oml",
            "hv_oml",
            "rho_zz",
            "theta_m",
        }
        stock_unsupported.update(
            f"{field}_{level}hPa"
            for field in (
                "relhum",
                "dewpoint",
                "temperature",
                "height",
                "uzonal",
                "umeridional",
                "w",
                "vorticity",
            )
            for level in (200, 250, 500, 700, 850, 925)
        )
        inactive_packages = {
            "mp_kessler_in",
            "mp_thompson_in",
            "mp_thompson_aers_in",
            "mp_wsm6_in",
            "cu_grell_freitas_in",
            "cu_kain_fritsch_in",
            "cu_ntiedtke_in",
            "bl_ysu_in",
            "bl_mynn_in",
            "sf_monin_obukhov_in",
            "sf_noahmp_in",
            "iau",
            "limited_area",
            "jedi_da",
            "ugwp_orog_stream",
            "ugwp_ngw_stream",
            "ugwp_diags_stream",
        }
        active_packages = (
            {"no_invariant_stream"}
            if streams_cfg is None
            or not any(spec.name == "invariant" for spec in streams_cfg.input_streams)
            else set()
        )

        def enabled(spec: Any) -> bool:
            if not spec.packages:
                return True
            packages = {package.strip() for package in spec.packages.split(";")}
            unknown = packages - inactive_packages - {"no_invariant_stream"}
            if unknown:
                raise MpasConfigError(f"Unknown stream packages: {sorted(unknown)}")
            if not packages & active_packages:
                skipped_streams[spec.name] = f"inactive packages: {sorted(packages)}"
                return False
            return True

        def schedule(interval: Any, *, initial: bool) -> list[int]:
            if interval is NEVER:
                return []
            if interval is INITIAL_ONLY:
                return [0] if initial else []
            quotient = interval / dt
            if not quotient.is_integer() or quotient < 1:
                raise MpasConfigError(
                    "Output cadence must be an integral multiple of dt"
                )
            return ([0] if initial else []) + list(
                range(int(quotient), nsteps + 1, int(quotient))
            )

        def add_plan(spec: Any, steps: list[int]) -> None:
            for step in steps:
                elapsed = step * dt
                if not elapsed.is_integer():
                    raise MpasConfigError(
                        "History/restart valid times must have whole-second precision"
                    )
                xtime = _xtime_for_step(base_time, dt, step)
                filename = (
                    spec.render(xtime) if spec is not None else _history_filename(xtime)
                )
                relative = Path(filename)
                if relative.is_absolute() or ".." in relative.parts:
                    raise MpasConfigError(
                        "Output stream paths must remain inside --output-dir"
                    )
                path = (output_dir / relative).resolve()
                if not path.is_relative_to(output_dir.resolve()):
                    raise MpasConfigError("Output stream path escapes --output-dir")
                if path in targets or path == summary_path:
                    raise MpasConfigError(f"Output path collision: {path}")
                clobber = spec is not None and spec.clobber_mode != "never_modify"
                if path.exists() and not clobber:
                    raise MpasConfigError(f"Output file already exists: {path}")
                targets.add(path)
                plans.setdefault(step, []).append((spec, path))

        if streams_cfg is None:
            steps = schedule(fallback_interval_seconds, initial=not resumed)
            if nsteps not in steps:
                steps.append(nsteps)
            add_plan(None, steps)
        else:
            for spec in streams_cfg.disabled_streams:
                skipped_streams[spec.name] = "type=none"
            for spec in streams_cfg.input_streams:
                if not enabled(spec):
                    continue
                if spec.input_interval_seconds not in {INITIAL_ONLY, NEVER}:
                    raise MpasConfigError(
                        f"Periodic external input is unsupported: {spec.name}"
                    )
                if (
                    spec.name not in {"input", "restart"}
                    and spec.input_interval_seconds is not NEVER
                ):
                    raise MpasConfigError(
                        f"Additional initial input stream is unsupported: {spec.name}"
                    )
            for spec in streams_cfg.output_streams:
                if not enabled(spec):
                    continue
                alarm = spec.output_interval_seconds
                if alarm is NEVER:
                    skipped_streams[spec.name] = "output_interval=none"
                    continue
                if spec.precision == "single" or spec.clobber_mode == "append":
                    raise MpasConfigError(
                        f"Unsupported active output precision/file policy: {spec.name}"
                    )
                file_interval = spec.filename_interval
                if file_interval != "output_interval":
                    parsed = parse_interval(file_interval)
                    if parsed is NEVER or parsed != alarm:
                        raise MpasConfigError(
                            f"Shared-file or unequal filename interval is unsupported: {spec.name}"
                        )
                if spec.name != "restart":
                    fields: list[str] = []
                    excluded = []
                    for field in spec.fields:
                        if field == "scalars":
                            fields.extend(history_scalar_fields(state))
                        elif field == "xtime":
                            continue  # Always written as the time coordinate.
                        elif (
                            field in HISTORY_FIELD_METADATA or field in MOISTURE_FIELDS
                        ):
                            fields.append(field)
                        elif field in stock_unsupported:
                            excluded.append(field)
                        else:
                            raise MpasConfigError(f"Unknown history field: {field!r}")
                    if excluded:
                        skipped_fields[spec.name] = excluded
                        LOGGER.warning(
                            "Stream %s omits fields outside the dycore history subset: %s",
                            spec.name,
                            excluded,
                        )
                    if spec.fields and not fields:
                        skipped_streams[spec.name] = "no supported history fields"
                        continue
                    stream_fields[spec.name] = (
                        list(dict.fromkeys(fields)) if spec.fields else None
                    )
                # Upstream startup suppresses restart output, and history output on resume.
                add_plan(
                    spec,
                    schedule(alarm, initial=spec.name != "restart" and not resumed),
                )
        written_history: list[str] = []
        written_restart: list[str] = []

        def write_step(step: int, step_state: dict[str, Any]) -> None:
            xtime = _xtime_for_step(base_time, dt, step)
            model_time_seconds = model_time_offset + step * dt
            for spec, path in plans.get(step, []):
                clobber = spec is not None and spec.clobber_mode != "never_modify"
                if spec is not None and spec.name == "restart":
                    write_restart(
                        step_state,
                        mesh,
                        path,
                        xtime,
                        config=config,
                        dt=dt,
                        model_time_seconds=model_time_seconds,
                        source_provenance=provenance,
                        clobber=clobber,
                    )
                    written_restart.append(str(path))
                else:
                    write_history(
                        path,
                        step_state,
                        mesh,
                        xtime=xtime,
                        fields=stream_fields.get(spec.name)
                        if spec is not None
                        else None,
                        config=config,
                        dt=dt,
                        model_time_seconds=model_time_seconds,
                        source_provenance=provenance,
                        clobber=clobber,
                    )
                    written_history.append(str(path))

        def initialize_output(prepared: dict[str, Any]) -> None:
            # Check every active history request before installing any output.
            checked: set[str | None] = set()
            for entries in plans.values():
                for spec, _path in entries:
                    if spec is not None and spec.name == "restart":
                        continue
                    name = spec.name if spec is not None else None
                    if name in checked:
                        continue
                    requested = stream_fields.get(name) if name is not None else None
                    _extract_history_fields(
                        prepared,
                        mesh,
                        requested_fields=requested
                        if requested is not None
                        else list(HISTORY_FIELD_METADATA),
                    )
                    checked.add(name)
            write_step(0, prepared)

        final_state = run_simulation(
            initial_state=clone_tree(state),
            mesh=clone_tree(mesh),
            config=dict(config),
            dt=dt,
            nsteps=nsteps,
            device=args.device,
            on_step=write_step,
            on_initial=initialize_output,
        )
        advanced_history = any(
            step > 0
            and any(spec is None or spec.name != "restart" for spec, _path in entries)
            for step, entries in plans.items()
        )
        if not advanced_history:
            LOGGER.warning(
                "No advanced-time history was produced; inspect stream schedules and skipped_streams"
            )
        summary = {
            "nsteps": nsteps,
            "dt": dt,
            "device": args.device,
            "dtype": args.dtype,
            "physics_mode": "dycore",
            "history_files": written_history,
            "restart_files": written_restart,
            "final_xtime": _xtime_for_step(base_time, dt, nsteps),
            "model_time_seconds": model_time_offset + nsteps * dt,
            "final_state_keys": sorted(final_state.keys()),
            "skipped_fields": skipped_fields,
            "skipped_streams": skipped_streams,
            "advanced_time_history_produced": advanced_history,
        }
        with atomic_output(summary_path, clobber=False) as temporary:
            temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return 0
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 2
        return int(code)
    except MpasError as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"MpasConfigError: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
