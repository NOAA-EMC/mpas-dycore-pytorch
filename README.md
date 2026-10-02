# MPAS-PyTorch

This code-only research preview has a bounded validation scope. Read
[validation scope](docs/VERIFIED_SCOPE.md) before using its results.

MPAS-PyTorch is a differentiable implementation of the MPAS-Atmosphere dynamics
core for CPU float64 research integrations. This code-only preview provides
forward integration, package history/restart I/O, and scoped JVP/VJP workflows.

## Install

Use a fresh Python 3.11 or 3.12 environment. On macOS:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
```

On Linux x86_64, install CPU Torch before the package:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install .
export ATEN_CPU_CAPABILITY=default
```

Set `ATEN_CPU_CAPABILITY=default` before importing Torch or running the CLI on
Linux x86_64. The nonlinear derivative example is qualified with this baseline
CPU dispatch; its AVX2 run reached the roundoff wall before satisfying the fixed
FD contract. Both raw outcomes are disclosed in the validation scope.

The supported dependency versions and tested platform conditions are recorded in
[validation scope](docs/VERIFIED_SCOPE.md). GPU, MPS and float32 are outside this
preview. There are no bundled meshes, ICs or reference trajectories.

## Run from MPAS inputs

A forecast requires separate init and compatible grid files plus a namelist:

```bash
mpas-pytorch --config namelist.atmosphere --init init.nc --grid grid.nc \
  --nsteps 2 --output-interval 00:12:00 --output-dir forecast
```

Without streams, the runner writes initialized t=0 history, scheduled intermediate
history and final history. With streams, it honors the declared supported schedules:

```bash
mpas-pytorch --config namelist.atmosphere --init init.nc --grid grid.nc \
  --streams streams.atmosphere --nsteps 2 --output-dir forecast-streams
```

Use fresh output directories: the default file policy preserves existing files.
The summary lists written files, skipped inactive fields and any output-disabled
streams. Durations must be positive integral multiples of dt; explicit `--nsteps`
takes precedence. `mpas-pytorch --help` lists required inputs and defaults.

Restart with the same mesh and configuration, supplying the package restart:

```bash
mpas-pytorch --config namelist.atmosphere --init init.nc --grid grid.nc \
  --restart-in forecast-streams/restart.2026-01-17_06.24.00.nc \
  --nsteps 2 --output-dir resumed
```

General upstream streams semantics are not all implemented. The preview supports
one complete time per history file, float64 precision, scalar expansion and
independent restart input/output schedules. Unsupported active semantics fail
before timestepping. Known inactive physics fields are reported explicitly.

## Run the recipient validation

[Prepare the public JW case](docs/INITIALIZATION.md), then run:

```bash
python examples/validate_model.py --case-dir jw/case --output-dir jw-check \
  --fortran-executable /path/to/MPAS-Model/atmosphere_model
```

This checks two-step CLI/API agreement and all twelve fields against freshly run
Fortran control and five independent one-ULP input twins,
three output times, continuous versus serialized 2+2 restart, edited-state
continuation, a JVP/VJP identity and a nonlinear wind finite-difference curve.
The forward/restart checks use the unmodified JW initial condition. The derivative
example uses a separate fixed control base: 0.1 m/s sinusoidal vertical motion and
initial wind rounded to 16 binary significant bits. This makes the prescribed
wind perturbations exactly representable through h=2^-36. Its accepted curve
points must keep all recorded hard branches unchanged; the script reports the
full curve, including branch changes and the roundoff wall. See
[API.md](docs/API.md#jvp-and-vjp) for the derivative scope.
Negative controls must fail. An unsuccessful check returns a nonzero exit status.
The reference point selection and numerical bounds were fixed before the PyTorch
comparison. Without `--fortran-executable`, the script uses the fixed macOS known-answer
table; those point values are not portable to freshly initialized Linux inputs.
The independent reference bounds and scope are described in the scope document.

[API.md](docs/API.md) gives function signatures, tensor shapes, scalar indexing,
padding rules, AD control domains and history/restart contracts.
[PRIVATE_HANDOFF.md](docs/PRIVATE_HANDOFF.md) describes local archive verification.

## License

New work copyright (c) 2026 IC Weather LLC. Distributed under Apache-2.0.
MPAS-derived portions retain their upstream terms in THIRD_PARTY_NOTICES.md.
