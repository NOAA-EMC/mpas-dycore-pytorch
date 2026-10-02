# Python API

Use CPU float64 tensors. The public functions accept dictionaries so geometry,
state and configuration can be inspected directly. The root `mpas_pytorch.__all__`
is the supported import surface. Missing scientific inputs are errors.

## Load a model

```python
from pathlib import Path
import mpas_pytorch as mpas

case = Path("jw/case")
config = mpas.load_config_from_namelist(case / "namelist.atmosphere")
state, mesh = mpas.load_initial_state(
    case / "x1.10242.init.nc", case / "x1.10242.grid.nc", config=config
)
final = mpas.run_simulation(state, mesh, config=config, dt=720.0, nsteps=2)
```

Signatures:

```python
load_config_from_namelist(namelist_path: str | Path, *, strict=True) -> dict
load_initial_state(init_path: str | Path, grid_path: str | Path,
                   mesh_support_path: str | Path | None = None, *,
                   device="cpu", config: dict | None = None) -> tuple[dict, dict]
run_timestep(state: dict, mesh: dict, config: dict | None = None,
             dt: float | None = None, device="cpu",
             differentiable=False) -> dict
run_simulation(initial_state: dict, mesh: dict, config: dict | None = None,
               dt: float | None = None, nsteps=1, device="cpu", on_step=None,
               differentiable=False, checkpoint_timesteps: bool | None = None,
               checkpoint_use_reentrant=False, checkpoint_materialize_final=True,
               on_phase=None, on_initial=None) -> dict
```

The grid is a separate, compatible MPAS grid/static file. Passing the init file
as the grid is unsupported. Raw input state is prepared by the model before its
first timestep: virtual potential temperature, Jacobian-weighted density and
coupled diagnostics are derived there. A raw loader return is not a complete
history or timestep boundary.

`dt=None` reads `config_dt`. Positive finite dt and integral nonnegative API step
counts are required. A zero-step API call performs no initialization or integration.
CLI forecasts require at least one step. `on_initial(state)` observes the prepared
boundary consumed by step one; `on_step(step_number, state)` observes each completed
step, numbered from one. Callbacks must not mutate the supplied dictionaries or
tensors. Copy any values you need to keep. Output observation does not change the
model's numerical integration.

The model fills missing supported configuration keys with defaults. A namelist
can contain familiar upstream settings, but unsupported active capabilities raise
errors. Inspect `mpas-pytorch --capabilities`; it describes supported requests,
while [validation scope](VERIFIED_SCOPE.md) describes measured coverage.

## State and controls

Let C, E, V and L be active cells, edges, vertices and vertical mass levels.
Loaded runtime tensors carry a trailing padding row. It represents missing-neighbor
connectivity and is excluded from scientific reductions and perturbations. Python
active entity and vertical indices start at zero; upstream connectivity is converted
by the loader. Do not rewrite connectivity or move the padding row.

| Key | Runtime shape | Meaning |
|---|---|---|
| `u` | `(E+1, L)` | Edge-normal horizontal velocity, m/s |
| `w` | `(C+1, L+1)` | Vertical velocity at W levels, m/s |
| `theta_m` | `(C+1, L)` | Virtual potential temperature, K |
| `rho_zz` | `(C+1, L)` | Jacobian-weighted dry density |
| `scalars` | `(C+1, L, S)` | Active scalar mixing ratios, kg/kg |
| `rt_diabatic_tend` | `(C+1, L)` when present | Carried thermodynamic tendency |

Moisture species order is `qv, qc, qr, qi, qs, qg`. When loading separate moisture
fields, the loader packs six slots and fills absent species with zeros. An input
already containing `scalars` retains its supplied slot count; it must follow this
order. The history writer supports the first six species in that order.
The loader exposes `qv` alongside the vapor component of `scalars`. Read evolved
vapor as `state["scalars"][..., 0]`; an auxiliary loader key need not remain in a
returned timestep boundary. Perturb the scalar slot once. Replacing a tensor by an independent leaf can change alias
ownership, so the examples differentiate the primary `scalars` tensor explicitly.

A returned complete timestep boundary also carries evolved diagnostics needed by
the next step. Preserve the complete returned state for continuation or serialize
it with `write_restart`. Unchanged boundaries reuse those diagnostics; changes to
prognostics invalidate that reuse and cause recomputation. Extracting only five
arrays is not an exact restart. Never edit internal boundary markers.
After editing primary tensors in a returned boundary, advance it through `run_timestep` before saving
that resulting boundary. `write_restart` rejects edited boundaries carrying stale
diagnostics. Editing diagnostic/support tensors or changing the continuation
mesh, configuration or timestep is rejected. The RK temperature alias follows the
primary temperature edit. New restart files bind the complete saved payload and
metadata; changing a saved field makes loading fail. Readers also check actual
float64 storage, finite data and supported metadata. Older files without the
complete integrity binding, including 0.1.0 restarts, must be regenerated.

The in-memory boundary binding retains immutable copies of numerical values,
sharing identical views. On x1.10242 this adds about 0.87 GiB per full boundary;
holding several boundaries increases memory use. Clones share the immutable
binding bytes. The binding catches edits through NumPy views and `.data` as well
as ordinary tensor edits. Disk restart integrity uses SHA-256. In-memory bindings
from an older runtime are unsupported; reload a supported restart instead.

## JVP and VJP

```python
run_simulation.jvp(state: dict, mesh: dict, perturbation: dict, **kwargs) -> dict
run_simulation.vjp(state: dict, mesh: dict, objective,
                   track_keys=None, **kwargs) -> dict
```

These differentiate one declared discrete integration with fixed mesh, dt,
configuration and all unperturbed state entries. They are not sensitivities to
terrain, connectivity, timestep or configuration switches. A derivative with respect
to the prepared state is different from a derivative with respect to raw IC fields.
For prepared-state controls, capture the initialized state through `on_initial`.

For a JVP, each perturbation must match its state tensor's shape, dtype and device.
Omitted keys are held fixed. Set padding perturbations to zero. The return contains
output tensors with propagated tangents. For a VJP, the objective must return a
scalar tensor connected to the model output; specify tracked keys explicitly when
using a subset. Defaults include the carried inter-step prognostics that are present.
Unused inputs may be absent from the returned gradient dictionary.

```python
import torch

# prepared is a copied on_initial boundary; geometry and config stay fixed.
direction = torch.zeros_like(prepared["u"])
direction[0, 27] = 1.0
kwargs = dict(config=config, dt=720.0, nsteps=1, checkpoint_timesteps=False)
tangent = mpas.run_simulation.jvp(prepared, mesh, {"u": direction}, **kwargs)
gradient = mpas.run_simulation.vjp(
    prepared, mesh, lambda output: output["u"][0, 27],
    track_keys=("u",), **kwargs
)
print(tangent["u"][0, 27], gradient["u"][0, 27])
```

`checkpoint_timesteps=None` selects the differentiated runtime's default checkpoint
policy. `False` retains the graph; `True` trades recomputation for graph storage.
The default non-reentrant path supports autograd gradients. The recipient example
uses `False` to make its short curve straightforward; release qualification checks
checkpointed and uncheckpointed maps separately. Memory requirements grow with mesh,
lead time, control domain and objective; checkpointing does not establish operational
memory bounds.

A JVP/VJP identity checks transposition of the chosen AD linearization. At exact
hard-limiter ties there may be no single linear derivative of the nonlinear model.
The forward/restart example uses the unmodified dry JW case. Its nonlinear
derivative check uses a separate fixed base with vertical motion
`w[k]=0.1*sin(pi*k/L)` m/s (bottom/top/padding zero) and initial `u` rounded to
16 binary significant bits. All other primary inputs remain fixed. The amplitude
direction is that base `u`; the scalar response projects output wind onto the
fixed normalized base wind. Every prescribed perturbation through `h=2^-36` must
match an independently computed exact dyadic value. The eleven-point central-FD
curve retains its original error bound and requires unchanged actual hard
branches at accepted points, then a roundoff wall. Earlier failed curves on the
unmodified base do not count as nonlinear verification. This example does not establish
a moist-qv adjoint at the dry boundary. See the bounded limitations in the scope file.

## History and restart I/O

```python
write_history(filepath: str | Path, state: dict, mesh: dict, *, xtime="",
              global_attrs=None, fields=None, config=None, dt=None,
              model_time_seconds=0.0, initial_time=None,
              source_provenance="caller-provided in-memory state",
              clobber=False) -> None
read_history(filepath: str | Path, *, validate=True) -> dict
write_restart(state: dict, mesh: dict, path: str | Path, xtime: str, *,
              initial_time=None, config=None, dt=None, model_time_seconds=0.0,
              source_provenance="caller-provided in-memory state",
              clobber=False) -> None
read_restart(path: str | Path, *, mesh=None, config=None, dt=None,
             device="cpu") -> tuple[dict, str]
```

Histories contain one time record per file, active entities only, float64 values,
`xtime`, and a numeric Gregorian time coordinate. `read_history` returns `fields`
(NumPy arrays), `dimensions` (sizes; unlimited dimensions are `None`) and `attributes`.
The twelve diagnostic fields are:

| Field | Active dimensions after Time | Units |
|---|---|---|
| `u` | E, L | m/s |
| `w` | C, L+1 | m/s |
| `theta` | C, L | K |
| `rho` | C, L | kg/m³ |
| `zz` | C, L | 1 |
| `qv` | C, L | kg/kg |
| `pressure` | C, L | Pa |
| `divergence` | C, L | 1/s |
| `vorticity` | V, L | 1/s |
| `ke` | C, L | m²/s² |
| `uReconstructZonal` | C, L | m/s |
| `uReconstructMeridional` | C, L | m/s |

`fields=["scalars"]` expands to active named species. `fields=None` writes available
supported fields. A requested required field must exist. Single-precision output,
shared-file append and general upstream Registry output are outside this preview.

For exact continuation, supply mesh, config and dt to `read_restart`. The reader
checks the complete saved payload, schema and those identities. Identity attributes
must be canonical SHA-256 strings; malformed or missing identity metadata raises
`MpasConfigError`. The reader accepts CPU tensors only and raises `MpasDeviceError`
for other devices before loading scientific arrays. Its second return is valid time; the state
includes `_restart_model_time_seconds` for the accumulated elapsed time. Pass that
offset into subsequent I/O. Package restarts are not interchangeable with Fortran
restarts. Compatibility with older package versions is stated in the changelog only
when a preserved-file continuation test has passed.

Writers default to preserving existing files. Explicit `clobber=True` permits
replacement only after a complete new file has been written and validated. Invalid
metadata or an interrupted replacement must preserve the existing target.

## Surface-pressure diagnostic

```python
from mpas_pytorch.diagnostics import physics_interface_surface_pressure

physics_interface_surface_pressure(
    *, nCells: int, rho_zz: Tensor, zz: Tensor, qv: Tensor,
    pressure_p: Tensor, pressure_base: Tensor, zgrid: Tensor,
    gravity: float = 9.80616,
) -> Tensor
```

This standalone helper evaluates the hydrostatic surface-pressure formula from
the MPAS physics interface on an evolved dynamics state. It does not run physics.
All density, metric, moisture and pressure inputs are cell-major `(C, L)` or
`(C+1, L)` tensors with matching extents; `zgrid` supplies W-level heights in
metres. At least two mass levels and three W levels are required. Use evolved
`scalars[..., 0]` for `qv`, the density `rho_zz`, metric `zz`, perturbation pressure
`pressure_p`, and base pressure `pressure_base` from the same state and mesh.
The result is one pressure in Pa per input cell row. If the inputs have trailing
padding, its result is zero and is excluded from hydrostatic calculations.
The helper supports PyTorch differentiation through these tensors.

## Exceptions

Catch `MpasError` for a typed public failure. Configuration-related errors also
inherit `ValueError`. Exported subclasses are `MpasConfigError`, `MpasDeviceError`,
`MpasMeshMismatchError`, `MpasMissingFieldError`, `MpasNumericalAgreementError`,
`MpasProvenanceError`, `MpasStrictModeError` and `MpasUnsupportedCapabilityError`.
A programming error can still raise a normal Python exception. The CLI reports
anticipated input/configuration errors on stderr and returns exit status 2.
