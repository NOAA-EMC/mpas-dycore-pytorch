# Validation scope and limitations

This is a CPU float64 dycore preview derived from MPAS-Atmosphere v8.3.1. The
capability output lists supported requests; a claim of measured behavior applies
only to the configuration, mesh, initial condition and duration actually tested.

## Qualification table

The checks below passed on version 0.1.7. The largest L2/Linf field-error
fraction across the six 50-step cases at steps 0/50 was 0.089578 of its
preregistered bound (JW vertical velocity). This does not extend qualification
to longer forecasts or larger meshes.

| Check | Conditions | Qualification |
|---|---|---|
| Forward accuracy | Six 50-step x1.10242 cases: January split/coupled RK2/RK3, July split RK3, JW case 2 split RK3 | Passed; fixed five-axis one-ULP bounds, all fields and conservation diagnostics |
| Options | 48 cells across 26 scenarios on x1.10242, at most five steps | Passed on the frozen runtime |
| Restart | Split/coupled RK3, continuous 50 versus serialized 25+25 steps | Passed; all 29 complete boundary tensors bitwise equal |
| Checkpoint equivalence | Split/coupled, two/five steps; default and explicit policies; two objectives | Passed; four real order-2 cases, all boundary tensors and five input VJPs bitwise equal |
| Derivative rules | All 12 custom functions; selected one/five-step detach paths; wrong-sign, detach and former-surrogate controls | Passed on the frozen 0.1.7 runtime |
| Recipient example | JW, two steps at dt=720; CLI/API, 2+2 restart, five-control identity and fixed wind FD curve | Passed in all four interpreter/platform cells; macOS fixed 108-point oracle and Linux same-input all-field Fortran oracle |
| Installation | Python 3.11/3.12, macOS arm64 and Debian Linux x86_64 | Same wheel passed all four full model workflows and 188 public tests per cell; Linux runs under emulation |

No new forecast qualification exceeds 50 steps or 10,242 cells. Earlier larger-mesh
and multi-day results remain historical evidence and are outside this preview's
current qualification. Option coverage is not 48 distinct options or every
mesh/order/lead. Complete derivative qualification across all control families and leads is not claimed.

The fixed macOS JW check covers 108 preregistered point values. Two independent
canonical builds agreed bitwise on all 25,378,026 values of twelve fields at steps
0/1/2. Fresh Linux initialization differs from that macOS case, and the original
pointwise check failed on Linux. The portable recipe instead uses freshly run
stock Fortran forecasts from the same inputs and independent five-axis one-ULP
twins, with the same factor-10 L2/Linf contract as the 50-step accuracy gates.
Twin perturbations affect nonzero active controls; exactly dry moisture remains
zero.
The original failed checks remain disclosed; no pointwise portability is claimed.

Tested dependencies are Torch 2.8.0 on macOS and 2.8.0+cpu on Linux, NumPy
2.0.2 and netCDF4 1.7.4. Interpreter versions were 3.11.15/3.12.14 on macOS
26.3 arm64 and 3.11.16/3.12.14 in Debian 12, glibc 2.36, x86_64 containers
under ARM-host emulation. Linux tests use baseline CPU dispatch. Broader
dependency ranges, older glibc and native Linux performance are unqualified.
The largest recipient JVP/VJP relative identity error across these cells was
2.6047544422179137e-14. The example reports its full FD curve and controls.

## Derivative limitations

Six finite-difference disagreements were measured on the preceding preview
runtime. Their raw scientific FD contract remains failed. The affected numerical
kernels are unchanged; this release's input and continuation guards do not
establish a new full-Jacobian qualification. Adjacent
float64 input experiments retain the same branches and stable AD values while
the nonlinear differences vary beyond the acceptance budget. This supports the
bounded resolution limitation; it does not convert those FD failures to passes.
Checkpoint equivalence and the nonlinear wind example passed independently.
Stable, resolvable disagreement is a derivative defect and blocks delivery. A JVP/VJP identity alone establishes transposition of the
selected linearization, not agreement with nonlinear perturbations.

The exactly dry JW moisture boundary contains tied hard-limiter operations. A
single linear Jacobian need not represent all directions there. The recipient
nonlinear example covers wind controls with moisture fixed; it does not claim a
moist-qv adjoint at the dry boundary. No blanket Jacobian statement follows from
this preview.

The recipient nonlinear wind example uses a separate, fixed derivative base:
0.1 m/s sinusoidal vertical motion and initial wind rounded to 16 binary
significant bits. This controls input rounding for the fixed amplitude sweep;
it does not modify the forward/reference case. The scope is this wind direction
and scalar projection on the stated base. Accepted points require unchanged
actual floating-state hard branches; fixed integer mesh topology is verified
separately before and after every nonlinear forecast. The original sparse and unmodified-amplitude curves failed
their numerical checks; they are not represented as passed. No rounding
correction computed with the JVP is used in nonlinear acceptance.

On Linux x86_64, the fixed wind FD example is tested with
`ATEN_CPU_CAPABILITY=default` (Torch reports `NO AVX`). Its isolated check passed
with errors 1.64e-10 and 2.91e-10 at h=2^-28 and 2^-32, below the unchanged
5.92e-10 bound. The same-input AVX2 check failed: the corresponding errors were
4.45e-11 and 1.45e-9. The complete baseline-dispatch Linux workflow passed on both Python versions,
including 25,378,026 field values at steps 0/1/2 against same-input Fortran
references and bitwise 29-tensor 2+2 restart. AVX2 derivative-example portability
is not claimed.

## What we are not claiming

- Forecast skill or operational deployment readiness.
- GPU/MPS, float32, MPI/distributed or regional execution.
- Atmospheric physics or I-JEDI/data-assimilation integration.
- Universal meshes, configurations, stream semantics or restart interchange with Fortran.
- Native Linux performance from an emulated Linux environment, or a fair Fortran speedup.
- Correct derivatives for every hard-switch point, control domain or integration length.

The RK implementation remains large and needs further structural cleanup. Its
scientific interfaces and routine map are documented; the broad refactor is deferred.
