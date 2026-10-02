# Changelog

## 0.1.7 — Public execution contracts (2026-10-02)

Configuration validation preserves continuation identity when repeated before
writing or resuming a restart.

- Revalidate current configuration at every public forecast entry point, including
  JVP/VJP dispatch; loader metadata does not authorize subsequent edits.
- Reject active physics, IAU, regional and distributed requests before preparation.
- Reject unsupported floating storage and devices before initialization.
- Preserve canonical split settings when revalidating loader-produced configurations.
- Replace internal issue labels with explanations of the scientific behavior.

### Erratum for 0.1.6

The namelist loader rejected unsupported capabilities, but the Python execution
APIs did not apply the same check. A mutable loader validation flag could bypass
revalidation after edits; an active physics request could return a dycore-only
forecast. Unsupported float32 storage and IAU requests also failed too late with
untyped exceptions. This preview rejects those requests before scientific preparation.
The valid CPU float64 numerical kernels are unchanged. Earlier valid-input results
remain historical evidence; the repaired runtime requires its own bounded qualification.

## 0.1.6 — Input and continuation integrity (2026-10-01)

- Reject missing, NaN and infinite consumed IC/mesh values before conversion.
- Reject nonfinite model results through the Python API.
- Preserve restart mesh/configuration/timestep identity after primary edits.
- Verify complete reference field dimensions and the actual five one-ULP raw twins,
  including unchanged variables, IDs and dry scalar zeros.

### Erratum for 0.1.5

The raw input loader discarded NetCDF missing-value masks. A missing required
value could therefore enter the model as a finite fill value and produce a
nonfinite forecast. Editing a primary restart field also bypassed the independent
mesh/configuration/timestep check. Both inputs are rejected by the corrected
preview. The recipient reference checker now verifies full field coverage and
raw twin perturbations; the earlier checker accepted incomplete synthetic
references. These findings do not invalidate the retained forecasts from valid
inputs, but those checks alone did not establish input or checker robustness.

## 0.1.5 — NOAA research preview (2026-10-01)

- Corrected default timestep checkpointing preserves evolved inter-step diagnostics.
- MPAS-input forecast recipes, visible initial/final history, scalar stream expansion
  and file-preserving writes.
- Complete restart integrity and rejection of stale edited boundaries.
- Public idealized-case preparation, executable forecast/restart/derivative checks,
  API/state contracts and a routine map.
- Reduced CPU temporary storage with ordered scalar-advection row blocks.
- Bounded qualification: six 50-step x1.10242 cases, short option coverage and
  scoped derivatives; tested Linux instructions require baseline CPU dispatch.

Restart files without the complete payload integrity binding are rejected. Generate
new restarts with this version; compatibility with 0.1.0 files is not supported.

### Erratum for 0.1.0

The default differentiable timestep-checkpoint path in 0.1.0 reconstructed some
evolved inter-step diagnostics. It therefore differentiated a slightly different
discrete map from the uncheckpointed forward model, with roundoff-sized numerical
differences. Its multi-step adjoint identities establish transposition of that
checkpointed map and cannot establish equivalence to the forward map. The correction
retains the exact evolved diagnostics; the corrected preview passed the documented split/coupled two/five-step
checkpoint comparisons and bounded scientific checks. The bounded finite-difference limitations are documented
separately in VERIFIED_SCOPE.md.

## 0.1.0

- Initial code-only release of the MPAS-Atmosphere dynamics-core runtime.
- Public forward, restart, history and automatic-differentiation interfaces.
