# Verify the supplied handoff

The private collaboration repository is `xtian15/mpas-pytorch`. `xtian15` owns
access management; `ian-icweather` is a collaborator. NOAA colleagues receive
access individually. Repository ownership can later transfer to the agreed NOAA
organization without replacing the delivery history. The delivered code is
Apache-2.0 with IC Weather LLC copyright and the retained upstream notices.

Each delivery identifies its canonical development commit and exported runtime
hashes in `RELEASES.json`. The maintained implementation remains in development;
this repository contains its validated code-only exports. Report proposed fixes
through issues or PRs so they can be validated in that implementation and included
in the next delivery. Do not commit inputs, output or reference data to this repo.

Use the supplied source archive and wheel with matching version and SHA256SUMS.
Verify the outer archive/wheel checksums before extraction, then the source-tree
checksums after extraction:

```bash
shasum -a 256 -c SHA256SUMS
```

On Linux, `sha256sum -c SHA256SUMS` is equivalent. Install the wheel in a fresh
virtual environment using the README's platform instructions. Run the supplied
public-case recipe and `examples/validate_model.py` from the extracted source
archive. The archive supplies examples/docs; the wheel supplies the runtime.
No private repository, model output archive or development environment is needed.

Read VERIFIED_SCOPE.md and CHANGELOG.md with the package. Record your Python,
Torch, NumPy and netCDF versions and platform when reporting a problem, along
with the command, error and run_summary.json. Keep generated inputs and output
under your chosen external data directory. This preview has no bundled
observations or IC data, and no transfer or publication step is part of setup.
