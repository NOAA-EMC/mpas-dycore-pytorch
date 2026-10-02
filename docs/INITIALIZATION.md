# Prepare the public idealized case

The benchmark uses spherical Jablonowski-Williamson case 2 on x1.10242, 55 vertical
levels and a 720-second timestep. The mesh fetch manifest records its download
location, byte counts and SHA-256 checksums. No forecast data is included.

An existing upstream MPAS-Atmosphere v8.3.1 `init_atmosphere_model` executable is
required. Use a double-precision build. If you need to build it, install matching
C/Fortran compilers, MPI, netCDF C/Fortran, parallel-netCDF and ParallelIO, following
[the upstream build instructions](https://www2.mmm.ucar.edu/projects/mpas/site/documentation/users_guide/building_mpas.html).
Set NETCDF, NETCDFF, PNETCDF and PIO to those installations. In a fresh build directory:

```bash
git clone --branch v8.3.1 --depth 1 https://github.com/MPAS-Dev/MPAS-Model.git MPAS-Model
cd MPAS-Model
test "$(git rev-parse HEAD)" = b9090a1434b8ceb2a92fc985f4d97e7947cd95a6
# Set these on the make command; the GNU target overrides FFLAGS_OPT.
make gfortran CORE=init_atmosphere PRECISION=double AUTOCLEAN=true -j1 \
  FFLAGS="-O3 -ffp-contract=off -ffree-line-length-none -fconvert=big-endian -ffree-form -fdefault-real-8 -fdefault-double-8" \
  CFLAGS="-O3 -ffp-contract=off" CXXFLAGS="-O3 -ffp-contract=off"
make gfortran CORE=atmosphere PRECISION=double AUTOCLEAN=true -j1 \
  FFLAGS="-O3 -ffp-contract=off -ffree-line-length-none -fconvert=big-endian -ffree-form -fdefault-real-8 -fdefault-double-8" \
  CFLAGS="-O3 -ffp-contract=off" CXXFLAGS="-O3 -ffp-contract=off"
```

Keep compiler arithmetic consistent with the reference conditions: disable
multiply-add contraction (`-ffp-contract=off`) and unsafe reassociation/fast-math.
Use the upstream make target for your compiler; the GNU target is shown above.
The initializer and its libraries remain external prerequisites, not bundled code.

From the MPAS-PyTorch distribution directory:

```bash
python examples/prepare_case.py \
  --initializer /path/to/MPAS-Model/init_atmosphere_model \
  --data-dir ./jw
```

The script refuses to reuse `jw`, verifies the downloaded grid, writes a single-rank
partition and global mesh metadata, prepares the upstream namelist/streams, runs
the initializer and checks its log and output. The grid starts on a unit sphere;
the initializer scales it once to Earth radius. Do not substitute a pre-scaled
real-data init/static file. The generated case remains exactly dry.

The result is `jw/case/` containing init.nc, its compatible grid, namelist and
streams. Forward/restart/derivative checks use this same case. Caller-supplied
compatible init/grid files use the same model APIs; the JW reference numbers apply
only to the documented case.

Initialization depends on compiler and platform arithmetic. The fixed numbers in
`known_answers.py` describe the validated macOS case. For a freshly initialized
Linux case, run the independent short Fortran checks from those same inputs:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python examples/validate_model.py --case-dir ./jw/case --output-dir ./validation \
  --fortran-executable /path/to/MPAS-Model/atmosphere_model
```

This runs six serialized two-step Fortran forecasts (control and five one-ULP
input twins), compares all twelve history fields at all three output times, and
runs the same Python API, restart and nonlinear derivative checks. Bounds are
10 times the largest independent twin difference, with fixed float64 norm floors
of 64 epsilon for L2 and 256 epsilon for Linf. A deliberately wrong field must
fail. Neither pointwise agreement near zero nor platform-independent initialization
is claimed. Allow roughly 24 GB RAM and 2 GB output space. Pin the command to one
CPU with `taskset -c CPU_ID` on Linux if comparing timings; this example makes no
performance claim. Debian multiarch netCDF installations may additionally need
`NETCDFLIBLOC=lib/x86_64-linux-gnu PNETCDFLIBLOC=lib/x86_64-linux-gnu` on both make
commands, and explicit netCDF C/Fortran include/library paths in the PIO build.

## Linux CPU arithmetic condition

For Linux x86_64, set `export ATEN_CPU_CAPABILITY=default` before importing Torch
or invoking `mpas-pytorch`. This selects the tested baseline CPU dispatch of
Torch 2.8.0. The AVX2 nonlinear wind example has a disclosed FD roundoff failure;
changing the probe or widening its bound is not part of this recipe. This is an
arithmetic condition for the preview, with no performance claim.


The paired reference receipt uses schema v2. Recreate older references with
`--fortran-executable`; retain the original files for provenance. Verification
checks all field dimensions and missing values, the five actual one-ULP raw IC
changes, all unchanged raw variables and IDs, dry scalar zeros, distinct histories,
and each run's input/executable/log/streams hashes. Keep the reference executable
at its recorded path when reusing `--reference-dir`.

These hashes establish integrity and consistency. They cannot authenticate
arbitrarily fabricated trajectories. Scientific independence requires the trusted
upstream source and compiler conditions above, and running that executable on the
recorded inputs. No reference forecasts or scientific data ship in the package.
