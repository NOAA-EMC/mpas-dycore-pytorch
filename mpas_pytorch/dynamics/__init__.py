"""
MPAS dynamics kernels.

This package contains PyTorch implementations of the MPAS-Atmosphere dynamics
time integration and associated subroutines.

Modules:
    vert_imp_coefs: Vertical implicit coefficients
    smlstep_pert: Small timestep perturbation
    acoustic_step: Acoustic mode solver
    divergence_damping: 3D divergence damping for acoustic noise control
    recover_large_step: Large timestep recovery
    advance_scalars: Scalar field advancement (explicit)
    advance_scalars_mono: Scalar field advancement (monotonic)
    dyn_tend: Dynamical tendency computation
    solve_diagnostics: Diagnostic variable computation
    moist_coefficients: Moist correction coefficients (cqw, cqu)
    stencil_ops: Reusable gather/CSR stencil operators for vectorized kernels
"""
