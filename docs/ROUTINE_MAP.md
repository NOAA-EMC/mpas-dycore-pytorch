# Dynamics routine map

These entry points help an MPAS developer navigate the Python implementation.
The corresponding upstream files are in MPAS-Atmosphere v8.3.1's atmosphere core.
The dynamics routines below are in `dynamics/mpas_atm_time_integration.F`.
Python work functions use explicit tensors in place of upstream derived types.

| Python module/function | Upstream routine or role |
|---|---|
| `atmosphere_model.run_simulation` | Core forecast loop and complete timestep boundary |
| `atmosphere_model.run_timestep` | Timestep initialization and dynamics driver |
| `dynamics/time_integration.atm_srk3` and `execute_rk_step` | `atm_srk3`, split/coupled RK stages |
| `dynamics/time_integration.atm_rk_integration_setup` | `atm_rk_integration_setup` |
| `dynamics/dyn_tend.atm_compute_dyn_tend_work` | `atm_compute_dyn_tend_work` |
| `dynamics/smlstep_pert.atm_set_smlstep_pert_variables_work` | `atm_set_smlstep_pert_variables_work` |
| `dynamics/acoustic_step.atm_advance_acoustic_step_work` | `atm_advance_acoustic_step_work` |
| `dynamics/divergence_damping.atm_divergence_damping_3d` | `atm_divergence_damping_3d` |
| `dynamics/recover_large_step.atm_recover_large_step_variables_work` | `atm_recover_large_step_variables_work` |
| `dynamics/solve_diagnostics.atm_compute_solve_diagnostics_work` | `atm_compute_solve_diagnostics_work` |
| `dynamics/vert_imp_coefs.atm_compute_vert_imp_coefs_work` | `atm_compute_vert_imp_coefs_work` |
| `dynamics/advance_scalars.atm_advance_scalars_work` | `atm_advance_scalars_work` |
| `dynamics/advance_scalars_mono.atm_advance_scalars_mono_work` | `atm_advance_scalars_mono_work` |
| `dynamics/vector_reconstruction` | MPAS vector reconstruction and declared wind-control maps |
| `diagnostics.physics_interface_surface_pressure` | `mpas_atmphys_interface.F` surface pressure formula |
| `initialization` and `dynamics/init_coupled_diagnostics` | Cold-start coupled diagnostic preparation |

The RK driver remains large. Decomposing it further is deferred until after this
preview. Checkpointing, static geometry support and differentiable linear solves
are used by the CPU/AD path; their presence does not claim GPU capability.
