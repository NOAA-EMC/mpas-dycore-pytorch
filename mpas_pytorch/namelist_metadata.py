"""Generated, package-owned MPAS namelist metadata.

The names in :data:`REGISTRY_OPTION_NAMES` were generated from the frozen
MPAS-Atmosphere v8.3.1 Registry identified by
``release_capabilities.REGISTRY_SHA256``.  Runtime configuration validation
must never discover a source checkout or parse Registry XML.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mpas_pytorch.config_defaults import FORTRAN_DEFAULTS, HONORED_OPTIONS, _config_bool

REGISTRY_OPTION_NAMES: frozenset[str] = frozenset(
    """
    camdim1 config_IAU_option config_IAU_window_length_s config_apply_lbcs
    config_apvm_upwinding config_block_decomp_file_prefix config_bucket_radt
    config_bucket_rainc config_bucket_rainnc config_bucket_update
    config_calendar_type config_camrad_abs_update config_coef_3rd_order
    config_conv_interval config_convection_scheme config_deepsoiltemp_update
    config_del4u_div_factor config_do_DAcycling config_do_restart config_dt
    config_dynamics_split_steps config_epssm config_explicit_proc_decomp
    config_frac_seaice config_gfconv_closure_deep config_gfconv_closure_shallow
    config_greeness_update config_gwdo_scheme config_h_ScaleWithMesh
    config_h_mom_eddy_visc2 config_h_mom_eddy_visc4 config_h_theta_eddy_visc2
    config_h_theta_eddy_visc4 config_halo_exch_method config_horiz_mixing
    config_jedi_da config_knob_ugwp_tauamp config_len_disp config_lsm_scheme
    config_microp_re config_microp_scheme config_mix_full config_monotonic
    config_mpas_cam_coef config_mynn_closure config_mynn_cloudpdf
    config_mynn_dheat_opt config_mynn_edmf config_mynn_edmf_dd
    config_mynn_edmf_mom config_mynn_edmf_output config_mynn_edmf_tke
    config_mynn_mixclouds config_mynn_mixlength config_mynn_mixqt
    config_mynn_mixscalars config_mynn_scaleaware config_mynn_stfunc
    config_mynn_tkeadvect config_mynn_tkebudget config_mynn_topdown
    config_n_microp config_ngw_scheme config_num_halos
    config_number_cam_damping_levels config_number_of_blocks
    config_number_of_sub_steps config_number_rayleigh_damp_u_levels
    config_nvertlevels config_o3climatology config_oml1d config_oml_gamma
    config_oml_hml0 config_oml_relaxation_time config_pbl_interval
    config_pbl_scheme config_physics_suite config_pio_num_iotasks
    config_pio_stride config_positive_definite config_print_detailed_minmax_vel
    config_print_global_minmax_sca config_print_global_minmax_vel
    config_proc_decomp_file_prefix config_radt_cld_dcorrlen
    config_radt_cld_overlap config_radt_cld_scheme config_radt_lw_scheme
    config_radt_sw_scheme config_radtlw_interval config_radtsw_interval
    config_rayleigh_damp_u config_rayleigh_damp_u_timescale_days
    config_relax_zone_divdamp_coef config_restart_timestamp_name
    config_run_duration config_scalar_adv_order config_scalar_advection
    config_scalar_vadv_order config_sfc_albedo config_sfc_snowalbedo
    config_sfclayer_scheme config_smagorinsky_coef config_smdiv
    config_sounding_interval config_split_dynamics_transport config_sst_update
    config_sstdiurn_update config_start_time config_stop_time
    config_theta_adv_order config_theta_vadv_order config_time_integration
    config_time_integration_order config_u_vadv_order config_ugwp_diags
    config_urban_physics config_v_mom_eddy_visc2 config_v_theta_eddy_visc2
    config_visc4_2dsmag config_w_adv_order config_w_vadv_order config_xnutr
    config_ysu_pblmix config_zd input_soil_data input_soil_temperature_lag
    months naerlev noznlev num_soil_layers
    """.split()  # noqa: SIM905 - generated names stay reviewable one-per-token
)


def known_unwired_namelist_keys() -> frozenset[str]:
    """Return valid Registry keys that are not part of the numerical config."""

    wired = set(FORTRAN_DEFAULTS) | set(HONORED_OPTIONS)
    return frozenset(REGISTRY_OPTION_NAMES - wired)


def _normalized_string(value: Any) -> str:
    return str(value).strip().strip("'\"").lower()


def active_unsupported_namelist_keys(raw: Mapping[str, Any]) -> list[str]:
    """Return known keys whose values request unsupported active modes."""

    active: list[str] = []
    suite = _normalized_string(raw.get("config_physics_suite", "none"))
    if suite not in {"", "none", "off"}:
        active.append("config_physics_suite")
    iau = _normalized_string(raw.get("config_IAU_option", "off"))
    if iau not in {"", "off", "none"}:
        active.append("config_IAU_option")
    for key in (
        "config_do_restart",
        "config_do_DAcycling",
        "config_jedi_da",
        "config_sst_update",
        "config_sstdiurn_update",
        "config_deepsoiltemp_update",
        "config_explicit_proc_decomp",
    ):
        if _config_bool(raw.get(key, False)):
            active.append(key)
    if _config_bool(raw.get("config_apply_lbcs", raw.get("apply_lbcs", False))):
        active.append("config_apply_lbcs")
    for key in sorted(REGISTRY_OPTION_NAMES):
        if (
            key.endswith("_scheme")
            and key in raw
            and _normalized_string(raw[key]) not in {"", "none", "off"}
        ):
            active.append(key)
    return active
