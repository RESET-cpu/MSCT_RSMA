import numpy as np


def default_sim_config():
    # Geometry and RF
    cfg = {}
    cfg["H_km"] = 600.0
    cfg["theta_max_deg"] = 30.0
    cfg["fc_hz"] = 2.0e9
    cfg["bw_hz"] = 50.0e6
    cfg["tn_kelvin"] = 290.0
    cfg["g_sat_dbi"] = 6.0
    cfg["g_ut_dbi"] = 0.0

    # Antennas
    cfg["Mx"] = 5
    cfg["My"] = 5
    cfg["Nx"] = 4
    cfg["Ny"] = 4
    cfg["dx_lambda"] = 1.0
    cfg["dy_lambda"] = 1.0
    cfg["dxb_lambda"] = 0.5
    cfg["dyb_lambda"] = 0.5

    # Rician K-factor: 3GPP TR 38.811 S-band Suburban LOS (Table 6.7.2-5a)
    # By default, K-factor is sampled based on elevation angle (use_3gpp_kappa=True).
    # The values below are legacy fallback when use_3gpp_kappa=False.
    # 3GPP 38.811 Suburban LOS S-band typical range: mu_K ~ 5.4-14.0 dB
    cfg["kappa_mean_db"] = 7.5   # Fallback: ~50 deg elevation typical value
    cfg["kappa_std_db"] = 3.0    # Moderate variance to avoid extreme values

    # Simulation
    cfg["seed"] = 7
    cfg["num_drops"] = 20
    cfg["num_mc_eval"] = 20000
    cfg["max_wmmse_iter"] = 40
    cfg["wmmse_tol"] = 1e-3

    # Power
    cfg["P_dBW_list"] = list(range(0, 26, 5))
    cfg["S_list"] = [1, 2, 3, 4, 5, 6]

    return cfg


def noise_variance(cfg):
    k_b = 1.380649e-23
    return k_b * cfg["tn_kelvin"] * cfg["bw_hz"]


def wavelength(cfg):
    c = 3.0e8
    return c / cfg["fc_hz"]


def linear_gain_from_dbi(g_dbi):
    return 10 ** (g_dbi / 10.0)


def km_to_m(x_km):
    return x_km * 1000.0
