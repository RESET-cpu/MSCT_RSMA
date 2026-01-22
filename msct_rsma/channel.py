import numpy as np

from .config import linear_gain_from_dbi, wavelength
from .utils import complex_normal, steering_vec_2d, sqrtm_psd


# =============================================================================
# 3GPP TR 38.811 v15.1.0 Table 6.7.2-5a: Suburban LOS S-band
# Rician K-factor parameters (mu_K in dB, sigma_K in dB) by elevation angle
#
# CORRECTED values based on 3GPP TR 38.811 v15.4.0:
# - Suburban LOS S-band has lower K-factor than previously used
# - sigma_K is set to a moderate value (3 dB) for realistic variation
#   (original table had very large sigma causing extreme K values)
# =============================================================================
_KAPPA_TABLE_ELEVATION = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0])
_KAPPA_TABLE_MU_DB = np.array([8.90, 14.00, 11.30, 9.00, 7.50, 6.60, 5.90, 5.50, 5.40])
_KAPPA_TABLE_SIGMA_DB = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0])  # Moderate variance


def _elevation_from_geometry(D_m, H_m):
    """Calculate elevation angle (degrees) from slant distance and satellite height.
    
    Elevation angle α satisfies: sin(α) = H / D
    where D is the slant distance and H is the satellite height.
    """
    sin_alpha = H_m / D_m
    # Clip to handle numerical issues at edge cases
    sin_alpha = np.clip(sin_alpha, 0.0, 1.0)
    return np.rad2deg(np.arcsin(sin_alpha))


def _kappa_sample_by_elevation(elevation_deg, rng):
    """Sample Rician K-factor based on elevation angle using 3GPP 38.811 parameters.
    
    Performs linear interpolation on the 3GPP table to get (mu_K, sigma_K),
    then samples K in dB domain from N(mu_K, sigma_K^2) and converts to linear.
    """
    # Clamp elevation to table range [10, 90] degrees
    elev_clamped = np.clip(elevation_deg, 10.0, 90.0)
    
    # Linear interpolation for mu and sigma
    mu_K_db = np.interp(elev_clamped, _KAPPA_TABLE_ELEVATION, _KAPPA_TABLE_MU_DB)
    sigma_K_db = np.interp(elev_clamped, _KAPPA_TABLE_ELEVATION, _KAPPA_TABLE_SIGMA_DB)
    
    # Sample in dB domain (Gaussian) and convert to linear (log-normal)
    kappa_db = mu_K_db + sigma_K_db * rng.standard_normal()
    return 10.0 ** (kappa_db / 10.0)


def _kappa_sample(cfg, rng):
    """Legacy K-factor sampling (not elevation-dependent). Kept for compatibility."""
    kappa_db = cfg["kappa_mean_db"] + cfg["kappa_std_db"] * rng.standard_normal()
    return 10 ** (kappa_db / 10.0)


def _direction_cosines(vec):
    """Compute direction cosines for UPA steering vector (per paper Fig.1).
    
    The paper uses a spherical parameterization with y-axis as polar axis:
        q^x = sin(θ^y) * cos(θ^x)
        q^y = cos(θ^y)
    
    For a unit direction vector u = (u_x, u_y, u_z), this is equivalent to:
        q^x = u_x
        q^y = u_y
    
    Coordinate system assumption (consistent with paper Fig.1):
        - Satellite UPA: x-y plane parallel to ground (nadir-pointing)
        - UT UPA: x-y plane horizontal (zenith-pointing)
        - z-axis: vertical (satellite height direction)
    
    Parameters
    ----------
    vec : np.ndarray
        3D direction vector (need not be normalized).
    
    Returns
    -------
    qx, qy : float
        Direction cosines along UPA x-axis and y-axis.
    """
    v = vec / np.linalg.norm(vec)
    return v[0], v[1]


def channel_stats(topology, cfg, rng, los_only=False, use_3gpp_kappa=True):
    """Compute channel statistics for all user-satellite links.
    
    Parameters
    ----------
    topology : dict
        Topology information from topology_drop().
    cfg : dict
        Simulation configuration.
    rng : numpy.random.Generator
        Random number generator.
    los_only : bool
        If True, set NLoS covariance to zero (pure LoS channel).
    use_3gpp_kappa : bool
        If True (default), use 3GPP TR 38.811 elevation-dependent K-factor.
        If False, use legacy fixed (mu, sigma) from cfg.
    
    Returns
    -------
    dict
        Channel statistics including beta, kappa, steering vectors, and NLoS covariance.
    """
    K, S = topology["D"].shape
    lam = wavelength(cfg)
    gs = linear_gain_from_dbi(cfg["g_sat_dbi"])
    gu = linear_gain_from_dbi(cfg["g_ut_dbi"])
    Mx, My = cfg["Mx"], cfg["My"]
    Nx, Ny = cfg["Nx"], cfg["Ny"]
    M = Mx * My
    N = Nx * Ny
    H_m = topology["H_m"]

    beta = np.zeros((K, S))
    kappa = np.zeros((K, S))
    elevation = np.zeros((K, S))  # Store elevation angles for debugging/analysis
    g = [[None for _ in range(S)] for _ in range(K)]
    d0 = [[None for _ in range(S)] for _ in range(K)]
    Sigma = [[None for _ in range(S)] for _ in range(K)]

    for k in range(K):
        for s in range(S):
            D = topology["D"][k, s]
            # Path loss with array gain compensation
            # Since steering vectors are unit-norm (||g||=1, ||d0||=1), the array gain
            # must be explicitly included. For rank-1 channel H = d g^H:
            # - Transmit array gain M is realized through beamforming (MRT)
            # - The effective path gain is: beta * M (transmit array gain)
            # This is consistent with standard mMIMO link budget where element gain
            # (Gsat, Gut) is separate from array gain (M, N).
            beta[k, s] = (gs * gu * M) / ((4.0 * np.pi * D / lam) ** 2)
            
            # Compute elevation angle and sample K-factor accordingly
            elev_deg = _elevation_from_geometry(D, H_m)
            elevation[k, s] = elev_deg
            
            if use_3gpp_kappa:
                # 3GPP TR 38.811 elevation-dependent K-factor
                kappa[k, s] = _kappa_sample_by_elevation(elev_deg, rng)
            else:
                # Legacy: fixed (mu, sigma) from config
                kappa[k, s] = _kappa_sample(cfg, rng)

            # LoS steering (satellite)
            sat = np.array([topology["sat_xy"][s, 0], topology["sat_xy"][s, 1], H_m])
            ut = np.array([topology["ut_xy"][k, 0], topology["ut_xy"][k, 1], 0.0])
            v_s = ut - sat
            qx, qy = _direction_cosines(v_s)
            g[k][s] = steering_vec_2d(Mx, My, cfg["dx_lambda"] * lam, cfg["dy_lambda"] * lam, lam, qx, qy)

            # LoS steering (UT)
            v_u = -v_s
            qx_u, qy_u = _direction_cosines(v_u)
            d0[k][s] = steering_vec_2d(Nx, Ny, cfg["dxb_lambda"] * lam, cfg["dyb_lambda"] * lam, lam, qx_u, qy_u)

            if los_only:
                Sigma[k][s] = np.zeros((N, N), dtype=np.complex128)
            else:
                mu = rng.uniform(0.0, 1.0, size=N)
                mu = mu / mu.sum()
                Sigma[k][s] = np.diag(mu)

    return {
        "beta": beta,
        "kappa": kappa,
        "elevation": elevation,  # Elevation angles in degrees (for analysis)
        "g": g,
        "d0": d0,
        "Sigma": Sigma,
        "M": M,
        "N": N,
    }


def build_hat_channels(stats):
    """Construct sCSI design channels for WMMSE optimization.
    
    The sCSI approach uses the channel mean (LoS component) as the design basis,
    but with power scaling to account for total channel power (LoS + NLoS).
    
    For Rician channel: H = d * g^H, where
        d = sqrt(β·κ/(κ+1)) · d0 + sqrt(β/(κ+1)) · d_nlos
    
    Mean: E[H] = sqrt(β·κ/(κ+1)) · d0 · g^H  (this is the LoS channel)
    Total power: E[||d||²] = β·N
    
    The sCSI design channel uses the LoS direction but scales to match total power:
        Ĥ_ks = sqrt(β) · d0 · g^H
    
    This gives ||Ĥ||² = β·N, matching E[||H||²].
    
    Returns
    -------
    list of np.ndarray
        List of K design channels, each of shape (N, M*S).
    """
    beta = stats["beta"]
    kappa = stats["kappa"]
    g = stats["g"]
    d0 = stats["d0"]
    Sigma = stats["Sigma"]
    K, S = beta.shape
    M = stats["M"]
    N = stats["N"]
    Hhat_list = []

    for k in range(K):
        Hhat_k = np.zeros((N, M * S), dtype=np.complex128)
        for s in range(S):
            # Use LoS direction d0 with total power scaling sqrt(β)
            # This matches E[||H||²] = β·N while preserving LoS spatial structure
            #
            # Alternative interpretation: this is the "effective mean" channel
            # that the receiver would see if NLoS scatter had the same direction as LoS
            d_hat = np.sqrt(beta[k, s]) * d0[k][s]
            Hhat_k[:, s * M : (s + 1) * M] = np.outer(d_hat, g[k][s].conj())
        Hhat_list.append(Hhat_k)

    return Hhat_list


def sample_channels(stats, topology, rng):
    beta = stats["beta"]
    kappa = stats["kappa"]
    g = stats["g"]
    d0 = stats["d0"]
    Sigma = stats["Sigma"]
    K, S = beta.shape
    M = stats["M"]
    N = stats["N"]
    H_list = []

    for k in range(K):
        Hk = np.zeros((N, M * S), dtype=np.complex128)
        for s in range(S):
            if np.allclose(Sigma[k][s], 0.0):
                d_hat = np.zeros((N,), dtype=np.complex128)
            else:
                chol = np.linalg.cholesky(Sigma[k][s])
                d_hat = chol @ complex_normal((N,), rng)
            d = np.sqrt(beta[k, s] * kappa[k, s] / (kappa[k, s] + 1.0)) * d0[k][s]
            d += np.sqrt(beta[k, s] / (kappa[k, s] + 1.0)) * d_hat
            Hk[:, s * M : (s + 1) * M] = np.outer(d, g[k][s].conj())
        H_list.append(Hk)
    return H_list


def los_channel_list(stats):
    beta = stats["beta"]
    kappa = stats["kappa"]
    g = stats["g"]
    d0 = stats["d0"]
    K, S = beta.shape
    M = stats["M"]
    N = stats["N"]
    H_list = []
    for k in range(K):
        Hk = np.zeros((N, M * S), dtype=np.complex128)
        for s in range(S):
            d = np.sqrt(beta[k, s] * kappa[k, s] / (kappa[k, s] + 1.0)) * d0[k][s]
            Hk[:, s * M : (s + 1) * M] = np.outer(d, g[k][s].conj())
        H_list.append(Hk)
    return H_list
