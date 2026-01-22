import numpy as np

from .channel import sample_channels
from .utils import fill_common_rates


def _rate_from_cov(signal, cov):
    # log2 det(I + cov^{-1} signal signal^H) for rank-1 signal
    sol = np.linalg.solve(cov, signal)
    val = np.real(signal.conj().T @ sol)
    return np.log2(1.0 + val)


def instantaneous_mmfr(q_c, q_p, H_list, sigma2, use_common=True):
    """Compute instantaneous MMFR using a given channel realization.
    
    This is the correct evaluation for iCSI (instantaneous CSI) scenarios,
    where the precoder is designed for a specific channel realization and
    evaluated on the SAME channel (not different MC samples).
    
    Parameters
    ----------
    q_c : np.ndarray
        Common stream precoder.
    q_p : list of np.ndarray
        Private stream precoders for each user.
    H_list : list of np.ndarray
        Channel matrices for each user (same as used for precoder design).
    sigma2 : float
        Noise variance.
    use_common : bool
        Whether to use common stream (RSMA) or not (SDMA).
    
    Returns
    -------
    float
        Instantaneous MMFR value.
    dict
        Detailed rate information.
    """
    K = len(q_p)
    f_c = np.zeros(K)
    f_p = np.zeros(K)
    
    for k, Hk in enumerate(H_list):
        N = Hk.shape[0]
        
        # Common stream rate (interference from all private streams)
        Sigma_c = sigma2 * np.eye(N, dtype=np.complex128)
        for q in q_p:
            Sigma_c += Hk @ np.outer(q, q.conj()) @ Hk.conj().T
        if use_common:
            sig_c = Hk @ q_c
            f_c[k] = _rate_from_cov(sig_c, Sigma_c)
        
        # Private stream rate (interference from other private streams only)
        Sigma_p = sigma2 * np.eye(N, dtype=np.complex128)
        for j, q in enumerate(q_p):
            if j == k:
                continue
            Sigma_p += Hk @ np.outer(q, q.conj()) @ Hk.conj().T
        sig_p = Hk @ q_p[k]
        f_p[k] = _rate_from_cov(sig_p, Sigma_p)
    
    if not use_common:
        return f_p.min(), {"f_p": f_p, "f_c": f_c, "r_c": np.zeros_like(f_p)}
    
    c_budget = f_c.min()
    r_c, mmfr = fill_common_rates(f_p, c_budget)
    return mmfr, {"f_p": f_p, "f_c": f_c, "r_c": r_c}


def ergodic_mmfr(q_c, q_p, stats, topology, sigma2, num_samples, rng, use_common=True):
    K = len(q_p)
    f_c = np.zeros(K)
    f_p = np.zeros(K)
    for _ in range(num_samples):
        H_list = sample_channels(stats, topology, rng)
        for k, Hk in enumerate(H_list):
            N = Hk.shape[0]
            Sigma_c = sigma2 * np.eye(N, dtype=np.complex128)
            for q in q_p:
                Sigma_c += Hk @ np.outer(q, q.conj()) @ Hk.conj().T
            if use_common:
                sig_c = Hk @ q_c
                f_c[k] += _rate_from_cov(sig_c, Sigma_c)

            Sigma_p = sigma2 * np.eye(N, dtype=np.complex128)
            for j, q in enumerate(q_p):
                if j == k:
                    continue
                Sigma_p += Hk @ np.outer(q, q.conj()) @ Hk.conj().T
            sig_p = Hk @ q_p[k]
            f_p[k] += _rate_from_cov(sig_p, Sigma_p)

    f_c /= num_samples
    f_p /= num_samples

    if not use_common:
        return f_p.min(), {"f_p": f_p, "f_c": f_c, "r_c": np.zeros_like(f_p)}

    c_budget = f_c.min()
    r_c, mmfr = fill_common_rates(f_p, c_budget)
    return mmfr, {"f_p": f_p, "f_c": f_c, "r_c": r_c}
