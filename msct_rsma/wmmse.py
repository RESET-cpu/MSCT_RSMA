import numpy as np
import cvxpy as cp

from .utils import block_indices


def _power_constraints(q_c, q_p_list, M, S, P):
    constraints = []
    for s in range(S):
        sl = block_indices(M, s)
        power = cp.sum_squares(cp.abs(q_c[sl]))
        for q_p in q_p_list:
            power += cp.sum_squares(cp.abs(q_p[sl]))
        constraints.append(power <= P)
    return constraints


def _apply_structure_constraints(q_p_list, assoc_ks, M):
    constraints = []
    S = len(assoc_ks)
    for s in range(S):
        sl = block_indices(M, s)
        served = set(assoc_ks[s])
        for k, q_p in enumerate(q_p_list):
            if k not in served:
                constraints.append(q_p[sl] == 0)
    return constraints


def _update_uv(H_list, q_c, q_p_list, sigma2):
    """Update auxiliary variables u and v for WMMSE.
    
    Added numerical stability:
    - MSE lower bound to prevent division by zero
    - Weight upper bound to prevent numerical overflow
    """
    K = len(H_list)
    dim = H_list[0].shape[0]
    u_c = []
    v_c = []
    u_p = []
    v_p = []
    
    # Numerical stability constants
    MSE_MIN = 1e-10  # Minimum MSE to prevent v explosion
    V_MAX = 1e10     # Maximum weight to prevent overflow
    
    for k in range(K):
        Hk = H_list[k]
        
        # Common stream
        Sigma_c = sigma2 * np.eye(dim, dtype=np.complex128)
        for q_p in q_p_list:
            Sigma_c += Hk @ np.outer(q_p, q_p.conj()) @ Hk.conj().T
        A_c = Hk @ np.outer(q_c, q_c.conj()) @ Hk.conj().T + Sigma_c
        uck = np.linalg.solve(A_c, Hk @ q_c)
        eck = np.abs(1.0 - uck.conj().T @ Hk @ q_c) ** 2 + uck.conj().T @ Sigma_c @ uck
        eck = max(np.real(eck), MSE_MIN)  # Numerical stability
        u_c.append(uck)
        v_c.append(min(1.0 / eck, V_MAX))  # Numerical stability

        # Private stream
        Sigma_p = sigma2 * np.eye(dim, dtype=np.complex128)
        for j, q_p in enumerate(q_p_list):
            if j == k:
                continue
            Sigma_p += Hk @ np.outer(q_p, q_p.conj()) @ Hk.conj().T
        A_p = Hk @ np.outer(q_p_list[k], q_p_list[k].conj()) @ Hk.conj().T + Sigma_p
        upk = np.linalg.solve(A_p, Hk @ q_p_list[k])
        epk = np.abs(1.0 - upk.conj().T @ Hk @ q_p_list[k]) ** 2 + upk.conj().T @ Sigma_p @ upk
        epk = max(np.real(epk), MSE_MIN)  # Numerical stability
        u_p.append(upk)
        v_p.append(min(1.0 / epk, V_MAX))  # Numerical stability
        
    return u_c, v_c, u_p, v_p


def _update_qr(H_list, u_c, v_c, u_p, v_p, sigma2, assoc_ks, P, M, use_common=True):
    K = len(H_list)
    MS = H_list[0].shape[1]

    q_p = [cp.Variable((MS,), complex=True) for _ in range(K)]
    if use_common:
        q_c = cp.Variable((MS,), complex=True)
        R_c = cp.Variable((K,))
    else:
        q_c = np.zeros((MS,), dtype=np.complex128)
        R_c = np.zeros((K,))
    R_p = cp.Variable((K,))
    t = cp.Variable()

    constraints = [R_p >= 0]
    if use_common:
        constraints.append(R_c >= 0)

    # MMFR epigraph
    if use_common:
        for k in range(K):
            constraints.append(R_c[k] + R_p[k] >= t)
    else:
        for k in range(K):
            constraints.append(R_p[k] >= t)

    # Rate constraints
    for k in range(K):
        Hk = H_list[k]
        if use_common:
            a_c = Hk.conj().T @ u_c[k]
            term1 = cp.square(cp.abs(1.0 - a_c.conj().T @ q_c))
            term2 = 0
            for qpk in q_p:
                term2 += cp.square(cp.abs(a_c.conj().T @ qpk))
            term2 += sigma2 * np.vdot(u_c[k], u_c[k]).real
            e_c = term1 + term2
            xi_c = np.log(v_c[k]) - v_c[k] * e_c + 1.0
            constraints.append(cp.sum(R_c) <= xi_c / np.log(2.0))

        a_p = Hk.conj().T @ u_p[k]
        term1_p = cp.square(cp.abs(1.0 - a_p.conj().T @ q_p[k]))
        term2_p = 0
        for j, qpj in enumerate(q_p):
            if j == k:
                continue
            term2_p += cp.square(cp.abs(a_p.conj().T @ qpj))
        term2_p += sigma2 * np.vdot(u_p[k], u_p[k]).real
        e_p = term1_p + term2_p
        xi_p = np.log(v_p[k]) - v_p[k] * e_p + 1.0
        constraints.append(R_p[k] <= xi_p / np.log(2.0))

    # Power and structure constraints
    if use_common:
        constraints += _power_constraints(q_c, q_p, M, len(assoc_ks), P)
    else:
        constraints += _power_constraints(cp.Constant(q_c), q_p, M, len(assoc_ks), P)
    constraints += _apply_structure_constraints(q_p, assoc_ks, M)

    prob = cp.Problem(cp.Maximize(t), constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    # Handle solver failure: return None values with -inf objective
    if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return None, None, -np.inf

    # Return objective value (MMFR) along with precoders
    obj_value = t.value if t.value is not None else -np.inf

    if use_common:
        q_c_val = q_c.value if q_c.value is not None else np.zeros((MS,), dtype=np.complex128)
        q_p_val = [q.value if q.value is not None else np.zeros((MS,), dtype=np.complex128) for q in q_p]
        return q_c_val, q_p_val, obj_value
    
    q_p_val = [q.value if q.value is not None else np.zeros((MS,), dtype=np.complex128) for q in q_p]
    return np.zeros((MS,), dtype=np.complex128), q_p_val, obj_value


def _mrt_init(H_list, assoc_ks, P, M, use_common):
    """Initialize precoders using MRT (Maximum Ratio Transmission) direction.
    
    This provides a deterministic, channel-aware initialization that:
    - Reduces variance from random initialization
    - Gives a good starting point for WMMSE convergence
    - Avoids poor local optima
    """
    K = len(H_list)
    S = len(assoc_ks)
    MS = M * S
    
    q_c = np.zeros((MS,), dtype=np.complex128)
    q_p = [np.zeros((MS,), dtype=np.complex128) for _ in range(K)]
    
    # For each user k, set q_p[k] proportional to H_k^H (MRT direction)
    for k in range(K):
        Hk = H_list[k]
        # MRT direction: H_k^H @ 1 (sum of columns, or use dominant direction)
        # For rank-1 channel H = d g^H, MRT is proportional to g
        mrt_dir = Hk.conj().T.sum(axis=1)  # Sum over receive antennas
        if np.linalg.norm(mrt_dir) > 1e-10:
            q_p[k] = mrt_dir / np.linalg.norm(mrt_dir)
    
    # For common stream, use average MRT direction
    if use_common:
        for k in range(K):
            q_c += q_p[k]
        if np.linalg.norm(q_c) > 1e-10:
            q_c = q_c / np.linalg.norm(q_c)
    
    # Apply structure constraints (zero out non-serving satellites)
    for s in range(S):
        sl = block_indices(M, s)
        served = set(assoc_ks[s])
        for k in range(K):
            if k not in served:
                q_p[k][sl] = 0.0
    
    # Normalize power per satellite
    for s in range(S):
        sl = block_indices(M, s)
        p = np.sum(np.abs(q_c[sl]) ** 2) if use_common else 0.0
        for q in q_p:
            p += np.sum(np.abs(q[sl]) ** 2)
        if p > 1e-10:
            scale = np.sqrt(P / p)
            if use_common:
                q_c[sl] *= scale
            for q in q_p:
                q[sl] *= scale
    
    return q_c, q_p


def wmmse_optimize(
    H_list,
    assoc_ks,
    P,
    sigma2,
    max_iter=40,
    tol=1e-3,
    use_common=True,
    q_init=None,
    rng=None,
    verbose=False,
    log_prefix="",
):
    K = len(H_list)
    MS = H_list[0].shape[1]
    M = MS // len(assoc_ks)

    # Use provided rng or create a default one for reproducibility
    if rng is None:
        rng = np.random.default_rng()

    if q_init is None:
        # Use MRT-based initialization for better convergence and reduced variance
        q_c, q_p = _mrt_init(H_list, assoc_ks, P, M, use_common)
    else:
        q_c, q_p = q_init

    last_obj = -np.inf
    prefix = log_prefix or ""
    for it in range(1, max_iter + 1):
        if verbose:
            print(f"{prefix}[WMMSE] iter {it}: update u/v")
        if use_common:
            u_c, v_c, u_p, v_p = _update_uv(H_list, q_c, q_p, sigma2)
        else:
            dummy_qc = np.zeros_like(q_p[0])
            u_c, v_c, u_p, v_p = _update_uv(H_list, dummy_qc, q_p, sigma2)
        if verbose:
            print(f"{prefix}[WMMSE] iter {it}: solve Q/R")
        q_c_new, q_p_new, current_obj = _update_qr(H_list, u_c, v_c, u_p, v_p, sigma2, assoc_ks, P, M, use_common)

        # Handle solver failure: keep previous solution
        if q_c_new is None or q_p_new is None:
            if verbose:
                print(f"{prefix}[WMMSE] iter {it}: solver failed, stop")
            break
        
        q_c, q_p = q_c_new, q_p_new

        # Convergence check based on objective function (MMFR) change
        if verbose:
            delta = np.abs(current_obj - last_obj) if np.isfinite(last_obj) else np.inf
            print(f"{prefix}[WMMSE] iter {it}: obj={current_obj:.6f}, delta={delta:.3e}")
        if np.abs(current_obj - last_obj) <= tol:
            if verbose:
                print(f"{prefix}[WMMSE] iter {it}: converged")
            break
        last_obj = current_obj

    return q_c, q_p
