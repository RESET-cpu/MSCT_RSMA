import numpy as np


def complex_normal(shape, rng):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def steering_vec_2d(nx, ny, dx, dy, lam, qx, qy):
    """Compute 2D UPA steering vector with unit norm normalization.
    
    Implements the paper's UPA array response (Eq. around system model):
        a(θ) ∈ C^{M}, where M = M_x × M_y
        
    The (m_x, m_y)-th element is:
        [a(θ)]_{m_x,m_y} = exp(-j 2π/λ (m_x d_x q^x + m_y d_y q^y))
    
    where q^x, q^y are direction cosines:
        q^x = sin(θ^y) cos(θ^x)
        q^y = cos(θ^y)
    
    The steering vector is normalized to unit norm: ||a||^2 = 1.
    
    Note on array gain:
        The paper's formula uses 1/nv normalization per axis, which would give
        ||a||^2 = 1/(nx*ny). However, this causes the beamforming array gain 
        (M for Tx, N for Rx) to be lost, leading to extremely low SNR.
        
        We use unit-norm normalization (||a||^2 = 1) to preserve the array gain
        in the precoding/combining design. This is consistent with standard
        MIMO precoding practice where beamforming gain is realized through
        coherent combining of antenna elements.
    
    Parameters
    ----------
    nx, ny : int
        Number of antenna elements in x and y directions (M_x, M_y).
    dx, dy : float
        Antenna spacing in x and y directions (in meters).
    lam : float
        Carrier wavelength λ (in meters).
    qx, qy : float
        Direction cosines along x-axis and y-axis.
    
    Returns
    -------
    np.ndarray
        Steering vector of shape (nx * ny,) with unit norm ||a||^2 = 1.
        Elements ordered as kron(a_x, a_y), i.e., y varies fastest.
    """
    # a_x[m_x] = exp(-j 2π d_x/λ q^x m_x)
    ex = np.exp(-1j * 2.0 * np.pi * dx / lam * qx * np.arange(nx))
    # a_y[m_y] = exp(-j 2π d_y/λ q^y m_y)
    ey = np.exp(-1j * 2.0 * np.pi * dy / lam * qy * np.arange(ny))
    # Kronecker product with unit-norm normalization to preserve array gain
    a = np.kron(ex, ey)
    return a / np.sqrt(nx * ny)  # ||a||^2 = 1


def sqrtm_psd(mat):
    evals, evecs = np.linalg.eigh((mat + mat.conj().T) * 0.5)
    evals = np.clip(evals, 0.0, None)
    return (evecs * np.sqrt(evals)) @ evecs.conj().T


def block_indices(block_size, block_id):
    start = block_id * block_size
    end = start + block_size
    return slice(start, end)


def fill_common_rates(f_p, c_budget):
    f_p = np.asarray(f_p).astype(float)
    k = f_p.size
    order = np.argsort(f_p)
    f_sorted = f_p[order]
    used = 0.0
    level = f_sorted[0]
    for i in range(k - 1):
        next_level = f_sorted[i + 1]
        need = (next_level - level) * (i + 1)
        if used + need >= c_budget:
            level = level + (c_budget - used) / (i + 1)
            used = c_budget
            break
        used += need
        level = next_level
    if used < c_budget:
        level = level + (c_budget - used) / k

    r_c = np.maximum(0.0, level - f_p)
    if r_c.sum() > c_budget + 1e-9:
        r_c *= c_budget / r_c.sum()
    return r_c, level
