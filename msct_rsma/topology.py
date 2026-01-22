import numpy as np

from .config import km_to_m


def _grid_positions(num_points, spacing):
    n = int(np.ceil(np.sqrt(num_points)))
    coords = []
    offset = (n - 1) / 2.0
    for i in range(n):
        for j in range(n):
            if len(coords) >= num_points:
                break
            coords.append(((i - offset) * spacing, (j - offset) * spacing))
    return np.array(coords)


def coverage_radius_m(H_m, theta_max_deg):
    theta = np.deg2rad(theta_max_deg)
    return H_m * np.tan(theta)


def dmax_m(H_m, theta_max_deg):
    theta = np.deg2rad(theta_max_deg)
    return H_m / np.cos(theta)


def topology_drop(S, K, cfg, rng):
    H_m = km_to_m(cfg["H_km"])
    r_cov = coverage_radius_m(H_m, cfg["theta_max_deg"])
    d_max = dmax_m(H_m, cfg["theta_max_deg"])

    # Satellite spacing: 0.5 × coverage radius to ensure significant overlap
    # This matches the paper's "overlapping service areas" description (Fig. 1)
    # With spacing=0.5*r_cov, ~83% of users fall in multi-satellite coverage
    spacing = r_cov * 0.5 if S > 1 else r_cov
    sat_xy = _grid_positions(S, spacing)

    # Sample UT positions uniformly in union of coverage discs
    min_xy = sat_xy.min(axis=0) - r_cov
    max_xy = sat_xy.max(axis=0) + r_cov
    ut_xy = []
    while len(ut_xy) < K:
        xy = rng.uniform(min_xy, max_xy)
        dists = np.linalg.norm(sat_xy - xy, axis=1)
        if np.any(dists <= r_cov):
            ut_xy.append(xy)
    ut_xy = np.array(ut_xy)

    # Distances and associations
    D = np.zeros((K, S))
    assoc_ks = [[] for _ in range(S)]
    assoc_sk = [[] for _ in range(K)]
    for k in range(K):
        for s in range(S):
            dx = ut_xy[k, 0] - sat_xy[s, 0]
            dy = ut_xy[k, 1] - sat_xy[s, 1]
            D[k, s] = np.sqrt(dx * dx + dy * dy + H_m * H_m)
            if D[k, s] <= d_max:
                assoc_ks[s].append(k)
                assoc_sk[k].append(s)

    # Ensure each UT is served by at least one satellite
    for k in range(K):
        if len(assoc_sk[k]) == 0:
            s = int(np.argmin(D[k]))
            assoc_ks[s].append(k)
            assoc_sk[k].append(s)

    return {
        "H_m": H_m,
        "sat_xy": sat_xy,
        "ut_xy": ut_xy,
        "D": D,
        "assoc_ks": assoc_ks,
        "assoc_sk": assoc_sk,
        "d_max": d_max,
    }


def nearest_only_association(D):
    K, S = D.shape
    assoc_ks = [[] for _ in range(S)]
    assoc_sk = [[] for _ in range(K)]
    for k in range(K):
        s = int(np.argmin(D[k]))
        assoc_ks[s].append(k)
        assoc_sk[k].append(s)
    return assoc_ks, assoc_sk
