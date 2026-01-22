from .topology import nearest_only_association
from .channel import channel_stats, build_hat_channels, sample_channels, los_channel_list
from .wmmse import wmmse_optimize


def optimize_scsi(topology, cfg, rng, P, use_common=True, noncoop=False, verbose=False, log_prefix=""):
    if noncoop:
        assoc_ks, assoc_sk = nearest_only_association(topology["D"])
        topology = dict(topology)
        topology["assoc_ks"] = assoc_ks
        topology["assoc_sk"] = assoc_sk
    stats = channel_stats(topology, cfg, rng, los_only=False)
    Hhat = build_hat_channels(stats)
    
    q_c, q_p = wmmse_optimize(
        Hhat,
        topology["assoc_ks"],
        P,
        cfg["sigma2"],
        cfg["max_wmmse_iter"],
        cfg["wmmse_tol"],
        use_common,
        rng=rng,
        verbose=verbose,
        log_prefix=log_prefix,
    )
    return stats, q_c, q_p, topology


def optimize_icsi(topology, cfg, rng, P, use_common=True, verbose=False, log_prefix=""):
    stats = channel_stats(topology, cfg, rng, los_only=False)
    H_list = sample_channels(stats, topology, rng)
    q_c, q_p = wmmse_optimize(
        H_list,
        topology["assoc_ks"],
        P,
        cfg["sigma2"],
        cfg["max_wmmse_iter"],
        cfg["wmmse_tol"],
        use_common,
        rng=rng,
        verbose=verbose,
        log_prefix=log_prefix,
    )
    return stats, q_c, q_p


def optimize_dcsi(topology, cfg, rng, P, use_common=True, verbose=False, log_prefix=""):
    stats = channel_stats(topology, cfg, rng, los_only=False)
    H_list = los_channel_list(stats)
    q_c, q_p = wmmse_optimize(
        H_list,
        topology["assoc_ks"],
        P,
        cfg["sigma2"],
        cfg["max_wmmse_iter"],
        cfg["wmmse_tol"],
        use_common,
        rng=rng,
        verbose=verbose,
        log_prefix=log_prefix,
    )
    return stats, q_c, q_p


# RSMA wrappers
def optimize_rsma_scsi(topology, cfg, rng, P, verbose=False, log_prefix=""):
    stats, q_c, q_p, _ = optimize_scsi(
        topology, cfg, rng, P, use_common=True, noncoop=False, verbose=verbose, log_prefix=log_prefix
    )
    return stats, q_c, q_p


def optimize_rsma_icsi(topology, cfg, rng, P, verbose=False, log_prefix=""):
    return optimize_icsi(topology, cfg, rng, P, use_common=True, verbose=verbose, log_prefix=log_prefix)


def optimize_rsma_dcsi(topology, cfg, rng, P, verbose=False, log_prefix=""):
    return optimize_dcsi(topology, cfg, rng, P, use_common=True, verbose=verbose, log_prefix=log_prefix)


def optimize_noncoop_rsma_scsi(topology, cfg, rng, P, verbose=False, log_prefix=""):
    stats, q_c, q_p, topo_nc = optimize_scsi(
        topology, cfg, rng, P, use_common=True, noncoop=True, verbose=verbose, log_prefix=log_prefix
    )
    return stats, q_c, q_p, topo_nc


# SDMA wrappers (use_common=False)
def optimize_sdma_scsi(topology, cfg, rng, P, verbose=False, log_prefix=""):
    stats, q_c, q_p, _ = optimize_scsi(
        topology, cfg, rng, P, use_common=False, noncoop=False, verbose=verbose, log_prefix=log_prefix
    )
    return stats, q_c, q_p


def optimize_sdma_icsi(topology, cfg, rng, P, verbose=False, log_prefix=""):
    return optimize_icsi(topology, cfg, rng, P, use_common=False, verbose=verbose, log_prefix=log_prefix)


def optimize_sdma_dcsi(topology, cfg, rng, P, verbose=False, log_prefix=""):
    return optimize_dcsi(topology, cfg, rng, P, use_common=False, verbose=verbose, log_prefix=log_prefix)


def optimize_noncoop_sdma_scsi(topology, cfg, rng, P, verbose=False, log_prefix=""):
    stats, q_c, q_p, topo_nc = optimize_scsi(
        topology, cfg, rng, P, use_common=False, noncoop=True, verbose=verbose, log_prefix=log_prefix
    )
    return stats, q_c, q_p, topo_nc
