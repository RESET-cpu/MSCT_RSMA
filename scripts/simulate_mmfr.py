import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Allow running without setting PYTHONPATH
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from msct_rsma.config import default_sim_config, noise_variance
from msct_rsma.topology import topology_drop
from msct_rsma.wmmse import wmmse_optimize
from msct_rsma.eval import ergodic_mmfr, instantaneous_mmfr


def _linear_power_watts(p_dbw):
    return 10 ** (p_dbw / 10.0)


def _parse_algs(algs):
    if algs is None:
        return None
    if isinstance(algs, (list, tuple)):
        raw = ",".join(algs)
    else:
        raw = str(algs)
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return set(parts) if parts else None


def _want(algs, key):
    return algs is None or key in algs


def run_P(cfg, verbose=False, algs=None):
    cfg["sigma2"] = noise_variance(cfg)
    S, K = 4, 10
    results = []
    start_time = time.time()
    print(f"[P] start: S={S}, K={K}, drops={cfg['num_drops']}", flush=True)
    
    # Import channel functions once
    from msct_rsma.channel import channel_stats, build_hat_channels, sample_channels, los_channel_list
    from msct_rsma.topology import nearest_only_association
    
    for p_dbw in cfg["P_dBW_list"]:
        P = _linear_power_watts(p_dbw)
        mmfr = {
            "P_dBW": p_dbw,
            "rsma_scsi": [],
            "sdma_scsi": [],
            "rsma_icsi": [],
            "sdma_icsi": [],
            "rsma_dcsi": [],
            "sdma_dcsi": [],
            "rsma_scsi_noncoop": [],
            "sdma_scsi_noncoop": [],
        }
        print(f"[P] P={p_dbw} dBW: start", flush=True)
        for drop_idx in range(1, cfg["num_drops"] + 1):
            if not verbose:
                ant = f"{cfg['Mx']}x{cfg['My']}|{cfg['Nx']}x{cfg['Ny']}"
                print(
                    f"[P] P={p_dbw} dBW, S={S}, K={K}, Ant={ant}: "
                    f"drop {drop_idx}/{cfg['num_drops']}",
                    flush=True,
                )
            
            # ============================================================
            # CRITICAL: Use deterministic per-drop seeds
            # This ensures different algorithm runs produce IDENTICAL channels
            # ============================================================
            # Base seed combines global seed, P index, and drop index
            p_idx = cfg["P_dBW_list"].index(p_dbw)
            drop_base_seed = cfg["seed"] * 10000 + p_idx * 100 + drop_idx
            
            # Separate RNGs for different purposes (ensures reproducibility across runs)
            topo_rng = np.random.default_rng(drop_base_seed)
            channel_rng = np.random.default_rng(drop_base_seed + 1)
            icsi_rng = np.random.default_rng(drop_base_seed + 2)
            eval_seed = drop_base_seed + 3
            alg_rng = np.random.default_rng(drop_base_seed + 4)
            
            # Generate topology and channel (deterministic for this drop)
            topo = topology_drop(S, K, cfg, topo_rng)
            stats = channel_stats(topo, cfg, channel_rng, los_only=False)
            Hhat = build_hat_channels(stats)
            
            # For non-cooperative: create modified topology with nearest-only association
            assoc_ks_nc, assoc_sk_nc = nearest_only_association(topo["D"])
            topo_nc = dict(topo)
            topo_nc["assoc_ks"] = assoc_ks_nc
            topo_nc["assoc_sk"] = assoc_sk_nc
            Hhat_nc = build_hat_channels(stats)
            
            # For iCSI: sample ONE channel realization
            H_icsi = sample_channels(stats, topo, icsi_rng)
            
            # For dCSI: use LoS-only channels (deterministic)
            H_dcsi = los_channel_list(stats)

            # sCSI RSMA (cooperative)
            if _want(algs, "rsma_scsi"):
                q_c, q_p = wmmse_optimize(
                    Hhat, topo["assoc_ks"], P, cfg["sigma2"],
                    cfg["max_wmmse_iter"], cfg["wmmse_tol"], True, rng=alg_rng
                )
                eval_rng = np.random.default_rng(eval_seed)
                val, _ = ergodic_mmfr(q_c, q_p, stats, topo, cfg["sigma2"], cfg["num_mc_eval"], eval_rng, True)
                mmfr["rsma_scsi"].append(val)

            # sCSI SDMA (cooperative)
            if _want(algs, "sdma_scsi"):
                q_c0, q_p0 = wmmse_optimize(
                    Hhat, topo["assoc_ks"], P, cfg["sigma2"],
                    cfg["max_wmmse_iter"], cfg["wmmse_tol"], False, rng=alg_rng
                )
                eval_rng = np.random.default_rng(eval_seed)
                val, _ = ergodic_mmfr(q_c0, q_p0, stats, topo, cfg["sigma2"], cfg["num_mc_eval"], eval_rng, False)
                mmfr["sdma_scsi"].append(val)

            # iCSI RSMA
            # NOTE: For iCSI, we use instantaneous_mmfr with the SAME channel used for design.
            # This is correct because iCSI assumes perfect instantaneous CSI is available,
            # so the precoder is optimized for the actual channel being used.
            if _want(algs, "rsma_icsi"):
                q_c_i, q_p_i = wmmse_optimize(
                    H_icsi, topo["assoc_ks"], P, cfg["sigma2"],
                    cfg["max_wmmse_iter"], cfg["wmmse_tol"], True, rng=alg_rng
                )
                # Use instantaneous_mmfr with the same channel H_icsi (not ergodic sampling!)
                val, _ = instantaneous_mmfr(q_c_i, q_p_i, H_icsi, cfg["sigma2"], True)
                mmfr["rsma_icsi"].append(val)
            
            # iCSI SDMA
            if _want(algs, "sdma_icsi"):
                q_c_i0, q_p_i0 = wmmse_optimize(
                    H_icsi, topo["assoc_ks"], P, cfg["sigma2"],
                    cfg["max_wmmse_iter"], cfg["wmmse_tol"], False, rng=alg_rng
                )
                # Use instantaneous_mmfr with the same channel H_icsi (not ergodic sampling!)
                val, _ = instantaneous_mmfr(q_c_i0, q_p_i0, H_icsi, cfg["sigma2"], False)
                mmfr["sdma_icsi"].append(val)

            # dCSI RSMA
            if _want(algs, "rsma_dcsi"):
                q_c_d, q_p_d = wmmse_optimize(
                    H_dcsi, topo["assoc_ks"], P, cfg["sigma2"],
                    cfg["max_wmmse_iter"], cfg["wmmse_tol"], True, rng=alg_rng
                )
                eval_rng = np.random.default_rng(eval_seed)
                val, _ = ergodic_mmfr(q_c_d, q_p_d, stats, topo, cfg["sigma2"], cfg["num_mc_eval"], eval_rng, True)
                mmfr["rsma_dcsi"].append(val)

            # dCSI SDMA
            if _want(algs, "sdma_dcsi"):
                q_c_d0, q_p_d0 = wmmse_optimize(
                    H_dcsi, topo["assoc_ks"], P, cfg["sigma2"],
                    cfg["max_wmmse_iter"], cfg["wmmse_tol"], False, rng=alg_rng
                )
                eval_rng = np.random.default_rng(eval_seed)
                val, _ = ergodic_mmfr(q_c_d0, q_p_d0, stats, topo, cfg["sigma2"], cfg["num_mc_eval"], eval_rng, False)
                mmfr["sdma_dcsi"].append(val)

            # non-cooperative sCSI RSMA
            if _want(algs, "rsma_scsi_noncoop"):
                q_c_nc, q_p_nc = wmmse_optimize(
                    Hhat_nc, topo_nc["assoc_ks"], P, cfg["sigma2"],
                    cfg["max_wmmse_iter"], cfg["wmmse_tol"], True, rng=alg_rng
                )
                eval_rng = np.random.default_rng(eval_seed)
                val, _ = ergodic_mmfr(q_c_nc, q_p_nc, stats, topo_nc, cfg["sigma2"], cfg["num_mc_eval"], eval_rng, True)
                mmfr["rsma_scsi_noncoop"].append(val)

            # non-cooperative sCSI SDMA
            if _want(algs, "sdma_scsi_noncoop"):
                q_c_nc0, q_p_nc0 = wmmse_optimize(
                    Hhat_nc, topo_nc["assoc_ks"], P, cfg["sigma2"],
                    cfg["max_wmmse_iter"], cfg["wmmse_tol"], False, rng=alg_rng
                )
                eval_rng = np.random.default_rng(eval_seed)
                val, _ = ergodic_mmfr(q_c_nc0, q_p_nc0, stats, topo_nc, cfg["sigma2"], cfg["num_mc_eval"], eval_rng, False)
                mmfr["sdma_scsi_noncoop"].append(val)

        results.append({k: float(np.mean(v)) if isinstance(v, list) else v for k, v in mmfr.items()})
        elapsed = time.time() - start_time
        print(f"[P] P={p_dbw} dBW done. elapsed={elapsed:.1f}s", flush=True)
    return results


def run_S(cfg, verbose=False, algs=None):
    cfg["sigma2"] = noise_variance(cfg)
    P = _linear_power_watts(15.0)
    K = 10
    results = []
    start_time = time.time()
    print(f"[S] start: P=15 dBW, K={K}, drops={cfg['num_drops']}", flush=True)

    # Import channel functions once
    from msct_rsma.channel import channel_stats, build_hat_channels, sample_channels, los_channel_list
    from msct_rsma.topology import nearest_only_association

    for S in cfg["S_list"]:
        mmfr = {
            "S": S,
            "rsma_scsi": [],
            "sdma_scsi": [],
            "rsma_icsi": [],
            "sdma_icsi": [],
            "rsma_dcsi": [],
            "sdma_dcsi": [],
            "rsma_scsi_noncoop": [],
            "sdma_scsi_noncoop": [],
        }
        print(f"[S] S={S}: start", flush=True)
        for drop_idx in range(1, cfg["num_drops"] + 1):
            if not verbose:
                ant = f"{cfg['Mx']}x{cfg['My']}|{cfg['Nx']}x{cfg['Ny']}"
                print(
                    f"[S] P=15 dBW, S={S}, K={K}, Ant={ant}: "
                    f"drop {drop_idx}/{cfg['num_drops']}",
                    flush=True,
                )
            
            # ============================================================
            # CRITICAL: Use deterministic per-drop seeds
            # This ensures different algorithm runs produce IDENTICAL channels
            # ============================================================
            s_idx = cfg["S_list"].index(S)
            drop_base_seed = cfg["seed"] * 10000 + s_idx * 100 + drop_idx
            
            # Separate RNGs for different purposes
            topo_rng = np.random.default_rng(drop_base_seed)
            channel_rng = np.random.default_rng(drop_base_seed + 1)
            icsi_rng = np.random.default_rng(drop_base_seed + 2)
            eval_seed = drop_base_seed + 3
            alg_rng = np.random.default_rng(drop_base_seed + 4)
            
            # Generate topology and channel (deterministic for this drop)
            topo = topology_drop(S, K, cfg, topo_rng)
            stats = channel_stats(topo, cfg, channel_rng, los_only=False)
            Hhat = build_hat_channels(stats)
            
            # For non-cooperative
            assoc_ks_nc, assoc_sk_nc = nearest_only_association(topo["D"])
            topo_nc = dict(topo)
            topo_nc["assoc_ks"] = assoc_ks_nc
            topo_nc["assoc_sk"] = assoc_sk_nc
            Hhat_nc = build_hat_channels(stats)
            
            # For iCSI
            H_icsi = sample_channels(stats, topo, icsi_rng)
            
            # For dCSI
            H_dcsi = los_channel_list(stats)

            # cooperative sCSI RSMA
            if _want(algs, "rsma_scsi"):
                q_c, q_p = wmmse_optimize(
                    Hhat, topo["assoc_ks"], P, cfg["sigma2"],
                    cfg["max_wmmse_iter"], cfg["wmmse_tol"], True, rng=alg_rng
                )
                eval_rng = np.random.default_rng(eval_seed)
                val, _ = ergodic_mmfr(q_c, q_p, stats, topo, cfg["sigma2"], cfg["num_mc_eval"], eval_rng, True)
                mmfr["rsma_scsi"].append(val)

            # cooperative sCSI SDMA
            if _want(algs, "sdma_scsi"):
                q_c0, q_p0 = wmmse_optimize(
                    Hhat, topo["assoc_ks"], P, cfg["sigma2"],
                    cfg["max_wmmse_iter"], cfg["wmmse_tol"], False, rng=alg_rng
                )
                eval_rng = np.random.default_rng(eval_seed)
                val, _ = ergodic_mmfr(q_c0, q_p0, stats, topo, cfg["sigma2"], cfg["num_mc_eval"], eval_rng, False)
                mmfr["sdma_scsi"].append(val)

            # non-cooperative sCSI RSMA
            if _want(algs, "rsma_scsi_noncoop"):
                q_c_nc, q_p_nc = wmmse_optimize(
                    Hhat_nc, topo_nc["assoc_ks"], P, cfg["sigma2"],
                    cfg["max_wmmse_iter"], cfg["wmmse_tol"], True, rng=alg_rng
                )
                eval_rng = np.random.default_rng(eval_seed)
                val, _ = ergodic_mmfr(q_c_nc, q_p_nc, stats, topo_nc, cfg["sigma2"], cfg["num_mc_eval"], eval_rng, True)
                mmfr["rsma_scsi_noncoop"].append(val)

            # non-cooperative sCSI SDMA
            if _want(algs, "sdma_scsi_noncoop"):
                q_c_nc0, q_p_nc0 = wmmse_optimize(
                    Hhat_nc, topo_nc["assoc_ks"], P, cfg["sigma2"],
                    cfg["max_wmmse_iter"], cfg["wmmse_tol"], False, rng=alg_rng
                )
                eval_rng = np.random.default_rng(eval_seed)
                val, _ = ergodic_mmfr(q_c_nc0, q_p_nc0, stats, topo_nc, cfg["sigma2"], cfg["num_mc_eval"], eval_rng, False)
                mmfr["sdma_scsi_noncoop"].append(val)

            # iCSI RSMA
            # NOTE: For iCSI, we use instantaneous_mmfr with the SAME channel used for design.
            if _want(algs, "rsma_icsi"):
                q_c_i, q_p_i = wmmse_optimize(
                    H_icsi, topo["assoc_ks"], P, cfg["sigma2"],
                    cfg["max_wmmse_iter"], cfg["wmmse_tol"], True, rng=alg_rng
                )
                # Use instantaneous_mmfr with the same channel H_icsi (not ergodic sampling!)
                val, _ = instantaneous_mmfr(q_c_i, q_p_i, H_icsi, cfg["sigma2"], True)
                mmfr["rsma_icsi"].append(val)

            # iCSI SDMA
            if _want(algs, "sdma_icsi"):
                q_c_i0, q_p_i0 = wmmse_optimize(
                    H_icsi, topo["assoc_ks"], P, cfg["sigma2"],
                    cfg["max_wmmse_iter"], cfg["wmmse_tol"], False, rng=alg_rng
                )
                # Use instantaneous_mmfr with the same channel H_icsi (not ergodic sampling!)
                val, _ = instantaneous_mmfr(q_c_i0, q_p_i0, H_icsi, cfg["sigma2"], False)
                mmfr["sdma_icsi"].append(val)

            # dCSI RSMA
            if _want(algs, "rsma_dcsi"):
                q_c_d, q_p_d = wmmse_optimize(
                    H_dcsi, topo["assoc_ks"], P, cfg["sigma2"],
                    cfg["max_wmmse_iter"], cfg["wmmse_tol"], True, rng=alg_rng
                )
                eval_rng = np.random.default_rng(eval_seed)
                val, _ = ergodic_mmfr(q_c_d, q_p_d, stats, topo, cfg["sigma2"], cfg["num_mc_eval"], eval_rng, True)
                mmfr["rsma_dcsi"].append(val)

            # dCSI SDMA
            if _want(algs, "sdma_dcsi"):
                q_c_d0, q_p_d0 = wmmse_optimize(
                    H_dcsi, topo["assoc_ks"], P, cfg["sigma2"],
                    cfg["max_wmmse_iter"], cfg["wmmse_tol"], False, rng=alg_rng
                )
                eval_rng = np.random.default_rng(eval_seed)
                val, _ = ergodic_mmfr(q_c_d0, q_p_d0, stats, topo, cfg["sigma2"], cfg["num_mc_eval"], eval_rng, False)
                mmfr["sdma_dcsi"].append(val)

        results.append({k: float(np.mean(v)) if isinstance(v, list) else v for k, v in mmfr.items()})
        elapsed = time.time() - start_time
        print(f"[S] S={S} done. elapsed={elapsed:.1f}s", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vs", choices=["P", "S", "both"], default="both")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--algs",
        default=None,
        help=(
            "Comma-separated algorithms to run. "
            "Examples: rsma_scsi,sdma_scsi or rsma_icsi. "
            "Default runs all."
        ),
    )
    args = parser.parse_args()

    cfg = default_sim_config()
    if args.quick:
        # Reduced parameters for quick testing
        # Note: num_drops=3 is too few for stable statistics; using 5 as minimum
        cfg["num_drops"] = 5
        cfg["num_mc_eval"] = 2000
        cfg["max_wmmse_iter"] = 40  # Usually converges in fewer iterations

    algs = _parse_algs(args.algs)
    results = {}
    if args.vs in ("P", "both"):
        results["P"] = run_P(cfg, verbose=args.verbose, algs=algs)
    if args.vs in ("S", "both"):
        results["S"] = run_S(cfg, verbose=args.verbose, algs=algs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _write_per_alg(kind, data, algs_selected):
        # data: list of dicts (per x point). algs_selected: set or None
        if not data:
            return
        # Determine all algorithm keys present
        example = data[0]
        alg_keys = [k for k in example.keys() if k not in ("P_dBW", "S")]
        if algs_selected is None:
            # Write combined "all" file
            out_all = out_dir / f"{kind}_all.json"
            out_all.write_text(json.dumps({kind: data}, indent=2))
            print(f"Saved: {out_all}")
            selected = alg_keys
        else:
            selected = [k for k in alg_keys if k in algs_selected]

        for alg in selected:
            alg_data = []
            for row in data:
                base = {"P_dBW": row["P_dBW"]} if kind == "P" else {"S": row["S"]}
                base[alg] = row.get(alg, None)
                alg_data.append(base)
            out_alg = out_dir / f"{kind}_{alg}.json"
            out_alg.write_text(json.dumps({kind: alg_data}, indent=2))
            print(f"Saved: {out_alg}")

    if "P" in results:
        _write_per_alg("P", results["P"], algs)
    if "S" in results:
        _write_per_alg("S", results["S"], algs)


if __name__ == "__main__":
    main()
