import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


# =============================================================================
# 绘图样式配置
# - RSMA: 蓝色系, SDMA: 红色系
# - 不同CSI类型用不同marker符号
# - 协作: 实线, 非协作: 虚线
# =============================================================================

_CURVE_STYLES = {
    # RSMA系列 - 蓝色
    "rsma_scsi":        {"color": "#1f77b4", "marker": "o", "linestyle": "-",  "label": "RSMA (sCSI)"},
    "rsma_icsi":        {"color": "#1f77b4", "marker": "s", "linestyle": "-",  "label": "RSMA (iCSI)"},
    "rsma_dcsi":        {"color": "#1f77b4", "marker": "^", "linestyle": "-",  "label": "RSMA (dCSI)"},
    "rsma_scsi_noncoop": {"color": "#1f77b4", "marker": "o", "linestyle": "--", "label": "RSMA (sCSI, non-coop)"},
    # SDMA系列 - 红色
    "sdma_scsi":        {"color": "#d62728", "marker": "o", "linestyle": "-",  "label": "SDMA (sCSI)"},
    "sdma_icsi":        {"color": "#d62728", "marker": "s", "linestyle": "-",  "label": "SDMA (iCSI)"},
    "sdma_dcsi":        {"color": "#d62728", "marker": "^", "linestyle": "-",  "label": "SDMA (dCSI)"},
    "sdma_scsi_noncoop": {"color": "#d62728", "marker": "o", "linestyle": "--", "label": "SDMA (sCSI, non-coop)"},
}

# 绘制顺序：确保图例有序且清晰
_PLOT_ORDER = [
    "rsma_scsi", "rsma_icsi", "rsma_dcsi", "rsma_scsi_noncoop",
    "sdma_scsi", "sdma_icsi", "sdma_dcsi", "sdma_scsi_noncoop",
]


def _plot_curve(ax, xs, ys, key):
    style = _CURVE_STYLES[key]
    ax.plot(
        xs, ys,
        color=style["color"],
        marker=style["marker"],
        linestyle=style["linestyle"],
        label=style["label"],
        markersize=7,
        linewidth=1.5,
    )


def _plot_P(p_data, out_path, algs=None):
    p_sorted = sorted(p_data, key=lambda x: x["P_dBW"])
    xs = [d["P_dBW"] for d in p_sorted]

    fig, ax = plt.subplots(figsize=(8, 6))
    for key in _PLOT_ORDER:
        if algs is not None and key not in algs:
            continue
        if key in p_sorted[0]:
            ys = [d.get(key, None) for d in p_sorted]
            if all(v is not None for v in ys):
                _plot_curve(ax, xs, ys, key)

    ax.set_xlabel("Transmit power P (dBW)", fontsize=12)
    ax.set_ylabel("Average MMFR (bps/Hz)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_S(s_data, out_path, algs=None):
    s_sorted = sorted(s_data, key=lambda x: x["S"])
    xs = [d["S"] for d in s_sorted]

    fig, ax = plt.subplots(figsize=(8, 6))
    for key in _PLOT_ORDER:
        if algs is not None and key not in algs:
            continue
        if key in s_sorted[0]:
            ys = [d.get(key, None) for d in s_sorted]
            if all(v is not None for v in ys):
                _plot_curve(ax, xs, ys, key)

    ax.set_xlabel("Number of satellites S", fontsize=12)
    ax.set_ylabel("Average MMFR (bps/Hz)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", default="results")
    parser.add_argument("--in", dest="in_path", default=None)
    parser.add_argument("--in_p", dest="in_p", default=None)
    parser.add_argument("--in_s", dest="in_s", default=None)
    parser.add_argument("--out_dir", default="plots")
    parser.add_argument("--vs", choices=["P", "S", "both"], default="both")
    parser.add_argument(
        "--algs",
        default=None,
        help="Comma-separated algorithms to plot. Default plots all.",
    )
    args = parser.parse_args()
    algs = None if args.algs is None else {p.strip() for p in args.algs.split(",") if p.strip()}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.vs == "both":
        in_p = Path(args.in_p) if args.in_p else Path(args.in_dir) / "P_all.json"
        in_s = Path(args.in_s) if args.in_s else Path(args.in_dir) / "S_all.json"
        data_p = json.loads(in_p.read_text())
        data_s = json.loads(in_s.read_text())
        if "P" in data_p:
            _plot_P(data_p["P"], out_dir / "P.png", algs=algs)
        if "S" in data_s:
            _plot_S(data_s["S"], out_dir / "S.png", algs=algs)
    elif args.vs == "P":
        in_path = Path(args.in_p) if args.in_p else (Path(args.in_path) if args.in_path else Path(args.in_dir) / "P_all.json")
        data = json.loads(in_path.read_text())
        if "P" in data:
            _plot_P(data["P"], out_dir / "P.png", algs=algs)
    elif args.vs == "S":
        in_path = Path(args.in_s) if args.in_s else (Path(args.in_path) if args.in_path else Path(args.in_dir) / "S_all.json")
        data = json.loads(in_path.read_text())
        if "S" in data:
            _plot_S(data["S"], out_dir / "S.png", algs=algs)

    print(f"Plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
