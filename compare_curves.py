# -*- coding: utf-8 -*-
"""
批量对比 target vs actual 曲线，按 ID 配对，逐张出图 + 汇总图
python compare_curves.py                 # 全部单图 + 汇总
python compare_curves.py --summary-only  # 只输出汇总图 + 终端指标表
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8})
COLORS = {"target": "#1a1a1a", "actual": "#d62728"}
HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
TICK_FREQS = [200, 500, 1000, 2000, 4000, 8000]
FREQS = [200,210,223,236,250,265,281,297,315,334,354,375,397,420,445,472,500,530,561,595,630,667,707,749,794,841,891,944,1000,
         1059,1122,1189,1260,1335,1414,1498,1587,1682,1782,1888,2000,2119,2245,2378,2520,2670,2828,2997,3175,3364,3564,3775,4000,
         4238,4490,4757,5040,5339,5657,5993,6350,6727,7127,7551,8000]


def load_results(path):
    with open(path) as f:
        data = json.load(f)
    results = {}
    items = data if isinstance(data, list) else data.get("results", [])
    for r in items:
        if r.get("curves") is not None:
            results[r["id"]] = r["curves"]
    freqs = data.get("frequencies", FREQS) if isinstance(data, dict) else FREQS
    return freqs, results


def compute_metrics(t, a):
    errors = [abs(ai - ti) for ai, ti in zip(a, t)]
    return {
        "mse": sum(e**2 for e in errors) / len(errors),
        "mae": sum(errors) / len(errors),
        "max_e": max(errors),
    }


def plot_single(freqs, cid, t, a, out_dir):
    """单条曲线图 + 差值图"""
    levels = ["50", "80", "90"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 6),
                              gridspec_kw={"height_ratios": [3, 1]})
    all_gains = [v for l in levels for v in t[l] + a[l]]
    y_min = np.floor(min(all_gains)/10)*10
    y_max = np.ceil(max(all_gains)/10)*10

    for col, lv in enumerate(levels):
        diff = [ai - ti for ai, ti in zip(a[lv], t[lv])]
        m = compute_metrics(t[lv], a[lv])

        ax1 = axes[0, col]
        ax1.plot(FREQS, t[lv], "-", color=COLORS["target"], lw=1.2, label="target")
        ax1.plot(FREQS, a[lv], "--", color=COLORS["actual"], lw=1.2, label="actual")
        ax1.set_title(f"{lv} dB")
        ax1.set_xscale("log")
        ax1.set_xlim(180, 9000)
        ax1.set_ylim(y_min, y_max)
        ax1.set_xticks(TICK_FREQS)
        ax1.set_xticklabels([f"{f:.0f}" for f in TICK_FREQS])
        ax1.grid(True, alpha=0.25, ls="--")
        if col == 0:
            ax1.set_ylabel("Gain (dB)")
        ax1.legend(fontsize=7, loc="lower right")

        ax2 = axes[1, col]
        bar_colors = [COLORS["actual"] if d < 0 else COLORS["target"] for d in diff]
        ax2.bar(FREQS, diff, color=bar_colors, width=10, edgecolor="none")
        ax2.axhline(0, color="black", lw=0.8)
        ax2.set_xscale("log")
        ax2.set_xlim(180, 9000)
        ax2.set_xticks(TICK_FREQS)
        ax2.set_xticklabels([f"{f:.0f}" for f in TICK_FREQS])
        ax2.set_ylabel("Diff (dB)")
        ax2.grid(True, alpha=0.25, ls="--", axis="y")
        d_max = max(abs(min(diff)), abs(max(diff)))
        ax2.set_ylim(-d_max - 2, d_max + 2)

    fig.suptitle(f"Case #{cid}  |  "
                 f"MSE={m['mse']:.2f}  MAE={m['mae']:.2f}  MaxE={m['max_e']:.2f}",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    out_path = out_dir / f"case_{cid:04d}.png"
    fig.savefig(out_path, dpi=120, facecolor="white")
    plt.close(fig)
    return m


def plot_summary(all_metrics, out_dir):
    """MSE/MAE/Max 直方图，3x3 网格"""
    levels = ["50", "80", "90"]
    metrics = ["mse", "mae", "max_e"]
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    for row, metric in enumerate(metrics):
        for col, lv in enumerate(levels):
            ax = axes[row, col]
            vals = [m[lv][metric] for m in all_metrics]
            color = "#aec7e8" if metric != "max_e" else "#ffbb78"
            ax.hist(vals, bins=30, color=color, edgecolor="white")
            ax.axvline(np.mean(vals), color="#d62728", lw=2, ls="--",
                      label=f"avg={np.mean(vals):.2f}")
            ax.set_title(f"{lv}dB {metric.upper()}")
            ax.set_xlabel(metric.upper())
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.25, ls="--", axis="y")
    n = len(all_metrics)
    fig.suptitle(f"Summary ({n} cases)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path = out_dir / "summary.png"
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    return out_path

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-only", action="store_true", help="skip per-case charts")
    args = parser.parse_args()
    targets = sorted(OUT.glob("target_results_*.json"), reverse=True)
    actuals = sorted(OUT.glob("sdk_results_*.json"), reverse=True)
    if not targets or not actuals:
        print("no result files in outputs/")
        return
    target_path = targets[0]
    actual_path = actuals[0]
    print(f"target: {target_path.name}")
    print(f"actual: {actual_path.name}")

    # extract suffix from batch_results filename, e.g. batch_results_0709_1.json -> 0709_1
    suffix = actual_path.stem.replace("batch_results_", "")
    pic_dir = OUT / f"picture_{suffix}"

    _, targets = load_results(target_path)
    _, actuals = load_results(actual_path)
    common = sorted(set(targets) & set(actuals))
    if not common:
        print("no common ids")
        return
    print(f"{len(common)} cases to compare")

    pic_dir.mkdir(exist_ok=True)
    all_metrics = []
    levels = ["50", "80", "90"]

    if not args.summary_only:
        for cid in common:
            t = targets[cid]
            a = actuals[cid]
            m_by_level = {}
            for lv in levels:
                m_by_level[lv] = compute_metrics(t[lv], a[lv])
            all_metrics.append(m_by_level)
            plot_single(FREQS, cid, t, a, pic_dir)
    else:
        for cid in common:
            t = targets[cid]
            a = actuals[cid]
            m_by_level = {}
            for lv in levels:
                m_by_level[lv] = compute_metrics(t[lv], a[lv])
            all_metrics.append(m_by_level)

    summary_path = plot_summary(all_metrics, pic_dir)

    print(f"\n{'Level':>6} {'Avg MSE':>10} {'Avg MAE':>10} {'Avg Max':>10}")
    print("-" * 40)
    for lv in levels:
        mse = np.mean([m[lv]["mse"] for m in all_metrics])
        mae = np.mean([m[lv]["mae"] for m in all_metrics])
        mx = np.mean([m[lv]["max_e"] for m in all_metrics])
        print(f"{lv:>4}dB {mse:>10.2f} {mae:>10.2f} {mx:>10.2f}")
    print(f"\n  per-case charts: {pic_dir}/case_*.png")
    print(f"  summary: {summary_path}")


if __name__ == "__main__":
    main()
