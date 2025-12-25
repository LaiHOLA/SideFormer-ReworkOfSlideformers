'''
python t1m2_time_report.py --run0 runs_t1m2\20251225_153129_T1M2_recomp0_seq256_bs1_ep1_spe30 `
                          --run1 runs_t1m2\20251225_153148_T1M2_recomp1_seq256_bs1_ep1_spe30 `
                          --out t1m2_report_out --skip-steps 1

'''
'''
python t1m2_time_report.py --run0 runs_t1m2\你的recomp0目录 `
                          --run1 runs_t1m2\你的recomp1目录 `
                          --out t1m2_report_out --skip-steps 2

'''
# -*- coding: utf-8 -*-
# t1m2_time_report.py
# Compare two runs (recompute OFF vs ON), generate paper-like plots + export CSV evidence.

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_run(run_dir: Path):
    step = pd.read_csv(run_dir / "step_metrics.csv")
    layer = pd.read_csv(run_dir / "layer_metrics.csv")
    epoch = pd.read_csv(run_dir / "epoch_metrics.csv")
    return step, layer, epoch


def paper_style():
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 160,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_df(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)


def _grid(ax):
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)


def plot_two_lines(x0, y0, x1, y1, lab0, lab1, title, xlabel, ylabel, out_png: Path, marker=False):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    if marker:
        ax.plot(x0, y0, marker="o", linewidth=1.6, label=lab0)
        ax.plot(x1, y1, marker="o", linewidth=1.6, label=lab1)
    else:
        ax.plot(x0, y0, linewidth=1.8, label=lab0)
        ax.plot(x1, y1, linewidth=1.8, label=lab1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _grid(ax)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_line_with_ci(x, mean0, lo0, hi0, mean1, lo1, hi1, title, xlabel, ylabel, out_png: Path, lab0="run0", lab1="run1"):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(x, mean0, linewidth=1.8, label=lab0)
    ax.fill_between(x, lo0, hi0, alpha=0.18)
    ax.plot(x, mean1, linewidth=1.8, label=lab1)
    ax.fill_between(x, lo1, hi1, alpha=0.18)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _grid(ax)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_epoch_distribution(a, b, out_png: Path):
    # box + jitter points (works even if n=1)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    data = [a, b]
    ax.boxplot(data, labels=["recompute=0", "recompute=1"], showfliers=False)
    # jitter scatter
    for i, arr in enumerate(data, start=1):
        if len(arr) == 0:
            continue
        xs = np.full_like(arr, fill_value=i, dtype=np.float64) + (np.random.rand(len(arr)) - 0.5) * 0.10
        ax.scatter(xs, arr, s=28, alpha=0.85)
    ax.set_title("Epoch wall-time distribution (box + points)")
    ax.set_ylabel("epoch wall time (s)")
    _grid(ax)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main():
    paper_style()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run0", type=str, required=True, help="recompute OFF run_dir")
    ap.add_argument("--run1", type=str, required=True, help="recompute ON run_dir")
    ap.add_argument("--out", type=str, default="t1m2_report_out")
    ap.add_argument("--skip-steps", type=int, default=2, help="drop warmup steps for stats")
    args = ap.parse_args()

    run0 = Path(args.run0)
    run1 = Path(args.run1)
    out = ensure_dir(Path(args.out))

    s0, l0, e0 = load_run(run0)
    s1, l1, e1 = load_run(run1)

    # sort by time to avoid any accidental disorder
    s0 = s0.sort_values("cum_wall_ms").reset_index(drop=True)
    s1 = s1.sort_values("cum_wall_ms").reset_index(drop=True)

    # =========================
    # Step-level comparisons
    # =========================
    # (A) wall time per step
    plot_two_lines(
        x0=s0["global_step"].values, y0=s0["wall_ms"].values,
        x1=s1["global_step"].values, y1=s1["wall_ms"].values,
        lab0="recompute=0", lab1="recompute=1",
        title="Step wall time comparison",
        xlabel="global step", ylabel="step wall time (ms)",
        out_png=out / "step_wall_ms_compare.png",
        marker=False
    )

    # (B) delta wall time per step (aligned by step index, taking min length)
    n = min(len(s0), len(s1))
    if n > 0:
        d = s1["wall_ms"].values[:n] - s0["wall_ms"].values[:n]
        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.plot(np.arange(n), d, linewidth=1.8)
        ax.axhline(0.0, linewidth=1.0)
        ax.set_title("Step wall-time delta (recompute ON - OFF)")
        ax.set_xlabel("step index (aligned)")
        ax.set_ylabel("delta wall time (ms)")
        _grid(ax)
        fig.tight_layout()
        fig.savefig(out / "step_wall_ms_delta.png")
        plt.close(fig)

    # =========================
    # Epoch-level distribution
    # =========================
    # With 1 epoch, distribution is trivial (still plotted as box+point).
    a = e0["epoch_wall_s"].values.astype(float)
    b = e1["epoch_wall_s"].values.astype(float)

    # epoch wall time curve (marker to avoid "empty-looking" single-point line)
    plot_two_lines(
        x0=e0["epoch"].values, y0=a,
        x1=e1["epoch"].values, y1=b,
        lab0="recompute=0", lab1="recompute=1",
        title="Epoch wall time comparison",
        xlabel="epoch", ylabel="epoch wall time (s)",
        out_png=out / "epoch_wall_time_compare.png",
        marker=True
    )

    plot_epoch_distribution(a, b, out / "epoch_wall_time_box_compare.png")

    # =========================
    # Memory timeline (time axis in 1000ms units => seconds)
    # Note: you only sample once per step, so the curve is piecewise-constant-ish.
    # =========================
    t0 = (s0["cum_wall_ms"].values / 1000.0).astype(float)
    t1 = (s1["cum_wall_ms"].values / 1000.0).astype(float)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(t0, s0["cpu_rss_gib"].values, linewidth=1.8, label="CPU_RSS r0")
    ax.plot(t0, s0["gpu_alloc_gib"].values, linewidth=1.8, label="GPU_alloc r0")
    ax.plot(t1, s1["cpu_rss_gib"].values, linewidth=1.8, label="CPU_RSS r1")
    ax.plot(t1, s1["gpu_alloc_gib"].values, linewidth=1.8, label="GPU_alloc r1")
    ax.set_title("Memory timeline (CPU RSS + GPU allocated)")
    ax.set_xlabel("time (1000ms units)")
    ax.set_ylabel("memory (GiB)")
    _grid(ax)
    ax.legend(frameon=True, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "mem_timeline_cpu_gpualloc_compare.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(t0, s0["cpu_rss_gib"].values, linewidth=1.8, label="CPU_RSS r0")
    ax.plot(t0, s0["gpu_resv_gib"].values, linewidth=1.8, label="GPU_reserved r0")
    ax.plot(t1, s1["cpu_rss_gib"].values, linewidth=1.8, label="CPU_RSS r1")
    ax.plot(t1, s1["gpu_resv_gib"].values, linewidth=1.8, label="GPU_reserved r1")
    ax.set_title("Memory timeline (CPU RSS + GPU reserved)")
    ax.set_xlabel("time (1000ms units)")
    ax.set_ylabel("memory (GiB)")
    _grid(ax)
    ax.legend(frameon=True, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "mem_timeline_cpu_gpureserved_compare.png")
    plt.close(fig)

    # =========================
    # Per-layer stats: mean + 95% CI (across steps after warmup)
    # =========================
    skip = max(int(args.skip_steps), 0)

    # keep steps after warmup by step order (not by global_step numeric gap)
    keep0 = set(s0["global_step"].values.tolist()[skip:])
    keep1 = set(s1["global_step"].values.tolist()[skip:])

    l0f = l0[l0["global_step"].isin(keep0)].copy()
    l1f = l1[l1["global_step"].isin(keep1)].copy()

    # define total per layer for this project
    l0f["total_ms"] = l0f["fwd_ms"] + l0f["recompute_fwd_ms"] + l0f["bwd_ms"]
    l1f["total_ms"] = l1f["fwd_ms"] + l1f["recompute_fwd_ms"] + l1f["bwd_ms"]

    def agg_ci(df, col):
        g = df.groupby("layer")[col]
        mean = g.mean()
        std = g.std(ddof=1)
        cnt = g.count().clip(lower=1)
        # 95% CI ~ 1.96 * std/sqrt(n) (small n 时只是近似，但比“没区间”强)
        ci = 1.96 * (std / np.sqrt(cnt))
        lo = mean - ci
        hi = mean + ci
        out = pd.DataFrame({"layer": mean.index.values, "mean": mean.values, "lo": lo.values, "hi": hi.values, "n": cnt.values})
        return out.sort_values("layer").reset_index(drop=True)

    f0 = agg_ci(l0f, "fwd_ms")
    f1 = agg_ci(l1f, "fwd_ms")
    b0 = agg_ci(l0f, "bwd_ms")
    b1 = agg_ci(l1f, "bwd_ms")
    t0s = agg_ci(l0f, "total_ms")
    t1s = agg_ci(l1f, "total_ms")

    merged = pd.DataFrame({"layer": t0s["layer"].values})
    merged["fwd_mean_r0"] = f0["mean"].values
    merged["fwd_mean_r1"] = f1["mean"].values
    merged["bwd_mean_r0"] = b0["mean"].values
    merged["bwd_mean_r1"] = b1["mean"].values
    merged["total_mean_r0"] = t0s["mean"].values
    merged["total_mean_r1"] = t1s["mean"].values
    merged["delta_total_ms"] = merged["total_mean_r1"] - merged["total_mean_r0"]

    save_df(f0, out / "layer_fwd_ci_r0.csv")
    save_df(f1, out / "layer_fwd_ci_r1.csv")
    save_df(b0, out / "layer_bwd_ci_r0.csv")
    save_df(b1, out / "layer_bwd_ci_r1.csv")
    save_df(t0s, out / "layer_total_ci_r0.csv")
    save_df(t1s, out / "layer_total_ci_r1.csv")
    save_df(merged, out / "layer_mean_merged.csv")

    x = merged["layer"].values

    plot_line_with_ci(
        x=x,
        mean0=f0["mean"].values, lo0=f0["lo"].values, hi0=f0["hi"].values,
        mean1=f1["mean"].values, lo1=f1["lo"].values, hi1=f1["hi"].values,
        title="Per-layer forward time (mean ± 95% CI)",
        xlabel="layer", ylabel="forward time (ms)",
        out_png=out / "layer_fwd_mean_ci.png",
        lab0="recompute=0", lab1="recompute=1"
    )

    plot_line_with_ci(
        x=x,
        mean0=b0["mean"].values, lo0=b0["lo"].values, hi0=b0["hi"].values,
        mean1=b1["mean"].values, lo1=b1["lo"].values, hi1=b1["hi"].values,
        title="Per-layer backward time (mean ± 95% CI)",
        xlabel="layer", ylabel="backward time (ms)",
        out_png=out / "layer_bwd_mean_ci.png",
        lab0="recompute=0", lab1="recompute=1"
    )

    plot_line_with_ci(
        x=x,
        mean0=t0s["mean"].values, lo0=t0s["lo"].values, hi0=t0s["hi"].values,
        mean1=t1s["mean"].values, lo1=t1s["lo"].values, hi1=t1s["hi"].values,
        title="Per-layer total time (mean ± 95% CI)",
        xlabel="layer", ylabel="total time (ms)",
        out_png=out / "layer_total_mean_ci.png",
        lab0="recompute=0", lab1="recompute=1"
    )

    # delta total
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(x, merged["delta_total_ms"].values, linewidth=1.9)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_title("Per-layer delta total time (recompute ON - OFF)")
    ax.set_xlabel("layer")
    ax.set_ylabel("delta total (ms)")
    _grid(ax)
    fig.tight_layout()
    fig.savefig(out / "layer_delta_total.png")
    plt.close(fig)

    # =========================
    # Summary evidence CSV (peaks, means, skip)
    # =========================
    summary = pd.DataFrame([{
        "run0": str(run0),
        "run1": str(run1),
        "skip_steps": int(skip),
        "mean_step_wall_ms_r0": float(np.mean(s0["wall_ms"].values[skip:])) if len(s0) > skip else float(np.mean(s0["wall_ms"].values)),
        "mean_step_wall_ms_r1": float(np.mean(s1["wall_ms"].values[skip:])) if len(s1) > skip else float(np.mean(s1["wall_ms"].values)),
        "peak_gpu_alloc_gib_r0": float(np.max(s0["peak_alloc_gib"].values)) if len(s0) else float("nan"),
        "peak_gpu_alloc_gib_r1": float(np.max(s1["peak_alloc_gib"].values)) if len(s1) else float("nan"),
        "peak_gpu_resv_gib_r0": float(np.max(s0["peak_resv_gib"].values)) if len(s0) else float("nan"),
        "peak_gpu_resv_gib_r1": float(np.max(s1["peak_resv_gib"].values)) if len(s1) else float("nan"),
        "peak_cpu_rss_gib_r0": float(np.nanmax(s0["cpu_rss_gib"].values)) if len(s0) else float("nan"),
        "peak_cpu_rss_gib_r1": float(np.nanmax(s1["cpu_rss_gib"].values)) if len(s1) else float("nan"),
        "mean_layer_total_ms_r0": float(np.nanmean(t0s["mean"].values)),
        "mean_layer_total_ms_r1": float(np.nanmean(t1s["mean"].values)),
    }])
    save_df(summary, out / "summary.csv")

    print("[OK] report saved to:", out)


if __name__ == "__main__":
    main()
