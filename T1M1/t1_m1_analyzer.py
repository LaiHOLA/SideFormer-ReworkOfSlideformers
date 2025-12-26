# -*- coding: utf-8 -*-
'''
T1-M1 paper-style analyzer.

Input:
  - --manifest MANIFEST.json (recommended)
  - or --runs label=path label=path ...

Outputs:
  - compare_summary.csv
  - figures/*.png (+ optional *.pdf)

Main figures:
  Fig01: wall-time distribution per iteration (boxplot + jitter)
  Fig02: time breakdown (stacked bars)
  Fig03: peak memory summary (GPU alloc/reserved + CPU RSS)
  Fig04: per-iter aligned memory timeline (mean +/- std over iters; percent-x)
  Fig05~07: layer-level profiles (forward / backward+recompute / optimizer)
'''
import os, json, argparse, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _read_csv(path):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _paper_rcparams():
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.0,
        "figure.dpi": 140,
        "savefig.dpi": 300,
    })

def _save(fig, path_png, export_pdf=False):
    fig.tight_layout()
    fig.savefig(path_png, bbox_inches="tight")
    if export_pdf:
        fig.savefig(os.path.splitext(path_png)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)

def _sem(x):
    x = np.asarray(x, dtype=float)
    if len(x) <= 1:
        return 0.0
    return float(np.std(x, ddof=1) / math.sqrt(len(x)))

def _coalesce_col(df, candidates, default=0.0):
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series([default] * len(df))

def load_runs_from_manifest(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    runs = [(r["label"], r["path"]) for r in m.get("runs", [])]
    if not runs:
        raise RuntimeError("manifest has no runs")
    return runs

def parse_runs(args_runs):
    runs = []
    for item in args_runs:
        if "=" not in item:
            raise ValueError(f"--runs item must be label=path, got: {item}")
        lb, p = item.split("=", 1)
        runs.append((lb.strip(), p.strip()))
    return runs

def load_one_run(label, run_dir, drop_first=False):
    it = _read_csv(os.path.join(run_dir, "metrics_iter.csv"))
    ly = _read_csv(os.path.join(run_dir, "metrics_layer.csv"))
    mt = _read_csv(os.path.join(run_dir, "mem_trace.csv"))
    tl = _read_csv(os.path.join(run_dir, "timeline.csv"))

    if it is None or ly is None:
        raise RuntimeError(f"missing metrics_iter.csv or metrics_layer.csv under {run_dir}")

    it["label"] = label
    ly["label"] = label
    if "mode" not in it.columns:
        it["mode"] = label
    if "mode" not in ly.columns:
        ly["mode"] = label

    if drop_first and "iter" in it.columns and len(it) >= 2:
        it = it[it["iter"] != it["iter"].min()].copy()

    # mem trace is optional; align t_ms within each iter
    if mt is not None and "t_ms" in mt.columns and "iter" in mt.columns:
        mt["label"] = label
        if "mode" not in mt.columns:
            mt["mode"] = label
        mt["t_ms"] = pd.to_numeric(mt["t_ms"], errors="coerce")
        mt = mt.dropna(subset=["t_ms"]).copy()
        mt = mt.sort_values(["iter", "t_ms"])
        mt["t_ms0"] = mt["t_ms"] - mt.groupby("iter")["t_ms"].transform("min")
        denom = mt.groupby("iter")["t_ms0"].transform("max").replace(0, np.nan)
        mt["t_pct"] = (mt["t_ms0"] / denom).fillna(0.0)
        if drop_first and mt["iter"].nunique() >= 2:
            mt = mt[mt["iter"] != mt["iter"].min()].copy()
    else:
        mt = None

    if tl is not None:
        tl["label"] = label
        if "mode" not in tl.columns:
            tl["mode"] = label

    return {"label": label, "run_dir": run_dir, "iter": it, "layer": ly, "mem": mt, "timeline": tl}

def summarize(runs):
    rows = []
    for r in runs:
        it = r["iter"]
        lb = r["label"]
        wall = _coalesce_col(it, ["wall_s"])
        fwd = _coalesce_col(it, ["forward_s"])
        bwd = _coalesce_col(it, ["backward_s"])
        optim = _coalesce_col(it, ["optim_s"])
        cpu_up = _coalesce_col(it, ["cpu_update_total_s"])
        peak_alloc = _coalesce_col(it, ["peak_alloc_gib"])
        peak_resv = _coalesce_col(it, ["peak_reserved_gib"])
        rss = _coalesce_col(it, ["cpu_rss_gib"])

        rows.append({
            "label": lb,
            "iters": int(len(it)),
            "wall_s_mean": float(wall.mean()),
            "wall_s_sem": _sem(wall),
            "forward_s_mean": float(fwd.mean()),
            "backward_s_mean": float(bwd.mean()),
            "optim_s_mean": float(optim.mean()),
            "cpu_update_total_s_mean": float(cpu_up.mean()),
            "peak_alloc_gib_mean": float(peak_alloc.mean()),
            "peak_reserved_gib_mean": float(peak_resv.mean()),
            "cpu_rss_gib_mean": float(rss.mean()),
            "wall_s_min": float(wall.min()),
            "wall_s_max": float(wall.max()),
        })
    return pd.DataFrame(rows).sort_values("label")

def fig_wall_distribution(runs, out_dir, export_pdf=False):
    data = [r["iter"]["wall_s"].to_numpy() for r in runs]
    labels = [r["label"] for r in runs]
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.boxplot(data, labels=labels, showfliers=False)
    for i, ys in enumerate(data, start=1):
        xs = np.random.normal(i, 0.04, size=len(ys))
        ax.scatter(xs, ys, s=18, alpha=0.6)
    ax.set_ylabel("Seconds")
    ax.set_title("Wall time per iteration")
    _save(fig, os.path.join(out_dir, "fig01_wall_distribution.png"), export_pdf)

def fig_time_breakdown(summary_df, out_dir, export_pdf=False):
    labels = summary_df["label"].tolist()
    fwd = summary_df["forward_s_mean"].to_numpy()
    bwd = summary_df["backward_s_mean"].to_numpy()
    optim = summary_df["optim_s_mean"].to_numpy()
    cpu_up = summary_df["cpu_update_total_s_mean"].to_numpy()
    wall = summary_df["wall_s_mean"].to_numpy()
    optim_total = np.maximum(optim, cpu_up)
    other = np.clip(wall - (fwd + bwd + optim_total), 0, None)

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    x = np.arange(len(labels))
    ax.bar(x, fwd, label="Forward")
    ax.bar(x, bwd, bottom=fwd, label="Backward")
    ax.bar(x, optim_total, bottom=fwd + bwd, label="Optimizer (GPU/CPU)")
    ax.bar(x, other, bottom=fwd + bwd + optim_total, label="Other/Overhead")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Seconds")
    ax.set_title("Time breakdown per iteration")
    ax.legend(ncol=2, frameon=False)
    _save(fig, os.path.join(out_dir, "fig02_time_breakdown.png"), export_pdf)

def fig_peak_memory(summary_df, out_dir, export_pdf=False):
    labels = summary_df["label"].tolist()
    alloc = summary_df["peak_alloc_gib_mean"].to_numpy()
    resv = summary_df["peak_reserved_gib_mean"].to_numpy()
    rss = summary_df["cpu_rss_gib_mean"].to_numpy()

    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.bar(x - w, alloc, width=w, label="GPU allocated (GiB)")
    ax.bar(x, resv, width=w, label="GPU reserved (GiB)")
    ax.bar(x + w, rss, width=w, label="CPU RSS (GiB)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("GiB")
    ax.set_title("Peak memory (end-of-iter sampling)")
    ax.legend(ncol=2, frameon=False)
    _save(fig, os.path.join(out_dir, "fig03_peak_memory.png"), export_pdf)

def _resample_pct_curve(df_iter, y_col, n=200):
    # Resample a single-iter curve to t_pct grid [0,1]
    if df_iter is None or len(df_iter) < 2:
        return None
    x = df_iter["t_pct"].to_numpy(dtype=float)
    y = df_iter[y_col].to_numpy(dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if x[-1] <= x[0]:
        return None
    grid = np.linspace(0.0, 1.0, n)
    yg = np.interp(grid, x, y)
    return grid, yg

def _aggregate_mem_curves(mt, y_col, n=200):
    # Aggregate over multiple iters: return grid, mean, std, n_iters_used
    if mt is None:
        return None
    iters = sorted(mt["iter"].unique().tolist())
    curves = []
    grid = None
    for it in iters:
        df = mt[mt["iter"] == it]
        res = _resample_pct_curve(df, y_col, n=n)
        if res is None:
            continue
        g, y = res
        grid = g
        curves.append(y)
    if not curves or grid is None:
        return None
    arr = np.stack(curves, axis=0)
    return grid, arr.mean(axis=0), arr.std(axis=0), len(curves)

def fig_memory_timeline(runs, out_dir, export_pdf=False):
    # Plot mean +/- std memory timeline over iters, on percent-x axis.
    # Solid: GPU reserved, dashed: CPU RSS.
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    any_plotted = 0
    for r in runs:
        mt = r["mem"]
        if mt is None:
            continue
        agg_gpu = _aggregate_mem_curves(mt, "gpu_reserved_gib", n=220)
        agg_cpu = _aggregate_mem_curves(mt, "cpu_rss_gib", n=220)
        if agg_gpu is None or agg_cpu is None:
            continue
        xg, mg, sg, ng = agg_gpu
        _, mc, sc, nc = agg_cpu

        ax.plot(xg, mg, label=f"{r['label']} GPU reserved (n={ng})")
        ax.fill_between(xg, mg - sg, mg + sg, alpha=0.12)
        ax.plot(xg, mc, linestyle="--", label=f"{r['label']} CPU RSS (n={nc})")
        ax.fill_between(xg, mc - sc, mc + sc, alpha=0.12)
        any_plotted += 1

    ax.set_xlabel("Progress within iteration (0 to 1)")
    ax.set_ylabel("GiB")
    ax.set_title("Memory timeline (mean +/- std over iters)")
    if any_plotted:
        ax.legend(frameon=False, ncol=2)
    _save(fig, os.path.join(out_dir, "fig04_memory_timeline.png"), export_pdf)

def fig_layer_profile(runs, out_dir, export_pdf=False):
    def _mean_by_layer(df, col):
        return df.groupby(["label", "layer"], as_index=False)[col].mean()

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    for r in runs:
        g = _mean_by_layer(r["layer"], "fwd_ms")
        sub = g[g["label"] == r["label"]]
        ax.plot(sub["layer"], sub["fwd_ms"], label=r["label"])
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Forward time (ms)")
    ax.set_title("Layer forward time (mean)")
    ax.legend(frameon=False)
    _save(fig, os.path.join(out_dir, "fig05_layer_forward.png"), export_pdf)

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    for r in runs:
        gR = _mean_by_layer(r["layer"], "recompute_ms")
        gB = _mean_by_layer(r["layer"], "bwd_ms")
        subR = gR[gR["label"] == r["label"]]
        subB = gB[gB["label"] == r["label"]]
        ax.plot(subB["layer"], subB["bwd_ms"], linestyle=":", label=f"{r['label']} bwd_ms")
        if float(subR["recompute_ms"].max()) > 0.001:
            ax.plot(subR["layer"], subR["recompute_ms"], linestyle="-", label=f"{r['label']} recompute_ms")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Layer backward and recomputation time (mean)")
    ax.legend(frameon=False, ncol=2)
    _save(fig, os.path.join(out_dir, "fig06_layer_bwd_recompute.png"), export_pdf)

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    any_plotted = 0
    for r in runs:
        gO = _mean_by_layer(r["layer"], "optim_ms")
        subO = gO[gO["label"] == r["label"]]
        if float(subO["optim_ms"].max()) <= 0.001:
            continue
        ax.plot(subO["layer"], subO["optim_ms"], label=r["label"])
        any_plotted += 1
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Optimizer time (ms)")
    ax.set_title("Per-layer optimizer/update time (mean)")
    if any_plotted:
        ax.legend(frameon=False)
    _save(fig, os.path.join(out_dir, "fig07_layer_optim.png"), export_pdf)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default="", help="manifest json produced by t1_m1_triple.py")
    ap.add_argument("--runs", nargs="*", default=[], help="explicit runs: label=path label=path ...")
    ap.add_argument("--out", type=str, default="t1m1_paper_out")
    ap.add_argument("--export_pdf", type=int, default=1)
    ap.add_argument("--drop_first", type=int, default=0, help="drop first iter as warmup if iters>=2")
    args = ap.parse_args()

    _paper_rcparams()
    runs_spec = load_runs_from_manifest(args.manifest) if args.manifest else parse_runs(args.runs)
    loaded = [load_one_run(lb, p, drop_first=bool(args.drop_first)) for lb, p in runs_spec]

    out_dir = args.out
    fig_dir = os.path.join(out_dir, "figures")
    _ensure_dir(fig_dir)

    summary = summarize(loaded)
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "compare_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    print("[SAVED]", summary_path)

    export_pdf = bool(args.export_pdf)
    fig_wall_distribution(loaded, fig_dir, export_pdf)
    fig_time_breakdown(summary, fig_dir, export_pdf)
    fig_peak_memory(summary, fig_dir, export_pdf)
    fig_memory_timeline(loaded, fig_dir, export_pdf)
    fig_layer_profile(loaded, fig_dir, export_pdf)

    print("[FIGURES]", fig_dir)

if __name__ == "__main__":
    main()
