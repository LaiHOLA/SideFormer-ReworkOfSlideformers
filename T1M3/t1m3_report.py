from __future__ import annotations
'''
python t1m3_report.py --runs runs_m3\20251225_231102_T1M3_headce_full_ckpt0_seq1024_bs2_bf16 runs_m3\20251225_231206_T1M3_headce_chunked_ckpt0_seq1024_bs2_bf16 --outdir runs_m3\_figs --skip_warmup 0 --paper double --formats png,pdf

'''

# -*- coding: utf-8 -*-
"""
T1M3 report plotter (publication-grade)

Expected per-run files (inside each run directory):
  - metrics_step.csv   (required)
  - config.json        (optional)
  - summary.json       (optional)
  - mem_trace.csv      (optional; for timeline curves)

Produces:
  - fig_mem_timeline.(png/pdf)
  - fig_peak_memory.(png/pdf)
  - fig_time_breakdown.(png/pdf)
  - fig_all_in_one.(png/pdf)
"""
import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# -------------------------
# Styling (paper-like)
# -------------------------
def apply_paper_style(paper: str = "double") -> None:
    """
    paper: 'single' (≈3.35in) or 'double' (≈6.8in)
    """
    # Fonts: keep default DejaVu (matplotlib builtin), reliable cross-platform.
    base_font = 10 if paper == "double" else 9
    small_font = base_font - 1

    mpl.rcParams.update({
        "font.size": base_font,
        "axes.titlesize": base_font + 1,
        "axes.labelsize": base_font,
        "xtick.labelsize": small_font,
        "ytick.labelsize": small_font,
        "legend.fontsize": small_font,
        "figure.titlesize": base_font + 2,

        "axes.linewidth": 1.0,
        "grid.linewidth": 0.6,
        "lines.linewidth": 2.0,

        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,

        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def paper_figsize(kind: str, paper: str) -> Tuple[float, float]:
    """
    Return figure size in inches tuned for papers.
    kind: 'wide', 'standard', 'tall'
    """
    width = 6.8 if paper == "double" else 3.35
    if kind == "wide":
        return (width, width * 0.42)
    if kind == "tall":
        return (width, width * 0.72)
    return (width, width * 0.55)


# -------------------------
# Data model
# -------------------------
@dataclass
class RunData:
    run_dir: Path
    label: str
    cfg: Dict
    summary: Dict
    steps: pd.DataFrame
    mem: Optional[pd.DataFrame]  # may be None


# -------------------------
# Helpers
# -------------------------
def read_json(path: Path) -> Dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def find_first_existing(run_dir: Path, candidates: List[str]) -> Optional[Path]:
    for name in candidates:
        p = run_dir / name
        if p.exists():
            return p
    return None


def shorten_label(raw: str) -> str:
    """
    Heuristic label shortener for names like:
      20251225_231102_T1M3_headce_full_ckpt0_seq1024_bs2_bf16
    """
    s = raw
    s = re.sub(r"^\d{8}_\d{6}_", "", s)  # strip timestamp
    s = s.replace("T1M3_", "")
    s = s.replace("__", "_")

    # key tokens
    seq = re.search(r"seq(\d+)", s)
    bs = re.search(r"bs(\d+)", s)
    ckpt = re.search(r"ckpt(\d+)", s)

    base = s
    # compact common patterns
    base = base.replace("_headce_", " head:")
    base = base.replace("_head_", " head:")
    base = base.replace("_full_", " full ")
    base = base.replace("_chunked_", " chunked ")
    base = base.replace("_", " ").strip()

    parts = [base]
    extra = []
    if seq:
        extra.append(f"seq={seq.group(1)}")
    if bs:
        extra.append(f"bs={bs.group(1)}")
    if ckpt:
        extra.append(f"ckpt={ckpt.group(1)}")
    if extra:
        parts.append("(" + ", ".join(extra) + ")")

    # avoid being too long
    out = " ".join(parts)
    out = re.sub(r"\s+", " ", out).strip()
    if len(out) > 40:
        out = out[:37] + "..."
    return out


def mean_ci95(x: np.ndarray) -> Tuple[float, float]:
    """
    Returns (mean, half-width of 95% CI). Uses normal approx if n>=2.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return (float("nan"), float("nan"))
    if n == 1:
        return (float(x[0]), 0.0)
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    se = sd / math.sqrt(n)
    # 1.96 is fine here (n is usually >= 5); keep simple & stable.
    return (mu, 1.96 * se)


def safe_get_col(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    if col not in df.columns:
        return np.full((len(df),), default, dtype=np.float64)
    return df[col].to_numpy(dtype=np.float64)


# -------------------------
# Loading
# -------------------------
def load_run(run_dir: Path, label: Optional[str], skip_warmup: int) -> RunData:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    cfg_p = run_dir / "config.json"
    sum_p = run_dir / "summary.json"
    steps_p = run_dir / "metrics_step.csv"

    if not steps_p.exists():
        raise FileNotFoundError(f"metrics_step.csv not found in {run_dir}")

    cfg = read_json(cfg_p) if cfg_p.exists() else {}
    summary = read_json(sum_p) if sum_p.exists() else {}

    steps = pd.read_csv(steps_p)
    # Basic sanity
    if "step" in steps.columns:
        steps = steps.sort_values("step").reset_index(drop=True)

    if skip_warmup > 0 and len(steps) > skip_warmup:
        steps = steps.iloc[skip_warmup:].reset_index(drop=True)

    mem_p = find_first_existing(run_dir, ["mem_trace.csv", "timeline.csv", "mem_timeline.csv"])
    mem = pd.read_csv(mem_p) if mem_p is not None else None

    # Build label
    if label and label.strip():
        lab = label.strip()
    else:
        # Prefer config's run_name if present, else folder name
        raw_name = cfg.get("run_name") or run_dir.name
        lab = shorten_label(str(raw_name))

    return RunData(run_dir=run_dir, label=lab, cfg=cfg, summary=summary, steps=steps, mem=mem)


# -------------------------
# Plotting
# -------------------------
def plot_mem_timeline(runs: List[RunData], outdir: Path, formats: List[str], paper: str) -> None:
    # Determine whether mem timeline exists
    usable = [r for r in runs if r.mem is not None and len(r.mem) > 0]
    if not usable:
        return

    fig = plt.figure(figsize=paper_figsize("wide", paper))
    ax = fig.add_subplot(1, 1, 1)

    # Decide columns
    # Common names in your project: t_ms, gpu_reserved_gib, gpu_alloc_gib, cpu_rss_gib
    # We'll plot reserved primarily; allocate if present.
    for r in usable:
        df = r.mem
        # time axis
        if "t_ms" in df.columns:
            t = df["t_ms"].to_numpy(dtype=np.float64)
        elif "time_ms" in df.columns:
            t = df["time_ms"].to_numpy(dtype=np.float64)
        elif "t" in df.columns:
            t = df["t"].to_numpy(dtype=np.float64)
        else:
            t = np.arange(len(df), dtype=np.float64)

        # reserved
        reserved_col = None
        for c in ["gpu_reserved_gib", "gpu_reserved", "reserved_gib", "reserved"]:
            if c in df.columns:
                reserved_col = c
                break

        alloc_col = None
        for c in ["gpu_alloc_gib", "gpu_alloc", "allocated_gib", "allocated"]:
            if c in df.columns:
                alloc_col = c
                break

        if reserved_col is not None:
            ax.plot(t, df[reserved_col].to_numpy(dtype=np.float64), label=f"{r.label} (reserved)")
        elif alloc_col is not None:
            ax.plot(t, df[alloc_col].to_numpy(dtype=np.float64), label=f"{r.label} (alloc)")

    ax.set_title("T1M3 GPU Memory Timeline")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Memory (GiB)")

    # Place legend outside (top) if crowded
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=2, frameon=False)
    ax.margins(x=0.01)

    for fmt in formats:
        fig.savefig(outdir / f"fig_mem_timeline.{fmt}")
    plt.close(fig)


def plot_peak_memory(runs: List[RunData], outdir: Path, formats: List[str], paper: str) -> None:
    labels = [r.label for r in runs]

    # Peak over steps (already warmup-trimmed)
    peak_alloc = []
    peak_reserved = []
    for r in runs:
        pa = np.nanmax(safe_get_col(r.steps, "peak_alloc_gib", np.nan))
        pr = np.nanmax(safe_get_col(r.steps, "peak_reserved_gib", np.nan))
        peak_alloc.append(pa)
        peak_reserved.append(pr)

    x = np.arange(len(runs), dtype=np.float64)
    width = 0.36

    fig = plt.figure(figsize=paper_figsize("standard", paper))
    ax = fig.add_subplot(1, 1, 1)

    b1 = ax.bar(x - width / 2, peak_alloc, width=width, label="Peak allocated")
    b2 = ax.bar(x + width / 2, peak_reserved, width=width, label="Peak reserved")

    ax.set_title("T1M3 Peak GPU Memory (max across steps)")
    ax.set_ylabel("GiB")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")

    # Annotate values
    def annotate(bars):
        for bar in bars:
            h = bar.get_height()
            if np.isfinite(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.2f}",
                        ha="center", va="bottom", fontsize=mpl.rcParams["xtick.labelsize"])

    annotate(b1)
    annotate(b2)

    ax.legend(frameon=False, loc="upper right")
    ax.margins(y=0.12)

    for fmt in formats:
        fig.savefig(outdir / f"fig_peak_memory.{fmt}")
    plt.close(fig)


def plot_time_breakdown(runs: List[RunData], outdir: Path, formats: List[str], paper: str) -> None:
    """
    Make a stacked-bar breakdown with wall time marker + CI.
    metrics_step.csv expected cols:
      wall_ms, fwd_ms, head_fwd_ms, bwd_ms, head_bwd_ms, optim_ms
    """
    labels = [r.label for r in runs]

    comp_names = [
        ("base forward", "fwd_ms"),
        ("head forward", "head_fwd_ms"),
        ("base backward", "bwd_ms"),
        ("head backward", "head_bwd_ms"),
        ("optim", "optim_ms"),
    ]

    # Compute mean per component
    means = {name: [] for name, _ in comp_names}
    wall_mean = []
    wall_ci = []

    for r in runs:
        w = safe_get_col(r.steps, "wall_ms", np.nan)
        mu_w, ci_w = mean_ci95(w)
        wall_mean.append(mu_w)
        wall_ci.append(ci_w)

        for name, col in comp_names:
            v = safe_get_col(r.steps, col, 0.0)
            means[name].append(float(np.nanmean(v)))

    x = np.arange(len(runs), dtype=np.float64)

    fig = plt.figure(figsize=paper_figsize("tall", paper))
    ax = fig.add_subplot(1, 1, 1)

    # Stacked bars
    bottom = np.zeros(len(runs), dtype=np.float64)
    bars = []
    for name, _ in comp_names:
        y = np.array(means[name], dtype=np.float64)
        b = ax.bar(x, y, bottom=bottom, label=name)
        bars.append(b)
        bottom = bottom + np.nan_to_num(y)

    # Wall markers (+ CI)
    ax.errorbar(x, wall_mean, yerr=wall_ci, fmt="o", capsize=3, linewidth=1.2, label="wall (mean ±95% CI)")

    ax.set_title("T1M3 Step Time Breakdown (mean, warmup skipped)")
    ax.set_ylabel("Time per step (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")

    # Legend: compact, outside
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False)

    ax.margins(y=0.15)

    for fmt in formats:
        fig.savefig(outdir / f"fig_time_breakdown.{fmt}")
    plt.close(fig)


def plot_all_in_one(runs: List[RunData], outdir: Path, formats: List[str], paper: str) -> None:
    """
    Single figure with three panels (timeline if available).
    """
    has_mem = any(r.mem is not None and len(r.mem) > 0 for r in runs)
    ncols = 1
    nrows = 3 if has_mem else 2

    # Wider for combined
    w = 6.8 if paper == "double" else 3.35
    fig = plt.figure(figsize=(w, w * (0.95 if has_mem else 0.75)))
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols, hspace=0.55)

    # Panel A: memory timeline
    row = 0
    if has_mem:
        ax0 = fig.add_subplot(gs[row, 0])
        usable = [r for r in runs if r.mem is not None and len(r.mem) > 0]
        for r in usable:
            df = r.mem
            if "t_ms" in df.columns:
                t = df["t_ms"].to_numpy(dtype=np.float64)
            elif "time_ms" in df.columns:
                t = df["time_ms"].to_numpy(dtype=np.float64)
            else:
                t = np.arange(len(df), dtype=np.float64)

            reserved_col = None
            for c in ["gpu_reserved_gib", "gpu_reserved", "reserved_gib", "reserved"]:
                if c in df.columns:
                    reserved_col = c
                    break

            if reserved_col is not None:
                ax0.plot(t, df[reserved_col].to_numpy(dtype=np.float64), label=r.label)

        ax0.set_title("A. GPU reserved timeline")
        ax0.set_xlabel("Time (ms)")
        ax0.set_ylabel("GiB")
        ax0.legend(frameon=False, loc="upper right")
        row += 1

    # Panel B: peak memory
    ax1 = fig.add_subplot(gs[row, 0])
    labels = [r.label for r in runs]
    peak_alloc = [float(np.nanmax(safe_get_col(r.steps, "peak_alloc_gib", np.nan))) for r in runs]
    peak_reserved = [float(np.nanmax(safe_get_col(r.steps, "peak_reserved_gib", np.nan))) for r in runs]
    x = np.arange(len(runs), dtype=np.float64)
    width = 0.36
    ax1.bar(x - width / 2, peak_alloc, width=width, label="Peak allocated")
    ax1.bar(x + width / 2, peak_reserved, width=width, label="Peak reserved")
    ax1.set_title("B. Peak GPU memory (max across steps)")
    ax1.set_ylabel("GiB")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=18, ha="right")
    ax1.legend(frameon=False, loc="upper right")
    row += 1

    # Panel C: time breakdown
    ax2 = fig.add_subplot(gs[row, 0])
    comp_names = [
        ("base fwd", "fwd_ms"),
        ("head fwd", "head_fwd_ms"),
        ("base bwd", "bwd_ms"),
        ("head bwd", "head_bwd_ms"),
        ("optim", "optim_ms"),
    ]
    means = {name: [] for name, _ in comp_names}
    wall_mean = []
    wall_ci = []
    for r in runs:
        wms = safe_get_col(r.steps, "wall_ms", np.nan)
        mu_w, ci_w = mean_ci95(wms)
        wall_mean.append(mu_w)
        wall_ci.append(ci_w)
        for name, col in comp_names:
            v = safe_get_col(r.steps, col, 0.0)
            means[name].append(float(np.nanmean(v)))

    bottom = np.zeros(len(runs), dtype=np.float64)
    for name, _ in comp_names:
        y = np.array(means[name], dtype=np.float64)
        ax2.bar(x, y, bottom=bottom, label=name)
        bottom = bottom + np.nan_to_num(y)

    ax2.errorbar(x, wall_mean, yerr=wall_ci, fmt="o", capsize=3, linewidth=1.2, label="wall ±95% CI")
    ax2.set_title("C. Step time breakdown (mean)")
    ax2.set_ylabel("ms")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=18, ha="right")
    ax2.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=3)

    for fmt in formats:
        fig.savefig(outdir / f"fig_all_in_one.{fmt}")
    plt.close(fig)


# -------------------------
# CLI
# -------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="Run directories to compare (each contains metrics_step.csv, etc.)")
    ap.add_argument("--labels", type=str, default="", help="Comma-separated labels, same length as runs (optional)")
    ap.add_argument("--outdir", type=str, default="t1m3_figs", help="Output directory")
    ap.add_argument("--skip_warmup", type=int, default=1, help="Skip first N steps for statistics")
    ap.add_argument("--paper", type=str, default="double", choices=["single", "double"], help="Figure width preset")
    ap.add_argument("--formats", type=str, default="png,pdf", help="Output formats, e.g., png,pdf")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    formats = [x.strip() for x in args.formats.split(",") if x.strip()]
    labels = [x.strip() for x in args.labels.split(",")] if args.labels.strip() else []

    apply_paper_style(args.paper)

    runs: List[RunData] = []
    for i, rd in enumerate(args.runs):
        lab = labels[i] if i < len(labels) else None
        runs.append(load_run(Path(rd), lab, args.skip_warmup))

    # Main figures
    plot_mem_timeline(runs, outdir, formats, args.paper)
    plot_peak_memory(runs, outdir, formats, args.paper)
    plot_time_breakdown(runs, outdir, formats, args.paper)
    plot_all_in_one(runs, outdir, formats, args.paper)

    # Simple console summary (for quick sanity)
    print("[OK] Saved figures to:", outdir.resolve())
    for r in runs:
        pa = float(np.nanmax(safe_get_col(r.steps, "peak_alloc_gib", np.nan)))
        pr = float(np.nanmax(safe_get_col(r.steps, "peak_reserved_gib", np.nan)))
        w = safe_get_col(r.steps, "wall_ms", np.nan)
        mu, ci = mean_ci95(w)
        print(f"  - {r.label}: peak_alloc={pa:.2f}GiB peak_reserved={pr:.2f}GiB wall={mu:.1f}±{ci:.1f} ms (95% CI)")


if __name__ == "__main__":
    main()
