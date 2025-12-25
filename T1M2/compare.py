import os, re, json, math, argparse
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# -------------------------
# Robust trace loader (handles truncated json)
# -------------------------

def _iter_traceevent_objtexts(text: str):
    m = re.search(r'"traceEvents"\s*:\s*\[', text)
    if not m:
        return
    i = m.end()
    n = len(text)

    in_str = False
    esc = False
    depth = 0
    obj_start = None

    while i < n:
        ch = text[i]
        if obj_start is None:
            if ch == "{":
                obj_start = i
                depth = 1
                in_str = False
                esc = False
            i += 1
            continue

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    yield text[obj_start:i + 1]
                    obj_start = None
        i += 1


def load_trace_events(path: str, max_events=None):
    meta = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        events = data.get("traceEvents", []) or []
        for k in ("schemaVersion", "trace_id", "cuda_driver_version", "cuda_runtime_version", "profile_memory"):
            if k in data:
                meta[k] = data[k]
        dp = data.get("deviceProperties") or []
        if dp and isinstance(dp, list) and isinstance(dp[0], dict):
            meta["device0"] = dp[0].get("name")
        if max_events:
            events = events[:max_events]
        return events, meta
    except Exception:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        m = re.search(r'"trace_id"\s*:\s*"([^"]+)"', text)
        if m: meta["trace_id"] = m.group(1)
        m = re.search(r'"cuda_driver_version"\s*:\s*(\d+)', text)
        if m: meta["cuda_driver_version"] = int(m.group(1))
        m = re.search(r'"cuda_runtime_version"\s*:\s*(\d+)', text)
        if m: meta["cuda_runtime_version"] = int(m.group(1))
        m = re.search(r'"profile_memory"\s*:\s*(\d+)', text)
        if m: meta["profile_memory"] = int(m.group(1))
        m = re.search(r'"deviceProperties"\s*:\s*\[\s*\{\s*[^}]*"name"\s*:\s*"([^"]+)"', text, flags=re.S)
        if m: meta["device0"] = m.group(1)

        events = []
        for obj_text in _iter_traceevent_objtexts(text):
            try:
                events.append(json.loads(obj_text))
            except Exception:
                continue
            if max_events and len(events) >= max_events:
                break
        return events, meta


# -------------------------
# Helpers / classification
# -------------------------

def get_ts_dur(e):
    ts = e.get("ts", None)
    dur = e.get("dur", None)
    if ts is None or dur is None:
        return None, None
    try:
        return float(ts), float(dur)
    except Exception:
        return None, None

def us_to_ms(x): return float(x) / 1000.0

def wall_bounds(events):
    tmin = float("inf")
    tmax = 0.0
    for e in events:
        ts, dur = get_ts_dur(e)
        if ts is None:
            continue
        tmin = min(tmin, ts)
        tmax = max(tmax, ts + dur)
    if not math.isfinite(tmin) or tmax <= tmin:
        return None, None
    return tmin, tmax

def is_cpu_op(e):
    return e.get("ph") == "X" and e.get("cat") == "cpu_op"

def is_gpu_kernel(e):
    if e.get("ph") != "X":
        return False
    cat = (e.get("cat") or "")
    if cat == "Kernel":
        return True
    cat_l = cat.lower()
    return ("kernel" in cat_l) and ("cuda" in cat_l or "gpu" in cat_l)

def is_cuda_runtime(e):
    if e.get("ph") != "X":
        return False
    cat = (e.get("cat") or "").lower()
    return ("cuda_runtime" in cat) or ("cuda_driver" in cat)

def extract_events_table(events, kind):
    """
    kind: 'cpu' | 'kernel' | 'runtime'
    returns df columns: name, count, dur_ms
    """
    pred = is_cpu_op if kind == "cpu" else (is_gpu_kernel if kind == "kernel" else is_cuda_runtime)
    agg = defaultdict(lambda: [0, 0.0])  # count, dur_us
    for e in events:
        if e.get("ph") != "X":
            continue
        if not pred(e):
            continue
        name = e.get("name", "") or ""
        ts, dur = get_ts_dur(e)
        if ts is None:
            continue
        agg[name][0] += 1
        agg[name][1] += dur

    rows = [(k, v[0], us_to_ms(v[1])) for k, v in agg.items()]
    df = pd.DataFrame(rows, columns=["name", "count", "dur_ms"])
    if len(df) == 0:
        return df
    df.sort_values(["dur_ms", "count"], ascending=[False, False], inplace=True)
    return df

def compare_tables(df0, df1):
    a = df0.rename(columns={"count": "count_0", "dur_ms": "dur_ms_0"})
    b = df1.rename(columns={"count": "count_1", "dur_ms": "dur_ms_1"})
    d = pd.merge(a, b, on="name", how="outer").fillna(0.0)
    d["count_diff"] = d["count_1"] - d["count_0"]
    d["dur_ms_diff"] = d["dur_ms_1"] - d["dur_ms_0"]
    d["dur_ms_max"] = d[["dur_ms_0", "dur_ms_1"]].max(axis=1)
    d["count_max"] = d[["count_0", "count_1"]].max(axis=1)

    total0 = float(d["dur_ms_0"].sum())
    total1 = float(d["dur_ms_1"].sum())
    d["share_0_%"] = (d["dur_ms_0"] / total0 * 100.0) if total0 > 0 else 0.0
    d["share_1_%"] = (d["dur_ms_1"] / total1 * 100.0) if total1 > 0 else 0.0
    d["share_diff_%"] = d["share_1_%"] - d["share_0_%"]

    # ratio (avoid div0)
    def ratio(n, dnm):
        if dnm <= 0:
            return float("inf") if n > 0 else 1.0
        return n / dnm
    d["dur_ratio_1_over_0"] = d.apply(lambda r: ratio(r["dur_ms_1"], r["dur_ms_0"]), axis=1)
    d["count_ratio_1_over_0"] = d.apply(lambda r: ratio(r["count_1"], r["count_0"]), axis=1)

    return d, total0, total1

def shorten(s, maxlen=90):
    s = str(s)
    return s if len(s) <= maxlen else (s[:maxlen-1] + "…")

def save_df(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig")

def plot_grouped_barh(df, value0, value1, title, xlabel, out_png, topk=25, sort_by="dur_ms_max"):
    if len(df) == 0:
        return
    d = df.sort_values(sort_by, ascending=False).head(topk).copy()
    d["label"] = d["name"].map(lambda x: shorten(x, 90))

    y = np.arange(len(d))
    h = 0.38

    plt.figure(figsize=(16, max(6, int(0.35 * len(d) + 2))))
    plt.barh(y - h/2, d[value0].values, height=h, label="recompute=0")
    plt.barh(y + h/2, d[value1].values, height=h, label="recompute=1")
    plt.yticks(y, d["label"].values)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close()

def plot_diff_barh(df, diff_col, title, xlabel, out_png, topk=25, sort_by_abs=True):
    if len(df) == 0:
        return
    d = df.copy()
    if sort_by_abs:
        d["_abs"] = d[diff_col].abs()
        d = d.sort_values("_abs", ascending=False).drop(columns=["_abs"])
    else:
        d = d.sort_values(diff_col, ascending=False)
    d = d.head(topk).copy()
    d["label"] = d["name"].map(lambda x: shorten(x, 90))

    y = np.arange(len(d))
    plt.figure(figsize=(16, max(6, int(0.35 * len(d) + 2))))
    plt.barh(y, d[diff_col].values)
    plt.yticks(y, d["label"].values)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close()


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace0", type=str, required=True)
    ap.add_argument("--trace1", type=str, required=True)
    ap.add_argument("--out", type=str, default="compare_out")
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--max_events", type=int, default=0, help="0=all")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    max_events = args.max_events if args.max_events > 0 else None

    ev0, meta0 = load_trace_events(args.trace0, max_events=max_events)
    ev1, meta1 = load_trace_events(args.trace1, max_events=max_events)

    t0_min, t0_max = wall_bounds(ev0)
    t1_min, t1_max = wall_bounds(ev1)
    wall0_ms = us_to_ms(t0_max - t0_min) if t0_min is not None else 0.0
    wall1_ms = us_to_ms(t1_max - t1_min) if t1_min is not None else 0.0

    # per-category tables
    cpu0 = extract_events_table(ev0, "cpu")
    cpu1 = extract_events_table(ev1, "cpu")
    ker0 = extract_events_table(ev0, "kernel")
    ker1 = extract_events_table(ev1, "kernel")
    rt0  = extract_events_table(ev0, "runtime")
    rt1  = extract_events_table(ev1, "runtime")

    cpu_cmp, cpu_total0, cpu_total1 = compare_tables(cpu0, cpu1)
    ker_cmp, ker_total0, ker_total1 = compare_tables(ker0, ker1)
    rt_cmp,  rt_total0,  rt_total1  = compare_tables(rt0,  rt1)

    # totals summary (解释“总时间/计算量”)
    # sum(dur_ms) 是“累计算子时间”，可能 > wall_ms（多线程/重叠）
    summary = pd.DataFrame([
        ["wall_ms (trace span)", wall0_ms, wall1_ms, wall1_ms - wall0_ms, (wall1_ms - wall0_ms) / wall0_ms * 100.0 if wall0_ms > 0 else float("inf")],
        ["CPU ops count", float(cpu_cmp["count_0"].sum()), float(cpu_cmp["count_1"].sum()), float(cpu_cmp["count_1"].sum() - cpu_cmp["count_0"].sum()),
         (cpu_cmp["count_1"].sum() - cpu_cmp["count_0"].sum()) / cpu_cmp["count_0"].sum() * 100.0 if cpu_cmp["count_0"].sum() > 0 else float("inf")],
        ["CPU ops accum_time_ms", cpu_total0, cpu_total1, cpu_total1 - cpu_total0, (cpu_total1 - cpu_total0) / cpu_total0 * 100.0 if cpu_total0 > 0 else float("inf")],
        ["GPU kernels count", float(ker_cmp["count_0"].sum()), float(ker_cmp["count_1"].sum()), float(ker_cmp["count_1"].sum() - ker_cmp["count_0"].sum()),
         (ker_cmp["count_1"].sum() - ker_cmp["count_0"].sum()) / ker_cmp["count_0"].sum() * 100.0 if ker_cmp["count_0"].sum() > 0 else float("inf")],
        ["GPU kernels accum_time_ms", ker_total0, ker_total1, ker_total1 - ker_total0, (ker_total1 - ker_total0) / ker_total0 * 100.0 if ker_total0 > 0 else float("inf")],
        ["CUDA runtime calls count", float(rt_cmp["count_0"].sum()), float(rt_cmp["count_1"].sum()), float(rt_cmp["count_1"].sum() - rt_cmp["count_0"].sum()),
         (rt_cmp["count_1"].sum() - rt_cmp["count_0"].sum()) / rt_cmp["count_0"].sum() * 100.0 if rt_cmp["count_0"].sum() > 0 else float("inf")],
        ["CUDA runtime accum_time_ms", rt_total0, rt_total1, rt_total1 - rt_total0, (rt_total1 - rt_total0) / rt_total0 * 100.0 if rt_total0 > 0 else float("inf")],
    ], columns=["metric", "recompute0", "recompute1", "delta", "pct_change_%"])

    # save csvs
    save_df(summary, os.path.join(args.out, "summary_totals.csv"))

    save_df(cpu0, os.path.join(args.out, "cpu_ops_recompute0.csv"))
    save_df(cpu1, os.path.join(args.out, "cpu_ops_recompute1.csv"))
    save_df(ker0, os.path.join(args.out, "gpu_kernels_recompute0.csv"))
    save_df(ker1, os.path.join(args.out, "gpu_kernels_recompute1.csv"))
    save_df(rt0,  os.path.join(args.out, "cuda_runtime_recompute0.csv"))
    save_df(rt1,  os.path.join(args.out, "cuda_runtime_recompute1.csv"))

    save_df(cpu_cmp.sort_values("dur_ms_max", ascending=False), os.path.join(args.out, "cpu_ops_compare.csv"))
    save_df(ker_cmp.sort_values("dur_ms_max", ascending=False), os.path.join(args.out, "gpu_kernels_compare.csv"))
    save_df(rt_cmp.sort_values("dur_ms_max", ascending=False),  os.path.join(args.out, "cuda_runtime_compare.csv"))

    # plots: same-op (OFF vs ON)
    plot_grouped_barh(
        cpu_cmp, "dur_ms_0", "dur_ms_1",
        title=f"CPU ops time by operator (Top {args.topk} by max time)",
        xlabel="Accumulated time per op (ms)",
        out_png=os.path.join(args.out, "cpu_ops_time_grouped.png"),
        topk=args.topk,
        sort_by="dur_ms_max",
    )
    plot_grouped_barh(
        ker_cmp, "dur_ms_0", "dur_ms_1",
        title=f"GPU kernels time by kernel name (Top {args.topk} by max time)",
        xlabel="Accumulated time per kernel (ms)",
        out_png=os.path.join(args.out, "gpu_kernels_time_grouped.png"),
        topk=args.topk,
        sort_by="dur_ms_max",
    )
    plot_grouped_barh(
        rt_cmp, "dur_ms_0", "dur_ms_1",
        title=f"CUDA runtime time by call (Top {args.topk} by max time)",
        xlabel="Accumulated time per runtime call (ms)",
        out_png=os.path.join(args.out, "cuda_runtime_time_grouped.png"),
        topk=args.topk,
        sort_by="dur_ms_max",
    )

    # plots: diffs (what increased most)
    plot_diff_barh(
        cpu_cmp, "dur_ms_diff",
        title=f"CPU ops time delta (recompute1 - recompute0), Top {args.topk}",
        xlabel="Delta time (ms)",
        out_png=os.path.join(args.out, "cpu_ops_time_delta.png"),
        topk=args.topk,
    )
    plot_diff_barh(
        ker_cmp, "dur_ms_diff",
        title=f"GPU kernels time delta (recompute1 - recompute0), Top {args.topk}",
        xlabel="Delta time (ms)",
        out_png=os.path.join(args.out, "gpu_kernels_time_delta.png"),
        topk=args.topk,
    )
    plot_diff_barh(
        rt_cmp, "dur_ms_diff",
        title=f"CUDA runtime time delta (recompute1 - recompute0), Top {args.topk}",
        xlabel="Delta time (ms)",
        out_png=os.path.join(args.out, "cuda_runtime_time_delta.png"),
        topk=args.topk,
    )

    # counts (compute amount proxy)
    plot_grouped_barh(
        cpu_cmp, "count_0", "count_1",
        title=f"CPU ops call count by operator (Top {args.topk} by max count)",
        xlabel="Call count",
        out_png=os.path.join(args.out, "cpu_ops_count_grouped.png"),
        topk=args.topk,
        sort_by="count_max",
    )
    plot_grouped_barh(
        ker_cmp, "count_0", "count_1",
        title=f"GPU kernel launch count by kernel name (Top {args.topk} by max count)",
        xlabel="Launch count",
        out_png=os.path.join(args.out, "gpu_kernels_count_grouped.png"),
        topk=args.topk,
        sort_by="count_max",
    )
    plot_grouped_barh(
        rt_cmp, "count_0", "count_1",
        title=f"CUDA runtime call count by API (Top {args.topk} by max count)",
        xlabel="Call count",
        out_png=os.path.join(args.out, "cuda_runtime_count_grouped.png"),
        topk=args.topk,
        sort_by="count_max",
    )

    # save meta
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"trace0": meta0, "trace1": meta1}, f, ensure_ascii=False, indent=2)

    # print summary to console
    print("==== TOTALS (trace span + accumulated per-category) ====")
    print(summary.to_string(index=False))
    print()
    print("[NOTE] accum_time_ms is SUM(dur) of events. It can be > wall_ms due to overlap/multi-thread/async GPU.")
    print("[SAVED]", args.out)


if __name__ == "__main__":
    main()
