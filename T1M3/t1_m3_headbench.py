
# -*- coding: utf-8 -*-
"""
t1_m3_headbench.py

Micro-benchmark for output-layer memory/time:
  - baseline: materialize logits + torch cross_entropy
  - chunked:  block-wise exact CE (no full logits)

This is the fastest way to validate T1M3 without touching your training pipeline.

Example:
  python t1_m3_headbench.py --vocab 152064 --hidden 2048 --seq 1024 --batch 2 --dtype bf16 --chunk 8192 --iters 20

Outputs:
  - printed summary
  - CSV in --outdir (metrics_headbench.csv)
"""
from __future__ import annotations
import argparse
import os
import time
import csv

import torch
import torch.nn.functional as F

from t1_m3_head import chunked_ce_loss


def _gib(x: int) -> float:
    return float(x) / (1024**3)


def bench_once(mode: str, hidden: torch.Tensor, weight: torch.Tensor, target: torch.Tensor,
               chunk: int, iters: int, warmup: int = 2):
    assert mode in ("full", "chunked")
    torch.cuda.synchronize()
    times = []
    peaks_alloc = []
    peaks_reserved = []

    for it in range(iters + warmup):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()

        # forward + backward (hidden requires grad)
        hidden2 = hidden.detach().requires_grad_(True)
        if mode == "full":
            logits = hidden2.matmul(weight.t())  # [N, V]
            loss = F.cross_entropy(logits.float(), target, reduction="mean")
        else:
            loss = chunked_ce_loss(hidden2, weight, target, chunk_size=chunk, reduction="mean")

        loss.backward()
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        if it >= warmup:
            times.append((t1 - t0) * 1000.0)
            peaks_alloc.append(_gib(torch.cuda.max_memory_allocated()))
            peaks_reserved.append(_gib(torch.cuda.max_memory_reserved()))

    return {
        "mode": mode,
        "ms_mean": sum(times)/len(times),
        "ms_p50": sorted(times)[len(times)//2],
        "ms_p95": sorted(times)[int(len(times)*0.95)-1],
        "peak_alloc_gib_mean": sum(peaks_alloc)/len(peaks_alloc),
        "peak_reserved_gib_mean": sum(peaks_reserved)/len(peaks_reserved),
        "peak_alloc_gib_max": max(peaks_alloc),
        "peak_reserved_gib_max": max(peaks_reserved),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=152064)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--outdir", type=str, default="t1m3_headbench_out")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = "cuda"

    N = args.batch * (args.seq - 1)
    print(f"[CFG] N={N} (batch={args.batch}, seq={args.seq})  V={args.vocab}  H={args.hidden}  dtype={dt}")

    # synthetic hidden/weight/target
    torch.manual_seed(0)
    hidden = torch.randn((N, args.hidden), device=device, dtype=dt) / (args.hidden ** 0.5)
    weight = torch.randn((args.vocab, args.hidden), device=device, dtype=dt) / (args.hidden ** 0.5)
    target = torch.randint(low=0, high=args.vocab, size=(N,), device=device, dtype=torch.long)

    # run
    full = bench_once("full", hidden, weight, target, args.chunk, args.iters)
    chunked = bench_once("chunked", hidden, weight, target, args.chunk, args.iters)

    # print
    def pr(x):
        return f"{x:.3f}"
    print("\n==== HeadBench Summary ====")
    for r in (full, chunked):
        print(f"[{r['mode']}] ms_mean={pr(r['ms_mean'])}  ms_p50={pr(r['ms_p50'])}  ms_p95={pr(r['ms_p95'])}  "
              f"peak_alloc_mean={pr(r['peak_alloc_gib_mean'])}GiB (max={pr(r['peak_alloc_gib_max'])})  "
              f"peak_reserved_mean={pr(r['peak_reserved_gib_mean'])}GiB (max={pr(r['peak_reserved_gib_max'])})")

    # save csv
    out_csv = os.path.join(args.outdir, "metrics_headbench.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(full.keys()))
        w.writeheader()
        w.writerow(full)
        w.writerow(chunked)
    print(f"[SAVED] {out_csv}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")
    main()
