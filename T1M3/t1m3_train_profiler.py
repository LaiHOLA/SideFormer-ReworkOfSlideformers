
# -*- coding: utf-8 -*-
"""
t1m3_train_profiler.py

T1M3: Practical output-layer memory mitigation
---------------------------------------------
Compare two head implementations:

1) ce_full    : materialize full logits [B*(S-1), V] then torch cross_entropy
2) ce_chunked : exact block-wise cross entropy without full logits (t1_m3_head.chunked_ce_loss)

We intentionally DETACH the vocab weight to isolate "output activation/logits memory".
(Otherwise the optimizer states for lm_head/embeddings dominate memory and obscure the effect.)

We obtain hidden states by calling the base model (model.model) to avoid computing logits twice.

Run examples:
  # baseline logits
  python t1m3_train_profiler.py --model .\\qwen --out runs_t1m3 --head ce_full --seq 1024 --batch 2 --steps 30 --dtype bf16

  # chunked (no logits)
  python t1m3_train_profiler.py --model .\\qwen --out runs_t1m3 --head ce_chunked --vocab-chunk 8192 --seq 1024 --batch 2 --steps 30 --dtype bf16

  # with gradient checkpointing
  python t1m3_train_profiler.py --model .\\qwen --out runs_t1m3 --head ce_chunked --ckpt 1 --seq 1024 --batch 2 --steps 30

Outputs in run dir:
  - config.json
  - metrics_step.csv
  - mem_trace.csv
  - summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from t1_m3_head import chunked_ce_loss


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def gib(x: int) -> float:
    return float(x) / (1024**3)


def cpu_rss_gib() -> float:
    try:
        import psutil
        p = psutil.Process(os.getpid())
        return p.memory_info().rss / (1024**3)
    except Exception:
        return float("nan")


class MemMonitor:
    def __init__(self, out_csv: str, interval_ms: int = 25):
        self.out_csv = out_csv
        self.interval_s = max(1, int(interval_ms)) / 1000.0
        self._stop = threading.Event()
        self._t0 = None
        self._thread = None

    def start(self, run_tag: str, step_idx: int):
        self._stop.clear()
        self._t0 = time.perf_counter()
        os.makedirs(os.path.dirname(self.out_csv), exist_ok=True)

        # append header if file not exists
        if not os.path.exists(self.out_csv):
            with open(self.out_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["run_tag", "step", "t_ms", "gpu_alloc_gib", "gpu_reserved_gib", "cpu_rss_gib"])

        def _loop():
            while not self._stop.is_set():
                t_ms = (time.perf_counter() - self._t0) * 1000.0
                ga = gib(torch.cuda.memory_allocated())
                gr = gib(torch.cuda.memory_reserved())
                cr = cpu_rss_gib()
                with open(self.out_csv, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow([run_tag, step_idx, f"{t_ms:.3f}", f"{ga:.6f}", f"{gr:.6f}", f"{cr:.6f}"])
                time.sleep(self.interval_s)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


@dataclass
class StepMetrics:
    step: int
    wall_ms: float
    fwd_ms: float
    head_fwd_ms: float
    head_bwd_ms: float
    bwd_ms: float
    optim_ms: float
    loss: float
    peak_alloc_gib: float
    peak_reserved_gib: float
    cpu_rss_gib: float


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(params, optim: str, lr: float):
    if optim == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if optim == "adam":
        return torch.optim.Adam(params, lr=lr)
    if optim == "adamw":
        return torch.optim.AdamW(params, lr=lr)
    raise ValueError(f"Unknown optim: {optim}")


def enable_grad_ckpt(model):
    # HF models differ slightly; this is the safest pattern.
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False


def compute_head_loss(hidden: torch.Tensor,
                      input_ids: torch.Tensor,
                      vocab_weight: torch.Tensor,
                      head: str,
                      vocab_chunk: int) -> torch.Tensor:
    """
    hidden: [B,S,H]
    input_ids: [B,S]
    vocab_weight: [V,H], DETACHED
    """
    if head == "mse":
        return hidden.float().pow(2).mean()

    # next-token CE
    h = hidden[:, :-1, :].contiguous().view(-1, hidden.size(-1))
    t = input_ids[:, 1:].contiguous().view(-1)

    if head == "ce_full":
        logits = h.matmul(vocab_weight.t())  # [N,V]
        return F.cross_entropy(logits.float(), t, reduction="mean")

    if head == "ce_chunked":
        return chunked_ce_loss(h, vocab_weight, t, chunk_size=vocab_chunk, reduction="mean")

    raise ValueError(f"Unknown head: {head}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="local model dir or HF repo id")
    ap.add_argument("--out", type=str, default="runs_t1m3")
    ap.add_argument("--run-name", type=str, default="", help="optional tag suffix")
    ap.add_argument("--head", type=str, default="ce_full", choices=["mse", "ce_full", "ce_chunked"])
    ap.add_argument("--vocab-chunk", type=int, default=8192)
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--ckpt", type=int, default=0, choices=[0, 1], help="enable HF gradient checkpointing")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--optim", type=str, default="sgd", choices=["sgd", "adam", "adamw"])
    ap.add_argument("--monitor-ms", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")

    set_seed(args.seed)

    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    run_tag = f"{now_tag()}_T1M3_head{args.head}_ckpt{args.ckpt}_seq{args.seq}_bs{args.batch}_{args.dtype}"
    if args.run_name:
        run_tag += f"_{args.run_name}"
    run_dir = os.path.join(args.out, run_tag)
    os.makedirs(run_dir, exist_ok=True)

    print(f"[RUN_DIR] {run_dir}")

    # Save config early
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    # Load model/tokenizer
    print("[LOAD] tokenizer/model ...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dt,
        device_map=None,
        trust_remote_code=True,
    ).to(args.device)

    model.train()
    if args.ckpt:
        enable_grad_ckpt(model)

    # Build optimizer (exclude vocab/emb weight states if you want to isolate)
    # Here we keep it simple: optimize all model params.
    optim = build_optimizer(model.parameters(), args.optim, args.lr)

    # Prepare synthetic inputs (no tokenization cost)
    V = int(getattr(model.config, "vocab_size", 32000))
    B, S = args.batch, args.seq
    print(f"[MODEL] vocab={V}  hidden={model.config.hidden_size if hasattr(model.config,'hidden_size') else 'unknown'}")
    input_ids = torch.randint(low=0, high=V, size=(B, S), device=args.device, dtype=torch.long)

    # DETACHED vocab weight (isolate logits/activation memory)
    vocab_weight = model.get_input_embeddings().weight.detach()

    metrics_path = os.path.join(run_dir, "metrics_step.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(StepMetrics.__annotations__.keys()))

    mem_mon = MemMonitor(os.path.join(run_dir, "mem_trace.csv"), interval_ms=args.monitor_ms)

    # warmup
    print(f"[WARMUP] {args.warmup} steps ...")
    for _ in range(args.warmup):
        optim.zero_grad(set_to_none=True)
        out = model.model(input_ids=input_ids, use_cache=False)
        hidden = out.last_hidden_state
        loss = compute_head_loss(hidden, input_ids, vocab_weight, args.head, args.vocab_chunk)
        loss.backward()
        optim.step()
    torch.cuda.synchronize()

    print(f"[RUN] steps={args.steps} ...")
    rows = []
    for step in range(args.steps):
        torch.cuda.reset_peak_memory_stats()
        optim.zero_grad(set_to_none=True)

        mem_mon.start(run_tag, step)

        t0 = time.perf_counter()

        # Forward base model
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        out = model.model(input_ids=input_ids, use_cache=False)
        hidden = out.last_hidden_state
        e1.record()
        torch.cuda.synchronize()
        fwd_ms = e0.elapsed_time(e1)

        # Head forward
        e2 = torch.cuda.Event(enable_timing=True)
        e3 = torch.cuda.Event(enable_timing=True)
        e2.record()
        loss = compute_head_loss(hidden, input_ids, vocab_weight, args.head, args.vocab_chunk)
        e3.record()
        torch.cuda.synchronize()
        head_fwd_ms = e2.elapsed_time(e3)

        # Head backward + model backward (we don't split them here; keep simple & stable)
        e4 = torch.cuda.Event(enable_timing=True)
        e5 = torch.cuda.Event(enable_timing=True)
        e4.record()
        loss.backward()
        e5.record()
        torch.cuda.synchronize()
        bwd_ms = e4.elapsed_time(e5)
        head_bwd_ms = float("nan")  # optional: see "split backward" in notes

        # Optim
        e6 = torch.cuda.Event(enable_timing=True)
        e7 = torch.cuda.Event(enable_timing=True)
        e6.record()
        optim.step()
        e7.record()
        torch.cuda.synchronize()
        optim_ms = e6.elapsed_time(e7)

        wall_ms = (time.perf_counter() - t0) * 1000.0
        mem_mon.stop()

        peak_alloc = gib(torch.cuda.max_memory_allocated())
        peak_reserved = gib(torch.cuda.max_memory_reserved())
        rss = cpu_rss_gib()

        m = StepMetrics(
            step=step,
            wall_ms=wall_ms,
            fwd_ms=fwd_ms,
            head_fwd_ms=head_fwd_ms,
            head_bwd_ms=head_bwd_ms,
            bwd_ms=bwd_ms,
            optim_ms=optim_ms,
            loss=float(loss.detach().cpu().item()),
            peak_alloc_gib=peak_alloc,
            peak_reserved_gib=peak_reserved,
            cpu_rss_gib=rss,
        )
        rows.append(m)

        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([getattr(m, k) for k in StepMetrics.__annotations__.keys()])

        print(f"[STEP {step:03d}] wall={wall_ms:.1f}ms fwd={fwd_ms:.1f} head_fwd={head_fwd_ms:.1f} "
              f"bwd={bwd_ms:.1f} optim={optim_ms:.1f} "
              f"peak_alloc={peak_alloc:.2f}GiB peak_reserved={peak_reserved:.2f}GiB loss={m.loss:.4f}")

    # summary
    def mean(xs):
        return sum(xs) / max(1, len(xs))

    summ = {
        "run_dir": run_dir,
        "head": args.head,
        "ckpt": args.ckpt,
        "dtype": args.dtype,
        "batch": args.batch,
        "seq": args.seq,
        "steps": args.steps,
        "mean_wall_ms": mean([r.wall_ms for r in rows]),
        "mean_fwd_ms": mean([r.fwd_ms for r in rows]),
        "mean_head_fwd_ms": mean([r.head_fwd_ms for r in rows]),
        "mean_bwd_ms": mean([r.bwd_ms for r in rows]),
        "mean_optim_ms": mean([r.optim_ms for r in rows]),
        "peak_alloc_gib_max": max([r.peak_alloc_gib for r in rows]),
        "peak_reserved_gib_max": max([r.peak_reserved_gib for r in rows]),
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2)

    print("[DONE] summary.json written.")


if __name__ == "__main__":
    main()
