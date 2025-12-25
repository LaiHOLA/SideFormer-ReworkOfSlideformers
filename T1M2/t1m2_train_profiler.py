## OFF
'''
python t1m2_train_profiler.py --model .\qwen --out runs_t1m2 --seq 256 --batch 1 `
  --epochs 1 --steps-per-epoch 30 --lr 1e-4 --optim sgd --dtype bf16 `
  --profiler 1 --prof-active 10 --sweep 1
'''
# Windows + torch2.7 + CUDA OK
# Outputs:
#   step_metrics.csv, layer_metrics.csv, epoch_metrics.csv, min_trace.json (+ optional profiler_trace.json)

import os
import json
import math
import time
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as ckpt

try:
    import psutil
except Exception:
    psutil = None

from transformers import AutoTokenizer, AutoModelForCausalLM


# -------------------------
# utils
# -------------------------
def now_str():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def gib(x: float) -> float:
    return float(x) / (1024**3)


def get_cpu_rss_gib() -> float:
    if psutil is not None:
        try:
            return gib(psutil.Process(os.getpid()).memory_info().rss)
        except Exception:
            pass
    # fallback (rough): not guaranteed on Windows without psutil
    return float("nan")


def parse_dtype(s: str) -> torch.dtype:
    s = s.lower().strip()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"bad --dtype: {s}")


def parse_layer_set(spec: str, n_layers: int) -> Set[int]:
    """
    spec examples:
      "" -> empty
      "all" -> {0..n-1}
      "0-13" -> 0..13
      "0-5,10,20-27"
    """
    spec = (spec or "").strip().lower()
    if spec == "":
        return set()
    if spec == "all":
        return set(range(n_layers))

    out: Set[int] = set()
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            a, b = int(a), int(b)
            lo, hi = min(a, b), max(a, b)
            for i in range(lo, hi + 1):
                out.add(i)
        else:
            out.add(int(p))
    # clamp
    out = {i for i in out if 0 <= i < n_layers}
    return out


def safe_from_pretrained(model_path: str, dtype: torch.dtype, device: str):
    # transformers 新版本可能更偏向 dtype=，老版本用 torch_dtype=
    kw = dict(low_cpu_mem_usage=True)
    try:
        m = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, **kw)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, **kw)
    m.to(device)
    return m


def write_csv(path: Path, header: List[str], rows: List[List]):
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join("" if v is None else str(v) for v in r) + "\n")


@dataclass
class StepRow:
    epoch: int
    step: int
    global_step: int
    wall_ms: float
    fwd_ms: float
    bwd_ms: float
    opt_ms: float
    loss: float
    tps: float
    cpu_rss_gib: float
    gpu_alloc_gib: float
    gpu_resv_gib: float
    peak_alloc_gib: float
    peak_resv_gib: float
    cum_wall_ms: float


@dataclass
class LayerRow:
    epoch: int
    global_step: int
    layer: int
    fwd_ms: float
    recompute_fwd_ms: float
    bwd_ms: float
    fwd_calls_phase_fwd: int
    fwd_calls_phase_bwd: int


# -------------------------
# layer timing hooks
# -------------------------
class LayerTimer:
    """
    Captures:
      - forward time during phase="fwd"
      - forward time during phase="bwd" (this is recompute-forward if checkpointing is on)
      - backward time by full_backward_pre_hook / full_backward_hook
    """
    def __init__(self, layers: nn.ModuleList, device: str):
        self.layers = layers
        self.device = device

        self.phase = "fwd"  # "fwd" or "bwd"

        # per-step temp storage (cuda events)
        self._fwd_starts: Dict[Tuple[int, str], torch.cuda.Event] = {}
        self._fwd_ends: Dict[Tuple[int, str], torch.cuda.Event] = {}
        self._bwd_starts: Dict[int, torch.cuda.Event] = {}
        self._bwd_ends: Dict[int, torch.cuda.Event] = {}

        # per-step counters
        self._fwd_calls_fwd: Dict[int, int] = {}
        self._fwd_calls_bwd: Dict[int, int] = {}

        # accumulators for current step results
        self.fwd_ms: Dict[int, float] = {}
        self.recompute_fwd_ms: Dict[int, float] = {}
        self.bwd_ms: Dict[int, float] = {}

        self._hooks = []
        self._install()

    def _install(self):
        for i, layer in enumerate(self.layers):
            # forward hooks
            def pre_hook(mod, inputs, idx=i):
                if self.device != "cuda":
                    return
                ev = torch.cuda.Event(enable_timing=True)
                ev.record()
                self._fwd_starts[(idx, self.phase)] = ev
                if self.phase == "fwd":
                    self._fwd_calls_fwd[idx] = self._fwd_calls_fwd.get(idx, 0) + 1
                else:
                    self._fwd_calls_bwd[idx] = self._fwd_calls_bwd.get(idx, 0) + 1

            def post_hook(mod, inputs, outputs, idx=i):
                if self.device != "cuda":
                    return
                ev = torch.cuda.Event(enable_timing=True)
                ev.record()
                self._fwd_ends[(idx, self.phase)] = ev

            # backward hooks
            def bwd_pre_hook(mod, grad_output, idx=i):
                if self.device != "cuda":
                    return
                ev = torch.cuda.Event(enable_timing=True)
                ev.record()
                self._bwd_starts[idx] = ev

            def bwd_post_hook(mod, grad_input, grad_output, idx=i):
                if self.device != "cuda":
                    return
                ev = torch.cuda.Event(enable_timing=True)
                ev.record()
                self._bwd_ends[idx] = ev

            self._hooks.append(layer.register_forward_pre_hook(pre_hook))
            self._hooks.append(layer.register_forward_hook(post_hook))
            # full backward hooks only exist in newer torch; torch2.7 has them
            self._hooks.append(layer.register_full_backward_pre_hook(bwd_pre_hook))
            self._hooks.append(layer.register_full_backward_hook(bwd_post_hook))

    def clear_step(self):
        self._fwd_starts.clear()
        self._fwd_ends.clear()
        self._bwd_starts.clear()
        self._bwd_ends.clear()
        self._fwd_calls_fwd.clear()
        self._fwd_calls_bwd.clear()
        self.fwd_ms.clear()
        self.recompute_fwd_ms.clear()
        self.bwd_ms.clear()

    def finalize_step(self):
        # must call after torch.cuda.synchronize()
        for i in range(len(self.layers)):
            # forward in phase fwd
            k_fwd = (i, "fwd")
            if k_fwd in self._fwd_starts and k_fwd in self._fwd_ends:
                self.fwd_ms[i] = self._fwd_starts[k_fwd].elapsed_time(self._fwd_ends[k_fwd])
            else:
                self.fwd_ms[i] = 0.0

            # forward in phase bwd -> recompute forward (if any)
            k_rec = (i, "bwd")
            if k_rec in self._fwd_starts and k_rec in self._fwd_ends:
                self.recompute_fwd_ms[i] = self._fwd_starts[k_rec].elapsed_time(self._fwd_ends[k_rec])
            else:
                self.recompute_fwd_ms[i] = 0.0

            # backward
            if i in self._bwd_starts and i in self._bwd_ends:
                self.bwd_ms[i] = self._bwd_starts[i].elapsed_time(self._bwd_ends[i])
            else:
                self.bwd_ms[i] = 0.0

    def calls_fwd(self, i: int) -> int:
        return self._fwd_calls_fwd.get(i, 0)

    def calls_bwd(self, i: int) -> int:
        return self._fwd_calls_bwd.get(i, 0)

    def close(self):
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks.clear()


# -------------------------
# checkpoint patching
# -------------------------
def patch_layers_checkpoint(layers: nn.ModuleList, ckpt_set: Set[int], use_reentrant: bool):
    """
    Wrap layer.forward with torch.utils.checkpoint.checkpoint for selected layers.
    Returns a list of (idx, original_forward) to restore later.
    """
    originals = []
    for i, layer in enumerate(layers):
        if i not in ckpt_set:
            continue
        orig = layer.forward
        originals.append((i, orig))

        def wrapped_forward(*args, __orig=orig, **kwargs):
            return ckpt(__orig, *args, use_reentrant=use_reentrant, **kwargs)

        layer.forward = wrapped_forward  # type: ignore
    return originals


def restore_layers_forward(layers: nn.ModuleList, originals):
    for (i, orig) in originals:
        layers[i].forward = orig  # type: ignore


# -------------------------
# minimal trace (chrome format)
# -------------------------
class MinTrace:
    def __init__(self):
        self.events = []
        self._t0_ns = time.perf_counter_ns()

    def ts_us(self) -> int:
        return (time.perf_counter_ns() - self._t0_ns) // 1000

    def add(self, name: str, ts_us: int, dur_us: int, cat: str, args: dict):
        self.events.append({
            "name": name,
            "cat": cat,
            "ph": "X",
            "ts": int(ts_us),
            "dur": int(max(0, dur_us)),
            "pid": 0,
            "tid": 0,
            "args": args or {}
        })

    def dump(self, path: Path):
        with path.open("w", encoding="utf-8") as f:
            json.dump({"traceEvents": self.events, "displayTimeUnit": "us"}, f)


# -------------------------
# core run
# -------------------------
def run_one(config, recompute_on: bool):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = parse_dtype(config.dtype)

    run_dir = ensure_dir(Path(config.out) / f"{now_str()}_T1M2_recomp{1 if recompute_on else 0}_seq{config.seq}_bs{config.batch}_ep{config.epochs}_spe{config.steps_per_epoch}")
    print(f"[RUN_DIR] {run_dir}")

    env = {
        "python": f"{os.sys.version}",
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "args": {
            **vars(config),
            "recompute": int(recompute_on),
        }
    }
    (run_dir / "env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    print("[ENV]", json.dumps(env, ensure_ascii=False))

    # load tokenizer/model
    print("[LOAD] tokenizer/model ...")
    tok = AutoTokenizer.from_pretrained(config.model, use_fast=True)
    model = safe_from_pretrained(config.model, dtype=dtype, device=device)
    model.train()

    # locate decoder layers
    layers = None
    base = getattr(model, "model", None)
    if base is not None and hasattr(base, "layers"):
        layers = base.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    if layers is None:
        raise RuntimeError("cannot find decoder layers (expect model.model.layers or model.layers)")
    if not isinstance(layers, (nn.ModuleList, list, tuple)):
        raise RuntimeError("decoder layers is not a ModuleList-like object")

    n_layers = len(layers)
    print(f"[INFO] detected decoder layers: {n_layers}")

    # recompute control
    ckpt_spec = (config.ckpt_layers or "").strip()
    if recompute_on:
        if ckpt_spec == "":
            ckpt_set = set(range(n_layers))  # default: checkpoint all layers
        else:
            ckpt_set = parse_layer_set(ckpt_spec, n_layers)
    else:
        ckpt_set = set()

    if recompute_on:
        # disable cache when checkpointing
        try:
            model.config.use_cache = False
        except Exception:
            pass

    originals = []
    if len(ckpt_set) > 0:
        originals = patch_layers_checkpoint(layers, ckpt_set, use_reentrant=bool(config.use_reentrant))
        print(f"[CFG] recompute=ON, ckpt_layers={sorted(list(ckpt_set))[:8]}{'...' if len(ckpt_set)>8 else ''}, use_reentrant={config.use_reentrant}")
    else:
        print("[CFG] recompute=OFF")

    # optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optim_name = config.optim.lower().strip()
    if optim_name == "sgd":
        opt = torch.optim.SGD(params, lr=config.lr)
    elif optim_name == "adam":
        opt = torch.optim.Adam(params, lr=config.lr)
    elif optim_name in ("adamw", "adamw_torch"):
        opt = torch.optim.AdamW(params, lr=config.lr)
    else:
        raise ValueError(f"bad --optim: {config.optim}")

    # random but deterministic data
    g = torch.Generator(device="cpu")
    g.manual_seed(config.seed)
    vocab = int(getattr(model.config, "vocab_size", 32000))
    input_ids_cpu = torch.randint(low=0, high=vocab, size=(config.batch, config.seq), generator=g, dtype=torch.long)
    input_ids = input_ids_cpu.to(device, non_blocking=True)

    # timers + profiler
    layer_timer = LayerTimer(layers, device=device) if device == "cuda" else None

    use_prof = bool(config.profiler)
    prof = None
    prof_trace_path = run_dir / "profiler_trace.json"
    if use_prof:
        # IMPORTANT: no on_trace_ready => we export ONCE at the end, no "Trace already saved"
        from torch.profiler import profile, ProfilerActivity, schedule
        act = [ProfilerActivity.CPU]
        if device == "cuda":
            act.append(ProfilerActivity.CUDA)
        prof = profile(
            activities=act,
            schedule=schedule(
                wait=max(0, int(config.prof_wait)),
                warmup=max(0, int(config.prof_warmup)),
                active=max(1, int(config.prof_active)),
                repeat=max(1, int(config.prof_repeat)),
            ),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )
        prof.__enter__()
        print("[PROF] torch.profiler enabled")

    # outputs
    step_rows: List[StepRow] = []
    layer_rows: List[LayerRow] = []
    epoch_rows: List[List] = []
    mintrace = MinTrace()

    # train
    torch.manual_seed(config.seed)
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    global_step = 0
    cum_wall_ms = 0.0

    for ep in range(config.epochs):
        print(f"\n========== [EPOCH {ep}] ==========")
        ep_wall_s_t0 = time.perf_counter()

        sum_wall_ms = 0.0
        sum_fwd_ms = 0.0
        sum_bwd_ms = 0.0
        sum_opt_ms = 0.0

        for st in range(config.steps_per_epoch):
            if layer_timer is not None:
                layer_timer.clear_step()

            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()

            # wall start
            t0_ns = time.perf_counter_ns()

            # forward timing
            if device == "cuda":
                fwd_s = torch.cuda.Event(enable_timing=True)
                fwd_e = torch.cuda.Event(enable_timing=True)
                bwd_s = torch.cuda.Event(enable_timing=True)
                bwd_e = torch.cuda.Event(enable_timing=True)
                opt_s = torch.cuda.Event(enable_timing=True)
                opt_e = torch.cuda.Event(enable_timing=True)

            # forward
            if layer_timer is not None:
                layer_timer.phase = "fwd"

            if device == "cuda":
                fwd_s.record()
            out = model(input_ids=input_ids, labels=input_ids)
            loss = out.loss
            if device == "cuda":
                fwd_e.record()

            # backward
            opt.zero_grad(set_to_none=True)

            if layer_timer is not None:
                layer_timer.phase = "bwd"

            if device == "cuda":
                bwd_s.record()
            loss.backward()
            if device == "cuda":
                bwd_e.record()

            # optimizer
            if device == "cuda":
                opt_s.record()
            opt.step()
            if device == "cuda":
                opt_e.record()

            # sync once for stable numbers
            if device == "cuda" and int(config.sync) == 1:
                torch.cuda.synchronize()

            # wall end
            t1_ns = time.perf_counter_ns()
            wall_ms = (t1_ns - t0_ns) / 1e6
            cum_wall_ms += wall_ms

            if device == "cuda":
                fwd_ms = float(fwd_s.elapsed_time(fwd_e))
                bwd_ms = float(bwd_s.elapsed_time(bwd_e))
                opt_ms = float(opt_s.elapsed_time(opt_e))
            else:
                fwd_ms = float("nan")
                bwd_ms = float("nan")
                opt_ms = float("nan")

            sum_wall_ms += wall_ms
            sum_fwd_ms += 0.0 if math.isnan(fwd_ms) else fwd_ms
            sum_bwd_ms += 0.0 if math.isnan(bwd_ms) else bwd_ms
            sum_opt_ms += 0.0 if math.isnan(opt_ms) else opt_ms

            # memory
            cpu_rss = get_cpu_rss_gib()
            if device == "cuda":
                gpu_alloc = gib(torch.cuda.memory_allocated())
                gpu_resv = gib(torch.cuda.memory_reserved())
                peak_alloc = gib(torch.cuda.max_memory_allocated())
                peak_resv = gib(torch.cuda.max_memory_reserved())
            else:
                gpu_alloc = gpu_resv = peak_alloc = peak_resv = float("nan")

            # tps
            wall_s = wall_ms / 1000.0
            tps = (config.batch * config.seq) / max(1e-9, wall_s)

            # finalize per-layer
            if layer_timer is not None:
                layer_timer.finalize_step()
                for li in range(n_layers):
                    layer_rows.append(LayerRow(
                        epoch=ep,
                        global_step=global_step,
                        layer=li,
                        fwd_ms=float(layer_timer.fwd_ms.get(li, 0.0)),
                        recompute_fwd_ms=float(layer_timer.recompute_fwd_ms.get(li, 0.0)),
                        bwd_ms=float(layer_timer.bwd_ms.get(li, 0.0)),
                        fwd_calls_phase_fwd=int(layer_timer.calls_fwd(li)),
                        fwd_calls_phase_bwd=int(layer_timer.calls_bwd(li)),
                    ))

            # minimal trace events (step level)
            ts0_us = mintrace.ts_us()
            mintrace.add(
                name=f"step_{global_step}",
                ts_us=ts0_us,
                dur_us=int(wall_ms * 1000),
                cat="step",
                args={
                    "epoch": ep,
                    "step": st,
                    "recompute": int(recompute_on),
                    "loss": float(loss.detach().float().cpu().item()),
                    "cpu_rss_gib": cpu_rss,
                    "gpu_alloc_gib": gpu_alloc,
                    "gpu_resv_gib": gpu_resv,
                }
            )

            step_rows.append(StepRow(
                epoch=ep,
                step=st,
                global_step=global_step,
                wall_ms=wall_ms,
                fwd_ms=fwd_ms,
                bwd_ms=bwd_ms,
                opt_ms=opt_ms,
                loss=float(loss.detach().float().cpu().item()),
                tps=tps,
                cpu_rss_gib=cpu_rss,
                gpu_alloc_gib=gpu_alloc,
                gpu_resv_gib=gpu_resv,
                peak_alloc_gib=peak_alloc,
                peak_resv_gib=peak_resv,
                cum_wall_ms=cum_wall_ms
            ))

            print(f"[STEP {global_step:04d}] wall={wall_ms:.2f}ms fwd={fwd_ms:.2f}ms bwd={bwd_ms:.2f}ms opt={opt_ms:.2f}ms "
                  f"loss={step_rows[-1].loss:.4f} tps={tps:.1f} CPU_RSS={cpu_rss:.2f}GiB GPU_alloc={gpu_alloc:.2f}GiB GPU_resv={gpu_resv:.2f}GiB")

            global_step += 1

            if prof is not None:
                prof.step()

        ep_wall_s = time.perf_counter() - ep_wall_s_t0
        epoch_rows.append([ep, ep_wall_s, sum_wall_ms, sum_fwd_ms, sum_bwd_ms, sum_opt_ms])
        print(f"[EPOCH {ep}] wall_s={ep_wall_s:.3f} sum_wall_ms={sum_wall_ms:.1f} sum_fwd_ms={sum_fwd_ms:.1f} sum_bwd_ms={sum_bwd_ms:.1f} sum_opt_ms={sum_opt_ms:.1f}")

    # profiler export (once)
    if prof is not None:
        try:
            prof.__exit__(None, None, None)
            prof.export_chrome_trace(str(prof_trace_path))
            print(f"[SAVED] profiler trace: {prof_trace_path}")
        except Exception as e:
            print(f"[WARN] export chrome trace failed: {repr(e)}")

    # write CSVs
    step_csv = run_dir / "step_metrics.csv"
    layer_csv = run_dir / "layer_metrics.csv"
    epoch_csv = run_dir / "epoch_metrics.csv"
    trace_json = run_dir / "min_trace.json"

    write_csv(
        step_csv,
        header=["epoch","step","global_step","wall_ms","fwd_ms","bwd_ms","opt_ms","loss","tps",
                "cpu_rss_gib","gpu_alloc_gib","gpu_resv_gib","peak_alloc_gib","peak_resv_gib","cum_wall_ms"],
        rows=[[r.epoch,r.step,r.global_step,f"{r.wall_ms:.6f}",f"{r.fwd_ms:.6f}",f"{r.bwd_ms:.6f}",f"{r.opt_ms:.6f}",
               f"{r.loss:.8f}",f"{r.tps:.6f}",
               f"{r.cpu_rss_gib:.6f}",f"{r.gpu_alloc_gib:.6f}",f"{r.gpu_resv_gib:.6f}",
               f"{r.peak_alloc_gib:.6f}",f"{r.peak_resv_gib:.6f}",f"{r.cum_wall_ms:.6f}"]
              for r in step_rows]
    )
    write_csv(
        layer_csv,
        header=["epoch","global_step","layer","fwd_ms","recompute_fwd_ms","bwd_ms","fwd_calls_phase_fwd","fwd_calls_phase_bwd"],
        rows=[[r.epoch,r.global_step,r.layer,f"{r.fwd_ms:.6f}",f"{r.recompute_fwd_ms:.6f}",f"{r.bwd_ms:.6f}",
               r.fwd_calls_phase_fwd,r.fwd_calls_phase_bwd]
              for r in layer_rows]
    )
    write_csv(
        epoch_csv,
        header=["epoch","epoch_wall_s","sum_wall_ms","sum_fwd_ms","sum_bwd_ms","sum_opt_ms"],
        rows=[[e, f"{ws:.6f}", f"{sw:.6f}", f"{sf:.6f}", f"{sb:.6f}", f"{so:.6f}"] for (e, ws, sw, sf, sb, so) in epoch_rows]
    )
    mintrace.dump(trace_json)

    print(f"[SAVED] {step_csv}")
    print(f"[SAVED] {layer_csv}")
    print(f"[SAVED] {epoch_csv}")
    print(f"[SAVED] {trace_json}")

    # restore forwards
    if len(originals) > 0:
        restore_layers_forward(layers, originals)
    if layer_timer is not None:
        layer_timer.close()

    return str(run_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--out", type=str, default="runs_t1m2")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--steps-per-epoch", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--optim", type=str, default="sgd", choices=["sgd","adam","adamw"])
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16","fp16","fp32"])
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--sync", type=int, default=1, help="cuda synchronize once per step for stable timing")
    ap.add_argument("--recompute", type=int, default=0, choices=[0,1])
    ap.add_argument("--ckpt-layers", type=str, default="", help='"" (default all when recompute=1), or "all", or "0-13,20-27"')
    ap.add_argument("--use-reentrant", type=int, default=0, choices=[0,1])

    ap.add_argument("--profiler", type=int, default=0, choices=[0,1])
    ap.add_argument("--prof-wait", type=int, default=0)
    ap.add_argument("--prof-warmup", type=int, default=0)
    ap.add_argument("--prof-active", type=int, default=10)
    ap.add_argument("--prof-repeat", type=int, default=1)

    ap.add_argument("--sweep", type=int, default=0, choices=[0,1], help="run recompute OFF then ON in one command")

    args = ap.parse_args()

    ensure_dir(Path(args.out))

    if args.sweep == 1:
        print("[BOOT] sweep mode: recompute OFF -> ON")
        d0 = run_one(args, recompute_on=False)
        d1 = run_one(args, recompute_on=True)
        print("[DONE] sweep finished")
        print("run0:", d0)
        print("run1:", d1)
    else:
        run_one(args, recompute_on=bool(args.recompute))


if __name__ == "__main__":
    main()
