# -*- coding: utf-8 -*-
# t1m2_train_profiler.py
# Run training micro-benchmark with recompute ablation + selective checkpointing.
# Exports:
#   run_dir/step_metrics.csv
#   run_dir/layer_metrics.csv
#   run_dir/epoch_metrics.csv
#   run_dir/env.json, args.json
# Optional: torch.profiler chrome traces under run_dir/prof/
'''
. 一键跑 OFF→ON
python t1m2_train_profiler.py --model .\qwen --out runs_t1m2 `
  --seq 256 --batch 1 --epochs 6 --steps-per-epoch 30 `
  --lr 1e-4 --optim adamw --dtype bf16 `
  --profiler 1 --prof-active 10 --prof-repeat 1 `
  --sweep 1

B. 手动控制“recompute层数”（比如只checkpoint前14层）
python t1m2_train_profiler.py --model .\qwen --out runs_t1m2 `
  --seq 256 --batch 1 --epochs 6 --steps-per-epoch 30 `
  --lr 1e-4 --optim adamw --dtype bf16 `
  --recompute 1 --ckpt-k 14


或者指定任意集合：

python t1m2_train_profiler.py --model .\qwen --out runs_t1m2 `
  --seq 256 --batch 1 --epochs 6 --steps-per-epoch 30 `
  --lr 1e-4 --optim adamw --dtype bf16 `
  --recompute 1 --ckpt-layers "0-7,20-27"
'''

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import torch

try:
    import psutil  # for CPU RSS
except Exception:
    psutil = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception as e:
    raise RuntimeError("transformers not found. Please install transformers in this env.") from e

from torch.utils.checkpoint import checkpoint


def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def cpu_rss_gib() -> float:
    if psutil is None:
        return float("nan")
    p = psutil.Process(os.getpid())
    return float(p.memory_info().rss) / (1024**3)


def cuda_mem_gib() -> Tuple[float, float, float, float]:
    if not torch.cuda.is_available():
        return (0.0, 0.0, 0.0, 0.0)
    alloc = torch.cuda.memory_allocated() / (1024**3)
    resv = torch.cuda.memory_reserved() / (1024**3)
    peak_alloc = torch.cuda.max_memory_allocated() / (1024**3)
    peak_resv = torch.cuda.max_memory_reserved() / (1024**3)
    return float(alloc), float(resv), float(peak_alloc), float(peak_resv)


def parse_dtype(s: str) -> torch.dtype:
    s = s.lower().strip()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"unsupported dtype: {s}")


def parse_layer_spec(spec: str, n_layers: int) -> Set[int]:
    """
    spec examples:
      ""          -> empty set
      "all"       -> {0..n-1}
      "none"      -> empty set
      "0-13"      -> 0..13
      "0-13,20,22-27"
    """
    spec = (spec or "").strip().lower()
    if spec in ("", "none"):
        return set()
    if spec == "all":
        return set(range(n_layers))
    out: Set[int] = set()
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            a = int(a.strip())
            b = int(b.strip())
            if a > b:
                a, b = b, a
            for i in range(a, b + 1):
                if 0 <= i < n_layers:
                    out.add(i)
        else:
            i = int(p)
            if 0 <= i < n_layers:
                out.add(i)
    return out


def get_decoder_layers(model) -> List[torch.nn.Module]:
    """
    Try common HF layouts:
      - model.model.layers (Llama/Qwen-like)
      - model.transformer.h (GPT2-like)
      - model.gpt_neox.layers (GPT-NeoX-like)
      - model.model.decoder.layers (T5-like, but causal LM unlikely)
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = list(model.model.layers)
        if layers:
            return layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = list(model.transformer.h)
        if layers:
            return layers
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        layers = list(model.gpt_neox.layers)
        if layers:
            return layers
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        layers = list(model.model.decoder.layers)
        if layers:
            return layers
    raise RuntimeError("Cannot find decoder layers list on this model; please adapt get_decoder_layers().")


@dataclass
class LayerStepBuffers:
    # CUDA event pairs for forward and recompute-forward
    fwd_pairs: Dict[int, List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = field(default_factory=dict)
    recomp_pairs: Dict[int, List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = field(default_factory=dict)
    # grad-ready events for outputs: idx=-1 is input to layer0
    grad_ev: Dict[int, torch.cuda.Event] = field(default_factory=dict)
    # internal stacks for starts
    _start_stack: Dict[Tuple[int, str], List[torch.cuda.Event]] = field(default_factory=dict)
    # make sure we hook layer0 input once
    _input0_hooked: bool = False


class LayerProbe:
    def __init__(self, layers: List[torch.nn.Module]):
        self.layers = layers
        self.n = len(layers)
        self.phase = "fwd"  # "fwd" or "bwd" (bwd phase contains recompute forwards)
        self.step_buf = LayerStepBuffers()
        self.enabled = torch.cuda.is_available()

    def begin_step(self):
        self.phase = "fwd"
        self.step_buf = LayerStepBuffers()

    def set_bwd_phase(self):
        self.phase = "bwd"

    def _push_start(self, layer_idx: int, kind: str):
        if not self.enabled:
            return
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        key = (layer_idx, kind)
        self.step_buf._start_stack.setdefault(key, []).append(ev)

    def _pop_pair(self, layer_idx: int, kind: str):
        if not self.enabled:
            return
        key = (layer_idx, kind)
        stack = self.step_buf._start_stack.get(key, [])
        if not stack:
            return
        st = stack.pop()
        ed = torch.cuda.Event(enable_timing=True)
        ed.record()
        if kind == "fwd":
            self.step_buf.fwd_pairs.setdefault(layer_idx, []).append((st, ed))
        else:
            self.step_buf.recomp_pairs.setdefault(layer_idx, []).append((st, ed))

    def pre_hook(self, layer_idx: int, module, inputs):
        # hook layer0 input grad-ready event
        if self.enabled and layer_idx == 0 and (not self.step_buf._input0_hooked):
            hs0 = None
            if isinstance(inputs, (tuple, list)) and len(inputs) > 0 and torch.is_tensor(inputs[0]):
                hs0 = inputs[0]
            if hs0 is not None and hs0.requires_grad:
                def _grad_hook(_grad):
                    ev = torch.cuda.Event(enable_timing=True)
                    ev.record()
                    self.step_buf.grad_ev[-1] = ev
                    return _grad
                hs0.register_hook(_grad_hook)
                self.step_buf._input0_hooked = True

        kind = "fwd" if self.phase == "fwd" else "recompute"
        self._push_start(layer_idx, kind)

    def post_hook(self, layer_idx: int, module, inputs, output):
        kind = "fwd" if self.phase == "fwd" else "recompute"
        self._pop_pair(layer_idx, kind)

        # only attach grad hooks during the real forward pass (avoid extra overhead during recompute)
        if self.enabled and self.phase == "fwd":
            hs = None
            if torch.is_tensor(output):
                hs = output
            elif isinstance(output, (tuple, list)) and len(output) > 0 and torch.is_tensor(output[0]):
                hs = output[0]
            if hs is not None and hs.requires_grad:
                def _grad_hook(_grad, idx=layer_idx):
                    ev = torch.cuda.Event(enable_timing=True)
                    ev.record()
                    self.step_buf.grad_ev[idx] = ev
                    return _grad
                hs.register_hook(_grad_hook)

    def finalize_layer_ms(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        returns fwd_ms[L], recomp_fwd_ms[L], bwd_ms[L]
        bwd_ms estimated by grad-ready deltas:
          bwd[i] = time(grad of output[i-1]) - time(grad of output[i])
        where output[-1] is input to layer0
        """
        L = self.n
        fwd_ms = np.full((L,), np.nan, dtype=np.float64)
        rcp_ms = np.full((L,), 0.0, dtype=np.float64)
        bwd_ms = np.full((L,), np.nan, dtype=np.float64)

        if not self.enabled:
            return fwd_ms, rcp_ms, bwd_ms

        # make sure all queued events are done
        torch.cuda.synchronize()

        for i in range(L):
            pairs = self.step_buf.fwd_pairs.get(i, [])
            if pairs:
                fwd_ms[i] = float(sum(st.elapsed_time(ed) for st, ed in pairs))
            else:
                fwd_ms[i] = float("nan")
            rpairs = self.step_buf.recomp_pairs.get(i, [])
            if rpairs:
                rcp_ms[i] = float(sum(st.elapsed_time(ed) for st, ed in rpairs))
            else:
                rcp_ms[i] = 0.0

        # grad-ready events
        for i in range(L):
            ev_i = self.step_buf.grad_ev.get(i, None)
            ev_prev = self.step_buf.grad_ev.get(i - 1, None)  # i-1, where -1 is layer0 input
            if (ev_i is not None) and (ev_prev is not None):
                bwd_ms[i] = float(ev_i.elapsed_time(ev_prev))
            else:
                bwd_ms[i] = float("nan")

        return fwd_ms, rcp_ms, bwd_ms


def apply_selective_checkpointing(layers: List[torch.nn.Module], ckpt_set: Set[int], use_reentrant: bool):
    """
    Patch layer.forward so that selected layers are checkpointed.
    Works across models with different forward signatures by:
      1) try checkpoint(orig, *args, **kwargs)
      2) fallback to closure checkpoint(fn, *args) capturing kwargs
    """
    for idx, layer in enumerate(layers):
        if hasattr(layer, "_t1m2_orig_forward"):
            continue
        layer._t1m2_orig_forward = layer.forward

        orig_fwd = layer._t1m2_orig_forward

        def make_wrapper(i, orig):
            def wrapped(*args, **kwargs):
                if (i in ckpt_set) and layer.training:
                    try:
                        return checkpoint(orig, *args, use_reentrant=use_reentrant, **kwargs)
                    except TypeError:
                        # older checkpoint API path (no kwargs)
                        def fn(*aa):
                            return orig(*aa, **kwargs)
                        return checkpoint(fn, *args, use_reentrant=use_reentrant)
                return orig(*args, **kwargs)
            return wrapped

        layer.forward = make_wrapper(idx, orig_fwd)


def hf_load(model_dir: str, dtype: torch.dtype, device: torch.device):
    # avoid transformers torch_dtype deprecation if possible
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token if tok.eos_token is not None else tok.unk_token

    kw = dict(local_files_only=True, trust_remote_code=True, low_cpu_mem_usage=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype, **kw)

    model.to(device)
    return tok, model


def make_optimizer(name: str, params, lr: float):
    name = name.lower().strip()
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.0)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
    raise ValueError(f"unsupported optimizer: {name}")


def build_run_dir(out_root: Path, recompute: int, seq: int, batch: int, epochs: int, spe: int,
                  optim: str, dtype: str, ckpt_layers: str) -> Path:
    tag = f"{now_ts()}_T1M2_recomp{recompute}_seq{seq}_bs{batch}_ep{epochs}_spe{spe}_{optim}_{dtype}"
    if ckpt_layers:
        tag += f"_ckpt{ckpt_layers.replace(',', '_').replace('-', 'to')}"
    return out_root / tag


def main_one(args, recompute_flag: int) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = parse_dtype(args.dtype)

    run_dir = build_run_dir(
        out_root=Path(args.out),
        recompute=recompute_flag,
        seq=args.seq,
        batch=args.batch,
        epochs=args.epochs,
        spe=args.steps_per_epoch,
        optim=args.optim,
        dtype=args.dtype,
        ckpt_layers=args.ckpt_layers if recompute_flag else ""
    )
    ensure_dir(run_dir)
    ensure_dir(run_dir / "prof")

    env = {
        "python": f"{os.sys.version}",
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "args": {**vars(args), "recompute": recompute_flag},
    }
    (run_dir / "env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    (run_dir / "args.json").write_text(json.dumps({**vars(args), "recompute": recompute_flag}, indent=2), encoding="utf-8")

    print("[RUN_DIR]", str(run_dir))
    print("[LOAD] tokenizer/model ...")
    tok, model = hf_load(args.model, dtype=dtype, device=device)

    model.train()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()

    layers = get_decoder_layers(model)
    n_layers = len(layers)
    print(f"[INFO] detected decoder layers: {n_layers}")
    print("[CFG] recompute=" + ("ON" if recompute_flag else "OFF"))

    # select which layers to checkpoint (recompute) if enabled
    ckpt_set: Set[int] = set()
    if recompute_flag:
        if args.ckpt_k > 0:
            ckpt_set = set(range(min(args.ckpt_k, n_layers)))
        else:
            ckpt_set = parse_layer_spec(args.ckpt_layers, n_layers)
            if not ckpt_set:
                ckpt_set = set(range(n_layers))  # default: all layers
        apply_selective_checkpointing(layers, ckpt_set, use_reentrant=bool(args.use_reentrant))
        print(f"[CFG] checkpoint layers count={len(ckpt_set)} example={sorted(list(ckpt_set))[:8]} ...")

    # attach layer probe hooks
    probe = LayerProbe(layers)
    handles = []
    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(lambda m, inp, idx=i: probe.pre_hook(idx, m, inp)))
        handles.append(layer.register_forward_hook(lambda m, inp, out, idx=i: probe.post_hook(idx, m, inp, out)))

    # fixed synthetic batch (stable benchmarking)
    vocab = int(getattr(model.config, "vocab_size", 32000))
    g = torch.Generator(device=device)
    g.manual_seed(args.seed + 999)

    input_ids = torch.randint(low=0, high=vocab, size=(args.batch, args.seq), generator=g, device=device, dtype=torch.long)
    attn_mask = torch.ones_like(input_ids, device=device, dtype=torch.long)
    labels = input_ids.clone()

    optim = make_optimizer(args.optim, model.parameters(), lr=args.lr)

    # optional torch.profiler
    prof = None
    if args.profiler and torch.cuda.is_available():
        from torch.profiler import profile, ProfilerActivity, schedule

        def trace_handler(p):
            # unique file per callback to avoid "Trace is already saved."
            fn = run_dir / "prof" / f"trace_step{p.step_num:04d}.json"
            try:
                p.export_chrome_trace(str(fn))
            except Exception as e:
                print("[WARN] export chrome trace failed:", repr(e))

        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=args.prof_wait, warmup=args.prof_warmup, active=args.prof_active, repeat=args.prof_repeat),
            on_trace_ready=trace_handler,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )
        prof.__enter__()
        print("[PROF] torch.profiler enabled")

    step_rows = []
    layer_rows = []
    epoch_rows = []

    global_step = 0
    run_t0 = time.perf_counter()
    cum_wall_ms = 0.0

    # autocast (only for cuda)
    from contextlib import nullcontext
    amp_ctx = nullcontext()
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        amp_ctx = torch.autocast(device_type="cuda", dtype=dtype)

    for ep in range(args.epochs):
        ep_t0 = time.perf_counter()
        ep_wall_ms_sum = 0.0
        ep_fwd_ms_sum = 0.0
        ep_bwd_ms_sum = 0.0
        ep_opt_ms_sum = 0.0

        print(f"\n========== [EPOCH {ep}] ==========")
        for si in range(args.steps_per_epoch):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            if args.sync and torch.cuda.is_available():
                torch.cuda.synchronize()

            probe.begin_step()

            # step wall clock
            wall_t0 = time.perf_counter()

            # forward time (cuda event)
            fwd_ms = float("nan")
            bwd_ms = float("nan")
            opt_ms = float("nan")

            if torch.cuda.is_available():
                ev_f0 = torch.cuda.Event(enable_timing=True)
                ev_f1 = torch.cuda.Event(enable_timing=True)
                ev_b0 = torch.cuda.Event(enable_timing=True)
                ev_b1 = torch.cuda.Event(enable_timing=True)
                ev_o0 = torch.cuda.Event(enable_timing=True)
                ev_o1 = torch.cuda.Event(enable_timing=True)
                ev_f0.record()

            with amp_ctx:
                out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                loss = out.loss

            if torch.cuda.is_available():
                ev_f1.record()

            # backward
            probe.set_bwd_phase()
            optim.zero_grad(set_to_none=True)

            if torch.cuda.is_available():
                ev_b0.record()
            loss.backward()
            if torch.cuda.is_available():
                ev_b1.record()

            # optimizer step
            if torch.cuda.is_available():
                ev_o0.record()
            optim.step()
            if torch.cuda.is_available():
                ev_o1.record()

            if args.sync and torch.cuda.is_available():
                torch.cuda.synchronize()

            wall_t1 = time.perf_counter()
            wall_ms = (wall_t1 - wall_t0) * 1000.0

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                fwd_ms = float(ev_f0.elapsed_time(ev_f1))
                bwd_ms = float(ev_b0.elapsed_time(ev_b1))
                opt_ms = float(ev_o0.elapsed_time(ev_o1))
            else:
                # CPU-only fallback (less accurate split)
                fwd_ms = float("nan")
                bwd_ms = float("nan")
                opt_ms = float("nan")

            # per-layer
            lfwd, lrecomp, lbwd = probe.finalize_layer_ms()

            # memory & rss
            rss = cpu_rss_gib()
            alloc, resv, peak_alloc, peak_resv = cuda_mem_gib()

            # tokens/s (rough)
            tokens = int(args.batch * args.seq)
            tps = tokens / max((wall_ms / 1000.0), 1e-6)

            cum_wall_ms = (time.perf_counter() - run_t0) * 1000.0

            step_rows.append({
                "global_step": global_step,
                "epoch": ep,
                "step_in_epoch": si,
                "wall_ms": wall_ms,
                "fwd_ms": fwd_ms,
                "bwd_ms": bwd_ms,
                "opt_ms": opt_ms,
                "loss": float(loss.detach().cpu().item()),
                "tokens": tokens,
                "tps": tps,
                "lr": args.lr,
                "cpu_rss_gib": rss,
                "gpu_alloc_gib": alloc,
                "gpu_resv_gib": resv,
                "peak_alloc_gib": peak_alloc,
                "peak_resv_gib": peak_resv,
                "cum_wall_ms": cum_wall_ms,
            })

            for li in range(n_layers):
                layer_rows.append({
                    "global_step": global_step,
                    "epoch": ep,
                    "layer": li,
                    "fwd_ms": float(lfwd[li]) if np.isfinite(lfwd[li]) else np.nan,
                    "recompute_fwd_ms": float(lrecomp[li]) if np.isfinite(lrecomp[li]) else 0.0,
                    "bwd_ms": float(lbwd[li]) if np.isfinite(lbwd[li]) else np.nan,
                })

            print(
                f"[STEP {global_step:04d}] wall={wall_ms:.2f}ms fwd={fwd_ms:.2f}ms bwd={bwd_ms:.2f}ms "
                f"opt={opt_ms:.2f}ms loss={loss.item():.4f} tps={tps:.1f} "
                f"CPU_RSS={rss:.2f}GiB GPU_alloc={alloc:.2f}GiB GPU_resv={resv:.2f}GiB"
            )

            # profiler step
            if prof is not None:
                try:
                    prof.step()
                except Exception as e:
                    print("[WARN] profiler step failed:", repr(e))

            ep_wall_ms_sum += wall_ms
            ep_fwd_ms_sum += (0.0 if np.isnan(fwd_ms) else fwd_ms)
            ep_bwd_ms_sum += (0.0 if np.isnan(bwd_ms) else bwd_ms)
            ep_opt_ms_sum += (0.0 if np.isnan(opt_ms) else opt_ms)

            global_step += 1

        ep_t1 = time.perf_counter()
        ep_wall_s = float(ep_t1 - ep_t0)
        epoch_rows.append({
            "epoch": ep,
            "epoch_wall_s": ep_wall_s,
            "sum_step_wall_ms": ep_wall_ms_sum,
            "sum_fwd_ms": ep_fwd_ms_sum,
            "sum_bwd_ms": ep_bwd_ms_sum,
            "sum_opt_ms": ep_opt_ms_sum,
        })
        print(f"[EPOCH {ep}] wall_s={ep_wall_s:.3f} sum_wall_ms={ep_wall_ms_sum:.1f} "
              f"sum_fwd_ms={ep_fwd_ms_sum:.1f} sum_bwd_ms={ep_bwd_ms_sum:.1f} sum_opt_ms={ep_opt_ms_sum:.1f}")

    # cleanup
    for h in handles:
        h.remove()

    if prof is not None:
        try:
            prof.__exit__(None, None, None)
        except Exception as e:
            print("[WARN] profiler exit failed:", repr(e))

    # save csv
    df_step = pd.DataFrame(step_rows)
    df_layer = pd.DataFrame(layer_rows)
    df_epoch = pd.DataFrame(epoch_rows)

    df_step.to_csv(run_dir / "step_metrics.csv", index=False)
    df_layer.to_csv(run_dir / "layer_metrics.csv", index=False)
    df_epoch.to_csv(run_dir / "epoch_metrics.csv", index=False)

    print("[SAVED]", str(run_dir / "step_metrics.csv"))
    print("[SAVED]", str(run_dir / "layer_metrics.csv"))
    print("[SAVED]", str(run_dir / "epoch_metrics.csv"))
    return run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="local model dir (HF format)")
    ap.add_argument("--out", type=str, default="runs_t1m2", help="output root dir")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--steps-per-epoch", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--optim", type=str, default="adamw", choices=["sgd", "adam", "adamw"])
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--sync", type=int, default=1, help="cuda synchronize for stable timing")
    ap.add_argument("--recompute", type=int, default=0, choices=[0, 1])
    ap.add_argument("--ckpt-layers", type=str, default="", help='e.g. "0-13,20-27" or "all"')
    ap.add_argument("--ckpt-k", type=int, default=0, help="checkpoint first K layers (overrides ckpt-layers)")
    ap.add_argument("--use-reentrant", type=int, default=0, choices=[0, 1], help="torch checkpoint reentrant mode")
    ap.add_argument("--profiler", type=int, default=0, choices=[0, 1])
    ap.add_argument("--prof-wait", type=int, default=0)
    ap.add_argument("--prof-warmup", type=int, default=0)
    ap.add_argument("--prof-active", type=int, default=10)
    ap.add_argument("--prof-repeat", type=int, default=1)
    ap.add_argument("--sweep", type=int, default=0, choices=[0, 1], help="run OFF then ON recompute")
    args = ap.parse_args()

    out_root = ensure_dir(Path(args.out))
    print("[BOOT] t1m2_train_profiler")
    if args.sweep:
        print("[BOOT] sweep mode: recompute OFF -> ON")
        r0 = main_one(args, 0)
        r1 = main_one(args, 1)
        print("[OK] sweep done:")
        print("  run0 =", str(r0))
        print("  run1 =", str(r1))
    else:
        r = main_one(args, int(args.recompute))
        print("[OK] run done:", str(r))


if __name__ == "__main__":
    main()
