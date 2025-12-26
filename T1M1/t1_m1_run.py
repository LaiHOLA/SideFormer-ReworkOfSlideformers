
# t1_m1_run.py
# -*- coding: utf-8 -*-
"""
T1-M1 runner with 3 modes:

1) --mode gpu   : pure GPU training (no checkpoint)
2) --mode ckpt  : PyTorch gradient checkpointing (layer-level) + GPU optimizer
3) --mode slide : SlideFormer-style layer streaming (CPU weights + 2-stage GPU) + CPU optimizer

All modes export:
- env.json
- log.txt
- metrics_iter.csv
- metrics_layer.csv
- mem_trace.csv

Designed for Qwen-family blocks called directly (layer granularity), so we explicitly construct
attention_mask / position_embeddings and pass into each layer call.

Example:
python t1_m1_run.py --mode gpu   --model ./qwen --seq 256 --batch 3 --iters 5 --layers 28 --train_layers 8 --optim adamw --adam_state_dtype bf16 --lr 1e-4 --logdir runs_m1
python t1_m1_run.py --mode ckpt  --model ./qwen --seq 256 --batch 3 --iters 5 --layers 28 --train_layers 8 --ckpt_layers 8 --optim adamw --adam_state_dtype bf16 --lr 1e-4 --logdir runs_m1
python t1_m1_run.py --mode slide --model ./qwen --seq 256 --batch 3 --iters 5 --layers 28 --train_layers 8 --prefetch 1 --optim adamw --adam_state_dtype bf16 --lr 1e-4 --logdir runs_m1
"""
from __future__ import annotations

import os, sys, time, json, copy, argparse, traceback, inspect
from datetime import datetime
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread, Event as ThreadEvent

import psutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------- utils ----------------
def gib(x: float) -> float:
    return float(x) / (1024 ** 3)

def cpu_rss_gib() -> float:
    p = psutil.Process(os.getpid())
    return gib(p.memory_info().rss)

def now_ms() -> float:
    return time.perf_counter() * 1000.0

class Logger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "w", encoding="utf-8")
    def log(self, msg: str):
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()
    def close(self):
        try: self.f.close()
        except Exception: pass

class MemoryMonitor:
    """
    Periodically sample CPU RSS and CUDA memory. Cheap enough for ~5-20ms interval.
    Writes samples to an internal list; call dump_csv(...) after stop().
    """
    def __init__(self, interval_ms: int = 10):
        self.interval_ms = max(1, int(interval_ms))
        self._stop = ThreadEvent()
        self._t: Thread | None = None
        self.samples: list[dict] = []
        self._t0 = 0.0
        self.iter_idx = -1
        self.mode = ""
    def start(self, iter_idx: int, mode: str):
        self.iter_idx = int(iter_idx)
        self.mode = str(mode)
        self.samples = []
        self._stop.clear()
        self._t0 = time.perf_counter()
        def loop():
            while not self._stop.is_set():
                t_ms = (time.perf_counter() - self._t0) * 1000.0
                cpu = cpu_rss_gib()
                if torch.cuda.is_available():
                    ga = gib(torch.cuda.memory_allocated())
                    gr = gib(torch.cuda.memory_reserved())
                else:
                    ga = 0.0
                    gr = 0.0
                self.samples.append({
                    "iter": self.iter_idx,
                    "mode": self.mode,
                    "t_ms": t_ms,
                    "cpu_rss_gib": cpu,
                    "gpu_alloc_gib": ga,
                    "gpu_reserved_gib": gr,
                })
                time.sleep(self.interval_ms / 1000.0)
        self._t = Thread(target=loop, daemon=True)
        self._t.start()
    def stop(self):
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=2.0)
    def dump_csv(self, path: str, append: bool = True):
        header = "iter,mode,t_ms,cpu_rss_gib,gpu_alloc_gib,gpu_reserved_gib\n"
        exists = os.path.exists(path)
        mode = "a" if (append and exists) else "w"
        with open(path, mode, encoding="utf-8") as f:
            if (mode == "w") or (not exists):
                f.write(header)
            for s in self.samples:
                f.write(f"{s['iter']},{s['mode']},{s['t_ms']:.3f},{s['cpu_rss_gib']:.6f},{s['gpu_alloc_gib']:.6f},{s['gpu_reserved_gib']:.6f}\n")


def pick_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers, model.model
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h, model.transformer
    raise RuntimeError("Cannot locate layers. Expected `model.model.layers` or `model.transformer.h`.")

def pick_embeddings_and_norm(core):
    embed = None
    norm = None
    for name in ["embed_tokens", "wte", "tok_embeddings", "embeddings"]:
        if hasattr(core, name):
            embed = getattr(core, name)
            break
    for name in ["norm", "final_layernorm", "ln_f"]:
        if hasattr(core, name):
            norm = getattr(core, name)
            break
    return embed, norm

def pick_rotary_emb(model, core):
    if hasattr(core, "rotary_emb"):
        return getattr(core, "rotary_emb")
    if hasattr(model, "rotary_emb"):
        return getattr(model, "rotary_emb")
    return None

def try_build_4d_causal_mask(attn_2d, hidden_states, logger: Logger):
    try:
        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
        bs, seqlen = attn_2d.shape
        input_shape = (bs, seqlen)
        try:
            m = _prepare_4d_causal_attention_mask(attn_2d, input_shape, hidden_states, past_key_values_length=0)
        except TypeError:
            m = _prepare_4d_causal_attention_mask(attn_2d, input_shape, hidden_states, 0)
        return m
    except Exception as e:
        logger.log(f"[WARN] cannot build 4D causal mask -> use None. err={repr(e)}")
        return None

def _sig_of(layer):
    try:
        return inspect.signature(layer.forward)
    except Exception:
        return None

def call_layer(layer, hidden_states, attention_mask, position_ids, position_embeddings):
    sig = _sig_of(layer)
    kwargs = {}
    if sig is not None:
        params = sig.parameters
        if "attention_mask" in params and attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if "position_ids" in params and position_ids is not None:
            kwargs["position_ids"] = position_ids
        if "position_embeddings" in params and position_embeddings is not None:
            kwargs["position_embeddings"] = position_embeddings
        if "use_cache" in params:
            kwargs["use_cache"] = False
        if "output_attentions" in params:
            kwargs["output_attentions"] = False
        if "output_hidden_states" in params:
            kwargs["output_hidden_states"] = False
    # compat fallbacks
    try:
        out = layer(hidden_states, **kwargs)
    except TypeError:
        kwargs.pop("position_embeddings", None)
        try:
            out = layer(hidden_states, **kwargs)
        except TypeError:
            kwargs.pop("attention_mask", None)
            kwargs.pop("position_ids", None)
            out = layer(hidden_states)
    if isinstance(out, (tuple, list)):
        return out[0]
    return out

def measure_cuda_ms(fn):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    out = fn()
    e.record()
    torch.cuda.synchronize()
    return float(s.elapsed_time(e)), out

def param_bytes(layer) -> int:
    b = 0
    for _, p in layer.named_parameters(recurse=True):
        if p is None: 
            continue
        b += p.numel() * p.element_size()
    return int(b)

def ensure_state_tensor(shape, dtype, device, pinned=False):
    t = torch.zeros(shape, dtype=dtype, device=device)
    if pinned and device == "cpu":
        t = t.pin_memory()
    return t

# ---------------- optim (custom) ----------------
@dataclass
class AdamWConfig:
    lr: float
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    state_dtype: str = "bf16"  # "bf16" or "fp32"

def _adam_state_dtype(param_dtype, state_dtype: str):
    if state_dtype == "fp32":
        return torch.float32
    # bf16 or fp16 states
    if param_dtype == torch.bfloat16:
        return torch.bfloat16
    if param_dtype == torch.float16:
        return torch.float16
    return torch.float32

def adamw_update_layer(layer, grads: dict, state: dict, cfg: AdamWConfig):
    """
    In-place AdamW update for a single layer. Works on CPU or CUDA depending on param device.
    Returns (ms, bytes_state_created).
    """
    bytes_new = 0
    is_cuda = next(layer.parameters()).is_cuda if any(True for _ in layer.parameters()) else False
    if is_cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    name2p = dict(layer.named_parameters(recurse=True))
    for name, g in grads.items():
        p = name2p.get(name, None)
        if p is None or g is None:
            continue
        st = state.get(name)
        if st is None:
            dt = _adam_state_dtype(p.data.dtype, cfg.state_dtype)
            st = {
                "m": ensure_state_tensor(p.data.shape, dt, device=p.device),
                "v": ensure_state_tensor(p.data.shape, dt, device=p.device),
                "step": 0
            }
            state[name] = st
            bytes_new += st["m"].numel() * st["m"].element_size() + st["v"].numel() * st["v"].element_size()

        st["step"] += 1
        step = st["step"]

        # compute in fp32 (even if state is bf16)
        g32 = g.float()
        p32 = p.data.float()
        m32 = st["m"].float()
        v32 = st["v"].float()

        if cfg.weight_decay != 0.0:
            g32 = g32.add(p32, alpha=cfg.weight_decay)

        m32.mul_(cfg.beta1).add_(g32, alpha=1.0 - cfg.beta1)
        v32.mul_(cfg.beta2).addcmul_(g32, g32, value=1.0 - cfg.beta2)

        bc1 = 1.0 - (cfg.beta1 ** step)
        bc2 = 1.0 - (cfg.beta2 ** step)
        mhat = m32 / bc1
        vhat = v32 / bc2
        upd = mhat / (vhat.sqrt().add_(cfg.eps))

        p32.add_(upd, alpha=-cfg.lr)
        p.data.copy_(p32.to(dtype=p.data.dtype))

        # write back
        if st["m"].dtype == torch.float32:
            st["m"].copy_(m32)
            st["v"].copy_(v32)
        else:
            st["m"].copy_(m32.to(st["m"].dtype))
            st["v"].copy_(v32.to(st["v"].dtype))

    if is_cuda:
        end.record()
        torch.cuda.synchronize()
        ms = float(start.elapsed_time(end))
        return ms, bytes_new
    else:
        # CPU timing (rough)
        return 0.0, bytes_new

def sgd_update_layer(layer, grads: dict, lr: float):
    is_cuda = next(layer.parameters()).is_cuda if any(True for _ in layer.parameters()) else False
    if is_cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    name2p = dict(layer.named_parameters(recurse=True))
    for name, g in grads.items():
        p = name2p.get(name, None)
        if p is None or g is None:
            continue
        if g.dtype != p.data.dtype:
            g = g.to(dtype=p.data.dtype)
        p.data.add_(g, alpha=-lr)
    if is_cuda:
        end.record()
        torch.cuda.synchronize()
        ms = float(start.elapsed_time(end))
        return ms
    return 0.0

# ---------------- slide helpers ----------------
def pin_layer_params_inplace(layer) -> int:
    pinned_bytes = 0
    for _, p in layer.named_parameters(recurse=True):
        if p is None:
            continue
        if p.device.type != "cpu":
            continue
        if not p.data.is_pinned():
            t = p.data.contiguous().pin_memory()
            pinned_bytes += t.numel() * t.element_size()
            p.data = t
    return pinned_bytes

def build_param_maps(stage_layer):
    return {n: p for n, p in stage_layer.named_parameters(recurse=True)}

def prefetch_layer_to_stage(cpu_layer, stage_param_map, stream, done_event, logger: Logger, tag=""):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record(stream)
        copied = 0
        bytes_copied = 0
        for name, p_cpu in cpu_layer.named_parameters(recurse=True):
            p_gpu = stage_param_map.get(name, None)
            if p_gpu is None:
                continue
            p_gpu.data.copy_(p_cpu.data, non_blocking=True)
            copied += 1
            bytes_copied += p_cpu.numel() * p_cpu.element_size()
        end.record(stream)
        done_event.record(stream)
    logger.log(f"[PREFETCH]{tag} queued params={copied}, bytes~{gib(bytes_copied):.3f}GiB")
    return start, end, bytes_copied, copied

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, required=True, choices=["gpu", "ckpt", "slide"],
                    help="gpu=纯GPU; ckpt=PyTorch checkpoint重计算; slide=SlideFormer式CPU权重+流水+重计算")
    ap.add_argument("--model", type=str, default="./qwen")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--layers", type=int, default=0, help="0=all; else first N layers (forward depth)")
    ap.add_argument("--train_layers", type=int, default=0, help="0=all used layers; else train only last N layers")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--logdir", type=str, default="runs_m1")
    ap.add_argument("--group", type=str, default="", help="optional run group tag to make multiple modes share a common prefix")
    ap.add_argument("--seed", type=int, default=0)

    # compat: triple may pass --plot 0; run itself doesn't plot (plots are in analyzer)
    ap.add_argument("--plot", type=int, default=0, help="compat only; ignored (use analyzer for figures)")

    # ckpt controls
    ap.add_argument("--ckpt_layers", type=int, default=0,
                    help="ckpt模式下：对 trainable 范围内最后 N 层做 checkpoint(0=全部trainable层)")
    ap.add_argument("--ckpt_use_reentrant", type=int, default=0,
                    help="checkpoint use_reentrant(0推荐, 兼容2.x+)")
    # slide controls
    ap.add_argument("--prefetch", type=int, default=1)
    ap.add_argument("--pin_mode", type=str, default="lazy", choices=["off", "lazy"])
    ap.add_argument("--cpu_workers", type=int, default=1, help="slide模式CPU更新线程数(1=最稳)")
    # optimizer
    ap.add_argument("--optim", type=str, default="adamw", choices=["sgd", "adamw"])
    ap.add_argument("--betas", type=str, default="0.9,0.999")
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--adam_state_dtype", type=str, default="bf16", choices=["bf16", "fp32"])

    # monitoring
    ap.add_argument("--monitor_ms", type=int, default=10, help="memory sampling interval (ms)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for T1-M1.")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    group = (args.group.strip() + "_") if args.group and args.group.strip() else ""
    run_dir = os.path.join(args.logdir, f"{stamp}_T1M1_{group}{args.mode}_seq{args.seq}_bs{args.batch}_iters{args.iters}_layers{args.layers or 'all'}_train{args.train_layers or 'all'}_{args.optim}")
    os.makedirs(run_dir, exist_ok=True)
    logger = Logger(os.path.join(run_dir, "log.txt"))
    logger.log(f"[RUN_DIR] {run_dir}")

    env = {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "dtype": str(dtype),
        "args": vars(args),
        "group": args.group,
    }
    with open(os.path.join(run_dir, "env.json"), "w", encoding="utf-8") as f:
        json.dump(env, f, ensure_ascii=False, indent=2)
    logger.log("[ENV] " + json.dumps(env, ensure_ascii=False))

    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    logger.log("[LOAD] tokenizer/model on CPU ...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map={"": "cpu"},
        torch_dtype=dtype,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    layers, core = pick_layers(model)
    embed, norm = pick_embeddings_and_norm(core)
    if embed is None:
        raise RuntimeError("Cannot locate embedding module.")

    rotary = pick_rotary_emb(model, core)
    if rotary is not None:
        rotary = rotary.to(device=device).eval()
        logger.log(f"[ROTARY] found: {type(rotary).__name__}, moved to cuda")
    else:
        logger.log("[WARN] rotary_emb not found. Some models may break without position_embeddings.")

    total_layers = len(layers)
    use_layers = total_layers if args.layers <= 0 else min(args.layers, total_layers)
    train_layers = use_layers if args.train_layers <= 0 else min(args.train_layers, use_layers)
    train_start = use_layers - train_layers
    logger.log(f"[CFG] total_layers={total_layers}, use_layers={use_layers}, train_layers={train_layers} (range {train_start}..{use_layers-1}) mode={args.mode}")

    # IO files
    metrics_iter = os.path.join(run_dir, "metrics_iter.csv")
    metrics_layer = os.path.join(run_dir, "metrics_layer.csv")
    mem_trace = os.path.join(run_dir, "mem_trace.csv")

    with open(metrics_iter, "w", encoding="utf-8") as f:
        f.write("iter,mode,wall_s,forward_s,backward_s,optim_s,cpu_update_total_s,peak_alloc_gib,peak_reserved_gib,cpu_rss_gib,total_prefetch_ms,total_fwd_ms,total_recompute_ms,total_bwd_ms\n")
    with open(metrics_layer, "w", encoding="utf-8") as f:
        f.write("iter,mode,layer,trainable,prefetch_ms,fwd_ms,recompute_ms,bwd_ms,grad_offload_gib,optim_ms\n")

    monitor = MemoryMonitor(interval_ms=args.monitor_ms)

    # prepare common inputs
    def build_ids():
        text = ("SlideFormer layer-streaming proof. " * 200).strip()
        ids = tok(text, return_tensors="pt", truncation=True, max_length=args.seq).input_ids
        if ids.size(1) < args.seq:
            pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0
            pad = torch.full((1, args.seq - ids.size(1)), pad_id, dtype=ids.dtype)
            ids = torch.cat([ids, pad], dim=1)
        ids = ids[:, :args.seq].repeat(args.batch, 1)
        return ids

    # optimizer configs
    beta1, beta2 = [float(x.strip()) for x in args.betas.split(",")]
    adam_cfg = AdamWConfig(
        lr=args.lr, beta1=beta1, beta2=beta2, eps=args.eps,
        weight_decay=args.weight_decay, state_dtype=("fp32" if args.adam_state_dtype == "fp32" else "bf16")
    )

    # per-layer adam state (dict per layer)
    # - gpu/ckpt: states on CUDA
    # - slide: states on CPU
    adam_state = [dict() for _ in range(use_layers)]

    # ---------------- mode: gpu / ckpt ----------------
    def run_gpu_or_ckpt():
        # move used modules to GPU
        logger.log("[MOVE] moving embedding/layers/norm to cuda ...")
        embed_cuda = embed.to(device=device, dtype=dtype)
        norm_cuda = norm.to(device=device, dtype=dtype) if norm is not None else None

        # move layers 0..use_layers-1 to GPU; keep others on CPU (unused)
        for i in range(use_layers):
            layers[i].to(device=device, dtype=dtype)
            layers[i].train(True if i >= train_start else False)
            # only train last train_layers
            for p in layers[i].parameters():
                p.requires_grad_(i >= train_start)

        # checkpoint selection within trainable layers
        ckpt_last_n = train_layers if args.ckpt_layers <= 0 else min(args.ckpt_layers, train_layers)
        ckpt_from = use_layers - ckpt_last_n  # checkpoint for [ckpt_from..use_layers)
        use_ckpt = (args.mode == "ckpt")

        # for backward timing hooks
        bwd_start_ev = {}
        bwd_end_ev = {}
        layer_id_to_idx = {id(layers[i]): i for i in range(train_start, use_layers)}

        def bwd_pre_hook(mod, grad_output):
            i = layer_id_to_idx.get(id(mod), None)
            if i is None: 
                return
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            bwd_start_ev[i] = ev

        def bwd_post_hook(mod, grad_input, grad_output):
            i = layer_id_to_idx.get(id(mod), None)
            if i is None:
                return
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            bwd_end_ev[i] = ev

        hooks = []
        for i in range(train_start, use_layers):
            hooks.append(layers[i].register_full_backward_pre_hook(bwd_pre_hook))
            hooks.append(layers[i].register_full_backward_hook(bwd_post_hook))

        # checkpoint recompute timing (only for ckpt)
        # store event pairs per layer for (fwd_no_grad) and (recompute_grad)
        ckpt_ev_fwd = {i: [] for i in range(train_start, use_layers)}
        ckpt_ev_re = {i: [] for i in range(train_start, use_layers)}

        from torch.utils.checkpoint import checkpoint

        for it in range(args.iters):
            logger.log(f"\n========== [ITER {it}] ==========")
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            monitor.start(it, args.mode)

            t0_wall = time.perf_counter()

            ids = build_ids().to(device, non_blocking=True)
            bs, seqlen = ids.shape
            position_ids = torch.arange(seqlen, device=device, dtype=torch.long).unsqueeze(0).expand(bs, -1)
            attn_2d = torch.ones((bs, seqlen), device=device, dtype=torch.long)

            with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                hidden = embed_cuda(ids)
            h_shape = hidden.shape

            attention_mask = try_build_4d_causal_mask(attn_2d, hidden, logger)

            if rotary is not None:
                with torch.no_grad():
                    try:
                        position_embeddings = rotary(hidden, position_ids)
                    except TypeError:
                        position_embeddings = rotary(position_ids)
            else:
                position_embeddings = None

            # prefix (frozen) forward, no grad
            t_fwd0 = time.perf_counter()
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                for i in range(train_start):
                    ms, hidden = measure_cuda_ms(lambda i=i, h=hidden: call_layer(layers[i], h, attention_mask, position_ids, position_embeddings))
                    # prefix layer forward times still useful for completeness, but mark trainable=0
                    with open(metrics_layer, "a", encoding="utf-8") as f:
                        f.write(f"{it},{args.mode},{i},0,0.0,{ms:.6f},0.0,0.0,0.0,0.0\n")

            # trainable forward (grad enabled for gpu; checkpoint for ckpt)
            fwd_ms_by_layer = {}
            for i in range(train_start, use_layers):
                if use_ckpt and i >= ckpt_from:
                    layer = layers[i]
                    def layer_fn(h, layer=layer, i=i):
                        # timed inside the function; differentiate fwd vs recompute by grad_enabled
                        s = torch.cuda.Event(enable_timing=True)
                        e = torch.cuda.Event(enable_timing=True)
                        s.record()
                        out = call_layer(layer, h, attention_mask, position_ids, position_embeddings)
                        e.record()
                        if torch.is_grad_enabled():
                            ckpt_ev_re[i].append((s, e))
                        else:
                            ckpt_ev_fwd[i].append((s, e))
                        return out
                    # measure wrapper time (forward pass call)
                    ms, hidden = measure_cuda_ms(lambda fn=layer_fn, h=hidden: checkpoint(fn, h, use_reentrant=bool(args.ckpt_use_reentrant)))
                    fwd_ms_by_layer[i] = ms
                else:
                    ms, hidden = measure_cuda_ms(lambda i=i, h=hidden: call_layer(layers[i], h, attention_mask, position_ids, position_embeddings))
                    fwd_ms_by_layer[i] = ms

            if norm_cuda is not None:
                with torch.autocast("cuda", dtype=dtype):
                    hidden = norm_cuda(hidden)

            t_fwd1 = time.perf_counter()
            forward_s = t_fwd1 - t_fwd0

            # simple loss
            with torch.autocast("cuda", dtype=dtype):
                loss = hidden.float().pow(2).mean()

            # backward
            t_bwd0 = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize()
            t_bwd1 = time.perf_counter()
            backward_s = t_bwd1 - t_bwd0

            # per-layer backward ms from hooks
            bwd_ms_by_layer = {i: 0.0 for i in range(train_start, use_layers)}
            for i in range(train_start, use_layers):
                s = bwd_start_ev.get(i, None)
                e = bwd_end_ev.get(i, None)
                if s is not None and e is not None:
                    bwd_ms_by_layer[i] = float(s.elapsed_time(e))

            # checkpoint recompute ms
            recompute_ms_by_layer = {i: 0.0 for i in range(train_start, use_layers)}
            if use_ckpt:
                torch.cuda.synchronize()
                for i in range(train_start, use_layers):
                    # sum recompute forwards (grad_enabled path)
                    total = 0.0
                    for s, e in ckpt_ev_re[i]:
                        total += float(s.elapsed_time(e))
                    recompute_ms_by_layer[i] = total

            # optimizer update
            t_opt0 = time.perf_counter()
            optim_ms_by_layer = {i: 0.0 for i in range(train_start, use_layers)}
            if args.optim == "sgd":
                for i in range(train_start, use_layers):
                    grads = {n: p.grad for n, p in layers[i].named_parameters(recurse=True) if p.grad is not None}
                    optim_ms_by_layer[i] = sgd_update_layer(layers[i], grads, args.lr)
            else:
                for i in range(train_start, use_layers):
                    grads = {n: p.grad for n, p in layers[i].named_parameters(recurse=True) if p.grad is not None}
                    ms, _ = adamw_update_layer(layers[i], grads, adam_state[i], adam_cfg)
                    optim_ms_by_layer[i] = ms
            # zero grads
            for i in range(train_start, use_layers):
                for p in layers[i].parameters():
                    p.grad = None
            torch.cuda.synchronize()
            optim_s = time.perf_counter() - t_opt0

            # finalize
            monitor.stop()
            monitor.dump_csv(mem_trace, append=True)

            wall_s = time.perf_counter() - t0_wall
            peak_alloc = gib(torch.cuda.max_memory_allocated())
            peak_reserved = gib(torch.cuda.max_memory_reserved())
            rss = cpu_rss_gib()

            total_fwd_ms = sum(fwd_ms_by_layer.values())
            total_recompute_ms = sum(recompute_ms_by_layer.values())
            total_bwd_ms = sum(bwd_ms_by_layer.values())

            # write per-layer rows for trainable layers
            with open(metrics_layer, "a", encoding="utf-8") as f:
                for i in range(train_start, use_layers):
                    f.write(f"{it},{args.mode},{i},1,0.0,{fwd_ms_by_layer.get(i,0.0):.6f},{recompute_ms_by_layer.get(i,0.0):.6f},{bwd_ms_by_layer.get(i,0.0):.6f},0.0,{optim_ms_by_layer.get(i,0.0):.6f}\n")

            with open(metrics_iter, "a", encoding="utf-8") as f:
                f.write(f"{it},{args.mode},{wall_s:.6f},{forward_s:.6f},{backward_s:.6f},{optim_s:.6f},0.0,{peak_alloc:.6f},{peak_reserved:.6f},{rss:.6f},0.0,{total_fwd_ms:.6f},{total_recompute_ms:.6f},{total_bwd_ms:.6f}\n")

            logger.log(f"[ITER {it}] wall_s={wall_s:.3f} forward_s={forward_s:.3f} backward_s={backward_s:.3f} optim_s={optim_s:.3f} "
                       f"peak_alloc={peak_alloc:.2f}GiB peak_reserved={peak_reserved:.2f}GiB CPU_RSS={rss:.2f}GiB "
                       f"fwd_ms(sum)={total_fwd_ms:.1f} recompute_ms(sum)={total_recompute_ms:.1f} bwd_ms(sum)={total_bwd_ms:.1f}")

        # cleanup hooks
        for h in hooks:
            try: h.remove()
            except Exception: pass

    # ---------------- mode: slide ----------------
    def run_slide():
        # keep original weights on CPU, build 2-stage GPU layers as copies of a template
        logger.log("[MOVE] moving embedding/norm to cuda (weights stay on CPU) ...")
        embed_cuda = embed.to(device=device, dtype=dtype).eval()
        norm_cuda = norm.to(device=device, dtype=dtype).eval() if norm is not None else None

        logger.log("[STAGE] building 2 GPU staging layers (ping-pong) ...")
        stage0 = copy.deepcopy(layers[0]).to(device=device, dtype=dtype).eval()
        stage1 = copy.deepcopy(layers[0]).to(device=device, dtype=dtype).eval()
        stage_param_maps = [build_param_maps(stage0), build_param_maps(stage1)]

        stream_prefetch = torch.cuda.Stream()
        stream_offload = torch.cuda.Stream()
        ev_done = [torch.cuda.Event(enable_timing=False), torch.cuda.Event(enable_timing=False)]

        pinned = set()
        def maybe_pin(i: int):
            if args.pin_mode == "off":
                return
            if i in pinned:
                return
            b = pin_layer_params_inplace(layers[i])
            pinned.add(i)
            logger.log(f"[PIN][L{i:02d}] pinned~{gib(b):.3f}GiB")

        for it in range(args.iters):
            logger.log(f"\n========== [ITER {it}] ==========")
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            monitor.start(it, args.mode)

            t0_wall = time.perf_counter()

            ids = build_ids().to(device, non_blocking=True)
            bs, seqlen = ids.shape
            position_ids = torch.arange(seqlen, device=device, dtype=torch.long).unsqueeze(0).expand(bs, -1)
            attn_2d = torch.ones((bs, seqlen), device=device, dtype=torch.long)

            # record per-layer stats
            prefetch_events = {}   # layer -> (start,end,bytes)
            fwd_ms_by_layer = {}
            recompute_ms_by_layer = {}
            bwd_ms_by_layer = {}
            grad_offload_gib_by_layer = {i: 0.0 for i in range(use_layers)}
            cpu_update_ms_by_layer = {i: 0.0 for i in range(use_layers)}

            # ---- forward streaming ----
            t_fwd0 = time.perf_counter()

            maybe_pin(0)
            s0, e0, b0, _ = prefetch_layer_to_stage(layers[0], stage_param_maps[0], stream_prefetch, ev_done[0], logger, tag=" L00->S0")
            prefetch_events[0] = (s0, e0, b0)
            torch.cuda.current_stream().wait_event(ev_done[0])

            if args.prefetch and use_layers > 1:
                maybe_pin(1)
                s1, e1, b1, _ = prefetch_layer_to_stage(layers[1], stage_param_maps[1], stream_prefetch, ev_done[1], logger, tag=" L01->S1")
                prefetch_events[1] = (s1, e1, b1)

            # offload activations for trainable range
            act_cpu = {}
            act_ev = {}

            with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                hidden = embed_cuda(ids)
                h_shape = hidden.shape
                attention_mask = try_build_4d_causal_mask(attn_2d, hidden, logger)
                if rotary is not None:
                    try:
                        position_embeddings = rotary(hidden, position_ids)
                    except TypeError:
                        position_embeddings = rotary(position_ids)
                else:
                    position_embeddings = None

                for i in range(use_layers):
                    sidx = i % 2
                    torch.cuda.current_stream().wait_event(ev_done[sidx])

                    if i >= train_start:
                        cpu_buf = torch.empty(h_shape, device="cpu", dtype=hidden.dtype, pin_memory=True)
                        ev = torch.cuda.Event(enable_timing=False)
                        with torch.cuda.stream(stream_offload):
                            cpu_buf.copy_(hidden, non_blocking=True)
                            ev.record(stream_offload)
                        act_cpu[i] = cpu_buf
                        act_ev[i] = ev

                    layer_mod = stage0 if sidx == 0 else stage1
                    ms, hidden = measure_cuda_ms(lambda mod=layer_mod, h=hidden: call_layer(mod, h, attention_mask, position_ids, position_embeddings))
                    fwd_ms_by_layer[i] = ms

                    if args.prefetch and (i + 1) < use_layers:
                        nidx = i + 1
                        dst_stage = nidx % 2
                        maybe_pin(nidx)
                        ss, ee, bb, _ = prefetch_layer_to_stage(layers[nidx], stage_param_maps[dst_stage], stream_prefetch, ev_done[dst_stage], logger, tag=f" L{nidx:02d}->S{dst_stage}")
                        prefetch_events[nidx] = (ss, ee, bb)

                if norm_cuda is not None:
                    hidden = norm_cuda(hidden)

            t_fwd1 = time.perf_counter()
            forward_s = t_fwd1 - t_fwd0

            # loss seed
            hidden_leaf = hidden.detach().requires_grad_(True)
            with torch.autocast("cuda", dtype=dtype):
                loss = hidden_leaf.float().pow(2).mean()
            g = torch.autograd.grad(loss, hidden_leaf, retain_graph=False, create_graph=False)[0].detach()

            # ---- backward streaming ----
            t_bwd0 = time.perf_counter()
            executor = ThreadPoolExecutor(max_workers=max(1, int(args.cpu_workers)))
            pending = []
            cpu_update_total_s = 0.0

            for i in range(use_layers - 1, train_start - 1, -1):
                sidx = i % 2
                act_ev[i].synchronize()
                torch.cuda.current_stream().wait_event(ev_done[sidx])

                h_in = act_cpu[i].to(device, non_blocking=True).detach().requires_grad_(True)
                layer_mod = stage0 if sidx == 0 else stage1

                # measure recompute forward only
                rec_ms, h_out = measure_cuda_ms(lambda mod=layer_mod, h=h_in: call_layer(mod, h, attention_mask, position_ids, position_embeddings))
                recompute_ms_by_layer[i] = rec_ms

                # measure full backward step for this layer: recompute+grad (separate from rec_ms)
                s_ev = torch.cuda.Event(enable_timing=True)
                e_ev = torch.cuda.Event(enable_timing=True)
                s_ev.record()
                named = list(layer_mod.named_parameters(recurse=True))
                p_tensors = [p for _, p in named]
                with torch.autocast("cuda", dtype=dtype):
                    grads = torch.autograd.grad(
                        outputs=h_out,
                        inputs=[h_in] + p_tensors,
                        grad_outputs=g,
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=True,
                    )
                e_ev.record()
                torch.cuda.synchronize()
                bwd_ms_by_layer[i] = float(s_ev.elapsed_time(e_ev))

                g = grads[0].detach()

                grads_cpu = {}
                bytes_grad = 0
                for (name, _), gg in zip(named, grads[1:]):
                    if gg is None:
                        continue
                    gg_cpu = gg.detach().to("cpu", non_blocking=True)
                    grads_cpu[name] = gg_cpu
                    bytes_grad += gg_cpu.numel() * gg_cpu.element_size()
                grad_offload_gib_by_layer[i] = gib(bytes_grad)

                # CPU optimizer update (async)
                def cpu_update(layer_idx=i, grads_cpu=grads_cpu):
                    t0 = time.perf_counter()
                    if args.optim == "sgd":
                        # in-place on pinned CPU params
                        name2p = dict(layers[layer_idx].named_parameters(recurse=True))
                        for n, g0 in grads_cpu.items():
                            p0 = name2p.get(n, None)
                            if p0 is None:
                                continue
                            if g0.dtype != p0.data.dtype:
                                g0 = g0.to(dtype=p0.data.dtype)
                            p0.data.add_(g0, alpha=-args.lr)
                    else:
                        # AdamW CPU
                        # reuse GPU-style adamw_update_layer but on CPU tensors
                        _ = adamw_update_layer(layers[layer_idx], grads_cpu, adam_state[layer_idx], adam_cfg)
                    t1 = time.perf_counter()
                    return layer_idx, (t1 - t0)

                pending.append(executor.submit(cpu_update))

                if args.prefetch and (i - 1) >= 0:
                    prev = i - 1
                    dst_stage = prev % 2
                    maybe_pin(prev)
                    ss, ee, bb, _ = prefetch_layer_to_stage(layers[prev], stage_param_maps[dst_stage], stream_prefetch, ev_done[dst_stage], logger, tag=f" L{prev:02d}->S{dst_stage}")
                    prefetch_events[prev] = (ss, ee, bb)

                del h_in, h_out, grads, grads_cpu

            for fut in as_completed(pending):
                layer_idx, upd_s = fut.result()
                cpu_update_total_s += upd_s
                cpu_update_ms_by_layer[layer_idx] = upd_s * 1000.0

            executor.shutdown(wait=True)
            t_bwd1 = time.perf_counter()
            backward_s = t_bwd1 - t_bwd0

            # finalize
            torch.cuda.synchronize()
            wall_s = time.perf_counter() - t0_wall
            peak_alloc = gib(torch.cuda.max_memory_allocated())
            peak_reserved = gib(torch.cuda.max_memory_reserved())
            rss = cpu_rss_gib()

            # prefetch ms
            total_prefetch_ms = 0.0
            prefetch_ms_by_layer = {i: 0.0 for i in range(use_layers)}
            for li, (ss, ee, bb) in prefetch_events.items():
                try:
                    ms = float(ss.elapsed_time(ee))
                except Exception:
                    ms = 0.0
                prefetch_ms_by_layer[li] = ms
                total_prefetch_ms += ms

            total_fwd_ms = sum(fwd_ms_by_layer.get(i, 0.0) for i in range(use_layers))
            total_recompute_ms = sum(recompute_ms_by_layer.get(i, 0.0) for i in range(train_start, use_layers))
            total_bwd_ms = sum(bwd_ms_by_layer.get(i, 0.0) for i in range(train_start, use_layers))

            monitor.stop()
            monitor.dump_csv(mem_trace, append=True)

            with open(metrics_layer, "a", encoding="utf-8") as f:
                for i in range(use_layers):
                    trainable = 1 if i >= train_start else 0
                    f.write(f"{it},{args.mode},{i},{trainable},{prefetch_ms_by_layer.get(i,0.0):.6f},{fwd_ms_by_layer.get(i,0.0):.6f},{recompute_ms_by_layer.get(i,0.0) if trainable else 0.0:.6f},{bwd_ms_by_layer.get(i,0.0) if trainable else 0.0:.6f},{grad_offload_gib_by_layer.get(i,0.0) if trainable else 0.0:.6f},{cpu_update_ms_by_layer.get(i,0.0) if trainable else 0.0:.6f}\n")

            with open(metrics_iter, "a", encoding="utf-8") as f:
                f.write(f"{it},{args.mode},{wall_s:.6f},{forward_s:.6f},{backward_s:.6f},0.0,{cpu_update_total_s:.6f},{peak_alloc:.6f},{peak_reserved:.6f},{rss:.6f},{total_prefetch_ms:.6f},{total_fwd_ms:.6f},{total_recompute_ms:.6f},{total_bwd_ms:.6f}\n")

            logger.log(f"[ITER {it}] wall_s={wall_s:.3f} forward_s={forward_s:.3f} backward_s={backward_s:.3f} cpu_update_total_s={cpu_update_total_s:.3f} "
                       f"peak_alloc={peak_alloc:.2f}GiB peak_reserved={peak_reserved:.2f}GiB CPU_RSS={rss:.2f}GiB "
                       f"prefetch_ms(sum)={total_prefetch_ms:.1f} fwd_ms(sum)={total_fwd_ms:.1f} recompute_ms(sum)={total_recompute_ms:.1f} bwd_ms(sum)={total_bwd_ms:.1f}")

    # dispatch
    if args.mode in ("gpu", "ckpt"):
        run_gpu_or_ckpt()
    else:
        run_slide()

    # write lightweight README for the run
    with open(os.path.join(run_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("# T1-M1 Run\n\n")
        f.write(f"- mode: `{args.mode}`\n")
        f.write(f"- model: `{args.model}`\n")
        f.write(f"- seq={args.seq}, batch={args.batch}, iters={args.iters}\n")
        f.write(f"- use_layers={use_layers}, train_layers={train_layers} (range {train_start}..{use_layers-1})\n")
        f.write(f"- optim={args.optim}, lr={args.lr}, adam_state_dtype={args.adam_state_dtype}\n\n")
        f.write("## Files\n")
        f.write("- env.json\n- log.txt\n- metrics_iter.csv\n- metrics_layer.csv\n- mem_trace.csv\n")

    logger.log(f"[SAVED] {metrics_iter}")
    logger.log(f"[SAVED] {metrics_layer}")
    logger.log(f"[SAVED] {mem_trace}")
    logger.close()
    print(run_dir)  # convenient for wrappers

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        msg = "[CRASH] " + repr(e) + "\n" + traceback.format_exc()
        print(msg, flush=True)
        raise
