
#python t1_m1_suite.py --model ./qwen --seq 256 --batch 1 --iters 2 --layers 28 --train_layers 28 --optim adamw --adam_state_dtype bf16 --lr 1e-4 --logdir runs_m1 --modes gpu,gpu_ckpt,sf
#all ways

#python t1_m1_suite.py --model ./qwen --seq 256 --batch 1 --iters 2 --layers 28 --train_layers 28 --optim sgd --adam_state_dtype bf16 --lr 1e-4 --logdir runs_m1 --modes sf,gpu
#slideformer and normal

#python t1_m1_suite.py --model ./qwen --seq 256 --batch 1 --iters 3 --layers 28 --train_layers 8 --ckpt_layers 8 --optim adamw --adam_state_dtype bf16 --lr 1e-4 --logdir runs_m1 --modes gpu,gpu_ckpt,sf
#partial compute

# t1_m1_suite.py
# One-shot suite for T1-M1:
#   - gpu      : pure GPU standard training (no checkpoint, no CPU offload)
#   - gpu_ckpt : pure GPU + PyTorch gradient checkpointing (manual per-layer checkpoint)
#   - sf       : SlideFormer-style CPU/GPU layer streaming + activation offload + explicit recompute + CPU optimizer
#
# Outputs per run:
#   log.txt, env.json, metrics_iter.csv, metrics_layer.csv, mem_trace.csv, timeline.csv, plots (*.png)
# And a final compare report:
#   compare_<stamp>/*

# t1_m1_suite.py
# One-shot suite for T1-M1:
#   - gpu      : pure GPU standard training (no checkpoint, no CPU offload)
#   - gpu_ckpt : pure GPU + PyTorch gradient checkpointing (manual per-layer checkpoint)
#   - sf       : SlideFormer-style CPU/GPU layer streaming + activation offload + explicit recompute + CPU optimizer
#
# Outputs per run:
#   log.txt, env.json, metrics_iter.csv, metrics_layer.csv, mem_trace.csv, timeline.csv, plots (*.png)
# And a final compare report:
#   compare_<stamp>/*

import os, sys, time, json, copy, argparse, traceback, inspect, math
from datetime import datetime
from dataclasses import dataclass
from threading import Thread, Event
from concurrent.futures import ThreadPoolExecutor, as_completed

import psutil
import torch
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM


# ----------------------------
# utils
# ----------------------------
def gib(x): return float(x) / (1024 ** 3)

def cpu_rss_gib():
    p = psutil.Process(os.getpid())
    return p.memory_info().rss / (1024 ** 3)

def now_ms(t0):
    return (time.perf_counter() - t0) * 1000.0

class Logger:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "w", encoding="utf-8")

    def log(self, msg):
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass

class MemMonitor:
    """
    Samples CPU RSS + GPU allocated/reserved over time.
    """
    def __init__(self, sample_ms=25):
        self.sample_s = max(sample_ms, 5) / 1000.0
        self._stop = Event()
        self.samples = []  # (t_ms, cpu_rss_gib, gpu_alloc_gib, gpu_reserved_gib)

    def start(self, t0):
        self.t0 = t0
        self.th = Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        while not self._stop.is_set():
            t = now_ms(self.t0)
            cpu = cpu_rss_gib()
            if torch.cuda.is_available():
                ga = gib(torch.cuda.memory_allocated())
                gr = gib(torch.cuda.memory_reserved())
            else:
                ga, gr = 0.0, 0.0
            self.samples.append((t, cpu, ga, gr))
            time.sleep(self.sample_s)

    def stop(self):
        self._stop.set()
        try:
            self.th.join(timeout=1.0)
        except Exception:
            pass


# ----------------------------
# model structure helpers
# ----------------------------
def pick_layers(model):
    # For HF causal LMs: model.model.layers (Qwen/Llama-like) or model.transformer.h (GPT2-like)
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

def measure_cuda(fn):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    out = fn()
    e.record()
    torch.cuda.synchronize()
    return float(s.elapsed_time(e)), out

def call_layer(layer, hidden_states, attention_mask, position_ids, position_embeddings):
    # cache signature flags (cheap enough; keep simple)
    try:
        sig = inspect.signature(layer.forward)
        params = sig.parameters
    except Exception:
        sig = None
        params = {}

    kwargs = {}
    if sig is not None:
        if "attention_mask" in params and attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if "position_ids" in params and position_ids is not None:
            kwargs["position_ids"] = position_ids
        if "position_embeddings" in params:
            kwargs["position_embeddings"] = position_embeddings
        if "use_cache" in params:
            kwargs["use_cache"] = False
        if "output_attentions" in params:
            kwargs["output_attentions"] = False
        if "output_hidden_states" in params:
            kwargs["output_hidden_states"] = False

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

def pin_layer_params_inplace(layer):
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

def prefetch_layer_to_stage(cpu_layer, stage_param_map, stream, done_event, logger, tag=""):
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
    return start, end, float(bytes_copied), int(copied)

def plot_save(path, title, xlabel, ylabel, xs, ys_list, labels):
    plt.figure(figsize=(14, 6))
    for ys, lb in zip(ys_list, labels):
        plt.plot(xs, ys, label=lb)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()

def bar_save(path, title, xlabel, ylabel, labels, values):
    plt.figure(figsize=(14, 6))
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


# ----------------------------
# input builder
# ----------------------------
def build_ids(tok, seq, batch):
    text = ("SlideFormer T1M1 suite. " * 400).strip()
    ids = tok(text, return_tensors="pt", truncation=True, max_length=seq).input_ids
    if ids.size(1) < seq:
        pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0
        pad = torch.full((1, seq - ids.size(1)), pad_id, dtype=ids.dtype)
        ids = torch.cat([ids, pad], dim=1)
    ids = ids[:, :seq].repeat(batch, 1)
    return ids


# ----------------------------
# Run: SlideFormer mode (sf)
# ----------------------------
def run_slideformer(args, run_dir, logger: Logger):
    assert torch.cuda.is_available(), "CUDA is required."
    device = torch.device("cuda")
    dtype = torch.bfloat16 if (args.dtype == "bf16" and torch.cuda.is_bf16_supported()) else torch.float16

    # env
    env = {
        "mode": "sf",
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "dtype": str(dtype),
        "args": vars(args),
    }
    with open(os.path.join(run_dir, "env.json"), "w", encoding="utf-8") as f:
        json.dump(env, f, ensure_ascii=False, indent=2)
    logger.log("[ENV] " + json.dumps(env, ensure_ascii=False))

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    logger.log("[LOAD] tokenizer/model on CPU ...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map={"": "cpu"},
        torch_dtype=dtype,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.train()  # for "training style" (dropout usually none here)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    layers, core = pick_layers(model)
    embed, norm = pick_embeddings_and_norm(core)
    if embed is None:
        raise RuntimeError("Cannot locate embedding module.")

    rotary = pick_rotary_emb(model, core)
    if rotary is None:
        logger.log("[WARN] rotary_emb not found. Some models (e.g. Qwen3) may fail.")
    else:
        rotary = rotary.to(device=device)
        rotary.train()
        logger.log(f"[ROTARY] found: {type(rotary).__name__}, moved to cuda")

    total_layers = len(layers)
    use_layers = total_layers if args.layers <= 0 else min(args.layers, total_layers)
    train_layers = use_layers if args.train_layers <= 0 else min(args.train_layers, use_layers)
    train_start = use_layers - train_layers

    logger.log(f"[CFG] total_layers={total_layers}, use_layers={use_layers}, train_layers={train_layers} (range {train_start}..{use_layers-1}), "
               f"prefetch={args.sf_prefetch}, pin_mode={args.pin_mode}, optim={args.optim}, cpu_workers={args.cpu_workers}")

    # move embed/norm to GPU (small)
    embed = embed.to(device=device, dtype=dtype).train()
    if norm is not None:
        norm = norm.to(device=device, dtype=dtype).train()

    # build 2 staging layers on GPU (ping-pong)
    logger.log("[STAGE] building 2 GPU staging layers (ping-pong) ...")
    stage0 = copy.deepcopy(layers[0]).to(device=device, dtype=dtype).train()
    stage1 = copy.deepcopy(layers[0]).to(device=device, dtype=dtype).train()
    stage_param_maps = [build_param_maps(stage0), build_param_maps(stage1)]

    stream_prefetch = torch.cuda.Stream()
    stream_offload = torch.cuda.Stream()
    ev_done = [torch.cuda.Event(enable_timing=False), torch.cuda.Event(enable_timing=False)]

    pinned = set()
    def maybe_pin(i):
        if args.pin_mode == "off":
            return 0
        if i in pinned:
            return 0
        b = pin_layer_params_inplace(layers[i])
        pinned.add(i)
        logger.log(f"[PIN][L{i:02d}] pinned~{gib(b):.3f}GiB")
        return b

    # CSVs
    metrics_iter = os.path.join(run_dir, "metrics_iter.csv")
    with open(metrics_iter, "w", encoding="utf-8") as f:
        f.write("iter,wall_s,forward_s,backward_s,cpu_update_total_s,peak_alloc_gib,peak_reserved_gib,cpu_rss_gib,total_prefetch_ms,total_fwd_ms,total_recompute_ms\n")

    metrics_layer = os.path.join(run_dir, "metrics_layer.csv")
    with open(metrics_layer, "w", encoding="utf-8") as f:
        f.write("iter,layer,stage,trainable,prefetch_ms,fwd_ms,recompute_ms,grad_offload_gib,cpu_update_ms\n")

    timeline_csv = os.path.join(run_dir, "timeline.csv")
    with open(timeline_csv, "w", encoding="utf-8") as f:
        f.write("iter,t_ms,phase,layer,detail\n")

    mem_csv = os.path.join(run_dir, "mem_trace.csv")
    with open(mem_csv, "w", encoding="utf-8") as f:
        f.write("iter,t_ms,cpu_rss_gib,gpu_alloc_gib,gpu_reserved_gib\n")

    # CPU optimizer (manual) - keep your existing idea, but parallelize
    def ensure_state_tensor(shape, dtype_):
        return torch.zeros(shape, dtype=dtype_, device="cpu")

    def cpu_sgd_update(layer_idx, cpu_layer, grads_cpu: dict, lr: float):
        t0 = time.perf_counter()
        name2p = dict(cpu_layer.named_parameters(recurse=True))
        for name, g in grads_cpu.items():
            p = name2p.get(name, None)
            if p is None or g is None:
                continue
            if g.dtype != p.data.dtype:
                g = g.to(dtype=p.data.dtype)
            p.data.add_(g, alpha=-lr)
        return layer_idx, (time.perf_counter() - t0)

    def cpu_adamw_update(layer_idx, cpu_layer, grads_cpu: dict, lr: float,
                         state: dict, beta1: float, beta2: float, eps: float,
                         weight_decay: float, state_dtype: str):
        """
        AdamW on CPU. state[name] = {"m": tensor, "v": tensor, "step": int}
        NOTE: CPU AdamW can be very slow; that's an important result by itself.
        """
        t0 = time.perf_counter()
        name2p = dict(cpu_layer.named_parameters(recurse=True))
        for name, g in grads_cpu.items():
            p = name2p.get(name, None)
            if p is None or g is None:
                continue
            st = state.get(name, None)
            if st is None:
                dt = torch.float32 if state_dtype == "fp32" else p.data.dtype
                st = {"m": ensure_state_tensor(p.data.shape, dt),
                      "v": ensure_state_tensor(p.data.shape, dt),
                      "step": 0}
                state[name] = st
            st["step"] += 1
            step = st["step"]

            # fp32 math
            g32 = g.float()
            p32 = p.data.float()
            m32 = st["m"].float()
            v32 = st["v"].float()

            if weight_decay != 0.0:
                g32 = g32.add(p32, alpha=weight_decay)

            m32.mul_(beta1).add_(g32, alpha=1.0 - beta1)
            v32.mul_(beta2).addcmul_(g32, g32, value=1.0 - beta2)

            bc1 = 1.0 - (beta1 ** step)
            bc2 = 1.0 - (beta2 ** step)
            mhat = m32 / bc1
            vhat = v32 / bc2
            upd = mhat / (vhat.sqrt().add_(eps))

            p32.add_(upd, alpha=-lr)
            p.data.copy_(p32.to(dtype=p.data.dtype))

            if st["m"].dtype == torch.float32:
                st["m"].copy_(m32)
                st["v"].copy_(v32)
            else:
                st["m"].copy_(m32.to(st["m"].dtype))
                st["v"].copy_(v32.to(st["v"].dtype))

        return layer_idx, (time.perf_counter() - t0)

    betas = [float(x.strip()) for x in args.betas.split(",")]
    beta1, beta2 = betas[0], betas[1]
    adam_state = [dict() for _ in range(use_layers)]  # per-layer dict

    # pre-build a README of what this mode means
    with open(os.path.join(run_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("## Mode = sf (SlideFormer-style)\n\n")
        f.write("- weights live on CPU; GPU holds only 2 staging transformer blocks (ping-pong)\n")
        f.write("- forward: H2D prefetch next layer while computing current layer\n")
        f.write("- trainable activations: offloaded to CPU pinned memory\n")
        f.write("- backward: layer-wise recomputation (explicit) + grad offload to CPU + CPU optimizer update\n")

    # run iters
    for it in range(args.iters):
        logger.log(f"\n========== [SF][ITER {it}] ==========")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        t0_wall = time.perf_counter()
        mon = MemMonitor(sample_ms=args.sample_ms)
        mon.start(t0_wall)

        ids = build_ids(tok, args.seq, args.batch).to(device, non_blocking=True)
        bs, seqlen = ids.shape
        position_ids = torch.arange(seqlen, device=device, dtype=torch.long).unsqueeze(0).expand(bs, -1)
        attn_2d = torch.ones((bs, seqlen), device=device, dtype=torch.long)

        # per-layer stats
        prefetch_events = {}
        fwd_ms_by_layer = {}
        recompute_ms_by_layer = {}
        grad_offload_gib_by_layer = {i: 0.0 for i in range(use_layers)}
        cpu_update_ms_by_layer = {i: 0.0 for i in range(use_layers)}
        act_cpu = {}
        act_ev = {}

        # initial prefetch
        maybe_pin(0)
        if args.sf_prefetch:
            ss, ee, bb, _ = prefetch_layer_to_stage(layers[0], stage_param_maps[0], stream_prefetch, ev_done[0], logger, tag=" L00->S0")
            prefetch_events[0] = (ss, ee, bb)
            with open(timeline_csv, "a", encoding="utf-8") as f:
                f.write(f"{it},{now_ms(t0_wall):.3f},prefetch_enqueue,0,S0\n")
        else:
            # directly copy (sync) if prefetch off
            stage_param_maps[0] = build_param_maps(stage0)

        torch.cuda.current_stream().wait_event(ev_done[0])

        if args.sf_prefetch and use_layers > 1:
            maybe_pin(1)
            ss, ee, bb, _ = prefetch_layer_to_stage(layers[1], stage_param_maps[1], stream_prefetch, ev_done[1], logger, tag=" L01->S1")
            prefetch_events[1] = (ss, ee, bb)
            with open(timeline_csv, "a", encoding="utf-8") as f:
                f.write(f"{it},{now_ms(t0_wall):.3f},prefetch_enqueue,1,S1\n")

        # forward (no_grad because staging weights overwrite would break a single autograd graph)
        t_fwd0 = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            hidden = embed(ids)
            h_shape = hidden.shape
            attention_mask = try_build_4d_causal_mask(attn_2d, hidden, logger)

            # rotary -> position_embeddings
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

                # offload activation for trainable layers
                if i >= train_start:
                    cpu_buf = torch.empty(h_shape, device="cpu", dtype=hidden.dtype, pin_memory=True)
                    ev = torch.cuda.Event(enable_timing=False)
                    with torch.cuda.stream(stream_offload):
                        cpu_buf.copy_(hidden, non_blocking=True)
                        ev.record(stream_offload)
                    act_cpu[i] = cpu_buf
                    act_ev[i] = ev
                    with open(timeline_csv, "a", encoding="utf-8") as f:
                        f.write(f"{it},{now_ms(t0_wall):.3f},act_offload_enqueue,{i},cpu_pinned\n")

                layer_mod = stage0 if sidx == 0 else stage1
                with open(timeline_csv, "a", encoding="utf-8") as f:
                    f.write(f"{it},{now_ms(t0_wall):.3f},fwd_start,{i},S{sidx}\n")

                ms, hidden = measure_cuda(lambda: call_layer(layer_mod, hidden, attention_mask, position_ids, position_embeddings))
                fwd_ms_by_layer[i] = ms
                logger.log(f"[FWD][L{i:02d}] stage={sidx} fwd_ms={ms:.3f} trainable={1 if i>=train_start else 0}")

                with open(timeline_csv, "a", encoding="utf-8") as f:
                    f.write(f"{it},{now_ms(t0_wall):.3f},fwd_end,{i},ms={ms:.3f}\n")

                if args.sf_prefetch and (i + 1) < use_layers:
                    nidx = i + 1
                    dst_stage = nidx % 2
                    maybe_pin(nidx)
                    ss, ee, bb, _ = prefetch_layer_to_stage(layers[nidx], stage_param_maps[dst_stage], stream_prefetch, ev_done[dst_stage], logger, tag=f" L{nidx:02d}->S{dst_stage}")
                    prefetch_events[nidx] = (ss, ee, bb)
                    with open(timeline_csv, "a", encoding="utf-8") as f:
                        f.write(f"{it},{now_ms(t0_wall):.3f},prefetch_enqueue,{nidx},S{dst_stage}\n")

            if norm is not None:
                hidden = norm(hidden)

        forward_s = time.perf_counter() - t_fwd0
        logger.log(f"[FWD] forward_s={forward_s:.3f} acts_saved(trainable)={len(act_cpu)} / {train_layers}")

        # toy loss to seed gradients (consistent across modes)
        hidden_leaf = hidden.detach().requires_grad_(True)
        with torch.autocast("cuda", dtype=dtype):
            loss = hidden_leaf.float().pow(2).mean()
        g = torch.autograd.grad(loss, hidden_leaf, retain_graph=False, create_graph=False)[0].detach()
        logger.log(f"[LOSS] loss={loss.item():.6f} grad_norm={float(g.float().norm().item()):.6f}")

        # backward streaming with explicit recompute
        t_bwd0 = time.perf_counter()
        executor = ThreadPoolExecutor(max_workers=max(1, int(args.cpu_workers)))
        pending = []
        cpu_update_total = 0.0

        for i in range(use_layers - 1, train_start - 1, -1):
            sidx = i % 2
            act_ev[i].synchronize()
            torch.cuda.current_stream().wait_event(ev_done[sidx])

            h_in = act_cpu[i].to(device, non_blocking=True).detach().requires_grad_(True)
            layer_mod = stage0 if sidx == 0 else stage1

            with open(timeline_csv, "a", encoding="utf-8") as f:
                f.write(f"{it},{now_ms(t0_wall):.3f},recompute_start,{i},S{sidx}\n")

            with torch.autocast("cuda", dtype=dtype):
                rec_ms, h_out = measure_cuda(lambda: call_layer(layer_mod, h_in, attention_mask, position_ids, position_embeddings))
            recompute_ms_by_layer[i] = rec_ms

            with open(timeline_csv, "a", encoding="utf-8") as f:
                f.write(f"{it},{now_ms(t0_wall):.3f},recompute_end,{i},ms={rec_ms:.3f}\n")

            named = list(layer_mod.named_parameters(recurse=True))
            p_tensors = [p for (_, p) in named]

            with torch.autocast("cuda", dtype=dtype):
                grads = torch.autograd.grad(
                    outputs=h_out,
                    inputs=[h_in] + p_tensors,
                    grad_outputs=g,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )

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

            # CPU update task
            if args.optim == "sgd":
                pending.append(executor.submit(cpu_sgd_update, i, layers[i], grads_cpu, args.lr))
            else:
                pending.append(executor.submit(
                    cpu_adamw_update, i, layers[i], grads_cpu, args.lr,
                    adam_state[i], beta1, beta2, args.eps, args.weight_decay,
                    "fp32" if args.adam_state_dtype == "fp32" else "bf16"
                ))

            logger.log(f"[BWD][L{i:02d}] stage={sidx} recompute_ms={rec_ms:.3f} grad_offload~{gib(bytes_grad):.3f}GiB pending_cpu_updates={len(pending)}")

            if args.sf_prefetch and (i - 1) >= 0:
                prev = i - 1
                dst_stage = prev % 2
                maybe_pin(prev)
                ss, ee, bb, _ = prefetch_layer_to_stage(layers[prev], stage_param_maps[dst_stage], stream_prefetch, ev_done[dst_stage], logger, tag=f" L{prev:02d}->S{dst_stage}")
                prefetch_events[prev] = (ss, ee, bb)
                with open(timeline_csv, "a", encoding="utf-8") as f:
                    f.write(f"{it},{now_ms(t0_wall):.3f},prefetch_enqueue,{prev},S{dst_stage}\n")

            del h_in, h_out, grads, grads_cpu

        # collect cpu updates
        for fut in as_completed(pending):
            layer_idx, upd_s = fut.result()
            cpu_update_total += upd_s
            cpu_update_ms_by_layer[layer_idx] = upd_s * 1000.0
            with open(timeline_csv, "a", encoding="utf-8") as f:
                f.write(f"{it},{now_ms(t0_wall):.3f},cpu_update_end,{layer_idx},ms={upd_s*1000.0:.3f}\n")
            logger.log(f"[CPU_UPDATE][L{layer_idx:02d}] update_s={upd_s:.4f}")

        executor.shutdown(wait=True)
        backward_s = time.perf_counter() - t_bwd0

        torch.cuda.synchronize()
        wall_s = time.perf_counter() - t0_wall
        peak_alloc = gib(torch.cuda.max_memory_allocated())
        peak_reserved = gib(torch.cuda.max_memory_reserved())
        rss = cpu_rss_gib()

        # stop monitor and dump mem samples
        mon.stop()
        with open(mem_csv, "a", encoding="utf-8") as f:
            for (tms, cpu, ga, gr) in mon.samples:
                f.write(f"{it},{tms:.3f},{cpu:.6f},{ga:.6f},{gr:.6f}\n")

        # prefetch ms
        total_prefetch_ms = 0.0
        for li, (ss, ee, bb) in prefetch_events.items():
            try:
                total_prefetch_ms += float(ss.elapsed_time(ee))
            except Exception:
                pass
        total_fwd_ms = sum(float(fwd_ms_by_layer.get(i, 0.0)) for i in range(use_layers))
        total_recompute_ms = sum(float(recompute_ms_by_layer.get(i, 0.0)) for i in range(train_start, use_layers))

        logger.log(f"[SF][ITER {it}] wall_s={wall_s:.3f} forward_s={forward_s:.3f} backward_s={backward_s:.3f} cpu_update_total_s={cpu_update_total:.3f} "
                   f"peak_alloc={peak_alloc:.2f}GiB peak_reserved={peak_reserved:.2f}GiB CPU_RSS={rss:.2f}GiB "
                   f"prefetch_ms(sum)={total_prefetch_ms:.1f} fwd_ms(sum)={total_fwd_ms:.1f} recompute_ms(sum)={total_recompute_ms:.1f}")

        with open(metrics_iter, "a", encoding="utf-8") as f:
            f.write(f"{it},{wall_s:.6f},{forward_s:.6f},{backward_s:.6f},{cpu_update_total:.6f},{peak_alloc:.6f},{peak_reserved:.6f},{rss:.6f},{total_prefetch_ms:.6f},{total_fwd_ms:.6f},{total_recompute_ms:.6f}\n")

        with open(metrics_layer, "a", encoding="utf-8") as f:
            for i in range(use_layers):
                sidx = i % 2
                trainable = 1 if i >= train_start else 0
                pm = 0.0
                if i in prefetch_events:
                    ss, ee, _ = prefetch_events[i]
                    try:
                        pm = float(ss.elapsed_time(ee))
                    except Exception:
                        pm = 0.0
                fm = float(fwd_ms_by_layer.get(i, 0.0))
                rm = float(recompute_ms_by_layer.get(i, 0.0)) if trainable else 0.0
                go = float(grad_offload_gib_by_layer.get(i, 0.0)) if trainable else 0.0
                cu = float(cpu_update_ms_by_layer.get(i, 0.0)) if trainable else 0.0
                f.write(f"{it},{i},{sidx},{trainable},{pm:.6f},{fm:.6f},{rm:.6f},{go:.6f},{cu:.6f}\n")

        # per-run plots (use this iter)
        if args.plot:
            xs = list(range(use_layers))
            prefetch_ms = []
            fwd_ms = []
            recompute_ms = []
            cpu_up_ms = []
            for i in xs:
                pm = 0.0
                if i in prefetch_events:
                    ss, ee, _ = prefetch_events[i]
                    try:
                        pm = float(ss.elapsed_time(ee))
                    except Exception:
                        pm = 0.0
                prefetch_ms.append(pm)
                fwd_ms.append(float(fwd_ms_by_layer.get(i, 0.0)))
                recompute_ms.append(float(recompute_ms_by_layer.get(i, 0.0)) if i >= train_start else 0.0)
                cpu_up_ms.append(float(cpu_update_ms_by_layer.get(i, 0.0)) if i >= train_start else 0.0)

            bar_save(os.path.join(run_dir, f"report_time_breakdown_iter{it}.png"),
                     "SF Time Breakdown (one iter)", "component", "seconds",
                     ["forward_s", "backward_s", "cpu_update_total_s", "wall_s"],
                     [forward_s, backward_s, cpu_update_total, wall_s])

            plot_save(os.path.join(run_dir, f"report_layer_prefetch_vs_compute_iter{it}.png"),
                      f"SF Layer Prefetch(H2D) vs Compute (train_start={train_start})",
                      "layer index", "milliseconds",
                      xs, [prefetch_ms, fwd_ms], ["prefetch_ms", "fwd_ms"])

            plot_save(os.path.join(run_dir, f"report_layer_recompute_iter{it}.png"),
                      f"SF Layer Backward Recomputation Time (trainable >= {train_start})",
                      "layer index", "milliseconds",
                      xs, [recompute_ms], ["recompute_ms"])

            plot_save(os.path.join(run_dir, f"report_layer_cpu_update_iter{it}.png"),
                      f"SF CPU Optimizer Update Time per Layer (trainable >= {train_start})",
                      "layer index", "milliseconds",
                      xs, [cpu_up_ms], ["cpu_update_ms"])

            # memory timeline: 2 lines (CPU RSS, GPU reserved)
            if len(mon.samples) > 2:
                tms = [x[0] for x in mon.samples]
                cpu_line = [x[1] for x in mon.samples]
                gpu_line = [x[3] for x in mon.samples]  # reserved
                plot_save(os.path.join(run_dir, f"report_mem_timeline_iter{it}.png"),
                          "SF Memory Timeline (CPU RSS vs GPU Reserved)",
                          "time (ms)", "GiB",
                          tms, [cpu_line, gpu_line], ["cpu_rss_gib", "gpu_reserved_gib"])

    logger.log(f"[SAVED] {metrics_iter}")
    logger.log(f"[SAVED] {metrics_layer}")
    logger.log(f"[SAVED] {mem_csv}")
    logger.log(f"[SAVED] {timeline_csv}")


# ----------------------------
# Run: GPU / GPU+checkpoint
# ----------------------------
def run_gpu_mode(args, run_dir, logger: Logger, use_ckpt: bool):
    assert torch.cuda.is_available(), "CUDA is required."
    device = torch.device("cuda")
    dtype = torch.bfloat16 if (args.dtype == "bf16" and torch.cuda.is_bf16_supported()) else torch.float16

    mode_name = "gpu_ckpt" if use_ckpt else "gpu"

    env = {
        "mode": mode_name,
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "dtype": str(dtype),
        "args": vars(args),
    }
    with open(os.path.join(run_dir, "env.json"), "w", encoding="utf-8") as f:
        json.dump(env, f, ensure_ascii=False, indent=2)
    logger.log("[ENV] " + json.dumps(env, ensure_ascii=False))

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    logger.log("[LOAD] tokenizer/model on GPU ...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # We load causal LM on GPU but only use the block stack (embed + layers + norm) for a consistent toy loss.
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).to(device)
    model.train()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    layers, core = pick_layers(model)
    embed, norm = pick_embeddings_and_norm(core)
    if embed is None:
        raise RuntimeError("Cannot locate embedding module.")

    rotary = pick_rotary_emb(model, core)
    if rotary is None:
        logger.log("[WARN] rotary_emb not found. Some models (e.g. Qwen3) may fail.")
    else:
        rotary = rotary.to(device=device).train()
        logger.log(f"[ROTARY] found: {type(rotary).__name__}")

    total_layers = len(layers)
    use_layers = total_layers if args.layers <= 0 else min(args.layers, total_layers)
    train_layers = use_layers if args.train_layers <= 0 else min(args.train_layers, use_layers)
    train_start = use_layers - train_layers

    # manual control for checkpoint: checkpoint last K trainable layers (default = all trainable)
    ckpt_layers = train_layers if args.ckpt_layers <= 0 else min(args.ckpt_layers, train_layers)
    ckpt_start = use_layers - ckpt_layers  # apply checkpoint for i in [ckpt_start, use_layers)
    if not use_ckpt:
        ckpt_start = use_layers + 1  # disable

    logger.log(f"[CFG] total_layers={total_layers}, use_layers={use_layers}, train_layers={train_layers} (range {train_start}..{use_layers-1}), "
               f"use_ckpt={int(use_ckpt)} ckpt_layers={ckpt_layers} (range {ckpt_start}..{use_layers-1}), optim={args.optim}")

    # build per-layer optimizer (to get per-layer step timing)
    def layer_params(mod):
        return [p for p in mod.parameters(recurse=True) if p.requires_grad]

    # we only "train" trainable blocks; embeddings/norm frozen for fair comparison with sf
    for p in embed.parameters(recurse=True):
        p.requires_grad_(False)
    if norm is not None:
        for p in norm.parameters(recurse=True):
            p.requires_grad_(False)

    # freeze layers < train_start by running them under no_grad + detach boundary
    # layers >= train_start require grad
    for i in range(use_layers):
        req = (i >= train_start)
        for p in layers[i].parameters(recurse=True):
            p.requires_grad_(req)

    opts = []
    opt_layer_ids = []
    if args.optim == "sgd":
        for i in range(train_start, use_layers):
            opts.append(torch.optim.SGD(layer_params(layers[i]), lr=args.lr))
            opt_layer_ids.append(i)
    else:
        betas = [float(x.strip()) for x in args.betas.split(",")]
        for i in range(train_start, use_layers):
            opts.append(torch.optim.AdamW(layer_params(layers[i]), lr=args.lr, betas=(betas[0], betas[1]),
                                          eps=args.eps, weight_decay=args.weight_decay))
            opt_layer_ids.append(i)

    from torch.utils.checkpoint import checkpoint as ckpt

    # CSVs
    metrics_iter = os.path.join(run_dir, "metrics_iter.csv")
    with open(metrics_iter, "w", encoding="utf-8") as f:
        f.write("iter,wall_s,forward_s,backward_s,optim_step_total_s,peak_alloc_gib,peak_reserved_gib,cpu_rss_gib\n")

    metrics_layer = os.path.join(run_dir, "metrics_layer.csv")
    with open(metrics_layer, "w", encoding="utf-8") as f:
        f.write("iter,layer,trainable,checkpointed,fwd_ms,bwd_ms,optim_step_ms\n")

    timeline_csv = os.path.join(run_dir, "timeline.csv")
    with open(timeline_csv, "w", encoding="utf-8") as f:
        f.write("iter,t_ms,phase,layer,detail\n")

    mem_csv = os.path.join(run_dir, "mem_trace.csv")
    with open(mem_csv, "w", encoding="utf-8") as f:
        f.write("iter,t_ms,cpu_rss_gib,gpu_alloc_gib,gpu_reserved_gib\n")

    # backward hooks for per-layer bwd time (includes recompute overhead if checkpointed)
    bwd_start_ev = {}
    bwd_end_ev = {}
    bwd_ms = {i: 0.0 for i in range(use_layers)}

    def make_bwd_hooks(i):
        def pre_hook(mod, grad_input):
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            bwd_start_ev[i] = ev
        def post_hook(mod, grad_input, grad_output):
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            bwd_end_ev[i] = ev
        return pre_hook, post_hook

    has_pre = hasattr(layers[0], "register_full_backward_pre_hook")
    for i in range(use_layers):
        pre, post = make_bwd_hooks(i)
        if has_pre:
            layers[i].register_full_backward_pre_hook(pre)
        layers[i].register_full_backward_hook(post)

    # run iters
    for it in range(args.iters):
        logger.log(f"\n========== [{mode_name.upper()}][ITER {it}] ==========")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        t0_wall = time.perf_counter()
        mon = MemMonitor(sample_ms=args.sample_ms)
        mon.start(t0_wall)

        ids = build_ids(tok, args.seq, args.batch).to(device, non_blocking=True)
        bs, seqlen = ids.shape
        position_ids = torch.arange(seqlen, device=device, dtype=torch.long).unsqueeze(0).expand(bs, -1)
        attn_2d = torch.ones((bs, seqlen), device=device, dtype=torch.long)

        # forward loop
        t_fwd0 = time.perf_counter()
        fwd_ms = {i: 0.0 for i in range(use_layers)}

        with torch.autocast("cuda", dtype=dtype):
            hidden = embed(ids)
            attention_mask = try_build_4d_causal_mask(attn_2d, hidden, logger)
            if rotary is not None:
                try:
                    position_embeddings = rotary(hidden, position_ids)
                except TypeError:
                    position_embeddings = rotary(position_ids)
            else:
                position_embeddings = None

            # run non-trainable prefix without grad, then detach
            if train_start > 0:
                with torch.no_grad():
                    for i in range(train_start):
                        with open(timeline_csv, "a", encoding="utf-8") as f:
                            f.write(f"{it},{now_ms(t0_wall):.3f},fwd_start,{i},nograd\n")
                        ms, hidden = measure_cuda(lambda: call_layer(layers[i], hidden, attention_mask, position_ids, position_embeddings))
                        fwd_ms[i] = ms
                        with open(timeline_csv, "a", encoding="utf-8") as f:
                            f.write(f"{it},{now_ms(t0_wall):.3f},fwd_end,{i},ms={ms:.3f}\n")
                hidden = hidden.detach().requires_grad_(True)

            # trainable suffix (maybe checkpointed)
            for i in range(train_start, use_layers):
                ckpted = (use_ckpt and i >= ckpt_start)
                with open(timeline_csv, "a", encoding="utf-8") as f:
                    f.write(f"{it},{now_ms(t0_wall):.3f},fwd_start,{i},ckpt={int(ckpted)}\n")

                def fn(h):
                    return call_layer(layers[i], h, attention_mask, position_ids, position_embeddings)

                if ckpted:
                    # use_reentrant=False is recommended on modern torch
                    ms, hidden = measure_cuda(lambda: ckpt(fn, hidden, use_reentrant=False))
                else:
                    ms, hidden = measure_cuda(lambda: fn(hidden))

                fwd_ms[i] = ms
                with open(timeline_csv, "a", encoding="utf-8") as f:
                    f.write(f"{it},{now_ms(t0_wall):.3f},fwd_end,{i},ms={ms:.3f}\n")

            if norm is not None:
                hidden = norm(hidden)

            # toy loss (same as sf)
            loss = hidden.float().pow(2).mean()

        forward_s = time.perf_counter() - t_fwd0
        logger.log(f"[FWD] forward_s={forward_s:.3f} loss={loss.item():.6f}")

        # backward
        t_bwd0 = time.perf_counter()
        for opt in opts:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.cuda.synchronize()
        backward_s = time.perf_counter() - t_bwd0

        # resolve per-layer bwd ms
        if has_pre:
            try:
                for i in range(use_layers):
                    if i in bwd_start_ev and i in bwd_end_ev:
                        bwd_ms[i] = float(bwd_start_ev[i].elapsed_time(bwd_end_ev[i]))
            except Exception:
                pass

        # per-layer optimizer step timing
        opt_ms = {i: 0.0 for i in range(use_layers)}
        t_opt0 = time.perf_counter()
        for layer_i, opt in zip(opt_layer_ids, opts):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            opt.step()
            e.record()
            torch.cuda.synchronize()
            opt_ms[layer_i] = float(s.elapsed_time(e))
            with open(timeline_csv, "a", encoding="utf-8") as f:
                f.write(f"{it},{now_ms(t0_wall):.3f},optim_step_end,{layer_i},ms={opt_ms[layer_i]:.3f}\n")
        optim_step_total_s = time.perf_counter() - t_opt0

        torch.cuda.synchronize()
        wall_s = time.perf_counter() - t0_wall
        peak_alloc = gib(torch.cuda.max_memory_allocated())
        peak_reserved = gib(torch.cuda.max_memory_reserved())
        rss = cpu_rss_gib()

        # stop monitor and dump mem samples
        mon.stop()
        with open(mem_csv, "a", encoding="utf-8") as f:
            for (tms, cpu, ga, gr) in mon.samples:
                f.write(f"{it},{tms:.3f},{cpu:.6f},{ga:.6f},{gr:.6f}\n")

        logger.log(f"[{mode_name.upper()}][ITER {it}] wall_s={wall_s:.3f} forward_s={forward_s:.3f} backward_s={backward_s:.3f} "
                   f"optim_step_total_s={optim_step_total_s:.3f} peak_alloc={peak_alloc:.2f}GiB peak_reserved={peak_reserved:.2f}GiB CPU_RSS={rss:.2f}GiB")

        with open(metrics_iter, "a", encoding="utf-8") as f:
            f.write(f"{it},{wall_s:.6f},{forward_s:.6f},{backward_s:.6f},{optim_step_total_s:.6f},{peak_alloc:.6f},{peak_reserved:.6f},{rss:.6f}\n")

        with open(metrics_layer, "a", encoding="utf-8") as f:
            for i in range(use_layers):
                trainable = 1 if i >= train_start else 0
                ckpted = 1 if (use_ckpt and i >= ckpt_start and i >= train_start) else 0
                f.write(f"{it},{i},{trainable},{ckpted},{float(fwd_ms.get(i,0.0)):.6f},{float(bwd_ms.get(i,0.0)):.6f},{float(opt_ms.get(i,0.0)):.6f}\n")

        # plots for this iter
        if args.plot:
            xs = list(range(use_layers))
            fwd_line = [float(fwd_ms.get(i, 0.0)) for i in xs]
            bwd_line = [float(bwd_ms.get(i, 0.0)) for i in xs]
            opt_line = [float(opt_ms.get(i, 0.0)) for i in xs]

            bar_save(os.path.join(run_dir, f"report_time_breakdown_iter{it}.png"),
                     f"{mode_name.upper()} Time Breakdown (one iter)", "component", "seconds",
                     ["forward_s", "backward_s", "optim_step_total_s", "wall_s"],
                     [forward_s, backward_s, optim_step_total_s, wall_s])

            plot_save(os.path.join(run_dir, f"report_layer_fwd_iter{it}.png"),
                      f"{mode_name.upper()} Layer Forward Time",
                      "layer index", "milliseconds",
                      xs, [fwd_line], ["fwd_ms"])

            plot_save(os.path.join(run_dir, f"report_layer_bwd_iter{it}.png"),
                      f"{mode_name.upper()} Layer Backward Time (includes recompute if checkpointed)",
                      "layer index", "milliseconds",
                      xs, [bwd_line], ["bwd_ms"])

            plot_save(os.path.join(run_dir, f"report_layer_optim_iter{it}.png"),
                      f"{mode_name.upper()} Layer Optimizer Step Time",
                      "layer index", "milliseconds",
                      xs, [opt_line], ["optim_step_ms"])

            # memory timeline: 2 lines (CPU RSS, GPU reserved)
            if len(mon.samples) > 2:
                tms = [x[0] for x in mon.samples]
                cpu_line = [x[1] for x in mon.samples]
                gpu_line = [x[3] for x in mon.samples]  # reserved
                plot_save(os.path.join(run_dir, f"report_mem_timeline_iter{it}.png"),
                          f"{mode_name.upper()} Memory Timeline (CPU RSS vs GPU Reserved)",
                          "time (ms)", "GiB",
                          tms, [cpu_line, gpu_line], ["cpu_rss_gib", "gpu_reserved_gib"])

    logger.log(f"[SAVED] {metrics_iter}")
    logger.log(f"[SAVED] {metrics_layer}")
    logger.log(f"[SAVED] {mem_csv}")
    logger.log(f"[SAVED] {timeline_csv}")


# ----------------------------
# compare report
# ----------------------------
def read_csv_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            rows.append((header, parts))
    return rows

def load_last_iter_metrics(run_dir):
    # metrics_iter.csv last row
    path = os.path.join(run_dir, "metrics_iter.csv")
    with open(path, "r", encoding="utf-8") as f:
        hdr = f.readline().strip().split(",")
        last = None
        for line in f:
            if line.strip():
                last = line.strip().split(",")
    if last is None:
        return None
    d = {k: float(v) if k != "iter" else int(float(v)) for k, v in zip(hdr, last)}
    return d

def load_layer_metrics(run_dir):
    path = os.path.join(run_dir, "metrics_layer.csv")
    with open(path, "r", encoding="utf-8") as f:
        hdr = f.readline().strip().split(",")
        rows = []
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split(",")
            row = {}
            for k, v in zip(hdr, parts):
                if k in ["iter", "layer", "stage", "trainable", "checkpointed"]:
                    row[k] = int(float(v))
                else:
                    try:
                        row[k] = float(v)
                    except Exception:
                        row[k] = v
            rows.append(row)
    return rows

def load_mem_trace(run_dir):
    path = os.path.join(run_dir, "mem_trace.csv")
    t, cpu, ga, gr = [], [], [], []
    with open(path, "r", encoding="utf-8") as f:
        _ = f.readline()
        for line in f:
            if not line.strip():
                continue
            it, tms, c, a, r = line.strip().split(",")
            t.append(float(tms))
            cpu.append(float(c))
            ga.append(float(a))
            gr.append(float(r))
    return t, cpu, ga, gr

def make_compare_report(compare_dir, run_infos):
    os.makedirs(compare_dir, exist_ok=True)

    # summary csv
    summary_csv = os.path.join(compare_dir, "compare_summary.csv")
    with open(summary_csv, "w", encoding="utf-8") as f:
        f.write("mode,run_dir,wall_s,forward_s,backward_s,update_s,peak_alloc_gib,peak_reserved_gib,cpu_rss_gib\n")
        for mode, rd in run_infos:
            it = load_last_iter_metrics(rd)
            if it is None:
                continue
            update_s = it.get("cpu_update_total_s", it.get("optim_step_total_s", 0.0))
            f.write(f"{mode},{rd},{it.get('wall_s',0):.6f},{it.get('forward_s',0):.6f},{it.get('backward_s',0):.6f},{update_s:.6f},"
                    f"{it.get('peak_alloc_gib',0):.6f},{it.get('peak_reserved_gib',0):.6f},{it.get('cpu_rss_gib',0):.6f}\n")

    # time breakdown bar
    modes = []
    wall = []
    fwd = []
    bwd = []
    upd = []
    peak_gpu = []
    peak_cpu = []
    for mode, rd in run_infos:
        it = load_last_iter_metrics(rd)
        if it is None:
            continue
        modes.append(mode)
        wall.append(it.get("wall_s", 0.0))
        fwd.append(it.get("forward_s", 0.0))
        bwd.append(it.get("backward_s", 0.0))
        upd.append(it.get("cpu_update_total_s", it.get("optim_step_total_s", 0.0)))
        peak_gpu.append(it.get("peak_reserved_gib", 0.0))
        peak_cpu.append(it.get("cpu_rss_gib", 0.0))

    bar_save(os.path.join(compare_dir, "compare_time_breakdown.png"),
             "Compare: One-Iter Time Breakdown", "mode", "seconds",
             [f"{m}\n(wall)" for m in modes], wall)

    bar_save(os.path.join(compare_dir, "compare_peak_gpu_reserved.png"),
             "Compare: Peak GPU Reserved", "mode", "GiB",
             modes, peak_gpu)

    bar_save(os.path.join(compare_dir, "compare_cpu_rss.png"),
             "Compare: CPU RSS (end of iter)", "mode", "GiB",
             modes, peak_cpu)

    # memory timeline overlay (gpu_reserved + cpu_rss)
    # (different durations; just plot each as-is)
    plt.figure(figsize=(14, 6))
    for mode, rd in run_infos:
        t, cpu, ga, gr = load_mem_trace(rd)
        if len(t) < 2:
            continue
        plt.plot(t, gr, label=f"{mode}_gpu_reserved")
    plt.title("Compare: GPU Reserved vs Time")
    plt.xlabel("time (ms)")
    plt.ylabel("GiB")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(compare_dir, "compare_gpu_reserved_timeline.png"), dpi=170)
    plt.close()

    plt.figure(figsize=(14, 6))
    for mode, rd in run_infos:
        t, cpu, ga, gr = load_mem_trace(rd)
        if len(t) < 2:
            continue
        plt.plot(t, cpu, label=f"{mode}_cpu_rss")
    plt.title("Compare: CPU RSS vs Time")
    plt.xlabel("time (ms)")
    plt.ylabel("GiB")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(compare_dir, "compare_cpu_rss_timeline.png"), dpi=170)
    plt.close()

    # per-layer comparison (fwd + bwd/recompute + update)
    # read only iter==0 rows for simplicity (or last iter)
    def layer_vec(mode, rd, key_candidates, use_iter=None):
        rows = load_layer_metrics(rd)
        if not rows:
            return None
        # pick max iter in file
        max_it = max(r["iter"] for r in rows)
        tgt_it = max_it if use_iter is None else use_iter
        rows = [r for r in rows if r["iter"] == tgt_it]
        max_layer = max(r["layer"] for r in rows)
        vec = [0.0] * (max_layer + 1)
        for r in rows:
            for k in key_candidates:
                if k in r:
                    vec[r["layer"]] = float(r[k])
                    break
        return vec

    # fwd
    plt.figure(figsize=(14, 6))
    for mode, rd in run_infos:
        v = layer_vec(mode, rd, ["fwd_ms"])
        if v is None:
            continue
        plt.plot(list(range(len(v))), v, label=mode)
    plt.title("Compare: Layer Forward Time")
    plt.xlabel("layer index")
    plt.ylabel("milliseconds")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(compare_dir, "compare_layer_fwd.png"), dpi=170)
    plt.close()

    # bwd (gpu modes) / recompute (sf)
    plt.figure(figsize=(14, 6))
    for mode, rd in run_infos:
        if mode == "sf":
            v = layer_vec(mode, rd, ["recompute_ms"])
        else:
            v = layer_vec(mode, rd, ["bwd_ms"])
        if v is None:
            continue
        plt.plot(list(range(len(v))), v, label=mode)
    plt.title("Compare: Layer Backward Cost (gpu: bwd_ms, sf: recompute_ms)")
    plt.xlabel("layer index")
    plt.ylabel("milliseconds")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(compare_dir, "compare_layer_bwd_or_recompute.png"), dpi=170)
    plt.close()

    # update (gpu: optim_step_ms, sf: cpu_update_ms)
    plt.figure(figsize=(14, 6))
    for mode, rd in run_infos:
        if mode == "sf":
            v = layer_vec(mode, rd, ["cpu_update_ms"])
        else:
            v = layer_vec(mode, rd, ["optim_step_ms"])
        if v is None:
            continue
        plt.plot(list(range(len(v))), v, label=mode)
    plt.title("Compare: Layer Update Cost (gpu: optim_step_ms, sf: cpu_update_ms)")
    plt.xlabel("layer index")
    plt.ylabel("milliseconds")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(compare_dir, "compare_layer_update.png"), dpi=170)
    plt.close()

    # total per layer (rough): fwd + bwd/recompute + update (+ prefetch for sf)
    plt.figure(figsize=(14, 6))
    for mode, rd in run_infos:
        fwdv = layer_vec(mode, rd, ["fwd_ms"]) or []
        if mode == "sf":
            bwdv = layer_vec(mode, rd, ["recompute_ms"]) or []
            updv = layer_vec(mode, rd, ["cpu_update_ms"]) or []
            prefv = layer_vec(mode, rd, ["prefetch_ms"]) or [0.0] * len(fwdv)
            L = min(len(fwdv), len(bwdv), len(updv), len(prefv))
            tot = [prefv[i] + fwdv[i] + bwdv[i] + updv[i] for i in range(L)]
        else:
            bwdv = layer_vec(mode, rd, ["bwd_ms"]) or [0.0] * len(fwdv)
            updv = layer_vec(mode, rd, ["optim_step_ms"]) or [0.0] * len(fwdv)
            L = min(len(fwdv), len(bwdv), len(updv))
            tot = [fwdv[i] + bwdv[i] + updv[i] for i in range(L)]
        if tot:
            plt.plot(list(range(len(tot))), tot, label=mode)
    plt.title("Compare: Layer Total Cost (rough sum)")
    plt.xlabel("layer index")
    plt.ylabel("milliseconds")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(compare_dir, "compare_layer_total.png"), dpi=170)
    plt.close()

    with open(os.path.join(compare_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("# Compare Report\n\n")
        f.write("This folder compares multiple modes for T1-M1.\n\n")
        f.write("- `compare_summary.csv`: final-iter summary per mode\n")
        f.write("- `compare_*png`: overlay plots\n")

# ----------------------------
# main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="./qwen")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--iters", type=int, default=2)

    ap.add_argument("--layers", type=int, default=0, help="0=all; else use first N layers")
    ap.add_argument("--train_layers", type=int, default=0, help="0=all used layers; else train only last N layers")

    # modes
    ap.add_argument("--modes", type=str, default="gpu,gpu_ckpt,sf",
                    help="comma-separated: gpu,gpu_ckpt,sf")

    # checkpoint control (gpu_ckpt)
    ap.add_argument("--ckpt_layers", type=int, default=0,
                    help="0=all trainable layers; else checkpoint only last K trainable layers")

    # slideformer controls
    ap.add_argument("--sf_prefetch", type=int, default=1, help="1=enable H2D prefetch in sf")
    ap.add_argument("--pin_mode", type=str, default="lazy", choices=["off", "lazy"])
    ap.add_argument("--cpu_workers", type=int, default=4, help="CPU threads for sf optimizer update")

    # optimizer
    ap.add_argument("--optim", type=str, default="adamw", choices=["sgd", "adamw"])
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--betas", type=str, default="0.9,0.999")
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--adam_state_dtype", type=str, default="bf16", choices=["bf16", "fp32"])

    # logging/report
    ap.add_argument("--logdir", type=str, default="runs_m1")
    ap.add_argument("--plot", type=int, default=1)
    ap.add_argument("--sample_ms", type=int, default=25, help="memory sampling interval (ms)")

    # dtype
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])

    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA is required for this suite."
    os.makedirs(args.logdir, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    run_infos = []
    for m in modes:
        run_dir = os.path.join(args.logdir, f"{stamp}_{m}_seq{args.seq}_bs{args.batch}_iters{args.iters}_layers{args.layers or 'all'}_train{args.train_layers or 'all'}")
        os.makedirs(run_dir, exist_ok=True)
        logger = Logger(os.path.join(run_dir, "log.txt"))
        logger.log(f"[BOOT] T1M1 suite start mode={m}")
        logger.log(f"[RUN_DIR] {run_dir}")

        try:
            if m == "sf":
                run_slideformer(args, run_dir, logger)
            elif m == "gpu":
                run_gpu_mode(args, run_dir, logger, use_ckpt=False)
            elif m == "gpu_ckpt":
                run_gpu_mode(args, run_dir, logger, use_ckpt=True)
            else:
                raise RuntimeError(f"Unknown mode={m}")
            run_infos.append((m, run_dir))
        finally:
            logger.close()

    # compare report
    if len(run_infos) >= 2:
        compare_dir = os.path.join(args.logdir, f"compare_{stamp}")
        make_compare_report(compare_dir, run_infos)
        print(f"[COMPARE_SAVED] {compare_dir}", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        msg = "[CRASH] " + repr(e) + "\n" + traceback.format_exc()
        print(msg, flush=True)
        raise
