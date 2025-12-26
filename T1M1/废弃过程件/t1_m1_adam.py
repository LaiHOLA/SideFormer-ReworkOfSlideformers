#full
#python t1_m1_pipeline.py --model ./qwen --seq 256 --batch 3 --iters 1 --layers 28 --train_layers 28 --optim adamw --adam_state_dtype bf16 --lr 1e-4 --logdir runs_m1
#8 layers validation
#python t1_m1_pipeline.py --model ./qwen --seq 256 --batch 3 --iters 1 --layers 28 --train_layers 8 --optim adamw --adam_state_dtype bf16 --lr 1e-4 --logdir runs_m1


import os, sys, time, json, copy, argparse, traceback, inspect, math
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import psutil
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM


def gib(x): return x / (1024 ** 3)

def cpu_rss_gib():
    p = psutil.Process(os.getpid())
    return p.memory_info().rss / (1024 ** 3)


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
    # return (start_event, end_event, bytes_copied, params_copied)
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


def measure_cuda(fn):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    out = fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e), out


def call_layer(layer, hidden_states, attention_mask, position_ids, position_embeddings, logger: Logger):
    sig = None
    try:
        sig = inspect.signature(layer.forward)
    except Exception:
        sig = None

    kwargs = {}
    if sig is not None:
        params = sig.parameters
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
        # progressively drop kwargs for compatibility
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


def param_bytes(layer):
    b = 0
    for _, p in layer.named_parameters(recurse=True):
        if p is None:
            continue
        b += p.numel() * p.element_size()
    return b


def ensure_state_tensor(shape, dtype, pinned=False):
    t = torch.zeros(shape, dtype=dtype, device="cpu")
    if pinned:
        t = t.pin_memory()
    return t


def cpu_sgd_update(layer_idx, cpu_layer, grads_cpu: dict, lr: float):
    t0 = time.perf_counter()
    name2p = dict(cpu_layer.named_parameters(recurse=True))
    for name, g in grads_cpu.items():
        p = name2p.get(name, None)
        if p is None or g is None:
            continue
        # in-place update; keep pinned buffer
        if g.dtype != p.data.dtype:
            g = g.to(dtype=p.data.dtype)
        p.data.add_(g, alpha=-lr)
    t1 = time.perf_counter()
    return layer_idx, (t1 - t0)


def cpu_adamw_update(layer_idx, cpu_layer, grads_cpu: dict, lr: float,
                     state: dict, beta1: float, beta2: float, eps: float,
                     weight_decay: float, state_dtype: str):
    """
    AdamW on CPU. state is per-layer dict:
      state[name] = {"m": tensor, "v": tensor, "step": int}
    To keep memory under control, state is allocated lazily per param when first seen.
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
            st = {
                "m": ensure_state_tensor(p.data.shape, dt, pinned=False),
                "v": ensure_state_tensor(p.data.shape, dt, pinned=False),
                "step": 0
            }
            state[name] = st

        st["step"] += 1
        step = st["step"]

        # compute in fp32 for stability
        g32 = g.float()
        p32 = p.data.float()
        m32 = st["m"].float()
        v32 = st["v"].float()

        if weight_decay != 0.0:
            g32 = g32.add(p32, alpha=weight_decay)

        m32.mul_(beta1).add_(g32, alpha=1.0 - beta1)
        v32.mul_(beta2).addcmul_(g32, g32, value=1.0 - beta2)

        # bias correction
        bc1 = 1.0 - (beta1 ** step)
        bc2 = 1.0 - (beta2 ** step)
        mhat = m32 / bc1
        vhat = v32 / bc2

        upd = mhat / (vhat.sqrt().add_(eps))

        # apply update to original p (keep pinned memory)
        p32.add_(upd, alpha=-lr)
        p.data.copy_(p32.to(dtype=p.data.dtype))

        # write back states
        if st["m"].dtype != torch.float32:
            st["m"].copy_(m32.to(st["m"].dtype))
            st["v"].copy_(v32.to(st["v"].dtype))
        else:
            st["m"].copy_(m32)
            st["v"].copy_(v32)

    t1 = time.perf_counter()
    return layer_idx, (t1 - t0)


def plot_save(path, title, xlabel, ylabel, xs, ys_list, labels):
    plt.figure(figsize=(13, 6))
    for ys, lb in zip(ys_list, labels):
        plt.plot(xs, ys, label=lb)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def bar_save(path, title, xlabel, ylabel, labels, values):
    plt.figure(figsize=(13, 6))
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="./qwen")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--layers", type=int, default=0, help="0=all; else first N layers (forward depth)")
    ap.add_argument("--train_layers", type=int, default=0, help="0=all used layers; else train only last N layers (selective recompute+update)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--logdir", type=str, default="runs_m1")
    ap.add_argument("--prefetch", type=int, default=1)
    ap.add_argument("--pin_mode", type=str, default="lazy", choices=["off", "lazy"])

    ap.add_argument("--optim", type=str, default="sgd", choices=["sgd", "adamw"])
    ap.add_argument("--betas", type=str, default="0.9,0.999")
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--adam_state_dtype", type=str, default="bf16", choices=["bf16", "fp32"])

    ap.add_argument("--plot", type=int, default=1, help="1=generate report plots")
    args = ap.parse_args()

    print("[BOOT] t1_m1_pipeline start", flush=True)
    assert torch.cuda.is_available(), "CUDA is required for T1-M1."

    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.logdir, f"{stamp}_T1M1_seq{args.seq}_bs{args.batch}_iters{args.iters}_layers{args.layers or 'all'}_train{args.train_layers or 'all'}_{args.optim}")
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
        dtype=dtype,
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
    if rotary is None:
        logger.log("[WARN] rotary_emb not found. Qwen3 may fail without position_embeddings.")
    else:
        rotary = rotary.to(device=device)
        rotary.eval()
        logger.log(f"[ROTARY] found: {type(rotary).__name__}, moved to cuda")

    total_layers = len(layers)
    use_layers = total_layers if args.layers <= 0 else min(args.layers, total_layers)

    train_layers = use_layers if args.train_layers <= 0 else min(args.train_layers, use_layers)
    train_start = use_layers - train_layers  # only train [train_start, use_layers)
    logger.log(f"[CFG] total_layers={total_layers}, use_layers={use_layers}, train_layers={train_layers} (range {train_start}..{use_layers-1}), pin_mode={args.pin_mode}, prefetch={args.prefetch}, optim={args.optim}")

    # estimate CPU memory footprint
    train_param_b = sum(param_bytes(layers[i]) for i in range(train_start, use_layers))
    if args.optim == "adamw":
        # m + v
        state_factor = 2.0
        state_dtype_bytes = 4 if args.adam_state_dtype == "fp32" else (2 if dtype == torch.bfloat16 else 2)
        # rough: states follow state dtype, not param dtype
        est_state_gib = gib(train_param_b / (dtype.itemsize) * state_dtype_bytes * state_factor) if dtype.itemsize else 0.0
        logger.log(f"[EST] train_param~{gib(train_param_b):.2f}GiB, adam_states(m+v)~{est_state_gib:.2f}GiB (state_dtype={args.adam_state_dtype})")
        if args.adam_state_dtype == "fp32":
            logger.log("[WARN] adam_state_dtype=fp32 may push CPU RAM high on 32GB. If OOM, switch to bf16 or reduce --train_layers.")
    else:
        logger.log(f"[EST] train_param~{gib(train_param_b):.2f}GiB, optim=sgd has no extra state")

    embed = embed.to(device=device, dtype=dtype)
    if norm is not None:
        norm = norm.to(device=device, dtype=dtype)

    logger.log("[STAGE] building 2 GPU staging layers (ping-pong) ...")
    stage0 = copy.deepcopy(layers[0]).to(device=device, dtype=dtype).eval()
    stage1 = copy.deepcopy(layers[0]).to(device=device, dtype=dtype).eval()
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

    def build_ids():
        text = ("SlideFormer layer-streaming proof. " * 200).strip()
        ids = tok(text, return_tensors="pt", truncation=True, max_length=args.seq).input_ids
        if ids.size(1) < args.seq:
            pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0
            pad = torch.full((1, args.seq - ids.size(1)), pad_id, dtype=ids.dtype)
            ids = torch.cat([ids, pad], dim=1)
        ids = ids[:, :args.seq].repeat(args.batch, 1)
        return ids

    metrics_iter = os.path.join(run_dir, "metrics_iter.csv")
    with open(metrics_iter, "w", encoding="utf-8") as f:
        f.write("iter,wall_s,forward_s,backward_s,cpu_update_total_s,peak_alloc_gib,peak_reserved_gib,cpu_rss_gib,total_prefetch_ms,total_fwd_ms,total_recompute_ms\n")

    metrics_layer = os.path.join(run_dir, "metrics_layer.csv")
    with open(metrics_layer, "w", encoding="utf-8") as f:
        f.write("iter,layer,stage,trainable,prefetch_ms,fwd_ms,recompute_ms,grad_offload_gib,cpu_update_ms\n")

    # optimizer states per layer (CPU)
    betas = [float(x.strip()) for x in args.betas.split(",")]
    beta1, beta2 = betas[0], betas[1]
    adam_state = [dict() for _ in range(use_layers)]  # per-layer dict

    for it in range(args.iters):
        logger.log(f"\n========== [ITER {it}] ==========")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        t0_wall = time.perf_counter()

        ids = build_ids().to(device, non_blocking=True)
        bs, seqlen = ids.shape
        position_ids = torch.arange(seqlen, device=device, dtype=torch.long).unsqueeze(0).expand(bs, -1)
        attn_2d = torch.ones((bs, seqlen), device=device, dtype=torch.long)

        # record per-layer stats
        prefetch_events = {}   # layer -> (start,end,bytes,params)
        fwd_ms_by_layer = {}
        recompute_ms_by_layer = {}
        grad_offload_gib_by_layer = {i: 0.0 for i in range(use_layers)}
        cpu_update_ms_by_layer = {i: 0.0 for i in range(use_layers)}

        # ---- Forward streaming (always no_grad to avoid graph with overwritten staging weights) ----
        t_fwd0 = time.perf_counter()

        maybe_pin(0)
        s0, e0, b0, _ = prefetch_layer_to_stage(layers[0], stage_param_maps[0], stream_prefetch, ev_done[0], logger, tag=" L00->S0")
        prefetch_events[0] = (s0, e0, b0)

        torch.cuda.current_stream().wait_event(ev_done[0])

        if args.prefetch and use_layers > 1:
            maybe_pin(1)
            s1, e1, b1, _ = prefetch_layer_to_stage(layers[1], stage_param_maps[1], stream_prefetch, ev_done[1], logger, tag=" L01->S1")
            prefetch_events[1] = (s1, e1, b1)

        act_cpu = {}   # only for trainable layers
        act_ev = {}

        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            hidden = embed(ids)
            h_shape = hidden.shape
            logger.log(f"[FWD] hidden_shape={tuple(h_shape)} dtype={hidden.dtype}")

            attention_mask = try_build_4d_causal_mask(attn_2d, hidden, logger)

            if rotary is not None:
                try:
                    position_embeddings = rotary(hidden, position_ids)
                except TypeError:
                    position_embeddings = rotary(position_ids)
                ok = isinstance(position_embeddings, (tuple, list)) and len(position_embeddings) == 2
                if ok:
                    c0, s0_ = position_embeddings
                    logger.log(f"[POS] position_embeddings OK cos={tuple(c0.shape)} sin={tuple(s0_.shape)} dtype={c0.dtype}")
                else:
                    logger.log(f"[WARN] rotary() returned unexpected type={type(position_embeddings)}")
            else:
                position_embeddings = None
                logger.log("[POS] position_embeddings=None (rotary not found)")

            for i in range(use_layers):
                sidx = i % 2
                torch.cuda.current_stream().wait_event(ev_done[sidx])

                # only save activations for trainable range
                if i >= train_start:
                    cpu_buf = torch.empty(h_shape, device="cpu", dtype=hidden.dtype, pin_memory=True)
                    ev = torch.cuda.Event(enable_timing=False)
                    with torch.cuda.stream(stream_offload):
                        cpu_buf.copy_(hidden, non_blocking=True)
                        ev.record(stream_offload)
                    act_cpu[i] = cpu_buf
                    act_ev[i] = ev

                layer_mod = stage0 if sidx == 0 else stage1
                ms, hidden = measure_cuda(lambda: call_layer(layer_mod, hidden, attention_mask, position_ids, position_embeddings, logger))
                fwd_ms_by_layer[i] = ms
                logger.log(f"[FWD][L{i:02d}] stage={sidx} fwd_ms={ms:.3f} trainable={1 if i>=train_start else 0}")

                if args.prefetch and (i + 1) < use_layers:
                    nidx = i + 1
                    dst_stage = nidx % 2
                    maybe_pin(nidx)
                    ss, ee, bb, _ = prefetch_layer_to_stage(layers[nidx], stage_param_maps[dst_stage], stream_prefetch, ev_done[dst_stage], logger, tag=f" L{nidx:02d}->S{dst_stage}")
                    prefetch_events[nidx] = (ss, ee, bb)

            if norm is not None:
                hidden = norm(hidden)

        t_fwd1 = time.perf_counter()
        forward_s = t_fwd1 - t_fwd0
        logger.log(f"[FWD] forward_s={forward_s:.3f} acts_saved(trainable)={len(act_cpu)} / {train_layers}")

        # ---- gradient seed from a simple loss (keeps pipeline minimal and stable) ----
        hidden_leaf = hidden.detach().requires_grad_(True)
        with torch.autocast("cuda", dtype=dtype):
            loss = hidden_leaf.float().pow(2).mean()
        g = torch.autograd.grad(loss, hidden_leaf, retain_graph=False, create_graph=False)[0].detach()
        logger.log(f"[LOSS] loss={loss.item():.6f} grad_norm={float(g.float().norm().item()):.6f}")

        # ---- Backward streaming: only for trainable layers ----
        t_bwd0 = time.perf_counter()
        executor = ThreadPoolExecutor(max_workers=1)
        pending = []
        cpu_update_total = 0.0

        for i in range(use_layers - 1, train_start - 1, -1):
            sidx = i % 2
            # ensure the activation we offloaded is ready
            act_ev[i].synchronize()
            torch.cuda.current_stream().wait_event(ev_done[sidx])

            h_in = act_cpu[i].to(device, non_blocking=True).detach().requires_grad_(True)
            layer_mod = stage0 if sidx == 0 else stage1

            with torch.autocast("cuda", dtype=dtype):
                rec_ms, h_out = measure_cuda(lambda: call_layer(layer_mod, h_in, attention_mask, position_ids, position_embeddings, logger))
            recompute_ms_by_layer[i] = rec_ms

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

            if args.optim == "sgd":
                pending.append(executor.submit(cpu_sgd_update, i, layers[i], grads_cpu, args.lr))
            else:
                pending.append(executor.submit(
                    cpu_adamw_update, i, layers[i], grads_cpu, args.lr,
                    adam_state[i], beta1, beta2, args.eps, args.weight_decay,
                    "fp32" if args.adam_state_dtype == "fp32" else "bf16"
                ))

            logger.log(f"[BWD][L{i:02d}] stage={sidx} recompute_ms={rec_ms:.3f} grad_offload~{gib(bytes_grad):.3f}GiB pending_cpu_updates={len(pending)}")

            if args.prefetch and (i - 1) >= 0:
                prev = i - 1
                dst_stage = prev % 2
                maybe_pin(prev)
                ss, ee, bb, _ = prefetch_layer_to_stage(layers[prev], stage_param_maps[dst_stage], stream_prefetch, ev_done[dst_stage], logger, tag=f" L{prev:02d}->S{dst_stage}")
                prefetch_events[prev] = (ss, ee, bb)

            del h_in, h_out, grads, grads_cpu

        for fut in as_completed(pending):
            layer_idx, upd_s = fut.result()
            cpu_update_total += upd_s
            cpu_update_ms_by_layer[layer_idx] = upd_s * 1000.0
            logger.log(f"[CPU_UPDATE][L{layer_idx:02d}] update_s={upd_s:.4f}")

        executor.shutdown(wait=True)
        t_bwd1 = time.perf_counter()
        backward_s = t_bwd1 - t_bwd0

        # finalize iteration stats
        torch.cuda.synchronize()
        wall_s = time.perf_counter() - t0_wall
        peak_alloc = gib(torch.cuda.max_memory_allocated())
        peak_reserved = gib(torch.cuda.max_memory_reserved())
        rss = cpu_rss_gib()

        # compute prefetch ms
        total_prefetch_ms = 0.0
        for li, (ss, ee, bb) in prefetch_events.items():
            try:
                ms = ss.elapsed_time(ee)
            except Exception:
                ms = 0.0
            total_prefetch_ms += ms

        total_fwd_ms = sum(fwd_ms_by_layer.get(i, 0.0) for i in range(use_layers))
        total_recompute_ms = sum(recompute_ms_by_layer.get(i, 0.0) for i in range(train_start, use_layers))

        logger.log(f"[ITER {it}] wall_s={wall_s:.3f} forward_s={forward_s:.3f} backward_s={backward_s:.3f} cpu_update_total_s={cpu_update_total:.3f} "
                   f"peak_alloc={peak_alloc:.2f}GiB peak_reserved={peak_reserved:.2f}GiB CPU_RSS={rss:.2f}GiB "
                   f"prefetch_ms(sum)={total_prefetch_ms:.1f} fwd_ms(sum)={total_fwd_ms:.1f} recompute_ms(sum)={total_recompute_ms:.1f}")

        with open(metrics_iter, "a", encoding="utf-8") as f:
            f.write(f"{it},{wall_s:.6f},{forward_s:.6f},{backward_s:.6f},{cpu_update_total:.6f},{peak_alloc:.6f},{peak_reserved:.6f},{rss:.6f},{total_prefetch_ms:.6f},{total_fwd_ms:.6f},{total_recompute_ms:.6f}\n")

        with open(metrics_layer, "a", encoding="utf-8") as f:
            for i in range(use_layers):
                sidx = i % 2
                trainable = 1 if i >= train_start else 0
                # prefetch ms may be missing for some layers depending on scheduling
                pm = 0.0
                if i in prefetch_events:
                    ss, ee, _ = prefetch_events[i]
                    try:
                        pm = ss.elapsed_time(ee)
                    except Exception:
                        pm = 0.0
                fm = fwd_ms_by_layer.get(i, 0.0)
                rm = recompute_ms_by_layer.get(i, 0.0) if trainable else 0.0
                go = grad_offload_gib_by_layer.get(i, 0.0) if trainable else 0.0
                cu = cpu_update_ms_by_layer.get(i, 0.0) if trainable else 0.0
                f.write(f"{it},{i},{sidx},{trainable},{pm:.6f},{fm:.6f},{rm:.6f},{go:.6f},{cu:.6f}\n")

        # --- report plots (use this iter only) ---
        if args.plot:
            # read layer metrics for this iter from in-memory dicts
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
                        pm = ss.elapsed_time(ee)
                    except Exception:
                        pm = 0.0
                prefetch_ms.append(pm)
                fwd_ms.append(fwd_ms_by_layer.get(i, 0.0))
                recompute_ms.append(recompute_ms_by_layer.get(i, 0.0) if i >= train_start else 0.0)
                cpu_up_ms.append(cpu_update_ms_by_layer.get(i, 0.0) if i >= train_start else 0.0)

            bar_save(
                os.path.join(run_dir, "report_time_breakdown.png"),
                "T1-M1 Time Breakdown (one iter)",
                "component", "seconds",
                ["forward_s", "backward_s", "cpu_update_total_s", "wall_s"],
                [forward_s, backward_s, cpu_update_total, wall_s]
            )

            plot_save(
                os.path.join(run_dir, "report_layer_prefetch_vs_compute.png"),
                f"Layer Prefetch(H2D) vs Compute (train_start={train_start})",
                "layer index", "milliseconds",
                xs,
                [prefetch_ms, fwd_ms],
                ["prefetch_ms", "fwd_ms"]
            )

            plot_save(
                os.path.join(run_dir, "report_layer_bwd_recompute.png"),
                f"Layer Backward Recomputation Time (only trainable layers >= {train_start})",
                "layer index", "milliseconds",
                xs,
                [recompute_ms],
                ["recompute_ms"]
            )

            plot_save(
                os.path.join(run_dir, "report_layer_cpu_update.png"),
                f"CPU Optimizer Update Time per Layer (only trainable layers >= {train_start})",
                "layer index", "milliseconds",
                xs,
                [cpu_up_ms],
                ["cpu_update_ms"]
            )

    # write README
    readme = os.path.join(run_dir, "README.md")
    with open(readme, "w", encoding="utf-8") as f:
        f.write("# T1-M1 Run Report\n\n")
        f.write(f"- run_dir: `{run_dir}`\n")
        f.write(f"- model: `{args.model}`\n")
        f.write(f"- seq={args.seq}, batch={args.batch}, use_layers={use_layers}, train_layers={train_layers} (range {train_start}..{use_layers-1})\n")
        f.write(f"- optim={args.optim}, lr={args.lr}\n")
        if args.optim == "adamw":
            f.write(f"- adam_state_dtype={args.adam_state_dtype}, betas={args.betas}, eps={args.eps}, weight_decay={args.weight_decay}\n")
        f.write("\n## Outputs\n")
        f.write("- `log.txt`: pipeline logs\n")
        f.write("- `metrics_iter.csv`: per-iter summary\n")
        f.write("- `metrics_layer.csv`: per-layer timings (prefetch/fwd/recompute/update)\n")
        if args.plot:
            f.write("- `report_time_breakdown.png`\n")
            f.write("- `report_layer_prefetch_vs_compute.png`\n")
            f.write("- `report_layer_bwd_recompute.png`\n")
            f.write("- `report_layer_cpu_update.png`\n")
        f.write("\n## How to interpret\n")
        f.write("- Prefetch uses 2-stage ping-pong GPU buffers (S0/S1): only two layer weights reside on GPU at any moment.\n")
        f.write("- Activations of trainable layers are offloaded to CPU pinned memory; backward recomputes those layers and offloads grads to CPU.\n")
        f.write("- CPU optimizer update is overlapped with GPU recomputation (see logs and cpu_update_total_s).\n")

    logger.log(f"[SAVED] {metrics_iter}")
    logger.log(f"[SAVED] {metrics_layer}")
    logger.log(f"[SAVED] {readme}")
    logger.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        msg = "[CRASH] " + repr(e) + "\n" + traceback.format_exc()
        print(msg, flush=True)
        try:
            os.makedirs("runs_m1", exist_ok=True)
            with open(os.path.join("runs_m1", "crash.txt"), "w", encoding="utf-8") as f:
                f.write(msg)
        except Exception:
            pass
        raise
