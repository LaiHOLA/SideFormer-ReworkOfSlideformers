#python t1_m1_pipeline.py --model ./qwen --seq 256 --batch 1 --iters 1 --layers 28 --lr 1e-4 --logdir runs_m1

import os, sys, time, json, copy, argparse, traceback, inspect
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import psutil
import torch
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
    # Qwen3 通常是 core.rotary_emb（全层共享），不是 layer.self_attn.rotary_emb
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
    with torch.cuda.stream(stream):
        copied = 0
        bytes_copied = 0
        for name, p_cpu in cpu_layer.named_parameters(recurse=True):
            p_gpu = stage_param_map.get(name, None)
            if p_gpu is None:
                continue
            p_gpu.data.copy_(p_cpu.data, non_blocking=True)
            copied += 1
            bytes_copied += p_cpu.numel() * p_cpu.element_size()
        done_event.record(stream)
    logger.log(f"[PREFETCH]{tag} queued params={copied}, bytes~{gib(bytes_copied):.3f}GiB")


def measure_cuda(fn):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    out = fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e), out


def call_layer(layer, hidden_states, attention_mask, position_ids, position_embeddings, logger: Logger):
    """
    关键：把 position_embeddings=(cos,sin) 传给 Qwen3 layer.forward。
    """
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

    # 多层 fallback：尽量别因多余 kwargs 崩掉
    try:
        out = layer(hidden_states, **kwargs)
    except TypeError as e:
        # 逐步删参数再试
        if "position_embeddings" in kwargs:
            kwargs.pop("position_embeddings", None)
            try:
                out = layer(hidden_states, **kwargs)
            except TypeError:
                kwargs.pop("attention_mask", None)
                kwargs.pop("position_ids", None)
                out = layer(hidden_states)
        else:
            kwargs.pop("attention_mask", None)
            kwargs.pop("position_ids", None)
            out = layer(hidden_states)

    if isinstance(out, (tuple, list)):
        return out[0]
    return out


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
    t1 = time.perf_counter()
    return layer_idx, (t1 - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="./qwen")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--layers", type=int, default=0, help="0=all; else first N layers")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--logdir", type=str, default="runs_m1")
    ap.add_argument("--prefetch", type=int, default=1)
    ap.add_argument("--pin_mode", type=str, default="lazy", choices=["off", "lazy"])
    args = ap.parse_args()

    print("[BOOT] t1_m1_pipeline start", flush=True)
    assert torch.cuda.is_available(), "CUDA is required for T1-M1."

    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.logdir, f"{stamp}_T1M1_seq{args.seq}_bs{args.batch}_iters{args.iters}_layers{args.layers or 'all'}")
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
        logger.log("[WARN] rotary_emb not found on core/model. Qwen3 may crash without position_embeddings.")
    else:
        rotary = rotary.to(device=device)
        rotary.eval()
        logger.log(f"[ROTARY] found: {type(rotary).__name__}, moved to cuda")

    total_layers = len(layers)
    use_layers = total_layers if args.layers <= 0 else min(args.layers, total_layers)
    logger.log(f"[CFG] total_layers={total_layers}, use_layers={use_layers}, pin_mode={args.pin_mode}, prefetch={args.prefetch}")

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

    metrics_path = os.path.join(run_dir, "metrics.csv")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("iter,wall_s,peak_alloc_gib,peak_reserved_gib,cpu_rss_gib,forward_s,backward_s,cpu_update_total_s\n")

    for it in range(args.iters):
        logger.log(f"\n========== [ITER {it}] ==========")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        t0_wall = time.perf_counter()

        ids = build_ids().to(device, non_blocking=True)
        bs, seqlen = ids.shape
        position_ids = torch.arange(seqlen, device=device, dtype=torch.long).unsqueeze(0).expand(bs, -1)
        attn_2d = torch.ones((bs, seqlen), device=device, dtype=torch.long)

        # --- Forward streaming (no_grad) ---
        t_fwd0 = time.perf_counter()

        maybe_pin(0)
        prefetch_layer_to_stage(layers[0], stage_param_maps[0], stream_prefetch, ev_done[0], logger, tag=" L00->S0")
        torch.cuda.current_stream().wait_event(ev_done[0])

        if args.prefetch and use_layers > 1:
            maybe_pin(1)
            prefetch_layer_to_stage(layers[1], stage_param_maps[1], stream_prefetch, ev_done[1], logger, tag=" L01->S1")

        acts_cpu = []
        acts_ev = []

        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            hidden = embed(ids)
            h_shape = hidden.shape
            logger.log(f"[FWD] hidden_shape={tuple(h_shape)} dtype={hidden.dtype}")

            attention_mask = try_build_4d_causal_mask(attn_2d, hidden, logger)

            # 关键：一次性算好 (cos, sin)
            if rotary is not None:
                try:
                    position_embeddings = rotary(hidden, position_ids)
                except TypeError:
                    position_embeddings = rotary(position_ids)
                ok = isinstance(position_embeddings, (tuple, list)) and len(position_embeddings) == 2
                if not ok:
                    logger.log(f"[WARN] rotary() returned unexpected type={type(position_embeddings)}")
                else:
                    c0, s0 = position_embeddings
                    logger.log(f"[POS] position_embeddings OK cos={tuple(c0.shape)} sin={tuple(s0.shape)} dtype={c0.dtype}")
            else:
                position_embeddings = None
                logger.log("[POS] position_embeddings=None (rotary not found)")

            for i in range(use_layers):
                sidx = i % 2
                torch.cuda.current_stream().wait_event(ev_done[sidx])

                # offload activation to CPU pinned
                act_cpu = torch.empty(h_shape, device="cpu", dtype=hidden.dtype, pin_memory=True)
                ev = torch.cuda.Event(enable_timing=False)
                with torch.cuda.stream(stream_offload):
                    act_cpu.copy_(hidden, non_blocking=True)
                    ev.record(stream_offload)
                acts_cpu.append(act_cpu)
                acts_ev.append(ev)

                layer_mod = stage0 if sidx == 0 else stage1
                fwd_ms, hidden = measure_cuda(lambda: call_layer(layer_mod, hidden, attention_mask, position_ids, position_embeddings, logger))
                logger.log(f"[FWD][L{i:02d}] stage={sidx} fwd_ms={fwd_ms:.3f}")

                if args.prefetch and (i + 1) < use_layers:
                    nidx = i + 1
                    dst_stage = nidx % 2
                    maybe_pin(nidx)
                    prefetch_layer_to_stage(layers[nidx], stage_param_maps[dst_stage], stream_prefetch, ev_done[dst_stage], logger, tag=f" L{nidx:02d}->S{dst_stage}")

            if norm is not None:
                hidden = norm(hidden)

        t_fwd1 = time.perf_counter()
        forward_s = t_fwd1 - t_fwd0
        logger.log(f"[FWD] forward_s={forward_s:.3f} acts_saved={len(acts_cpu)}")

        # --- init grad from simple loss ---
        hidden_leaf = hidden.detach().requires_grad_(True)
        with torch.autocast("cuda", dtype=dtype):
            loss = hidden_leaf.float().pow(2).mean()
        g = torch.autograd.grad(loss, hidden_leaf, retain_graph=False, create_graph=False)[0].detach()
        logger.log(f"[LOSS] loss={loss.item():.6f} grad_norm={float(g.float().norm().item()):.6f}")

        # --- Backward streaming: recompute + CPU update overlap ---
        t_bwd0 = time.perf_counter()
        executor = ThreadPoolExecutor(max_workers=1)
        pending = []
        cpu_update_total = 0.0

        for i in range(use_layers - 1, -1, -1):
            sidx = i % 2
            acts_ev[i].synchronize()
            torch.cuda.current_stream().wait_event(ev_done[sidx])

            h_in = acts_cpu[i].to(device, non_blocking=True).detach().requires_grad_(True)
            layer_mod = stage0 if sidx == 0 else stage1

            with torch.autocast("cuda", dtype=dtype):
                rec_ms, h_out = measure_cuda(lambda: call_layer(layer_mod, h_in, attention_mask, position_ids, position_embeddings, logger))

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

            pending.append(executor.submit(cpu_sgd_update, i, layers[i], grads_cpu, args.lr))
            logger.log(f"[BWD][L{i:02d}] stage={sidx} recompute_ms={rec_ms:.3f} grad_offload~{gib(bytes_grad):.3f}GiB pending_cpu_updates={len(pending)}")

            if args.prefetch and (i - 1) >= 0:
                prev = i - 1
                dst_stage = prev % 2
                maybe_pin(prev)
                prefetch_layer_to_stage(layers[prev], stage_param_maps[dst_stage], stream_prefetch, ev_done[dst_stage], logger, tag=f" L{prev:02d}->S{dst_stage}")

            del h_in, h_out, grads, grads_cpu

        for fut in as_completed(pending):
            layer_idx, upd_s = fut.result()
            cpu_update_total += upd_s
            logger.log(f"[CPU_UPDATE][L{layer_idx:02d}] update_s={upd_s:.4f}")

        executor.shutdown(wait=True)
        t_bwd1 = time.perf_counter()
        backward_s = t_bwd1 - t_bwd0

        torch.cuda.synchronize()
        wall_s = time.perf_counter() - t0_wall
        peak_alloc = gib(torch.cuda.max_memory_allocated())
        peak_reserved = gib(torch.cuda.max_memory_reserved())
        rss = cpu_rss_gib()

        logger.log(f"[ITER {it}] wall_s={wall_s:.3f} forward_s={forward_s:.3f} backward_s={backward_s:.3f} cpu_update_total_s={cpu_update_total:.3f} peak_alloc={peak_alloc:.2f}GiB peak_reserved={peak_reserved:.2f}GiB CPU_RSS={rss:.2f}GiB")

        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(f"{it},{wall_s:.6f},{peak_alloc:.6f},{peak_reserved:.6f},{rss:.6f},{forward_s:.6f},{backward_s:.6f},{cpu_update_total:.6f}\n")

    logger.log(f"[SAVED] {metrics_path}")
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
