#python recompute_proof.py --model ./qwen --recompute 0 --steps 3 --seq 256 --batch 1 --profile 1
#python recompute_proof.py --model ./qwen --recompute 1 --steps 3 --seq 256 --batch 1 --profile 1

import os, time, argparse, csv
from datetime import datetime

import psutil
import torch
import torch.utils.checkpoint as cp
from transformers import AutoTokenizer, AutoModelForCausalLM

def gib(x): return x / (1024**3)

def cpu_rss_gib():
    p = psutil.Process(os.getpid())
    return p.memory_info().rss / (1024**3)

@torch.no_grad()
def build_batch(tok, seq_len, batch_size, device):
    text = ("In this experiment we profile memory and recompute. " * 200).strip()
    ids = tok(text, return_tensors="pt", truncation=True, max_length=seq_len).input_ids
    if ids.size(1) < seq_len:
        pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0
        pad = torch.full((1, seq_len - ids.size(1)), pad_id, dtype=ids.dtype)
        ids = torch.cat([ids, pad], dim=1)
    ids = ids[:, :seq_len].repeat(batch_size, 1).to(device)
    labels = ids.clone()
    return ids, labels

def find_blocks(model):
    # 对 Qwen3：通常是 model.model.layers
    try:
        layers = model.model.layers
        return "model.layers", layers
    except Exception:
        pass
    # 兜底：找最长 ModuleList
    best = None
    for n, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 8:
            if best is None or len(mod) > len(best[1]):
                best = (n, mod)
    if best:
        return best[0], best[1]
    return None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="./qwen")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--recompute", type=int, default=0, help="0=no, 1=HF gradient_checkpointing")
    ap.add_argument("--profile", type=int, default=0, help="1=export chrome trace")
    ap.add_argument("--outdir", type=str, default="runs")
    ap.add_argument("--trace", type=int, default=0, help="same as --profile (compat)")
    args = ap.parse_args()

    if args.trace:
        args.profile = 1

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    device = "cuda"

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_recomp{args.recompute}_seq{args.seq}_bs{args.batch}"
    run_dir = os.path.join(args.outdir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map="cuda",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.train()
    model.config.use_cache = False

    # ====== 开关 recompute（HF 的 gradient checkpointing）======
    if args.recompute:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        print("[CFG] gradient_checkpointing=ON")
    else:
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass
        print("[CFG] gradient_checkpointing=OFF")

    # ====== 1)  checkpoint 是否被调用：monkeypatch cp.checkpoint ======
    ckpt_calls = {"n": 0}
    _orig_checkpoint = cp.checkpoint

    def _wrapped_checkpoint(function, *args, **kwargs):
        ckpt_calls["n"] += 1
        return _orig_checkpoint(function, *args, **kwargs)

    cp.checkpoint = _wrapped_checkpoint  # patch

    # ====== 2)  saved tensors 减少（这比 forward hook 更可靠）======
    saved_stat = {"count": 0, "bytes": 0}

    def pack_hook(t):
        if torch.is_tensor(t):
            saved_stat["count"] += 1
            saved_stat["bytes"] += t.numel() * t.element_size()
        return t

    def unpack_hook(t):
        return t

    # ====== 3) forward hooks：统计 block / attn / mlp forward 次数======
    block_path, blocks = find_blocks(model)
    print(f"[INFO] Detected blocks: {block_path}, num_blocks={len(blocks) if blocks is not None else 'NA'}")

    fwd_counts = {"block": 0, "attn": 0, "mlp": 0}
    hooks = []

    if blocks is not None:
        for blk in blocks:
            hooks.append(blk.register_forward_hook(lambda m, i, o: fwd_counts.__setitem__("block", fwd_counts["block"] + 1)))
            # 抓 blk.self_attn / blk.attn
            for attn_name in ["self_attn", "attn", "attention"]:
                if hasattr(blk, attn_name):
                    hooks.append(getattr(blk, attn_name).register_forward_hook(
                        lambda m, i, o: fwd_counts.__setitem__("attn", fwd_counts["attn"] + 1)
                    ))
                    break
            # 尝试抓 blk.mlp / blk.ffn
            for mlp_name in ["mlp", "ffn", "feed_forward"]:
                if hasattr(blk, mlp_name):
                    hooks.append(getattr(blk, mlp_name).register_forward_hook(
                        lambda m, i, o: fwd_counts.__setitem__("mlp", fwd_counts["mlp"] + 1)
                    ))
                    break

    # 优化器：SGD
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)

    # base memory
    torch.cuda.synchronize()
    base_alloc = gib(torch.cuda.memory_allocated())
    base_reserved = gib(torch.cuda.memory_reserved())
    base_rss = cpu_rss_gib()
    print(f"[BASE] GPU_alloc={base_alloc:.2f}GiB  GPU_reserved={base_reserved:.2f}GiB  CPU_RSS={base_rss:.2f}GiB")

    csv_path = os.path.join(run_dir, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "step","loss","time_s","tokens_per_s",
            "gpu_alloc_start_gib","gpu_alloc_peak_gib","gpu_alloc_peak_delta_gib",
            "gpu_reserved_peak_gib",
            "cpu_rss_gib",
            "ckpt_calls",
            "saved_tensors_count","saved_tensors_gib",
            "fwd_block","fwd_attn","fwd_mlp"
        ])

        def one_step(step_idx):
            # reset per-step counters
            ckpt_calls["n"] = 0
            saved_stat["count"] = 0
            saved_stat["bytes"] = 0
            fwd_counts["block"] = 0
            fwd_counts["attn"] = 0
            fwd_counts["mlp"] = 0

            inp, labels = build_batch(tok, args.seq, args.batch, device)

            torch.cuda.synchronize()
            alloc_start = gib(torch.cuda.memory_allocated())
            torch.cuda.reset_peak_memory_stats()

            t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=dtype):
                out = model(input_ids=inp, labels=labels)
                loss = out.loss

            # saved_tensors_hooks 包在 backward 周围，统计“保存了多少中间张量”
            with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
                loss.backward()

            opt.step()
            opt.zero_grad(set_to_none=True)

            torch.cuda.synchronize()
            t1 = time.perf_counter()

            alloc_peak = gib(torch.cuda.max_memory_allocated())
            reserved_peak = gib(torch.cuda.max_memory_reserved())
            peak_delta = max(0.0, alloc_peak - alloc_start)
            rss = cpu_rss_gib()

            toks = args.batch * args.seq
            tps = toks / max(1e-9, (t1 - t0))

            saved_gib = gib(saved_stat["bytes"])

            print(
                f"[STEP {step_idx}] loss={loss.item():.4f} time={t1-t0:.3f}s tps={tps:.1f} "
                f"delta_alloc={peak_delta:.2f}GiB reserved_peak={reserved_peak:.2f}GiB "
                f"ckpt_calls={ckpt_calls['n']} saved={saved_stat['count']}({saved_gib:.2f}GiB) "
                f"fwd(block/attn/mlp)={fwd_counts['block']}/{fwd_counts['attn']}/{fwd_counts['mlp']}"
            )

            w.writerow([
                step_idx, float(loss.item()), float(t1-t0), float(tps),
                float(alloc_start), float(alloc_peak), float(peak_delta),
                float(reserved_peak),
                float(rss),
                int(ckpt_calls["n"]),
                int(saved_stat["count"]), float(saved_gib),
                int(fwd_counts["block"]), int(fwd_counts["attn"]), int(fwd_counts["mlp"])
            ])
            f.flush()

        if args.profile:
            activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            with torch.profiler.profile(
                activities=activities,
                profile_memory=True,
                with_stack=False,
                record_shapes=False,
            ) as prof:
                for i in range(args.steps):
                    one_step(i)
                    prof.step()
            trace_path = os.path.join(run_dir, "trace.json")
            prof.export_chrome_trace(trace_path)
            print("[TRACE]", trace_path)
        else:
            for i in range(args.steps):
                one_step(i)

    # cleanup hooks and patch
    for h in hooks:
        h.remove()
    cp.checkpoint = _orig_checkpoint

    print("[SAVED]", csv_path)
    print("[RUN_DIR]", run_dir)

if __name__ == "__main__":
    main()
