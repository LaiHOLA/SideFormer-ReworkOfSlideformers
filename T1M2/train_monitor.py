#python 03_train_monitor.py --model ./qwen --recompute 0 --steps 10 --seq 1024 --batch 1 --plot 1
#python 03_train_monitor.py --model ./qwen --recompute 1 --steps 10 --seq 1024 --batch 1 --plot 1

import os, time, argparse, csv
from datetime import datetime

import psutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def gib(x): return x / (1024**3)

def cpu_rss_gib():
    p = psutil.Process(os.getpid())
    return p.memory_info().rss / (1024**3)

@torch.no_grad()
def build_batch(tok, seq_len, batch_size, device):
    # 合成文本：保证每次输入稳定，便于对比系统开销
    text = ("In this experiment we profile memory and recompute. " * 200).strip()
    ids = tok(text, return_tensors="pt", truncation=True, max_length=seq_len).input_ids
    if ids.size(1) < seq_len:
        pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0
        pad = torch.full((1, seq_len - ids.size(1)), pad_id, dtype=ids.dtype)
        ids = torch.cat([ids, pad], dim=1)
    ids = ids[:, :seq_len].repeat(batch_size, 1).to(device)
    labels = ids.clone()
    return ids, labels

def find_transformer_blocks(model):
    """
    尝试找到“按层的 block 列表”，用于统计 forward 调用次数。
    不同架构名字不同，这里做几个常见路径探测。
    """
    candidates = []
    paths = [
        ("model.layers", lambda m: getattr(getattr(m, "model", None), "layers", None)),
        ("transformer.h", lambda m: getattr(getattr(m, "transformer", None), "h", None)),
        ("gpt_neox.layers", lambda m: getattr(getattr(getattr(m, "gpt_neox", None), "layers", None), None)),
    ]
    for name, fn in paths:
        try:
            x = fn(model)
            if x is not None and hasattr(x, "__len__"):
                candidates.append((name, x))
        except Exception:
            pass
    # 兜底：找出所有 ModuleList 里最长的那个（常见就是 blocks）
    if not candidates:
        best = None
        for n, mod in model.named_modules():
            if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 8:
                if best is None or len(mod) > len(best[1]):
                    best = (n, mod)
        if best:
            candidates.append(best)
    return candidates[0] if candidates else (None, None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="./qwen")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--recompute", type=int, default=0, help="0=no, 1=gradient checkpointing")
    ap.add_argument("--profile", type=int, default=0, help="1=export chrome trace")
    ap.add_argument("--outdir", type=str, default="runs")
    ap.add_argument("--plot", type=int, default=1, help="1=save plots")
    args = ap.parse_args()

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

    # 为了先跑稳：用 SGD（无状态/低状态）
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)

    # 统计“block forward 调用次数”，看 checkpoint 是否触发重算
    block_path, blocks = find_transformer_blocks(model)
    fwd_counts = {}
    hooks = []
    if blocks is not None:
        print(f"[INFO] Detected blocks: {block_path}, num_blocks={len(blocks)}")
        for i, blk in enumerate(blocks):
            fwd_counts[i] = 0
            def make_hook(idx):
                def _hook(_m, _inp, _out):
                    fwd_counts[idx] += 1
                return _hook
            hooks.append(blk.register_forward_hook(make_hook(i)))
    else:
        print("[WARN] Cannot find transformer blocks; will skip forward-count proof.")

    # 记录基座显存：模型加载完成后的常驻显存
    torch.cuda.synchronize()
    base_alloc = gib(torch.cuda.memory_allocated())
    base_reserved = gib(torch.cuda.memory_reserved())
    base_rss = cpu_rss_gib()
    print(f"[BASE] GPU_alloc={base_alloc:.2f}GiB  GPU_reserved={base_reserved:.2f}GiB  CPU_RSS={base_rss:.2f}GiB")

    csv_path = os.path.join(run_dir, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "step","loss","time_s",
            "gpu_alloc_start_gib","gpu_alloc_peak_gib","gpu_alloc_end_gib",
            "gpu_alloc_peak_delta_gib",
            "gpu_reserved_start_gib","gpu_reserved_peak_gib","gpu_reserved_end_gib",
            "cpu_rss_gib",
            "blocks_fwd_total"
        ])

        def one_step(step_idx):
            # 每 step 前清零计数
            for k in fwd_counts:
                fwd_counts[k] = 0

            inp, labels = build_batch(tok, args.seq, args.batch, device)

            torch.cuda.synchronize()
            alloc_start = gib(torch.cuda.memory_allocated())
            reserved_start = gib(torch.cuda.memory_reserved())

            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()

            with torch.autocast("cuda", dtype=dtype):
                out = model(input_ids=inp, labels=labels)
                loss = out.loss

            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

            torch.cuda.synchronize()
            t1 = time.perf_counter()

            alloc_peak = gib(torch.cuda.max_memory_allocated())
            reserved_peak = gib(torch.cuda.max_memory_reserved())
            alloc_end = gib(torch.cuda.memory_allocated())
            reserved_end = gib(torch.cuda.memory_reserved())
            rss = cpu_rss_gib()

            # 关键：增量峰值（更能看出 checkpoint 节省的 activation）
            peak_delta = max(0.0, alloc_peak - alloc_start)

            blocks_total = sum(fwd_counts.values()) if fwd_counts else -1

            print(
                f"[STEP {step_idx}] loss={loss.item():.4f}  time={t1-t0:.3f}s  "
                f"alloc_start={alloc_start:.2f}GiB  peak={alloc_peak:.2f}GiB  delta={peak_delta:.2f}GiB  "
                f"reserved_peak={reserved_peak:.2f}GiB  CPU_RSS={rss:.2f}GiB  blocks_fwd={blocks_total}"
            )

            w.writerow([
                step_idx, float(loss.item()), float(t1-t0),
                float(alloc_start), float(alloc_peak), float(alloc_end),
                float(peak_delta),
                float(reserved_start), float(reserved_peak), float(reserved_end),
                float(rss),
                int(blocks_total),
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

    # 清理 hooks
    for h in hooks:
        h.remove()

    print("[SAVED]", csv_path)

    if args.plot:
        try:
            import pandas as pd
            import matplotlib.pyplot as plt

            df = pd.read_csv(csv_path)
            # 1) 显存增量峰值（最能反映 checkpoint 节省激活）
            plt.figure()
            plt.plot(df["step"], df["gpu_alloc_peak_delta_gib"])
            plt.xlabel("step")
            plt.ylabel("GPU peak delta (GiB)")
            plt.title("Peak GPU allocated delta per step")
            p1 = os.path.join(run_dir, "gpu_peak_delta.png")
            plt.savefig(p1, dpi=150)

            # 2) GPU reserved peak
            plt.figure()
            plt.plot(df["step"], df["gpu_reserved_peak_gib"])
            plt.xlabel("step")
            plt.ylabel("GPU reserved peak (GiB)")
            plt.title("GPU reserved peak per step")
            p2 = os.path.join(run_dir, "gpu_reserved_peak.png")
            plt.savefig(p2, dpi=150)

            # 3) CPU RSS
            plt.figure()
            plt.plot(df["step"], df["cpu_rss_gib"])
            plt.xlabel("step")
            plt.ylabel("CPU RSS (GiB)")
            plt.title("CPU RSS per step")
            p3 = os.path.join(run_dir, "cpu_rss.png")
            plt.savefig(p3, dpi=150)

            # 4) step time
            plt.figure()
            plt.plot(df["step"], df["time_s"])
            plt.xlabel("step")
            plt.ylabel("time (s)")
            plt.title("Step time")
            p4 = os.path.join(run_dir, "step_time.png")
            plt.savefig(p4, dpi=150)

            print("[PLOTS]", p1, p2, p3, p4)
        except Exception as e:
            print("[PLOT_WARN]", repr(e))

if __name__ == "__main__":
    main()
