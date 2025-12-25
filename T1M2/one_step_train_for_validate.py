#python 02_one_step_train.py --recompute 0 --steps 3 --seq 256 --batch 1 --profile 0
#python 02_one_step_train.py --recompute 1 --steps 3 --seq 256 --batch 1 --profile 0

import os, time, argparse
import psutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def gib(x): return x / (1024**3)

def cpu_rss_gib():
    p = psutil.Process(os.getpid())
    return p.memory_info().rss / (1024**3)

@torch.no_grad()
def build_batch(tok, seq_len, batch_size, device):
    text = ("In this experiment we profile memory and recompute. " * 200).strip()
    #I don't really use a dataset since i just need to use it as a example for discovering
    #I will try that later to validate the training process in Term2.
    ids = tok(text, return_tensors="pt", truncation=True, max_length=seq_len).input_ids
    if ids.size(1) < seq_len:
        pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0
        pad = torch.full((1, seq_len - ids.size(1)), pad_id, dtype=ids.dtype)
        ids = torch.cat([ids, pad], dim=1)
    ids = ids[:, :seq_len].repeat(batch_size, 1).to(device)
    labels = ids.clone()
    return ids, labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="./qwen")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--recompute", type=int, default=0, help="0=no, 1=gradient checkpointing")
    ap.add_argument("--profile", type=int, default=0, help="1=export chrome trace")
    ap.add_argument("--trace_dir", type=str, default="traces")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    device = "cuda"

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,                    # 用 dtype，避免 torch_dtype deprecated
        device_map="cuda",
        attn_implementation="sdpa",     # 避免 flash-attn 依赖
        trust_remote_code=True,
    )
    model.train()
    model.config.use_cache = False  # checkpointing/训练都建议关掉 cache

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

    # 先用无状态/低状态优化器，避免 AdamW 在 32GB/16GB 上直接爆
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)

    # warmup + measure
    os.makedirs(args.trace_dir, exist_ok=True)

    def one_step(step_idx):
        inp, labels = build_batch(tok, args.seq, args.batch, device)
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

        peak_alloc = gib(torch.cuda.max_memory_allocated())
        peak_reserved = gib(torch.cuda.max_memory_reserved())
        rss = cpu_rss_gib()
        print(f"[STEP {step_idx}] loss={loss.item():.4f}  time={t1-t0:.3f}s  "
              f"GPU_alloc={peak_alloc:.2f}GiB  GPU_reserved={peak_reserved:.2f}GiB  CPU_RSS={rss:.2f}GiB")

    if args.profile:
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(
            activities=activities,
            record_shapes=False,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            for i in range(args.steps):
                one_step(i)
                prof.step()
        trace_path = os.path.join(args.trace_dir, f"trace_recompute{args.recompute}_seq{args.seq}_bs{args.batch}.json")
        prof.export_chrome_trace(trace_path)
        print("[TRACE]", trace_path)
    else:
        for i in range(args.steps):
            one_step(i)

if __name__ == "__main__":
    main()
