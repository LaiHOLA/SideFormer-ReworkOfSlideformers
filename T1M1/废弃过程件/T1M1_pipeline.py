import os, time, argparse, json
from datetime import datetime

import torch
import pandas as pd
from transformers import AutoModelForCausalLM

def gib(x): return x / (1024**3)

def find_layers(model):
    # 兼容：model.layers / model.model.layers / transformer.h
    if hasattr(model, "layers"):
        return model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError("Cannot find layers list. Tried: model.layers / model.model.layers / model.transformer.h")

def try_layer_forward(layer, hidden, position_ids=None, attention_mask=None):
    # 尽量适配不同 decoder layer 的 forward 形式
    try:
        out = layer(hidden_states=hidden, position_ids=position_ids, attention_mask=attention_mask, use_cache=False)
    except TypeError:
        try:
            out = layer(hidden, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
        except TypeError:
            out = layer(hidden)

    if isinstance(out, tuple):
        return out[0]
    if hasattr(out, "hidden_states"):
        return out.hidden_states
    return out

def pin_module_(m: torch.nn.Module):
    # 将参数/缓冲区替换为 pinned CPU tensor（更利于 non_blocking H2D）
    for p in m.parameters(recurse=True):
        if p.device.type == "cpu":
            p.data = p.data.pin_memory()
    for b in m.buffers(recurse=True):
        if b.device.type == "cpu":
            b.data = b.data.pin_memory()

def module_to_(m: torch.nn.Module, device: str):
    # inplace move
    m.to(device)
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="./qwen")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--mode", type=str, default="overlap", choices=["naive", "overlap"])
    ap.add_argument("--pin", type=int, default=0, help="1=pin CPU params for async H2D")
    ap.add_argument("--profile", type=int, default=1, help="1=export chrome trace")
    ap.add_argument("--outdir", type=str, default="runs_m1")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    assert torch.cuda.is_available(), "CUDA not available"
    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # load on CPU to allow streaming layers
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map={"": "cpu"},
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    layers = find_layers(model)
    n_layers = len(layers)

    # Create fixed input hidden on GPU (block-only benchmark)
    hidden_size = getattr(model.config, "hidden_size", None) or getattr(model.config, "n_embd", None)
    assert hidden_size is not None, "Cannot get hidden_size from config"
    hidden = torch.randn(args.batch, args.seq, hidden_size, device=device, dtype=dtype)

    position_ids = torch.arange(args.seq, device=device).unsqueeze(0).repeat(args.batch, 1)

    # Optional: pin CPU params to make H2D copy truly async
    if args.pin:
        for i in range(n_layers):
            pin_module_(layers[i])

    # Streams
    prefetch_stream = torch.cuda.Stream()
    offload_stream = torch.cuda.Stream()
    main_stream = torch.cuda.default_stream()

    # run dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.outdir, f"{ts}_m1_{args.mode}_seq{args.seq}_bs{args.batch}")
    os.makedirs(run_dir, exist_ok=True)

    env = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "dtype": str(dtype),
        "mode": args.mode,
        "pin": args.pin,
        "seq": args.seq,
        "batch": args.batch,
        "iters": args.iters,
        "n_layers": n_layers,
    }
    with open(os.path.join(run_dir, "env.json"), "w", encoding="utf-8") as f:
        json.dump(env, f, ensure_ascii=False, indent=2)

    # Helper: measure wall time + peak mem
    def one_iter(iter_idx: int):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # make a fresh hidden each iter to avoid weird caching effects
        h = hidden

        # ensure layer0 on GPU
        with torch.cuda.stream(prefetch_stream):
            with torch.profiler.record_function("LAYER_0_PREFETCH"):
                module_to_(layers[0], device)
        prefetch_stream.synchronize()

        for k in range(n_layers):
            # prefetch next
            if args.mode == "overlap" and k + 1 < n_layers:
                with torch.cuda.stream(prefetch_stream):
                    with torch.profiler.record_function(f"LAYER_{k+1}_PREFETCH"):
                        module_to_(layers[k+1], device)

            # compute current
            with torch.profiler.record_function(f"LAYER_{k}_COMPUTE"):
                with torch.autocast("cuda", dtype=dtype):
                    h = try_layer_forward(layers[k], h, position_ids=position_ids, attention_mask=None)

            # offload current (after compute enqueued)
            with torch.cuda.stream(offload_stream):
                with torch.profiler.record_function(f"LAYER_{k}_OFFLOAD"):
                    module_to_(layers[k], "cpu")

            # make sure next is ready before next compute (only overlap mode)
            if args.mode == "overlap" and k + 1 < n_layers:
                prefetch_stream.synchronize()

        # ensure all done
        offload_stream.synchronize()
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        peak_alloc = gib(torch.cuda.max_memory_allocated())
        peak_resv  = gib(torch.cuda.max_memory_reserved())
        wall_ms = (t1 - t0) * 1000.0
        print(f"[ITER {iter_idx}] wall_ms={wall_ms:.2f}  peak_alloc={peak_alloc:.2f}GiB  peak_reserved={peak_resv:.2f}GiB")
        return {"iter": iter_idx, "wall_ms": wall_ms, "peak_alloc_gib": peak_alloc, "peak_reserved_gib": peak_resv}

    rows = []

    if args.profile:
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(
            activities=activities,
            record_shapes=False,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            for i in range(args.iters):
                with torch.profiler.record_function(f"STEP_{i}"):
                    rows.append(one_iter(i))
                prof.step()
        trace_path = os.path.join(run_dir, "trace.json")
        prof.export_chrome_trace(trace_path)
        print("[TRACE]", trace_path)
    else:
        for i in range(args.iters):
            rows.append(one_iter(i))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(run_dir, "metrics.csv"), index=False, encoding="utf-8-sig")
    print("[SAVED]", os.path.join(run_dir, "metrics.csv"))
    print("[RUN_DIR]", run_dir)

if __name__ == "__main__":
    main()
