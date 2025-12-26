import os
import time
import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import psutil
import torch
from torch import nn
from torch.func import functional_call
from transformers import AutoModelForCausalLM

# -----------------------------
# Basic utils
# -----------------------------
def gib(x: float) -> float:
    return x / (1024**3)

def cpu_rss_gib() -> float:
    p = psutil.Process(os.getpid())
    return p.memory_info().rss / (1024**3)

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def move_buffers_to(module: nn.Module, device: str) -> None:
    # Move only buffers (e.g., rotary inv_freq) to CUDA. Keep parameters on CPU.
    for name, buf in list(module.named_buffers(recurse=True)):
        if buf is None:
            continue
        if buf.device.type == device:
            continue
        parent = module
        parts = name.split(".")
        for k in parts[:-1]:
            parent = getattr(parent, k)
        bname = parts[-1]
        parent._buffers[bname] = buf.to(device, non_blocking=True)

# -----------------------------
# SlideFormer-like engine (CPU param store + GPU compute + async prefetch)
# -----------------------------
class SlideEngine:
    """
    Term1-M1 objective:
      1) CPU/GPU coordinated pipeline at layer granularity
      2) H2D prefetch (async) overlapped with GPU compute
      3) CPU-side parameter update (SGD) overlapped with GPU backward (via thread pool)

    Design choices for Windows-friendly reproducibility:
      - Do NOT require GPUDirect Storage.
      - Keep master parameters on CPU (bf16 by default).
      - Stage only CURRENT / NEXT layer weights on GPU ("sliding" weights).
      - Use custom autograd.Function to do backward recompute and CPU update.
    """

    def __init__(self, model, num_layers: int, lr: float, pin_cpu: bool):
        self.model = model
        self.base = getattr(model, "model", model)
        self.layers = self.base.layers
        self.num_layers = min(num_layers, len(self.layers))
        self.lr = float(lr)
        self.pin_cpu = bool(pin_cpu)

        self.prefetch_stream = torch.cuda.Stream()
        self.cpu_pool = ThreadPoolExecutor(max_workers=2)
        self._pending_updates = []

        # Small CUDA-resident buffers needed by each layer (rotary, etc.)
        for i in range(self.num_layers):
            move_buffers_to(self.layers[i], "cuda")

        # CPU master param refs for each layer (no duplication)
        self.cpu_params = []  # list[list[(name, param_ref)]]
        for i in range(self.num_layers):
            items = []
            for n, p in self.layers[i].named_parameters(recurse=True):
                items.append((n, p))
            self.cpu_params.append(items)

        # Forward/backward prefetch caches (only keep 1-2 layers)
        self.fwd_cache = {}  # layer_idx -> dict[name] = GPU weight tensor (no grad)
        self.bwd_cache = {}  # layer_idx -> dict[name] = GPU weight tensor (no grad)

        # Runtime masks used by layer forward; updated per batch/seq
        self.attn_mask = None
        self.pos_ids = None

        # Autocast dtype
        self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # ---------- Prefetch helpers ----------
    def _cpu_to_gpu_dict(self, layer_idx: int) -> dict:
        """Copy CPU master params of one layer to GPU tensors (non-grad)."""
        w = {}
        for name, p in self.cpu_params[layer_idx]:
            src = p.detach()
            # Stage through pinned CPU buffer to allow non_blocking H2D
            if self.pin_cpu:
                cpu_buf = torch.empty_like(src, device="cpu", pin_memory=True)
            else:
                cpu_buf = torch.empty_like(src, device="cpu")
            cpu_buf.copy_(src, non_blocking=False)
            w[name] = cpu_buf.to("cuda", non_blocking=True)
        return w

    def prefetch_fwd(self, layer_idx: int) -> None:
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return
        if layer_idx in self.fwd_cache:
            return
        with torch.cuda.stream(self.prefetch_stream):
            self.fwd_cache[layer_idx] = self._cpu_to_gpu_dict(layer_idx)
            # keep cache small (only current/next)
            for k in list(self.fwd_cache.keys()):
                if k < layer_idx - 0 or k > layer_idx + 1:
                    del self.fwd_cache[k]

    def prefetch_bwd(self, layer_idx: int) -> None:
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return
        if layer_idx in self.bwd_cache:
            return
        with torch.cuda.stream(self.prefetch_stream):
            self.bwd_cache[layer_idx] = self._cpu_to_gpu_dict(layer_idx)
            for k in list(self.bwd_cache.keys()):
                if k < layer_idx - 1 or k > layer_idx + 0:
                    del self.bwd_cache[k]

    def take_fwd_weights(self, layer_idx: int) -> dict:
        """Wait for prefetch, then take the prefetched weights (and delete from cache)."""
        torch.cuda.current_stream().wait_stream(self.prefetch_stream)
        w = self.fwd_cache.pop(layer_idx, None)
        if w is None:
            # sync fetch if missing
            w = self._cpu_to_gpu_dict(layer_idx)
        return w

    def take_bwd_weights(self, layer_idx: int) -> dict:
        torch.cuda.current_stream().wait_stream(self.prefetch_stream)
        w = self.bwd_cache.pop(layer_idx, None)
        if w is None:
            w = self._cpu_to_gpu_dict(layer_idx)
        return w

    # ---------- CPU update ----------
    def submit_cpu_update(self, layer_idx: int, grad_dict: dict) -> None:
        """Apply SGD update to CPU master params in a background thread."""
        def _job():
            for name, p in self.cpu_params[layer_idx]:
                g = grad_dict.get(name, None)
                if g is None:
                    continue
                # move grad to CPU and update CPU master
                g_cpu = g.detach().to("cpu")
                p.data.add_(g_cpu, alpha=-self.lr)

        self._pending_updates.append(self.cpu_pool.submit(_job))

    def wait_cpu_updates(self) -> None:
        for f in self._pending_updates:
            f.result()
        self._pending_updates.clear()

    # ---------- Layer call ----------
    def build_masks(self, batch: int, seq: int) -> None:
        # 2D mask works for most HF decoder layers; if a layer requires 4D,
        # it will raise and we will fallback without mask.
        self.attn_mask = torch.ones((batch, seq), device="cuda", dtype=torch.long)
        self.pos_ids = torch.arange(seq, device="cuda").unsqueeze(0).expand(batch, -1)

    def layer_forward_opaque(self, layer_idx: int, hidden_states: torch.Tensor, w_gpu: dict) -> torch.Tensor:
        """
        Opaque forward (recorded only by our custom autograd, not by PyTorch autograd):
        - use functional_call with provided weights
        - under autocast
        """
        layer = self.layers[layer_idx]
        kwargs_try = [
            {"attention_mask": self.attn_mask, "position_ids": self.pos_ids, "use_cache": False},
            {"attention_mask": self.attn_mask, "position_ids": self.pos_ids},
            {"position_ids": self.pos_ids},
            {},
        ]
        with torch.autocast("cuda", dtype=self.amp_dtype):
            last_err = None
            for kw in kwargs_try:
                try:
                    out = functional_call(layer, w_gpu, (hidden_states,), kw, strict=False)
                    if isinstance(out, (tuple, list)):
                        return out[0]
                    return out
                except Exception as e:
                    last_err = e
            raise last_err

    def layer_recompute_and_grads(self, layer_idx: int, x: torch.Tensor, grad_out: torch.Tensor, w_gpu_no_grad: dict):
        """
        Recompute forward under grad, then compute grads wrt x and weights.
        w_gpu_no_grad: GPU weights (no grad); we create leaf copies for autograd.
        """
        # Leaf weights
        w_leaf = {k: v.detach().requires_grad_(True) for k, v in w_gpu_no_grad.items()}
        x_req = x.detach().requires_grad_(True)

        # Forward
        y = self.layer_forward_opaque(layer_idx, x_req, w_leaf)

        # Backward for this layer only: compute dx and dw
        inputs = [x_req] + list(w_leaf.values())
        grads = torch.autograd.grad(
            outputs=y,
            inputs=inputs,
            grad_outputs=grad_out,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        dx = grads[0]
        dw_list = grads[1:]
        grad_dict = {}
        for (name, _), g in zip(self.cpu_params[layer_idx], dw_list):
            if g is not None:
                grad_dict[name] = g
        return dx, grad_dict

# -----------------------------
# Custom autograd.Function
# -----------------------------
class SlideBlockFn(torch.autograd.Function):
    ENGINE = None  # set before forward()

    @staticmethod
    def forward(ctx, x: torch.Tensor, layer_idx_t: torch.Tensor) -> torch.Tensor:
        eng: SlideEngine = SlideBlockFn.ENGINE
        layer_idx = int(layer_idx_t.item())

        # Ensure current layer weights are prefetched; then consume them.
        eng.prefetch_fwd(layer_idx)
        w = eng.take_fwd_weights(layer_idx)

        # Schedule next-layer prefetch early to overlap with compute
        eng.prefetch_fwd(layer_idx + 1)

        # Opaque forward (no autograd recording for internal ops)
        y = eng.layer_forward_opaque(layer_idx, x, w)

        # Save minimal state for backward (activation only)
        ctx.layer_idx = layer_idx
        ctx.save_for_backward(x.detach())

        # Free weight tensors ASAP (reference dropped)
        del w
        return y

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        eng: SlideEngine = SlideBlockFn.ENGINE
        layer_idx = ctx.layer_idx
        (x_saved,) = ctx.saved_tensors

        # While computing backward for this layer, prefetch previous layer to overlap
        eng.prefetch_bwd(layer_idx - 1)

        # Get weights for this layer (from bwd cache or sync load)
        w = eng.take_bwd_weights(layer_idx)

        # Recompute + grads
        dx, grad_dict = eng.layer_recompute_and_grads(layer_idx, x_saved, grad_out, w)

        # CPU update is async (overlap with remaining GPU backward)
        eng.submit_cpu_update(layer_idx, grad_dict)

        # Free weights
        del w
        return dx, None

# -----------------------------
# Main (T1-M1 runnable script)
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="./qwen")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--pin", type=int, default=1)
    ap.add_argument("--profile", type=int, default=1)
    ap.add_argument("--outdir", type=str, default="runs")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    run_dir = ensure_dir(Path(args.outdir) / f"{now_stamp()}_t1m1_layerstream_seq{args.seq}_bs{args.batch}_L{args.layers}")
    trace_path = run_dir / "trace.json"
    metrics_path = run_dir / "metrics.csv"
    env_path = run_dir / "env.json"

    env = {
        "python": os.sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "bf16_supported": bool(torch.cuda.is_bf16_supported()) if torch.cuda.is_available() else False,
    }
    env_path.write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")

    # Load model on CPU: CPU is parameter store
    dtype_cpu = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_cpu,
        device_map="cpu",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.train()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # Engine
    eng = SlideEngine(model=model, num_layers=args.layers, lr=args.lr, pin_cpu=bool(args.pin))
    SlideBlockFn.ENGINE = eng

    # Synthetic hidden states: focus on layer streaming pipeline (avoid embedding/lm_head for T1-M1)
    # Hidden size inferred from the first layer
    hidden_size = getattr(model.config, "hidden_size", None) or getattr(model.config, "dim", None)
    if hidden_size is None:
        # fallback: try read from first layer norm weight
        first = eng.layers[0]
        for _, p in first.named_parameters(recurse=True):
            if p.dim() == 1:
                hidden_size = int(p.numel())
                break
    if hidden_size is None:
        raise RuntimeError("Cannot infer hidden_size; please check model config.")

    loss_fn = nn.MSELoss()

    # Base memory snapshot
    base_gpu_alloc = gib(torch.cuda.memory_allocated())
    base_gpu_res = gib(torch.cuda.memory_reserved())
    base_cpu = cpu_rss_gib()
    print(f"[BASE] GPU_alloc={base_gpu_alloc:.2f}GiB GPU_reserved={base_gpu_res:.2f}GiB CPU_RSS={base_cpu:.2f}GiB hidden={hidden_size}")

    rows = []

    def run_one_step(step_idx: int):
        # Create random input on GPU (bf16/fp16)
        dtype_in = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        x = torch.randn((args.batch, args.seq, hidden_size), device="cuda", dtype=dtype_in)

        eng.build_masks(args.batch, args.seq)
        torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        h = x
        # Forward through streamed layers
        for i in range(eng.num_layers):
            idx = torch.tensor(i, device="cuda", dtype=torch.int64)
            h = SlideBlockFn.apply(h, idx)

        # Dummy loss on final hidden states (keeps project focused on block streaming)
        loss = loss_fn(h.float(), torch.zeros_like(h, dtype=torch.float32))
        loss.backward()

        # Ensure CPU updates are finished before next step
        eng.wait_cpu_updates()
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        peak_alloc = gib(torch.cuda.max_memory_allocated())
        peak_res = gib(torch.cuda.max_memory_reserved())
        rss = cpu_rss_gib()

        print(f"[STEP {step_idx}] loss={loss.item():.6f} time={t1-t0:.3f}s "
              f"GPU_alloc_peak={peak_alloc:.2f}GiB GPU_reserved_peak={peak_res:.2f}GiB CPU_RSS={rss:.2f}GiB")

        rows.append({
            "step": step_idx,
            "loss": float(loss.item()),
            "time_s": float(t1 - t0),
            "gpu_alloc_peak_gib": float(peak_alloc),
            "gpu_reserved_peak_gib": float(peak_res),
            "cpu_rss_gib": float(rss),
            "seq": int(args.seq),
            "batch": int(args.batch),
            "layers": int(eng.num_layers),
            "pin_cpu": int(args.pin),
            "lr": float(args.lr),
        })

        # Release references
        del h, x, loss

    if args.profile:
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(
            activities=activities,
            record_shapes=False,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            for i in range(args.steps):
                run_one_step(i)
                prof.step()
        prof.export_chrome_trace(str(trace_path))
        print("[TRACE]", trace_path)
    else:
        for i in range(args.steps):
            run_one_step(i)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print("[SAVED]", metrics_path)
    print("[RUN_DIR]", run_dir)

if __name__ == "__main__":
    main()
