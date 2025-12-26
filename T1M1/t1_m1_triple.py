# -*- coding: utf-8 -*-
'''
T1-M1 experiment orchestrator (run any subset of modes + optional analysis).
#python t1_m1_triple.py --model ./qwen --seq 256 --batch 1 --iters 10 --layers 28 --train_layers 28 --optim sgd --lr 1e-4 --modes gpu,ckpt,slide --analyze 1

Modes:
  - gpu   : baseline pure-GPU training (no checkpointing, no CPU offload)
  - ckpt  : baseline pure-GPU training + PyTorch gradient checkpointing (recompute)
  - slide : SlideFormer-style layer streaming (CPU param store + staged H2D, activation offload, per-layer recompute, CPU optimizer update)

Key goals:
  1) pick optimizer: SGD vs AdamW (use --optim sgd|adamw)
  2) run flexible subsets: --modes gpu,ckpt,slide  (any combination)
  3) write a manifest JSON so the analyzer can create paper-style figures from any subset
'''
import os, sys, json, argparse, subprocess, re
from datetime import datetime

def _run(cmd, cwd=None):
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    out_lines = []
    for line in p.stdout:
        print(line, end="")
        out_lines.append(line)
    rc = p.wait()
    return rc, "".join(out_lines)

def _extract_run_dir(text):
    m = re.search(r"\[RUN_DIR\]\s+(.+)", text)
    return m.group(1).strip() if m else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="./qwen")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--train_layers", type=int, default=0)

    ap.add_argument("--ckpt_layers", type=int, default=0,
                    help="only for ckpt mode: checkpoint last N trainable layers (0=all trainable layers)")

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--logdir", type=str, default="runs_m1")
    ap.add_argument("--modes", type=str, default="gpu,ckpt,slide",
                    help="comma list: gpu,ckpt,slide (any subset)")

    # optimizer
    ap.add_argument("--optim", type=str, default="sgd", choices=["sgd", "adamw"])
    ap.add_argument("--betas", type=str, default="0.9,0.999")
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--adam_state_dtype", type=str, default="bf16", choices=["bf16", "fp32"])

    # slide-only knobs (forwarded; harmless for other modes)
    ap.add_argument("--prefetch", type=int, default=1)
    ap.add_argument("--pin_mode", type=str, default="lazy", choices=["off", "lazy"])
    ap.add_argument("--monitor_ms", type=int, default=5)

    ap.add_argument("--analyze", type=int, default=1, help="1=run analyzer after runs")
    ap.add_argument("--analyzer_out", type=str, default="",
                    help="optional analyzer output dir; default under logdir")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    group = f"T1M1_{stamp}"
    os.makedirs(args.logdir, exist_ok=True)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    valid = {"gpu", "ckpt", "slide"}
    for m in modes:
        if m not in valid:
            raise SystemExit(f"Unknown mode: {m}. valid={sorted(valid)}")

    runs = []
    cwd = os.path.dirname(os.path.abspath(__file__))
    for mode in modes:
        cmd = [
            sys.executable, "-u", "t1_m1_run.py",
            "--model", args.model,
            "--seq", str(args.seq),
            "--batch", str(args.batch),
            "--iters", str(args.iters),
            "--layers", str(args.layers),
            "--train_layers", str(args.train_layers),
            "--lr", str(args.lr),
            "--logdir", args.logdir,
            "--mode", mode,
            "--group", group,

            "--optim", args.optim,
            "--betas", args.betas,
            "--eps", str(args.eps),
            "--weight_decay", str(args.weight_decay),
            "--adam_state_dtype", args.adam_state_dtype,

            "--prefetch", str(args.prefetch),
            "--pin_mode", args.pin_mode,
            "--monitor_ms", str(args.monitor_ms),
            "--ckpt_layers", str(args.ckpt_layers),
            "--plot", "0",
        ]

        print(f"\n===== RUN MODE: {mode} =====")
        rc, out = _run(cmd, cwd=cwd)
        if rc != 0:
            raise SystemExit(f"Run failed: mode={mode}, rc={rc}")

        run_dir = _extract_run_dir(out)
        if not run_dir:
            raise SystemExit(f"Cannot find [RUN_DIR] in output for mode={mode}")
        runs.append({"label": mode, "path": run_dir})

    manifest = {
        "group": group,
        "created_at": stamp,
        "cmdline": " ".join(sys.argv),
        "runs": runs,
        "args": vars(args),
    }
    manifest_path = os.path.join(args.logdir, f"{group}_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n[MANIFEST] {manifest_path}")
    for r in runs:
        print(f"  - {r['label']}: {r['path']}")

    if args.analyze:
        out_dir = args.analyzer_out.strip() or os.path.join(args.logdir, f"{group}_paper")
        cmd = [
            sys.executable, "-u", "t1_m1_analyzer.py",
            "--manifest", manifest_path,
            "--out", out_dir,
            "--export_pdf", "1",
        ]
        print("\n===== ANALYZE =====")
        rc, _ = _run(cmd, cwd=cwd)
        if rc != 0:
            raise SystemExit(f"Analyzer failed rc={rc}")
        print(f"[ANALYZER_OUT] {out_dir}")

if __name__ == "__main__":
    main()
