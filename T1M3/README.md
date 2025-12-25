
# T1M3 Suite — Output-layer Memory Mitigation (Exact Chunked CE)

This mini-suite provides a **practical** output-layer memory mitigation:
**compute exact cross-entropy without materializing full logits** `[B*(S-1), V]`.

It contains:
- `t1_m3_head.py` — chunked CE implementation (exact, no CUDA compile)
- `t1_m3_headbench.py` — isolated head benchmark (best for quick validation)
- `t1m3_train_profiler.py` — training-step profiler using HF `model.model(...)` (no custom layer code)
- `t1m3_report.py` — paper-style plots from run dirs

## 1) Quick validation (recommended first)

```bash
python t1_m3_headbench.py --vocab 152064 --hidden 2048 --seq 1024 --batch 2 --dtype bf16 --chunk 8192 --iters 20
```

You should see:
- `chunked` uses **much lower peak allocated/reserved** than `full`
- time may increase (extra passes), which is expected

## 2) End-to-end step profiling (your local model)

### Baseline (materialize logits)
```bash
python t1m3_train_profiler.py --model .\qwen --out runs_t1m3 --head ce_full --seq 1024 --batch 2 --steps 30 --dtype bf16 --optim sgd
```

### Chunked (no full logits)
```bash
python t1m3_train_profiler.py --model .\qwen --out runs_t1m3 --head ce_chunked --vocab-chunk 8192 --seq 1024 --batch 2 --steps 30 --dtype bf16 --optim sgd
```

### With gradient checkpointing (optional)
```bash
python t1m3_train_profiler.py --model .\qwen --out runs_t1m3 --head ce_chunked --ckpt 1 --seq 1024 --batch 2 --steps 30 --dtype bf16 --optim sgd
```

## 3) Generate report

```bash
python t1m3_report.py --runs runs_t1m3\20251225_231102_T1M3_headce_full_ckpt0_seq1024_bs2_bf16 runs_t1m3\20251225_231206_T1M3_headce_chunked_ckpt0_seq1024_bs2_bf16 --out t1m3_report_out
```

Outputs in `t1m3_report_out/`:
- `summary_table.csv`
- `fig_time_breakdown.png`
- `fig_peak_memory.png`
- `fig_mem_timeline.png` (if mem_trace exists)

## Notes

- We **detach** the vocab weight by default (inside `t1m3_train_profiler.py`) to isolate the effect of logits/activation memory.
  If you later want to train the head weight as well, we can extend `chunked_ce_loss` to return/accumulate `grad_weight` and include it in the optimizer.
- Best to increase `seq`/`batch` when demonstrating memory savings; logits size scales with `B*(S-1)*V`.
