# T1-M1 Suite (gpu / ckpt / slide) + Paper Analyzer

## 文件
- `t1_m1_run.py`：单个模式运行器，输出原始 CSV（metrics_iter/layer、mem_trace、timeline、env.json、log.txt）
- `t1_m1_triple.py`：实验编排器，可自由选择模式组合，并输出 manifest
- `t1_m1_analyzer.py`：论文风格分析器，读 manifest 或 run_dir 列表，输出 figures + compare_summary.csv

## 三个模式含义
- `gpu`：纯 GPU 标准训练（保留激活，不做 checkpoint，不做 CPU offload）
- `ckpt`：纯 GPU + PyTorch Gradient Checkpoint（反向阶段重算）
- `slide`：仿 SlideFormer：参数常驻 CPU，H2D 分层预取；激活 offload 到 CPU；反向逐层 recompute；梯度回传到 CPU；CPU 侧做优化器更新

## 一键跑三种 + 出论文图
```bash
python t1_m1_triple.py --model ./qwen --seq 256 --batch 3 --iters 3 --layers 28 --train_layers 28 \
  --optim adamw --adam_state_dtype bf16 --lr 1e-4 --modes gpu,ckpt,slide --analyze 1
```

## 自由组合：只跑 ckpt vs slide
```bash
python t1_m1_triple.py --model ./qwen --seq 256 --batch 3 --iters 3 --layers 28 --train_layers 28 \
  --optim sgd --lr 1e-4 --modes ckpt,slide --analyze 1
```

## 控制 recompute 深度
- ckpt：`--ckpt_layers N`（只对“最后 N 个可训练层”做 checkpoint；0=全部可训练层）
- slide：recompute 深度 = train 深度；用 `--train_layers N` 控制（即只训练/重算最后 N 层）

## 只做分析（已有 run_dir）
```bash
python t1_m1_analyzer.py --runs gpu=RUN_DIR_GPU ckpt=RUN_DIR_CKPT slide=RUN_DIR_SLIDE --out t1m1_paper_out
```
输出在 `t1m1_paper_out/figures/`，包含 PNG+PDF（可用 `--export_pdf 0` 关闭 PDF）。

```bash
python t1_m1_triple.py --model ./qwen --seq 256 --batch 1 --iters 10 \
  --layers 28 --train_layers 28 \
  --optim sgd --lr 1e-4 \
  --modes gpu,ckpt,slide --analyze 1

```