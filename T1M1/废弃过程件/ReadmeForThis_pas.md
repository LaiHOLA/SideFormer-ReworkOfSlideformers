在T1M2里面我们是测试了重计算的可行度,现在阶段我们实现的是CPU和GPU之间的混合流水,看一下Recompute的效果.
入参里面的--optim adamw（CPU 上维护 Adam 状态 m/v，内存分布会明显）
入参里面的--train_layers N（只训练最后 N 层 → 只对这些层保存激活/做 recompute/算梯度/更新；前面层冻结，等价于“手动控制需要重算的层数”）
仍然严格保持SlideFormer的核心形态：CPU 为主存储与更新，GPU 只保留 2 层 staging + 当前计算临时张量

参考的代码如下先用 AdamW状态用 bf16
python t1_m1_adam.py --model ./qwen --seq 256 --batch 3 --iters 1 --layers 28 --train_layers 28 --optim adamw --adam_state_dtype bf16 --lr 1e-4 --logdir runs_m1

只训练最后几层,做更像“控制重算层数”的ablation
python t1_m1_pipeline.py --model ./qwen --seq 256 --batch 3 --iters 1 --layers 28 --train_layers 8 --optim adamw --adam_state_dtype bf16 --lr 1e-4 --logdir runs_m1

生成的内容里面有如下数据

log.txt：完整流水线日志（你现在看到的那种）

metrics_iter.csv：每个 iter 的 wall/forward/backward/CPU_update/显存峰值/CPU_RSS 等

metrics_layer.csv：逐层的 prefetch_ms / fwd_ms / recompute_ms / grad_offload / cpu_update_ms

report_time_breakdown.png：时间分解（forward/backward/cpu update/wall）

report_layer_prefetch_vs_compute.png：逐层 H2D 预取 vs 计算时间

report_layer_bwd_recompute.png：逐层 recompute 时间

report_layer_cpu_update.png：逐层 CPU 更新耗时
