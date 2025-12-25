# SideFormer-ReworkOfSlideformers
M1:Implement a minimal CPU/GPU coordinated pipeline at layer granularity
M2:Run a recompute ablation: no recompute vs standard gradient checkpointing
M3:Add a practical output-layer memory mitigation
Please check T1M1/T1M2/T1M3 for any code or outputs.
The introduce papers are in the top layer of this project


In this project, I used Qwen3 1.7b(https://modelscope.cn/models/Qwen/Qwen3-1.7B or https://huggingface.co/Qwen/Qwen3-1.7B ,  Ali Team) for the experiment.
Thanks for the contribution of ALi Team to LLM field.

If you want to try this code, you could download qwen31.7b to each T1MX folder, and leave them inside ./qwen folder.
Furthermore, the configuration is in Pytorch 2.7 + Cuda 12.8 + RTX3080@16g and 32G ram, so check your config detailly.
