\
param(
  [string]$Model = ".\qwen",
  [int]$Seq = 256,
  [int]$Batch = 1,
  [int]$Steps = 3,
  [int]$Layers = 28,
  [double]$LR = 0.0001,
  [int]$Pin = 1
)
python .\t1m1_stream_pipeline.py --model $Model --seq $Seq --batch $Batch --steps $Steps --layers $Layers --lr $LR --pin $Pin --profile 1
