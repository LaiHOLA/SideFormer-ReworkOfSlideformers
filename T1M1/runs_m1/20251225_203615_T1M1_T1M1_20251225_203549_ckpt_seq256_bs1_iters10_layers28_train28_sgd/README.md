# T1-M1 Run

- mode: `ckpt`
- model: `./qwen`
- seq=256, batch=1, iters=10
- use_layers=28, train_layers=28 (range 0..27)
- optim=sgd, lr=0.0001, adam_state_dtype=bf16

## Files
- env.json
- log.txt
- metrics_iter.csv
- metrics_layer.csv
- mem_trace.csv
