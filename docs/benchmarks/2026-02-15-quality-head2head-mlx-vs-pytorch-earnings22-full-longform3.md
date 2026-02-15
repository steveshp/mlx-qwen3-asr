# MLX vs PyTorch Quality Head-to-Head (manifest-quality-v1, n=3)

- model: `Qwen/Qwen3-ASR-0.6B`
- samples: `3`
- reference_max_chunk_sec: `20.0`
- MLX primary: `0.1322`
- PyTorch primary: `0.1509`
- Delta primary (MLX-Ref): `-0.0187`

| System | Primary | WER | CER | Mean latency (s) |
|---|---:|---:|---:|---:|
| MLX | 0.1322 | 0.1322 | 0.0698 | 296.7379 |
| PyTorch ref | 0.1509 | 0.1509 | 0.0858 | 674.7257 |
