# MLX vs PyTorch Quality Head-to-Head (manifest-quality-v1, n=10)

- model: `Qwen/Qwen3-ASR-0.6B`
- samples: `10`
- MLX primary: `0.1156`
- PyTorch primary: `0.1799`
- Delta primary (MLX-Ref): `-0.0642`

| System | Primary | WER | CER | Mean latency (s) |
|---|---:|---:|---:|---:|
| MLX | 0.1156 | 0.1671 | 0.0704 | 11.1300 |
| PyTorch ref | 0.1799 | 0.2431 | 0.1197 | 55.1165 |
