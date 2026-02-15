# MLX vs PyTorch Quality Head-to-Head (manifest-quality-v1, n=200)

- model: `Qwen/Qwen3-ASR-0.6B`
- samples: `200`
- MLX primary: `0.2323`
- PyTorch primary: `0.2304`
- Delta primary (MLX-Ref): `+0.0019`

| System | Primary | WER | CER | Mean latency (s) |
|---|---:|---:|---:|---:|
| MLX | 0.2323 | 0.2323 | 0.1642 | 1.3420 |
| PyTorch ref | 0.2304 | 0.2304 | 0.1631 | 4.3866 |
