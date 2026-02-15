# MLX vs PyTorch Quality Head-to-Head (manifest-quality-v1, n=40)

- model: `Qwen/Qwen3-ASR-0.6B`
- samples: `40`
- MLX primary: `0.1417`
- PyTorch primary: `0.1417`
- Delta primary (MLX-Ref): `+0.0000`

| System | Primary | WER | CER | Mean latency (s) |
|---|---:|---:|---:|---:|
| MLX | 0.1417 | 0.1417 | 0.0927 | 0.8502 |
| PyTorch ref | 0.1417 | 0.1417 | 0.0938 | 3.6674 |
