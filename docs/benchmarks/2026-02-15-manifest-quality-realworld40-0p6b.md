# Real-World Manifest Quality (0.6B fp16, n=40)

- model: `Qwen/Qwen3-ASR-0.6B`
- manifest: `2026-02-15-realworld-manifest-40.jsonl`
- composition:
  - `ami-ihm-test`: 20 samples, 10 speakers
  - `earnings22-chunked-test`: 20 samples, 10 speakers
  - mean clip duration: `5.88s`

| Metric | Value |
|---|---:|
| Primary error rate (WER) | 0.1417 |
| WER | 0.1417 |
| CER | 0.0927 |
| Mean latency (MLX) | 0.8502s |

Companion head-to-head artifact:
- `2026-02-15-quality-head2head-mlx-vs-pytorch-realworld40.md`
