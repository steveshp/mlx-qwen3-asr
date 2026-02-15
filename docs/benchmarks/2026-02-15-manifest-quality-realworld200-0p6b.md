# Real-World Manifest Quality (0.6B fp16, n=200)

- model: `Qwen/Qwen3-ASR-0.6B`
- manifest: `2026-02-15-realworld-manifest-200.jsonl`
- composition:
  - `ami-ihm-test`: 100 samples, 16 speakers
  - `earnings22-chunked-test`: 100 samples, 50 speakers
  - mean clip duration: `~5.9s`

| Metric | Value |
|---|---:|
| Primary error rate (WER) | 0.2323 |
| WER | 0.2323 |
| CER | 0.1642 |
| Mean latency (MLX) | 1.3420s |

Companion head-to-head artifact:
- `2026-02-15-quality-head2head-mlx-vs-pytorch-realworld200.md`
