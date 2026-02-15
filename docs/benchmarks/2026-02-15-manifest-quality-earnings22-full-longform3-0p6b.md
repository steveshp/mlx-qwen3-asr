# Real-World Long-Form Quality (Earnings22 full, 0.6B fp16, n=3)

- model: `Qwen/Qwen3-ASR-0.6B`
- manifest: `2026-02-15-earnings22-full-longform3-manifest.jsonl`
- audio source: non-synthetic full-length Earnings22 recordings
- selection filter: `850s <= duration <= 1400s`
- total audio duration: `3902s` (`~65.0 min`)
- mean clip duration: `1300.7s` (`~21.7 min`)

| Metric | Value |
|---|---:|
| Primary error rate (WER) | 0.1322 |
| WER | 0.1322 |
| CER | 0.0698 |
| Mean latency (MLX) | 296.74s |
| Real-time factor (derived) | 0.228 |

Companion head-to-head artifact:
- `2026-02-15-quality-head2head-mlx-vs-pytorch-earnings22-full-longform3.md`
