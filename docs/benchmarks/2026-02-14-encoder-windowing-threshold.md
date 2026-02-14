# Encoder Windowing Threshold Benchmark

Date: 2026-02-14

## Change

Use segmented per-window encoder execution for long sequences, while keeping
dense masked execution for smaller window counts.

- New threshold: `num_windows >= 20` switches to segmented execution.
- Rationale: avoid short-input regressions while removing long-input quadratic
  overhead from dense `(L, L)` block masks.

## Method

- Synthetic benchmark with identical layer weights and inputs.
- Compared:
  - `dense_mask`: full-sequence execution with block-diagonal additive mask
  - `windowed_segments`: exact per-window execution then concatenate
- Config:
  - `d_model=256`, `num_heads=8`, `ffn_dim=1024`, `layers=8`, `window_len=104`
- Accuracy check: max absolute output diff recorded for each sequence length.

## Results

| Seq Len | Windows | Max Abs Diff | Dense Mean (s) | Windowed Mean (s) | Speedup |
|---:|---:|---:|---:|---:|---:|
| 832 | 8 | 1.91e-06 | 0.00941 | 0.01267 | 0.74x |
| 1664 | 16 | 2.62e-06 | 0.02994 | 0.02825 | 1.06x |
| 2080 | 20 | 2.50e-06 | 0.05472 | 0.04659 | 1.17x |
| 3120 | 30 | 2.62e-06 | 0.10998 | 0.07408 | 1.48x |
| 8320 | 80 | 2.62e-06 | 0.79896 | 0.19145 | 4.17x |

## Artifact

- `docs/benchmarks/2026-02-14-encoder-windowing-threshold.json`
