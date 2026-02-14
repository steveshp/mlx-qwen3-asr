# Reference Parity Suite (Smoke v2)

- Suite: `reference-parity-suite-v1`
- Model: `Qwen/Qwen3-ASR-0.6B`
- Subsets: `test-clean`, `test-other`
- Samples per subset: `1`
- Synthetic long mixes: `1` (3 segments)
- Synthetic noise variants: `SNR=10dB`

## Summary

| Metric | Value |
|---|---:|
| Total samples | 5 |
| Token-exact matches | 2 |
| Token match rate | 0.4000 |
| Normalized-text matches | 3 |
| Text match rate | 0.6000 |
| Mean latency (MLX) | 1.9532s |
| Mean latency (reference) | 8.9474s |

## By Subset

| Subset | Samples | Token match rate | Text match rate | MLX mean latency | Reference mean latency |
|---|---:|---:|---:|---:|---:|
| test-clean | 1 | 0.0000 | 1.0000 | 1.3445s | 5.7780s |
| test-other | 1 | 0.0000 | 0.0000 | 1.7313s | 7.3153s |
| synthetic-longmix | 1 | 0.0000 | 0.0000 | 3.7523s | 18.5144s |
| test-clean-noise-snr10 | 1 | 1.0000 | 1.0000 | 1.3588s | 5.7852s |
| test-other-noise-snr10 | 1 | 1.0000 | 1.0000 | 1.5790s | 7.3440s |

## Interpretation

- Token-level parity remains incomplete on broader and longer conditions.
- Text-level parity is higher than token-level parity on punctuation-sensitive cases.
- Keep this lane exploratory and use first-mismatch rows for targeted debugging.
