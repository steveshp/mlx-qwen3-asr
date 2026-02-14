# Reference Parity Suite (English 22-sample)

- Suite: `reference-parity-suite-v1`
- Model: `Qwen/Qwen3-ASR-0.6B`
- Subsets: `test-clean`, `test-other`
- Samples per subset: `5`
- Synthetic long mixes: `2` (4 segments each)
- Synthetic noise variants: `SNR=10dB` on selected base samples

## Summary

| Metric | Value |
|---|---:|
| Total samples | 22 |
| Token-exact matches | 16 |
| Token match rate | 0.7273 |
| Normalized-text matches | 18 |
| Text match rate | 0.8182 |
| Mean latency (MLX) | 1.5403s |
| Mean latency (reference) | 6.8751s |

## By Subset

| Subset | Samples | Token match rate | Text match rate | MLX mean latency | Reference mean latency |
|---|---:|---:|---:|---:|---:|
| test-clean | 5 | 0.8000 | 1.0000 | 1.1849s | 5.3487s |
| test-other | 5 | 0.8000 | 0.8000 | 1.0364s | 4.8129s |
| synthetic-longmix | 2 | 0.0000 | 0.0000 | 5.6808s | 24.3410s |
| test-clean-noise-snr10 | 5 | 1.0000 | 1.0000 | 1.2378s | 5.5877s |
| test-other-noise-snr10 | 5 | 0.6000 | 0.8000 | 1.0458s | 4.7645s |

## Interpretation

- Base single-speaker short utterances are close but not fully token-exact.
- Long mixed-speaker synthetic clips remain a clear weak spot.
- Text parity is better than token parity on punctuation-sensitive cases, but strict token parity still needs targeted fixes before promotion to a hard release gate.
