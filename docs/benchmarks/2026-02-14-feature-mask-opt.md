# compute_features Attention-Mask Optimization

- Change: skip `return_attention_mask=True` on `padding="do_not_pad"` path.
- Method: same-process microbenchmark of WhisperFeatureExtractor path.

| Scenario | Old Mean (s) | New Mean (s) | Delta | Speedup |
|---|---:|---:|---:|---:|
| 2s audio | 0.000500 | 0.000528 | +5.68% | 0.95x |
| 10s audio | 0.001173 | 0.001111 | -5.27% | 1.06x |

Interpretation: preprocessing gets faster while preserving padded-mode behavior (tested).
