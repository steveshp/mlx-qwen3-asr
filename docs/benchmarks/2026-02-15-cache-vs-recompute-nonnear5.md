# Non-near 5: Cache vs Recompute Consistency

- Date: 2026-02-15
- Artifact (JSON): `docs/benchmarks/2026-02-15-cache-vs-recompute-nonnear5.json`

## Summary
- Samples: **5**
- Cache-equal rows: **5/5**
- Result: MLX cached decode path matches full-prefix recompute path on all five priority rows.

## Per-sample
| sample_id | lang | equal | first_mismatch_cache_vs_recompute | len_cached | len_recompute |
|---|---|---|---:|---:|---:|
| en_us-394135166243682296 | English | True | -1 | 26 | 26 |
| ja_jp-8922261396806111795 | Japanese | True | -1 | 65 | 65 |
| de_de-11906634980733046933 | German | True | -1 | 49 | 49 |
| ar_eg-14219101187915533421 | Arabic | True | -1 | 52 | 52 |
| hi_in-17876469696694013955 | Hindi | True | -1 | 118 | 118 |

## Interpretation
- Residual multilingual non-near mismatches are not explained by KV cache rollback/update logic.
- Follow-up should focus on model-path parity (encoder/decoder numerics and layer behavior), not cache semantics.
