# Non-near Parity Matrix

- Date: 2026-02-15
- Model: `Qwen/Qwen3-ASR-0.6B`
- Source samples: `docs/benchmarks/2026-02-14-fleurs-multilingual-100-manifest.jsonl`
- Sample count: **5**

## Summary
- This is a bounded parity matrix on the fixed non-near subset to avoid open-ended tuning.

## Case Results
| case | dtype | token_match_rate | mismatch_rows | mean_first_mismatch_index | mlx_latency_mean_sec |
|---|---|---:|---:|---:|---:|
| baseline-f16 | float16 | 0.000 | 5 | 28.6 | 1.878 |
| baseline-f32 | float32 | 0.000 | 5 | 28.6 | 1.804 |
| force-dense-f16 | float16 | 0.000 | 5 | 28.6 | 1.766 |
| force-segmented-f16 | float16 | 0.000 | 5 | 28.6 | 1.820 |
| force-attn-fallback-f16 | float16 | 0.000 | 5 | 28.6 | 1.911 |

## Decision
- Outcome for this run: no case improved mismatch_rows; bounded exploration stops here.
