# Non-near 5: Stage-Level Parity Probe

- Date: 2026-02-15
- Artifact (JSON): `docs/benchmarks/2026-02-15-nonnear5-stage-parity.json`

## Summary
- Samples: **5**
- Mean audio-feature cosine (MLX vs reference): **0.9389**
- Mean audio-feature MAE: **0.004149**
- Teacher-forced top1 parity remains high overall, but each sample has a deterministic first top1 divergence at the same step as its observed decode mismatch index.

## Per-sample
| sample_id | lang | audio_cosine | audio_mae | teacher_forced_top1_rate | first_top1_mismatch_step |
|---|---|---:|---:|---:|---:|
| en_us-394135166243682296 | English | 0.951920 | 0.003073 | 0.960000 | 17 |
| ja_jp-8922261396806111795 | Japanese | 0.893303 | 0.007014 | 0.939394 | 7 |
| de_de-11906634980733046933 | German | 0.948103 | 0.003705 | 0.980392 | 29 |
| ar_eg-14219101187915533421 | Arabic | 0.944082 | 0.003906 | 0.944444 | 29 |
| hi_in-17876469696694013955 | Hindi | 0.957045 | 0.003048 | 0.982906 | 61 |

## Interpretation
- Residual non-near mismatches are reproducible under teacher forcing and are not caused by decode cache drift.
- Encoder outputs are close but not identical to reference (cosine ~0.89-0.96 on this subset), which is a plausible contributor to later decode-rank flips.
