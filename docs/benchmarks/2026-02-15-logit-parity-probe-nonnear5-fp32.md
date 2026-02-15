# Logit Parity Probe: non-near priority rows in fp32

- Date: 2026-02-15
- Probe artifact (JSON): `docs/benchmarks/2026-02-15-logit-parity-probe-nonnear5-fp32.json`

## Summary
- Rows probed: **5**
- Near-tie rows (`ref_margin<=0.5` OR `mlx_margin<=0.5`): **0/5**
- Strict top1/top2 cross-swaps: **3/5**
- Interpretation: residual non-near mismatches remain in fp32; these are not an fp16 rounding artifact.

## Rows
| sample_id | language | mismatch_index | ref_top | mlx_top | ref_margin | mlx_margin | strict_cross_swap |
|---|---|---:|---:|---:|---:|---:|---:|
| en_us-394135166243682296 | English | 17 | 1378 | 220 | 0.718750 | 1.396873 | 1 |
| ja_jp-8922261396806111795 | Japanese | 7 | 17587 | 11387 | 1.453125 | 0.512505 | 0 |
| de_de-11906634980733046933 | German | 29 | 92979 | 58534 | 2.671875 | 3.118046 | 0 |
| ar_eg-14219101187915533421 | Arabic | 29 | 131211 | 86941 | 0.718750 | 1.637304 | 1 |
| hi_in-17876469696694013955 | Hindi | 61 | 113 | 105 | 1.218750 | 0.728312 | 1 |
