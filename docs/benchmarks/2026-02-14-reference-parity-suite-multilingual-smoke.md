# Reference Parity Suite (Multilingual Smoke)

- Suite: `reference-parity-suite-v1`
- Model: `Qwen/Qwen3-ASR-0.6B`
- Source manifest: `docs/benchmarks/2026-02-14-fleurs-multilingual-smoke-manifest.jsonl`
- Dataset source: `google/fleurs` (`test` split)
- Language configs: `en_us`, `cmn_hans_cn`, `ja_jp`, `de_de`
- Samples per language: `1`
- Run mode: release-gate manifest lane (`subsets=''`, manifest-driven)

## Summary

| Metric | Value |
|---|---:|
| Total samples | 4 |
| Token-exact matches | 2 |
| Token match rate | 0.5000 |
| Normalized-text matches | 2 |
| Text match rate | 0.5000 |
| Mean latency (MLX) | 1.4889s |
| Mean latency (reference) | 7.2721s |

## By Subset

| Subset | Samples | Token match rate | Text match rate | MLX mean latency | Reference mean latency |
|---|---:|---:|---:|---:|---:|
| fleurs-test-en_us | 1 | 1.0000 | 1.0000 | 0.8448s | 3.8703s |
| fleurs-test-cmn_hans_cn | 1 | 1.0000 | 1.0000 | 1.6625s | 6.5127s |
| fleurs-test-ja_jp | 1 | 0.0000 | 0.0000 | 1.3127s | 5.1716s |
| fleurs-test-de_de | 1 | 0.0000 | 0.0000 | 2.1355s | 13.5339s |

## Interpretation

- Multilingual parity infrastructure is now reproducible and artifact-backed.
- On this smoke sample, strict parity holds for English and Chinese, but not for Japanese/German.
- Keep this lane exploratory and expand per-language sample counts before promoting any multilingual parity claim.
