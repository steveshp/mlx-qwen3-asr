# Diarization Quality Benchmark (2026-02-15)

- Script: `scripts/eval_diarization.py`
- Manifest: `docs/benchmarks/2026-02-15-diarization-manifest-20.jsonl`
- JSON artifact: `docs/benchmarks/2026-02-15-diarization-quality-20.json`
- Model: `Qwen/Qwen3-ASR-0.6B` (`float16`)

## Summary

Current accepted run (`docs/benchmarks/2026-02-15-diarization-quality-20-v5.json`):

- Samples: `20`
- DER: `0.5051`
- JER: `0.4646`
- Mean latency/sample: `1.8981s`
- Runtime: `38.59s`

## Run history

| Artifact | DER | JER | Mean latency/sample | Status |
|---|---:|---:|---:|---|
| `2026-02-15-diarization-quality-20.json` | `0.6762` | `0.6065` | `6.2255s` | Initial baseline |
| `2026-02-15-diarization-quality-20-v2.json` | `0.7628` | `0.6636` | `3.0611s` | Rejected (quality regression) |
| `2026-02-15-diarization-quality-20-v3.json` | `0.6762` | `0.6065` | `2.7288s` | Baseline verification |
| `2026-02-15-diarization-quality-20-v4.json` | `0.5653` | `0.4930` | `1.8630s` | Interim accepted improvement |
| `2026-02-15-diarization-quality-20-v5.json` | `0.5051` | `0.4646` | `1.8981s` | Accepted improvement |

Latency is environment-sensitive (cold/warm cache), so adoption decisions are
quality-first (DER/JER) and then latency.

## Evaluation settings

- `frame_step_sec=0.02`
- `collar_sec=0.25`
- `ignore_overlap=true`

## Dataset notes

- Built from single-speaker real-world clips via
  `scripts/build_diarization_manifest.py`.
- Mix style: sequential two-speaker concatenation with short silence gaps.
- This artifact establishes an experimental baseline for future diarization
  improvements and release-gate thresholds.
