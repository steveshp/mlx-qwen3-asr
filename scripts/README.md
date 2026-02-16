# Scripts Layout

`scripts/` mixes user-facing utilities and internal evaluation tooling.

## User-facing utilities

- `scripts/convert.py` — convert and quantize model weights for MLX.
- `scripts/generate_mel_filters.py` — regenerate mel filter assets.
- `scripts/publish_quantized.py` — publish pre-quantized artifacts.
- `scripts/update_readme_stats.py` — refresh README test/LOC metadata from repo state.

## Evaluation and benchmarking tooling

- `scripts/eval/metrics.py` — shared normalization + WER/CER helpers.
- `scripts/eval/analyze_reference_parity_mismatches.py` — parity mismatch categorization.
- `scripts/eval/benchmark_quantization_matrix.py` — quantization sweep + eval/latency matrix.
- `scripts/eval_streaming_metrics.py` — streaming stability/rollback/finalization diagnostics.
- `scripts/eval_streaming_manifest.py` — multi-file streaming quality lane from JSONL manifest.
- `scripts/eval_diarization.py` — DER/JER evaluation from diarization manifests.
- `scripts/build_diarization_manifest.py` — deterministic multi-speaker mix manifest builder.
- `scripts/build_realworld_manifest.py` — deterministic AMI/Earnings real-world manifest curation.
- `scripts/build_earnings22_longform_manifest.py` — deterministic non-synthetic long-form manifest builder.
- Root-level `eval_*.py`, `benchmark_*.py`, and `analyze_*.py` scripts are dev tooling.

The long-term direction is to keep user-facing scripts at root and migrate
infrastructure scripts under `scripts/eval/` (and related dev-only subdirs)
without breaking existing command paths.
