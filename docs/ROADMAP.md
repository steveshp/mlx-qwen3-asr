# Roadmap to North Star

Target: `pip install mlx-qwen3-asr` is the obvious/default way to run
Qwen3-ASR on Apple Silicon.

## Status by Priority

1. Correctness validation vs official PyTorch (token parity, greedy)
- In progress.
- Added optional integration test scaffold:
  - `tests/test_reference_parity.py`
  - Enabled via `RUN_REFERENCE_PARITY=1`.
- Added manual CI workflow: `.github/workflows/reference-parity.yml`.
- Added explicit gate runner: `scripts/quality_gate.py` with fast/release modes.
- Added policy doc: `docs/QUALITY_GATE.md`.
- Added scheduled regression lane:
  - `.github/workflows/nightly-regression.yml`
  - `scripts/eval_librispeech.py`
  - `docs/GOLDEN_DATASET.md`

2. Long audio preprocessing (no 30s feature truncation)
- Done.
- `compute_features()` now uses `truncation=False` and defaults to
  `padding="do_not_pad"`.

3. Quantized model artifacts on HuggingFace (4-bit / 8-bit)
- In progress (runtime path now validated; artifact publishing pending).
- Code-level quantization utility exists (`mlx_qwen3_asr.convert.quantize_model`).
- Added publishing script: `scripts/publish_quantized.py`.
- Added manual CI workflow: `.github/workflows/publish-quantized.yml`.
- Fixed quantized runtime loading in `load_model()`:
  - detects quantized tensors,
  - quantizes model modules before loading weights,
  - supports optional `quantization_config.json` metadata.
- Local validated benchmark point (4-bit, 0.6B, Apple M4 Pro):
  - short fixture: mean `0.1591s`, RTF `0.0628`
  - 10s clip: mean `0.2831s`, RTF `0.0283`

Performance progress:
- Done (low-risk optimization): tokenizer caching in `transcribe()` hot path.
- Measured local result (Apple M4 Pro, `Qwen/Qwen3-ASR-0.6B`, short fixture):
  - mean latency `1.7217s` -> `0.5464s` (`-68.3%`)
  - RTF `0.6796` -> `0.2157`
- Done (decoder-path optimization):
  - preallocated KV cache updates,
  - direct GQA fused attention path (no explicit K/V repeat).
- Current measured point (Apple M4 Pro, `Qwen/Qwen3-ASR-0.6B`, float16):
  - short fixture: mean `0.5303s`, RTF `0.2093`
  - 10s clip: mean `0.9420s`, RTF `0.0942`

4. Forced aligner timestamps
- In progress.
- Timestamps are enabled through optional `qwen-asr` backend integration.
- Native MLX aligner remains a future optimization.

5. Discoverability (README polish + PyPI)
- In progress.
- README validation section added; package metadata links updated.
- Benchmark process documented (`scripts/benchmark_asr.py`, `docs/BENCHMARKING.md`).

## Acceptance Gates

- Token parity: deterministic greedy parity test passes on reference fixtures.
- Reliability: long audio regression tests pass (`>30s` mel lengths).
- Golden quality: LibriSpeech sampled WER/CER trend remains within policy threshold.
- Packaging: reproducible release checklist (version bump, build, publish).
