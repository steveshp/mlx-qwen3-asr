# Execution Tracker (2026-02-14)

North-star execution notes for the current optimization/refactor wave.

## Goal

Ship the highest-quality native MLX Qwen3-ASR path while preserving correctness gates.

## Current Priorities

1. Decode/session architecture decoupling for production-grade core inference.
2. Keep all claims benchmark-backed and quality-gate clean.
3. Prioritize offline quality/speed/timestamp rigor over streaming feature expansion.

## Verified Findings

### 1) Custom mel parity status (updated)

Measured against current `compute_features()` (HF WhisperFeatureExtractor):

- Frame count mismatch:
  - HF path returns `N` frames for `N*160` samples (e.g., 2s -> 200).
  - Current custom `log_mel_spectrogram()` returns `N+1` frames (e.g., 2s -> 201).
- Value mismatch after truncating custom to HF frame count:
  - MAE around `0.153-0.155` on normalized log-mel.
  - Correlation around `0.09-0.12`.

Status update:
- Fixed by (a) regenerating filterbank to match Whisper exactly and
  (b) trimming final STFT frame before normalization.
- Parity now passes on deterministic random + fixture set:
  - max MAE: `2.83e-07`
  - max abs diff: `1.13e-04`
  - exact frame-count parity.
- Artifacts:
  - `docs/benchmarks/2026-02-14-mel-parity.json`
  - `docs/benchmarks/2026-02-14-mel-parity.md`

### 2) Mel filter mismatch root cause (resolved)

Comparing `assets/mel_filters.npz` against `WhisperFeatureExtractor.mel_filters`
(transposed to shape `(128, 201)`):

- MAE: `0.0002265`
- Max abs diff: `0.05816`

Resolution:
- `scripts/generate_mel_filters.py` now generates Whisper-compatible filters
  (`mel_filter_bank(..., norm="slaney", mel_scale="slaney")`).
- `compute_features(..., padding="do_not_pad")` now uses the corrected custom
  mel path by default.

### 3) Official streaming semantics context

From upstream `qwen_asr/inference/qwen3_asr.py`:

- Streaming path re-feeds accumulated audio context each chunk.
- Uses `unfixed_chunk_num` and `unfixed_token_num` rollback controls.
- Streaming in official stack is backend-limited (`vllm`) and no timestamps.

We already aligned our rolling streaming controls to the same knobs and added
bounded-context behavior for stable per-chunk cost.

### 4) Encoder long-context execution strategy

- Dense block-mask encoder attention is fine for small window counts but scales
  poorly as sequence length grows.
- A segmented per-window execution path is numerically equivalent (max-abs diff
  around `1e-6` in synthetic checks) and significantly faster for long contexts.
- Hybrid threshold selected from benchmark sweep:
  - use segmented path when `num_windows >= 20`.
- Artifacts:
  - `docs/benchmarks/2026-02-14-encoder-windowing-threshold.json`
  - `docs/benchmarks/2026-02-14-encoder-windowing-threshold.md`

### 5) Output parse robustness hardening

- Implemented upstream-style repetition cleanup in ASR output parsing.
- Added explicit handling for:
  - `language None<asr_text>` empty-audio case,
  - forced-language parsing path,
  - consistent trailing special-token cleanup.
- Added regression tests in tokenizer/transcribe lanes.

## Decision Gates

### Gate A: Mel backend switch

Status: COMPLETE

Switched only after all were true:

1. Frame-count parity is exact across short and long clips.
2. Value parity threshold:
   - MAE <= `1e-3` (target), or a justified threshold validated by transcript parity.
3. Transcript parity:
   - No regression in reference parity lane.
   - No measurable degradation in sampled LibriSpeech WER/CER lane.

### Gate B: Streaming architecture claims

Keep streaming marked experimental until:

1. Resumable decode cache path is implemented and benchmarked.
2. Streaming benchmark lane shows predictable latency under long sessions.
3. Regression tests cover multilingual rollback behavior and state transitions.

## Immediate Next Tasks

1. (Complete) Added explicit `Session` API skeleton for explicit state ownership.
2. (Complete) Introduced model-level `prefill/step` interfaces so `generate()` no longer reaches into model internals.
3. (In progress) Expand golden-set benchmark coverage (quality + latency) for release and quantized artifacts.
4. Keep streaming experimental; only revisit deeper streaming work after core gates are saturated.
