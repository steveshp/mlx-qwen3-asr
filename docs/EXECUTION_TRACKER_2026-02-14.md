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

### 6) Streaming input hardening and dead-state cleanup

- Removed unused `previous_tokens` field from `StreamingState`.
- Added explicit input normalization in `feed_audio(...)`:
  - accepts non-1D arrays (flattens),
  - converts `int16` PCM to `float32` in `[-1, 1]`.
- Added init-time validation for streaming parameters:
  - `chunk_size_sec > 0`, `max_context_sec > 0`, `sample_rate > 0`.
- Added regression tests for these behaviors.

### 7) Speculative decoding prototype (target/draft) landed

- Added cache trim support in `KVCache` and model `step_many(...)` API for
  batched verification steps.
- Added experimental speculative generation path (`generate_speculative`) with
  strict greedy parity behavior and cache rewind logic.
- Wired optional draft-model controls through Python API/CLI:
  - `transcribe(..., draft_model=..., num_draft_tokens=...)`
  - `mlx-qwen3-asr --draft-model ... --num-draft-tokens ...`
- Added benchmark harness:
  - `scripts/benchmark_speculative.py`
- Current measured status:
  - parity: `text_match=true`,
  - performance: slower on tested short/10s workloads vs baseline.
  - therefore remains experimental and non-default.

### 8) Multi-model cache residency hardening

- `_ModelHolder` cache now keys by `(model_path, dtype)` instead of storing
  only one global model instance.
- Why this matters:
  - speculative runs often use both target and draft models,
  - single-entry cache caused avoidable reload thrash across repeated calls.
- Added regression coverage to ensure two model IDs remain cached without
  evicting each other in steady-state usage.

### 9) Final paper/repo review synthesis (actionability)

Primary-source refresh across Qwen3-ASR, Swift ports, and ASR decoding papers:

- Lossless speculative decoding papers remain directionally relevant, but local
  evidence still shows no win on current short/10s workloads; keep experimental.
- Whisper/WhisperX-style long-form segmentation + alignment remains the highest
  practical algorithmic lane for post-recording quality stability.
- No paper-backed shortcut was found that can safely replace the current
  correctness-first gates for MRoPE/audio injection/long-context handling.

### 10) Post-change correctness checkpoint

- Ran release gate after final cache/research pass:
  - `RUN_REFERENCE_PARITY=1 python scripts/quality_gate.py --mode release`
  - Result: PASS (full pytest + reference parity lane green).

### 11) Native max_length mel path hardening

- `compute_features(..., padding="max_length")` now uses the native custom mel
  path (with deterministic zero-padding to 3000 frames for short clips), rather
  than routing through HF extractor.
- Compatibility fallback to HF remains for uncommon padding modes.
- Result:
  - reduced HF extractor dependency in common feature paths,
  - preserved `feature_lens` behavior and existing tests.

### 12) Quantization matrix refresh with speaker-balanced defaults

- Ran `scripts/benchmark_quantization_matrix.py` with:
  - configs: `fp16,4:64,8:64`
  - benchmark runs: `5`
  - eval lane: `test-clean`, `n=100`, `speaker_round_robin`
- Artifacts:
  - `docs/benchmarks/2026-02-14-quant-matrix-speaker100.json`
  - `docs/benchmarks/2026-02-14-quant-matrix-speaker100.md`
- Snapshot:
  - `4bit-g64` long speedup vs fp16: `4.68x`, WER delta `+0.004316`
  - `8bit-g64` long speedup vs fp16: `3.11x`, WER delta `+0.000432`

### 13) Audio frontend cache cleanup (native path hot-loop)

- Added cache for mel filterbank asset loading (`_mel_filters_np`) so repeated
  chunk processing does not re-open/parse `mel_filters.npz`.
- Added cache for Hann window creation (`_hann_window`) used by STFT.
- Added regression tests to ensure cache behavior is active.

### 14) Latest post-change release checkpoint

- Re-ran full release gate after native max_length path + audio frontend cache
  cleanups:
  - `RUN_REFERENCE_PARITY=1 python scripts/quality_gate.py --mode release`
  - Result: PASS.

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
