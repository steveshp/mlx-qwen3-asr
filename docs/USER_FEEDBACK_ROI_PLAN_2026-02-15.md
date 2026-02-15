# User Feedback ROI Plan (2026-02-15)

This document maps user-facing feedback to ROI, implementation status, and next actions.

## Completed in this patch

| Feedback item | ROI | Status | Notes |
|---|---|---|---|
| Progress indication for long files | High | Done | Added chunk-level progress/ETA in CLI and callback events in Python API. |
| Stdout vs file output control (`--quiet`, `--stdout-only`) | High | Done | Added `--quiet`, `--stdout-only`, and `--no-output-file` alias. |
| Stdout-only without file clutter | High | Done | `--stdout-only` skips output file writes entirely. |
| Silent first-run download | High | Done | CLI now prints explicit first-run model download notice before model resolution. |
| `--list-languages` discoverability | High | Done | Added `--list-languages` with canonical names + aliases/codes. |
| Microphone live transcription (`--mic`) | Very high | Done | Added `--mic` capture flow with optional device/duration/sample-rate controls. |
| Subtitle grouping (word-level -> phrase-level) | High | Done | Added subtitle cue grouping heuristics for SRT/VTT writers. |
| SRT fallback without timestamps is unusable | High | Done | CLI auto-enables timestamps for SRT/VTT offline output; writers now fail fast if segments are missing. |
| Python progress hooks | High | Done | Added `on_progress` callback to `transcribe()` and `Session.transcribe()`. |
| Chunk-level iteration without timestamps | High | Done | Added `return_chunks=True` and `TranscriptionResult.chunks`. |
| Session model metadata/introspection | Medium | Done | Added `Session.model_info` property. |
| Async/await support | High | Done | Added `transcribe_async`, `transcribe_batch_async`, and `Session.transcribe_async`. |
| Batch transcription API | High | Done | Added `transcribe_batch()` with shared loaded components. |
| Language output consistency | Medium | Done | Canonicalized detected language naming. |
| Verbose RTF speed reporting | Medium | Done | Added RTF in verbose CLI output. |

## Deferred / external actions

| Feedback item | ROI | Status | Reason |
|---|---|---|---|
| Language detection confidence score | Medium | Deferred | Current model output path does not expose a calibrated confidence metric; needs design/validation work. |
| Mixed-language code-switch support | Medium | Deferred | Requires decoder/prompting strategy redesign and multilingual segmentation policy. |
| Speaker diarization ("who said what") | Very high | Done (experimental) | Native-only optional extra (`[diarize]`) shipped in offline API/CLI baseline; DER/JER quality gates remain the stabilization gate before non-experimental status. |
| Built-in model comparison CLI | Low-Medium | Deferred | Useful but non-blocking; current API now supports scripting this with `transcribe_batch()`. |
| CLI one-command quantization (`--quantize`) | Medium | Deferred | Existing conversion path is in `scripts/convert.py`; integrating conversion UX into runtime CLI is separate work. |
| Pre-quantized model IDs on HuggingFace | High | External | Requires publishing artifacts to HF (release/distribution process), not just local code changes. |

## Validation

- Lint: `uv run ruff check mlx_qwen3_asr tests`
- Tests: `uv run pytest -q` (all passing in this workspace)

## Diarization docs bundle

- `docs/DIARIZATION_BRIEF_2026-02-15.md`
- `docs/DIARIZATION_NATIVE_PLAN_2026-02-15.md`
- `docs/DIARIZATION_RESEARCH_2026-02-15.md`
- `docs/DIARIZATION_IMPLEMENTATION_SPEC_2026-02-15.md`
