# Native Speaker Diarization Plan (2026-02-15)

## Verdict

Your proposal is directionally correct and high-ROI:

- Speaker diarization is a top user request after baseline ASR.
- Native MLX implementation aligns with project identity.
- Packaging as an optional extra (`mlx-qwen3-asr[diarize]`) keeps core install lean.
- Wrapping pyannote/silero would weaken the project's "fully native" differentiation.

Recommendation: proceed with native-only diarization, but gate rollout by measurable diarization quality and latency targets.

Companion research document: `docs/DIARIZATION_RESEARCH_2026-02-15.md` (GitHub
and paper survey + implementation rationale).
Execution spec: `docs/DIARIZATION_IMPLEMENTATION_SPEC_2026-02-15.md` (concrete
API/module/test/eval plan).
One-page index: `docs/DIARIZATION_BRIEF_2026-02-15.md`.

## Status Update (Post-Implementation, 2026-02-15)

- Experimental offline diarization baseline is now landed in runtime:
  - API: `transcribe(..., diarize=True, ...)` and `Session.transcribe(..., diarize=True, ...)`
  - CLI: `--diarize`, `--num-speakers`, `--min-speakers`, `--max-speakers`, `--diarization-window-sec`, `--diarization-hop-sec`
  - Output: `TranscriptionResult.speaker_segments`, optional speaker labels on word `segments`, JSON writer support
- Guardrails landed: diarization is offline-only and explicitly rejected for `--streaming` and `--mic`.
- Remaining stabilization work is quality-gate focused (DER/JER + performance envelope + overlap handling), not contract plumbing.

## Initial Review Findings (Pre-Implementation Snapshot)

### High severity

1. Quality risk without explicit diarization metrics/gates
- Current quality gates focus on ASR/parity/timestamps, not diarization quality.
- Shipping diarization without DER/JER policy will create support churn quickly.

2. Output contract was not diarization-ready
- Initial state: `TranscriptionResult` exposed `text`, `language`, `segments`, `chunks`.
- Initial state: no schema for speaker labels or speaker-attributed segments.
- Current status: resolved in baseline via `speaker_segments` and speaker labels on timestamp segments; JSON writer includes diarization fields.

3. Streaming/mic paths are incompatible with first-pass diarization scope
- CLI explicitly disallows timestamps in streaming/mic modes.
- Diarization should initially target offline transcription only.

### Medium severity

1. Speaker count estimation can degrade quickly in real audio
- Unconstrained clustering over noisy embeddings can over/under-cluster.
- Need explicit support for `num_speakers` override and bounded auto-estimation (`min/max`).

2. Overlap speech handling should be staged
- First release should declare non-overlap-primary behavior.
- Full overlap-aware assignment can come in a later phase.

## Architecture Fit (Current Codebase)

Natural insertion points:

1. Offline pipeline hook (`mlx_qwen3_asr/transcribe.py`)
- Current flow: ASR decode -> optional forced aligner -> return result.
- Diarization should run after or alongside timestamp alignment to attach speaker IDs to time spans.

2. Session parity (`mlx_qwen3_asr/session.py`)
- Any diarization option added to `transcribe()` should be mirrored in `Session.transcribe()`.

3. CLI contract (`mlx_qwen3_asr/cli.py`)
- Add offline-only flags (`--diarize`, `--num-speakers`, `--min-speakers`, `--max-speakers`, `--diarization-window-sec`, `--diarization-hop-sec`).
- Keep explicit errors for unsupported combinations (`--streaming --diarize`, `--mic --diarize`) in v1.

4. Output writers (`mlx_qwen3_asr/writers.py`)
- JSON should include speaker-labeled segments.
- SRT/VTT/TSV should optionally emit speaker prefixes/columns.

## Native-Only Technical Approach

## 1) Acoustic embedding backend (native)

- Current baseline (implemented): lightweight native acoustic embeddings from
  windowed spectral/statistical features (dependency-light, no external model weights).
- Future upgrade path: port a compact ECAPA-TDNN-style speaker encoder to MLX
  behind the same diarization contract.
- Keep runtime path native, consistent with ASR/aligner philosophy.

## 2) Segmentation + embedding extraction

- Use fixed windows over waveform (e.g., 1.5s window, 0.75s hop) with simple energy filtering.
- Extract L2-normalized embeddings per window.

## 3) Clustering

- Baseline uses:
  - fixed speaker count: cosine k-means,
  - auto speaker count: thresholded online clustering with min/max bounds.
- Provide:
  - fixed `num_speakers` mode (user-provided),
  - bounded auto mode (`min_speakers`, `max_speakers`) using a distance threshold.

## 4) Attribution to transcript

- Map clustered speaker turns onto aligned word segments (when timestamps available).
- Build contiguous speaker-attributed utterance segments:
  - `{speaker, start, end, text}`.
- Keep original word segments for detailed use cases.

## 5) API surface (implemented baseline)

- `transcribe(..., diarize=False, diarization_num_speakers=None, diarization_min_speakers=1, diarization_max_speakers=8)`
- Extend `TranscriptionResult` with:
  - `speaker_segments: Optional[list[dict]]`
  - optional speaker labels on word `segments`

## 6) Packaging

- Add optional dependency group:
  - `diarize = [...]`
- Core package remains unchanged for users who do not need diarization.

Note: no pyannote/silero runtime wrappers in core diarization path.

## Acceptance Gates Before "Stable"

1. Quality
- Add DER/JER benchmark lane on at least one public diarization dataset.
- Define release thresholds and regression policy in `docs/QUALITY_GATE.md`.

2. Performance
- Measure overhead on 10s, 60s, and long-form clips; publish artifacts in `docs/benchmarks/`.

3. Reliability
- Deterministic tests for speaker assignment stability and segment monotonicity.
- CLI/API tests for unsupported mode combinations and output schema guarantees.

## Phased Rollout

1. Phase 1: Experimental offline diarization
- Native acoustic embedding baseline + clustering + JSON speaker segments.
- No streaming/mic diarization.

2. Phase 2: Output parity and UX hardening
- SRT/VTT/TSV speaker presentation.
- Better auto speaker count heuristics and error messages.

3. Phase 3: Advanced behavior
- Overlap-aware diarization, streaming-compatible strategy, stronger multilingual robustness.

## Final Assessment

"Native-only + optional extra" is the right product decision for this repository.
The key execution requirement is to treat diarization as a quality-gated subsystem
(like timestamps), not just a feature toggle.

## My Final Thoughts

- This feature can be a true differentiator only if it is native and integrated
  deeply with your existing timestamp/alignment path.
- The fastest "good" implementation is a cascaded embedding+clustering design,
  not end-to-end neural diarization on day one.
- The main failure mode is not model size; it is weak evaluation discipline.
  Treat DER/JER gates like you treated ASR parity gates.
