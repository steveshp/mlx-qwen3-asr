# Native Diarization Implementation Spec (2026-02-15)

This document translates the diarization research/review into a concrete build
plan for `mlx-qwen3-asr`.

Related docs:
- Strategy and product assessment: `docs/DIARIZATION_NATIVE_PLAN_2026-02-15.md`
- GitHub/paper research: `docs/DIARIZATION_RESEARCH_2026-02-15.md`

## Implementation status snapshot (2026-02-15)

- M1 (API/contracts) is complete and merged.
- M2 baseline is complete and merged with native dependency-light acoustic embeddings + clustering.
- M3 is complete: DER/JER evaluation harness + quality-gate hook are landed and
  initial versioned benchmark artifacts are committed.
- Latest verification rerun
  (`docs/benchmarks/2026-02-15-diarization-quality-20-v3.json`) matches the
  canonical baseline DER/JER after rejecting a regressed experiment
  (`docs/benchmarks/2026-02-15-diarization-quality-20-v2.json`).
- Runtime update accepted in
  `docs/benchmarks/2026-02-15-diarization-quality-20-v5.json`:
  DER `0.5051` / JER `0.4646` (improved from DER `0.6762` / JER `0.6065`).
- M4 remains open: stabilization hardening and graduation criteria.

## Scope

## In scope (v1)

- Offline diarization only (non-streaming).
- Native dependency-light acoustic embeddings (no external diarization runtime stack).
- Clustering-based speaker assignment.
- Speaker-attributed transcript outputs in Python API + CLI JSON output.
- Optional dependency group (`[diarize]`) only if needed for lightweight utilities.

## Out of scope (v1)

- Runtime wrappers around pyannote/silero.
- Streaming/mic diarization.
- Full overlap-aware multi-label speaker assignment.
- EEND/MSDD model training and deployment.

## Public API Changes

## Python API (`transcribe`, `Session.transcribe`)

Implemented kwargs:

- `diarize: bool = False`
- `diarization_num_speakers: int | None = None`
- `diarization_min_speakers: int = 1`
- `diarization_max_speakers: int = 8`
- `diarization_window_sec: float = 1.5`
- `diarization_hop_sec: float = 0.75`

Validation rules:

- `diarization_num_speakers` must be `>=1` if provided.
- `diarization_min_speakers >= 1`.
- `diarization_max_speakers >= diarization_min_speakers`.
- If `diarization_num_speakers` is set, fixed-speaker clustering is used.

## `TranscriptionResult` additions

- `speaker_segments: list[dict] | None = None`
- optional `"speaker"` field on timestamp `segments` when both timestamps and
  diarization are enabled.

`speaker_segments` item shape:

```json
{
  "speaker": "SPEAKER_00",
  "start": 12.34,
  "end": 16.02,
  "text": "Thanks everyone for joining."
}
```

## CLI Changes

New flags:

- `--diarize`
- `--num-speakers`
- `--min-speakers`
- `--max-speakers`
- `--diarization-window-sec`
- `--diarization-hop-sec`

Mode constraints:

- Error for `--streaming --diarize`.
- Error for `--mic --diarize`.

## Output Contract Changes

## JSON (`writers.py`)

When diarization is enabled:

- include `speaker_segments`.
- include speaker labels on `segments` if word timestamps exist.

## SRT/VTT/TSV

v1 recommendation:

- keep default behavior unchanged,
- add optional speaker-prefix formatting in a follow-up phase.

Reason: preserve compatibility and keep first release minimal-risk.

## Internal Module Plan

Implemented module:

1. `mlx_qwen3_asr/diarization.py`
- windowing + embedding extraction,
- clustering and speaker count estimation,
- assignment to timestamped words/chunks,
- merge to speaker turns.

Touch points:

- `mlx_qwen3_asr/transcribe.py`
- `mlx_qwen3_asr/session.py`
- `mlx_qwen3_asr/cli.py`
- `mlx_qwen3_asr/writers.py`

## Pipeline Integration Order

1. ASR decode (existing).
2. Optional forced alignment (existing timestamp path).
3. Diarization over raw chunk/full audio.
4. Speaker assignment onto aligned words.
5. Merge words into speaker turns.
6. Return structured output.

If timestamps are disabled and diarization is requested:

- either auto-enable timestamps,
- or perform chunk-level assignment only with explicit warning.

Recommended default: auto-enable timestamps when `diarize=True`.

## Clustering Strategy

v1 baseline:

- fixed-speaker mode: cosine k-means.
- auto-speaker mode: thresholded online clustering with bounded min/max fallback.
- temporal smoothing is applied only in auto-speaker mode.

v1.1 options:

- add spectral clustering backend for hard cases.
- add simple temporal smoothing to reduce rapid speaker flips.

Scaling controls:

- cap embedding windows per recording via stride and silence filtering.
- chunk long recordings, then reconcile cluster IDs across chunks.

## Model/Asset Strategy

- Current baseline uses local acoustic feature embeddings (no external model assets).
- Future upgrade path: swap embedding backend behind the same diarization contract.

## Evaluation and Quality Gates

Metrics:

- DER
- JER
- B3 Precision/Recall/F1
- speaker-flip rate on contiguous text spans (project-specific stability metric)

Benchmark lanes:

1. Public comparability lane:
- AMI / CALLHOME / DIHARD where licensing permits.

2. Product lane:
- meeting/interview/podcast-like long-form internal fixtures.

3. Performance lane:
- 10s, 60s, and multi-minute latency overhead artifacts.

Release gate recommendation:

- No release without published versioned diarization benchmark artifact in
  `docs/benchmarks/`.
- Tooling implemented:
  - `scripts/eval_diarization.py` (DER/JER evaluator from JSONL manifests).
  - `scripts/quality_gate.py` hook via `RUN_DIARIZATION_QUALITY_EVAL=1`.
- Baseline artifacts committed:
  - `docs/benchmarks/2026-02-15-diarization-manifest-20.jsonl`
  - `docs/benchmarks/2026-02-15-diarization-quality-20.json`
  - `docs/benchmarks/2026-02-15-diarization-quality-20.md`
- Quality-history artifacts:
  - `docs/benchmarks/2026-02-15-diarization-quality-20-v2.json` (regressed; not adopted)
  - `docs/benchmarks/2026-02-15-diarization-quality-20-v3.json` (post-rollback verification)
  - `docs/benchmarks/2026-02-15-diarization-quality-20-v4.json` (accepted runtime improvement)
  - `docs/benchmarks/2026-02-15-diarization-quality-20-v5.json` (accepted runtime improvement)

## Test Plan

Unit tests:

- windowing math and boundary correctness.
- clustering determinism on synthetic embeddings.
- speaker turn merge logic.
- CLI argument validation and unsupported mode combinations.
- writer schema includes diarization fields when expected.

Integration tests:

- `transcribe(..., diarize=True)` happy path with mocked components.
- `Session.transcribe(..., diarize=True)` parity with top-level API.
- JSON output schema regression tests.

Non-regression tests:

- ensure existing non-diarization flows are unchanged.

## Milestone Breakdown

## M1: API + data contracts [DONE]

- Add kwargs, dataclass fields, CLI flags, JSON schema.
- Add validation and mode guardrails.

## M2: Native embedding + baseline clustering [DONE]

- Land native embedding inference path.
- Implement clustering and speaker turn construction.

## M3: Evaluation harness + benchmark artifacts [DONE]

- Add scripts and docs for DER/JER reporting. [DONE]
- Commit first baseline benchmark artifacts. [DONE]

## M4: Stabilization [PENDING]

- Long-form scaling improvements.
- speaker-flip reduction.
- docs/README promotion from experimental to stable (only if gates pass).

## Risks and Mitigations

Risk: poor speaker count auto-estimation.
- Mitigation: expose `--num-speakers` and bounded min/max.

Risk: overlap-heavy audio degrades attribution.
- Mitigation: declare limitation in v1 and add overlap lane to roadmap.

Risk: long-form clustering runtime grows too much.
- Mitigation: chunk-level embedding aggregation and optional backend upgrades.

## Go/No-Go Checklist

- [x] Native-only runtime path (no external diarization wrapper dependency).
- [x] Offline API + CLI contracts finalized and tested.
- [x] DER/JER benchmark lane implemented.
- [x] Versioned benchmark artifacts committed.
- [x] Documentation updated (README/API/quality gate notes).
