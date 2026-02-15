# Speaker Diarization Research Review (2026-02-15)

Research objective: identify the best implementation strategy for native MLX
speaker diarization in `mlx-qwen3-asr`, with practical guidance for an initial
production-quality release.

Implementation companion: `docs/DIARIZATION_IMPLEMENTATION_SPEC_2026-02-15.md`.

Verification date: 2026-02-15.
Method: direct review of upstream GitHub repositories, framework docs, and
primary papers listed in "Primary Sources".

## Verified Snapshot (Source-Backed)

- `pyannote-audio` currently presents production-ready diarization pipelines and
  benchmark reporting, and pyannote tooling/docs expose explicit speaker-count
  controls (`num_speakers`, `min_speakers`, `max_speakers`).
- NeMo diarization configs expose multiscale controls (`window_length_in_sec`,
  `shift_length_in_sec`, `multiscale_weights`) and clustering controls
  (`oracle_num_speakers`, `max_num_speakers`) in a modular pipeline.
- VBx repo documents the common AHC -> VB-HMM flow and explicitly notes AHC
  runtime limitations for very long recordings.
- `dscore` provides standard diarization metrics used in evaluation reporting,
  including DER, JER, and B-cubed.
- `whisperX` exposes diarization UX in ASR pipelines, including CLI flags for
  bounded speaker count estimation (`--min_speakers`, `--max_speakers`).
- ECAPA-TDNN and TitaNet remain strong practical references for compact speaker
  embedding backbones; EEND-family remains valuable but higher-risk for v1.

## Executive Conclusion

Best near-term path is a **native cascaded pipeline**:

1. speech activity segmentation,
2. MLX speaker embedding extraction (ECAPA/TitaNet-style),
3. clustering + optional refinement,
4. assignment onto aligned words/chunks.

This is the best fit for this repository because it is:
- compatible with your current offline ASR + forced aligner pipeline,
- feasible without large diarization-label training infrastructure,
- incrementally shippable behind an optional extra (`[diarize]`),
- consistent with your "native-only" product position.

End-to-end diarization (EEND-family) should be a later R&D lane, not v1.

## GitHub/Framework Findings

## 1) pyannote ecosystem

Observed pattern:
- Segmentation + speaker embeddings + global agglomerative clustering.
- Supports known/unknown speaker count controls in production UX.
- Upstream materials include benchmark reporting for current pipeline variants.

Why it matters:
- Confirms that a cascaded architecture is still state-of-practice.
- Provides a reference for speaker-count UX (`num/min/max speakers`).

## 2) NVIDIA NeMo diarization stack

Observed pattern:
- Explicit module graph: VAD, speaker embeddings, clustering, ASR.
- Multiscale embedding windows and shifts are first-class config.
- Long-form controls exist (chunked clustering over embeddings).
- Includes both clustering and neural MSDD paths.
- Config surfaces explicit speaker-count controls (oracle and bounded auto).

Why it matters:
- Strong template for a config-driven diarization subsystem.
- Multiscale embeddings are a practical quality lever worth adopting.

## 3) VBx recipe (BUTSpeechFIT)

Observed pattern:
- x-vector extraction -> AHC initialization -> VB-HMM refinement.
- Widely used benchmark baseline.
- AHC becomes slow on long recordings; even upstream docs note this.

Why it matters:
- Two-stage clustering is robust, but long-form scaling must be explicit.
- Suggests keeping refinement optional in v1.

## 4) SpectralCluster

Observed pattern:
- Practical constrained spectral clustering with runtime knobs and autotuning.
- Includes explicit speed/quality tradeoff guidance for large inputs.

Why it matters:
- Good design reference for clustering APIs and scalability controls.
- Useful fallback if plain AHC underperforms on difficult data.

## 5) WhisperX integration lessons

Observed pattern:
- ASR -> forced alignment -> diarization -> word-speaker assignment.
- Explicit user controls for min/max speakers.
- Publicly documented limitations on overlap and diarization quality.

Why it matters:
- Confirms assignment stage and UX users expect.
- Reinforces need for clear quality caveats in early releases.

## 6) WeSpeaker ecosystem

Observed pattern:
- Speaker-embedding-centric toolkit with diarization task support.
- Recent updates include UMAP/HDBSCAN recipe changes for VoxConverse.

Why it matters:
- Embedding-centric design is mature and production-friendly.
- Clustering backends should be pluggable, not hard-coded.

## Paper Findings (Actionable)

## 1) ECAPA-TDNN (Interspeech 2020)

Key point:
- Strong speaker embedding architecture built on x-vector/TDNN family with
  Res2Net + SE + improved statistics pooling.

Action for MLX:
- Prefer ECAPA-style embedding encoder for v1 portability/quality balance.

## 2) TitaNet (2021)

Key point:
- Strong diarization-relevant performance with lightweight variants
  (TitaNet-S ~6M parameters).

Action for MLX:
- Keep TitaNet-S as a second embedding-backbone candidate if ECAPA latency is
  too high.

## 3) EEND line (2019-2024+)

Key points:
- EEND directly models diarization and overlap, avoiding clustering limits.
- Known practical challenge: generalization to unseen speaker counts.

Action for MLX:
- Keep as R&D lane for overlap-heavy scenarios, not initial release path.

## 4) VBx (2022 + 2023 discriminative training)

Key points:
- VBx remains a widely adopted baseline and challenge reference.
- Discriminative training improves correlation to DER objective.

Action for MLX:
- Use VBx ideas to motivate optional refinement stage in later versions, after
  baseline clustering is stable.

## 5) Sortformer (ICML 2025)

Key points:
- Introduces Sort Loss for permutation-resolved supervision.
- Explicitly frames diarization-to-ASR bridging ("timestamps to tokens").

Action for MLX:
- Confirms strategic value of tight ASR+diarization integration in this repo.
- Keep as future direction once baseline diarization ships.

## Recommended MLX Implementation

## Phase 1 (ship-worthy baseline)

1. Offline-only scope:
- No streaming/mic diarization in v1.

2. Embedding model:
- Port compact ECAPA-TDNN-like encoder to MLX.

3. Segmentation:
- Start from speech regions already available in pipeline context and/or simple
  VAD; add neural VAD later if needed.
- Support multiscale windows (inspired by NeMo config patterns).

4. Clustering:
- Start with cosine AHC + thresholding.
- Add bounded speaker-count controls:
  - `num_speakers`,
  - `min_speakers`,
  - `max_speakers`.

5. Attribution:
- Assign speaker IDs to word-level aligned segments.
- Build contiguous `speaker_segments` for final transcript rendering.

6. Output/API:
- Add diarization fields in `TranscriptionResult` and JSON output.
- Keep backward compatibility for existing fields.

## Phase 2 (quality and long-form hardening)

1. Long recording scalability:
- Add chunked embedding/clustering controls.

2. Clustering upgrades:
- Add spectral/constraint-based alternative and compare.

3. Optional refinement:
- Evaluate VB-style temporal smoothing/refinement if speaker flips remain high.

## Phase 3 (advanced)

1. Overlap-aware diarization and overlap word assignment.
2. Research lane for EEND/Sortformer-style speaker-aware ASR integration.

## Evaluation Plan (Must-Have)

Use `dscore`-style metrics and report at minimum:
- DER,
- JER,
- B3 Precision/Recall/F1.

Report both:
- strict overlap-included settings,
- and collar/ignore-overlap variants (for comparability to external reports).

Datasets to target for public comparability:
- AMI,
- CALLHOME,
- DIHARD (where licensing/availability allows),
- plus your own long-form regression fixtures for runtime behavior.

## Source Notes

- Performance numbers across papers/repos are not directly comparable unless
  collars, overlap handling, VAD assumptions, and evaluation protocol match.
- "Best implementation" here is inferred for `mlx-qwen3-asr` constraints:
  native runtime, maintainable pipeline, and optional dependency policy.

## Primary Sources

GitHub / docs:
- https://github.com/pyannote/pyannote-audio
- https://docs.pyannote.ai/features
- https://www.isca-archive.org/interspeech_2023/bredin23_interspeech.html
- https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_diarization/configs.html
- https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_meeting.yaml
- https://github.com/BUTSpeechFIT/VBx
- https://github.com/wq2012/SpectralCluster
- https://github.com/nryant/dscore
- https://github.com/m-bain/whisperX
- https://github.com/wenet-e2e/wespeaker
- https://github.com/huggingface/diarizers

Papers:
- https://arxiv.org/abs/2005.07143 (ECAPA-TDNN)
- https://arxiv.org/abs/2110.04410 (TitaNet)
- https://arxiv.org/abs/2003.02966 (EEND)
- https://arxiv.org/abs/2310.02732 (Discriminative VBx)
- https://arxiv.org/abs/2409.06656 (Sortformer)
- https://arxiv.org/abs/2203.17068 (EEND-SS)
- https://arxiv.org/abs/2309.06672 (AED-EEND)
