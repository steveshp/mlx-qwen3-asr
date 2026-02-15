# Evaluation Coverage Matrix

This document tracks what is measured today versus what remains before we can
make broad "production-grade across languages/conditions" quality claims.

## Current Measured Coverage

- Token parity:
  - English mixed-condition lane (clean/other/long/noise): 22 samples.
  - Multilingual manifest lanes:
    - smoke: 4 samples (EN/CN/JA/DE),
    - medium: 20 samples (10 languages x 2),
    - expanded: 100 samples (10 languages x 10).
  - Long-form multilingual parity lane:
    - 10 synthetic concatenated clips (~75-90s each, 10 languages),
    - strict token/text parity currently `0.0` (all mismatch; early divergence).
- Aligner parity:
  - 50 LibriSpeech test-clean samples (MLX vs official qwen_asr backend).
- WER/CER lane:
  - LibriSpeech test-clean, 100 samples, speaker-balanced, 0.6B model
    (`fp16`, `4-bit`, `8-bit`).
  - LibriSpeech test-other, 100 samples, speaker-balanced, 0.6B model
    (`fp16`, `4-bit`, `8-bit`).
  - LibriSpeech test-clean + test-other, 100 samples each, 1.7B model
    (`fp16`).
  - FLEURS multilingual manifest-quality lane:
    - 100 short-form samples (10 languages x 10),
    - refreshed for both 0.6B and 1.7B (`fp16`),
    - Unicode-safe WER/CER + language-aware primary metric.
  - Long-form multilingual manifest-quality lane:
    - 10 synthetic concatenated clips (~75-90s each, 10 languages),
    - Unicode-safe WER/CER with language-aware primary metric
      (CER for zh/ja/ko; WER otherwise).
- Streaming diagnostics lane:
  - Per-session quality metrics exposed from runtime state:
    `partial_stability`, `rewrite_rate`, `finalization_delta_chars`.
  - Tooling:
    - `scripts/eval_streaming_metrics.py` (single-run diagnostics probe),
    - `scripts/benchmark_streaming.py` (`streaming_quality` summary payload).

## Valid Gaps (prioritized)

1. `P0` Non-English quality lane (beyond token parity)
- Why: token parity != ground-truth accuracy. Need quality metrics per language.
- Status: partial. We now have multilingual quality lanes for both short-form
  (`n=100`) and long-form synthetic (`n=10`) across 10 languages, but still
  need larger-scale and non-synthetic ground-truth coverage.

2. `P0` Long-form quality lane (`>30s`, multi-minute)
- Why: chunking/context behavior can drift differently than short utterances.
- Status: partial. Synthetic long-form quality lane now exists (`n=10`), but
  we still need larger and/or real-world long-form ground-truth datasets.

3. `P1` Direct MLX-vs-PyTorch quality comparison on same samples
- Why: closes the loop on backend parity at transcript quality level.
- Status: partially closed. Multilingual-100 head-to-head artifact exists; broader
  domains (test-other, long-form real-world) still missing.

4. `P1` Real-world audio lane (meetings/podcasts/accented speech)
- Why: LibriSpeech-only is too clean for deployment confidence.
- Status: not yet part of committed evaluation matrix.

5. `P1` Streaming quality dataset lane (stability/rollback on real conversational audio)
- Why: offline WER/CER does not fully capture live incremental UX quality.
- Status: instrumentation landed; benchmarked dataset coverage is not yet committed.

## Follow-up Order

1. Add larger and/or real-world long-form ground-truth lane (multi-minute, non-synthetic).
2. Expand multilingual quality beyond 10-language x 10-sample manifest.
3. Expand backend quality comparison lanes (MLX vs PyTorch) beyond multilingual-100.
4. Add real-world curated lane with fixed artifact set and versioned manifests.
5. Add streaming-quality dataset lane and commit versioned artifacts for
   `partial_stability`, `rewrite_rate`, and `finalization_delta_chars`.
