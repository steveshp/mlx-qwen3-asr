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
  - FLEURS multilingual manifest-quality lane:
    - 100 short-form samples (10 languages x 10),
    - Unicode-safe WER/CER + language-aware primary metric.
  - Long-form multilingual manifest-quality lane:
    - 10 synthetic concatenated clips (~75-90s each, 10 languages),
    - Unicode-safe WER/CER with language-aware primary metric
      (CER for zh/ja/ko; WER otherwise).

## Valid Gaps (prioritized)

1. `P0` WER/CER on `Qwen/Qwen3-ASR-1.7B`
- Why: verifies larger-model quality envelope, not just 0.6B.
- Status: not yet run in committed artifacts.

2. `P0` WER/CER on LibriSpeech `test-other`
- Why: harder-noise/accent lane needed for robustness claims.
- Status: parity exists; WER lane not yet expanded.

3. `P0` Non-English quality lane (beyond token parity)
- Why: token parity != ground-truth accuracy. Need quality metrics per language.
- Status: partial. We now have multilingual quality lanes for both short-form
  (`n=100`) and long-form synthetic (`n=10`) across 10 languages, but still
  need larger-scale and non-synthetic ground-truth coverage.

4. `P0` Long-form quality lane (`>30s`, multi-minute)
- Why: chunking/context behavior can drift differently than short utterances.
- Status: partial. Synthetic long-form quality lane now exists (`n=10`), but
  we still need larger and/or real-world long-form ground-truth datasets.

5. `P1` Direct MLX-vs-PyTorch quality comparison on same samples
- Why: closes the loop on backend parity at transcript quality level.
- Status: token parity is in place; WER-side backend comparison lane missing.

6. `P1` Real-world audio lane (meetings/podcasts/accented speech)
- Why: LibriSpeech-only is too clean for deployment confidence.
- Status: not yet part of committed evaluation matrix.

## Follow-up Order

1. Add `test-other` WER/CER (0.6B, fp16 + key quant profiles).
2. Add 1.7B WER/CER on test-clean + test-other.
3. Add long-form lane with deterministic multi-minute concatenations and WER/CER.
4. Add multilingual quality lane with a pinned dataset/manifest and language-wise scores.
5. Add backend quality comparison lane (MLX vs PyTorch, same eval set).
6. Add real-world curated lane with fixed artifact set and versioned manifests.
