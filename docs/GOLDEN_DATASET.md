# Golden Dataset Strategy

This repository uses a two-level quality evaluation strategy:

1. Fast gate (always on PR): unit/integration tests for correctness regressions.
2. Golden regression lane (scheduled/manual): benchmark-quality WER/CER checks.

## Why this exists

- Unit tests catch logic bugs quickly.
- Golden regression catches real transcription quality drift.
- Both are required for enterprise-grade reliability.

## Golden Suite v1

Current default suite: **LibriSpeech test-clean** (OpenSLR 12), deterministic
speaker-balanced sample.

- Source archive: `https://www.openslr.org/resources/12/test-clean.tar.gz`
- Selection: deterministic **speaker-balanced round-robin** over sorted speaker IDs
  (fallback mode `sequential` retained for compatibility/testing)
- Metric: corpus WER + CER
- Intended model for gate: `Qwen/Qwen3-ASR-0.6B` (faster, stable CI runtime)

## Command

```bash
python scripts/eval_librispeech.py \
  --subset test-clean \
  --samples 100 \
  --sampling speaker_round_robin \
  --model Qwen/Qwen3-ASR-0.6B \
  --dtype float16 \
  --json-output docs/benchmarks/golden-librispeech.json
```

Fail gate on quality regression:

```bash
python scripts/eval_librispeech.py \
  --subset test-clean \
  --samples 100 \
  --sampling speaker_round_robin \
  --model Qwen/Qwen3-ASR-0.6B \
  --fail-wer-above 0.05
```

## CI / Nightly policy

- PR CI stays fast and required.
- Nightly regression runs:
  - `scripts/quality_gate.py --mode fast`
  - `scripts/eval_librispeech.py` on deterministic speaker-balanced subset
  - `scripts/benchmark_asr.py` for latency + RTF trend tracking
- Artifacts are uploaded per run for historical comparison.

## Future expansions

- Add multilingual golden lane once non-gated, script-free datasets are pinned.
- Add release-only full split evaluation (`test-clean`, `test-other`) before tags.
- Add forced-aligner backend parity lane (`mlx` vs `qwen_asr`) as dataset
  coverage expands beyond English deterministic samples.

## Aligner Parity Command

```bash
python scripts/eval_aligner_parity.py \
  --subset test-clean \
  --samples 50 \
  --model Qwen/Qwen3-ForcedAligner-0.6B \
  --fail-text-match-rate-below 1.0 \
  --fail-timing-mae-ms-above 60.0 \
  --json-output docs/benchmarks/aligner-parity.json
```
