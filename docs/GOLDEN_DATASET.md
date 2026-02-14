# Golden Dataset Strategy

This repository uses a two-level quality evaluation strategy:

1. Fast gate (always on PR): unit/integration tests for correctness regressions.
2. Golden regression lane (scheduled/manual): benchmark-quality WER/CER checks.

## Why this exists

- Unit tests catch logic bugs quickly.
- Golden regression catches real transcription quality drift.
- Both are required for enterprise-grade reliability.

## Golden Suite v1

Current default suite: **LibriSpeech test-clean** (OpenSLR 12), deterministic subset.

- Source archive: `https://www.openslr.org/resources/12/test-clean.tar.gz`
- Selection: first `N` utterances in stable lexical order from `*.trans.txt`
- Metric: corpus WER + CER
- Intended model for gate: `Qwen/Qwen3-ASR-0.6B` (faster, stable CI runtime)

## Command

```bash
python scripts/eval_librispeech.py \
  --subset test-clean \
  --samples 100 \
  --model Qwen/Qwen3-ASR-0.6B \
  --dtype float16 \
  --json-output docs/benchmarks/golden-librispeech.json
```

Fail gate on quality regression:

```bash
python scripts/eval_librispeech.py \
  --subset test-clean \
  --samples 100 \
  --model Qwen/Qwen3-ASR-0.6B \
  --fail-wer-above 0.12
```

## CI / Nightly policy

- PR CI stays fast and required.
- Nightly regression runs:
  - `scripts/quality_gate.py --mode fast`
  - `scripts/eval_librispeech.py` on a smaller deterministic subset
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
  --samples 20 \
  --model Qwen/Qwen3-ForcedAligner-0.6B \
  --json-output docs/benchmarks/aligner-parity.json
```
