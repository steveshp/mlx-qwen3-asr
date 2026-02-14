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
- Add release-only full split quality evaluation (`test-clean`, `test-other`) before tags.
- Add forced-aligner backend parity lane (`mlx` vs `qwen_asr`) as dataset
  coverage expands beyond English deterministic samples.

## Reference Parity Suite Command

```bash
python scripts/eval_reference_parity_suite.py \
  --model Qwen/Qwen3-ASR-0.6B \
  --subsets test-clean,test-other \
  --samples-per-subset 5 \
  --include-long-mixes \
  --include-noise-variants \
  --noise-snrs-db 10,5 \
  --long-mixes 2 \
  --json-output docs/benchmarks/reference-parity-suite.json
```

Optional multilingual extension:
- pass `--manifest-jsonl path/to/manifest.jsonl` with records containing
  `audio_path` and optional `language` labels.
- optional fail thresholds:
  - `--fail-match-rate-below` for strict token parity
  - `--fail-text-match-rate-below` for Unicode-safe normalized-text parity

### Reproducible multilingual manifest (FLEURS)

Build deterministic multilingual samples (local WAV + JSONL manifest):

```bash
python scripts/build_multilingual_manifest.py \
  --languages en_us,zh_cn,ja_jp,de_de,fr_fr,es_419,ru_ru,ar_eg,hi_in,ko_kr \
  --samples-per-language 2 \
  --output-manifest docs/benchmarks/fleurs-multilingual-manifest.jsonl
```

Implementation note:
- `build_multilingual_manifest.py` downloads FLEURS TSV/audio tar assets directly
  from the Hugging Face Hub (no `datasets` script loader dependency).

Run manifest-only parity suite:

```bash
python scripts/eval_reference_parity_suite.py \
  --subsets '' \
  --samples-per-subset 1 \
  --manifest-jsonl docs/benchmarks/fleurs-multilingual-manifest.jsonl \
  --json-output docs/benchmarks/reference-parity-suite-multilingual.json
```

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
