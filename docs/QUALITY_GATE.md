# Quality Gate

This repo follows a strict order:

1. Match official quality/correctness.
2. Then optimize for Apple Silicon runtime.
3. Then port stable behavior to Swift.

## Modes

### Fast Gate (required for PRs)

```bash
python scripts/quality_gate.py --mode fast
```

Checks:
- Ruff lint on tracked Python files.
- Full pytest suite.
- Typed-core `mypy` gate on selected modules:

```bash
python -m mypy --follow-imports=skip --ignore-missing-imports \
  mlx_qwen3_asr/config.py \
  mlx_qwen3_asr/chunking.py \
  mlx_qwen3_asr/attention.py \
  mlx_qwen3_asr/encoder.py \
  mlx_qwen3_asr/decoder.py \
  mlx_qwen3_asr/model.py
```

### Release Gate (required before tags/releases)

```bash
RUN_REFERENCE_PARITY=1 python scripts/quality_gate.py --mode release
```

Checks:
- Everything in fast gate.
- Reference parity test (`tests/test_reference_parity.py`) with `RUN_REFERENCE_PARITY=1`.
  - By default, missing `torch`/`qwen_asr` now fails release gate.
  - Override only if intentionally running in lightweight mode:
    - `REQUIRE_REFERENCE_PARITY_DEPS=0`
- Default quality-metrics lane (deterministic LibriSpeech sample):
  - runs `scripts/eval_librispeech.py`
  - enforces both WER and CER thresholds (`--fail-wer-above`, `--fail-cer-above`)
  - default envs:
    - `QUALITY_EVAL_SUBSET=test-clean`
    - `QUALITY_EVAL_SAMPLES=20`
    - `QUALITY_EVAL_FAIL_WER_ABOVE=0.10`
    - `QUALITY_EVAL_FAIL_CER_ABOVE=0.06`
  - disable only when explicitly needed:
    - `RUN_QUALITY_EVAL=0`
- Optional aligner parity lane when explicitly enabled:
  - `RUN_ALIGNER_PARITY=1`
  - runs `scripts/eval_aligner_parity.py` on deterministic LibriSpeech samples.

### Strict Release Profile (recommended for highest bar)

Use this before publishing quality/performance claims:

```bash
RUN_STRICT_RELEASE=1 \
MANIFEST_QUALITY_EVAL_JSONL=docs/benchmarks/2026-02-14-fleurs-multilingual-100-manifest.jsonl \
python scripts/quality_gate.py --mode release
```

What strict profile turns on by default:
- Manifest quality lane (`RUN_MANIFEST_QUALITY_EVAL=1`) with required manifest.
- Reference parity suite (`RUN_REFERENCE_PARITY_SUITE=1`) with multilingual defaults:
  - `REFERENCE_PARITY_SUITE_SUBSETS=''`
  - `REFERENCE_PARITY_SUITE_MANIFEST_JSONL` defaults to
    `docs/benchmarks/2026-02-14-fleurs-multilingual-100-manifest.jsonl` if present.
  - regression floors:
    - `REFERENCE_PARITY_SUITE_FAIL_MATCH_RATE_BELOW=0.58`
    - `REFERENCE_PARITY_SUITE_FAIL_TEXT_MATCH_RATE_BELOW=0.61`
- Performance lane (`RUN_PERF_BENCHMARK=1`) using `scripts/benchmark_asr.py`:
  - default audio: `tests/fixtures/test_speech.wav`
  - default fail thresholds:
    - `PERF_BENCH_FAIL_RTF_ABOVE=0.50`
    - `PERF_BENCH_FAIL_LATENCY_MEAN_ABOVE=2.00`
- Streaming quality lane (`RUN_STREAMING_QUALITY_EVAL=1`) using
  `scripts/eval_streaming_metrics.py`:
  - default audio: `tests/fixtures/test_speech.wav`
  - default endpointing modes: `fixed,energy`
  - default fail thresholds:
    - `STREAMING_QUALITY_FAIL_PARTIAL_STABILITY_BELOW=0.85`
    - `STREAMING_QUALITY_FAIL_REWRITE_RATE_ABOVE=0.30`
    - `STREAMING_QUALITY_FAIL_FINALIZATION_DELTA_CHARS_ABOVE=32`

Relevant perf env overrides:
- `PERF_BENCH_AUDIO`
- `PERF_BENCH_MODEL`
- `PERF_BENCH_DTYPE`
- `PERF_BENCH_WARMUP_RUNS`
- `PERF_BENCH_RUNS`
- `PERF_BENCH_MAX_NEW_TOKENS`
- `PERF_BENCH_FAIL_RTF_ABOVE`
- `PERF_BENCH_FAIL_LATENCY_MEAN_ABOVE`
- `PERF_BENCH_JSON_OUTPUT`

Relevant streaming-quality env overrides:
- `RUN_STREAMING_QUALITY_EVAL`
- `STREAMING_QUALITY_AUDIO`
- `STREAMING_QUALITY_MODEL`
- `STREAMING_QUALITY_DTYPE`
- `STREAMING_QUALITY_CHUNK_SIZE_SEC`
- `STREAMING_QUALITY_MAX_CONTEXT_SEC`
- `STREAMING_QUALITY_FINALIZATION_MODE`
- `STREAMING_QUALITY_ENDPOINTING_MODES` (comma-separated; e.g. `fixed,energy`)
- `STREAMING_QUALITY_UNFIXED_CHUNK_NUM`
- `STREAMING_QUALITY_UNFIXED_TOKEN_NUM`
- `STREAMING_QUALITY_FAIL_PARTIAL_STABILITY_BELOW`
- `STREAMING_QUALITY_FAIL_REWRITE_RATE_ABOVE`
- `STREAMING_QUALITY_FAIL_FINALIZATION_DELTA_CHARS_ABOVE`

### Nightly Regression Lane (scheduled/manual)

`Nightly Regression` workflow runs on macOS and tracks:
- Fast gate
- Golden quality sample (`scripts/eval_librispeech.py`, deterministic
  `speaker_round_robin`, default `n=100`)
- Latency/RTF benchmark (`scripts/benchmark_asr.py`)

This lane is intentionally separate from PR CI so day-to-day development stays fast.
Current schedule cadence is weekly (plus manual `workflow_dispatch`) to keep operational overhead low.

### Optional Native Aligner Parity Gate

Use this before changing timestamp backend defaults:

```bash
RUN_REFERENCE_PARITY=1 RUN_ALIGNER_PARITY=1 ALIGNER_PARITY_SAMPLES=10 \
python scripts/quality_gate.py --mode release
```

Default thresholds in the gate hook:
- text-match rate: `>= 1.0`
- timing MAE: `<= 60ms`

### Optional Reference Parity Suite Gate (broader coverage)

Use this to validate token-level parity beyond a single fixture:

```bash
RUN_REFERENCE_PARITY=1 RUN_REFERENCE_PARITY_SUITE=1 \
REFERENCE_PARITY_SUITE_SUBSETS=test-clean,test-other \
REFERENCE_PARITY_SUITE_SAMPLES_PER_SUBSET=3 \
REFERENCE_PARITY_SUITE_INCLUDE_LONG_MIXES=1 \
REFERENCE_PARITY_SUITE_INCLUDE_NOISE_VARIANTS=1 \
REFERENCE_PARITY_SUITE_NOISE_SNRS_DB=10,5 \
python scripts/quality_gate.py --mode release
```

What it adds:
- deterministic multi-speaker sampling from `test-clean` and `test-other`,
- optional synthetic long/mixed-speaker clips,
- optional synthetic noisy variants (white noise at configured SNR values),
- strict token-level match-rate threshold (default `>= 1.0`),
- optional normalized-text match threshold via
  `REFERENCE_PARITY_SUITE_FAIL_TEXT_MATCH_RATE_BELOW`.

Optional envs:
- `REFERENCE_PARITY_SUITE_MANIFEST_JSONL=/abs/path/manifest.jsonl`
- `REFERENCE_PARITY_SUITE_JSON_OUTPUT=docs/benchmarks/reference-parity-suite.json`

For multilingual lanes, generate a deterministic manifest first:

```bash
python scripts/build_multilingual_manifest.py \
  --languages en_us,zh_cn,ja_jp,de_de \
  --samples-per-language 1 \
  --output-manifest docs/benchmarks/fleurs-multilingual-smoke.jsonl

RUN_REFERENCE_PARITY=1 RUN_REFERENCE_PARITY_SUITE=1 \
REFERENCE_PARITY_SUITE_SUBSETS='' \
REFERENCE_PARITY_SUITE_INCLUDE_LONG_MIXES=0 \
REFERENCE_PARITY_SUITE_MANIFEST_JSONL=docs/benchmarks/fleurs-multilingual-smoke.jsonl \
REFERENCE_PARITY_SUITE_FAIL_MATCH_RATE_BELOW=0.0 \
REFERENCE_PARITY_SUITE_FAIL_TEXT_MATCH_RATE_BELOW=0.0 \
python scripts/quality_gate.py --mode release
```

Current status:
- this lane is required by default when `RUN_STRICT_RELEASE=1` and otherwise
  remains opt-in for lighter release checks.

### Optional Manifest Quality Gate (multilingual/long-form WER/CER)

Use this when you have a JSONL manifest with `audio_path` + `reference_text`
and want direct quality metrics instead of token parity:

```bash
RUN_MANIFEST_QUALITY_EVAL=1 \
MANIFEST_QUALITY_EVAL_JSONL=docs/benchmarks/2026-02-14-fleurs-longform-10x75-manifest.jsonl \
MANIFEST_QUALITY_EVAL_FAIL_PRIMARY_ABOVE=0.35 \
python scripts/quality_gate.py --mode release
```

Behavior:
- runs `scripts/eval_manifest_quality.py`,
- computes Unicode-safe WER and CER,
- uses language-aware primary metric:
  - CER for Chinese/Japanese/Korean,
  - WER otherwise.

Optional envs:
- `MANIFEST_QUALITY_EVAL_MODEL` (default `Qwen/Qwen3-ASR-0.6B`)
- `MANIFEST_QUALITY_EVAL_DTYPE` (default `float16`)
- `MANIFEST_QUALITY_EVAL_MAX_NEW_TOKENS` (default `1024`)
- `MANIFEST_QUALITY_EVAL_FAIL_WER_ABOVE`
- `MANIFEST_QUALITY_EVAL_FAIL_CER_ABOVE`
- `MANIFEST_QUALITY_EVAL_FAIL_PRIMARY_ABOVE` (default `0.35`)
- `MANIFEST_QUALITY_EVAL_LIMIT` (evaluate first N rows)
- `MANIFEST_QUALITY_EVAL_JSON_OUTPUT`

## Pass Criteria

- `fast` gate must pass on every pull request.
- `release` gate must pass before publishing releases/artifacts.
- Nightly regression should remain green; red runs block performance claims until investigated.
- Any optimization PR that changes decoding/model math must include:
  - parity evidence (release gate pass or equivalent),
  - benchmark before/after results.

## Why This Exists

- Prevents silent quality regressions while optimizing.
- Keeps claims honest: parity first, speed second.
- Creates a clear handoff path for later Swift porting.
