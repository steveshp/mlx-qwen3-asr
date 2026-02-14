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

### Release Gate (required before tags/releases)

```bash
RUN_REFERENCE_PARITY=1 python scripts/quality_gate.py --mode release
```

Checks:
- Everything in fast gate.
- Reference parity test (`tests/test_reference_parity.py`).
- Optional aligner parity lane when explicitly enabled:
  - `RUN_ALIGNER_PARITY=1`
  - runs `scripts/eval_aligner_parity.py` on deterministic LibriSpeech samples.

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

Current status:
- this lane is exploratory and intended for gap discovery before promotion to a
  required release gate.

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
