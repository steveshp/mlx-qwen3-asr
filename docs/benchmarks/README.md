# Benchmarks Directory

Store benchmark JSON outputs here.

Suggested naming:

- `baseline_<machine>_<model>_<dtype>.json`
- `pr<id>_<machine>_<model>_<dtype>.json`
- `latest.json` for quick local runs
- `nightly-latency.json` for scheduled runtime trend
- `nightly-librispeech.json` for scheduled quality trend

## Recorded Results

Tokenizer cache optimization benchmark (2026-02-14):

- Machine: Apple M4 Pro, macOS 26.2
- Model: `Qwen/Qwen3-ASR-0.6B`, dtype `float16`
- Audio: `tests/fixtures/test_speech.wav`
- Runs: warmup=1, measured=3

Artifacts:

- `docs/benchmarks/2026-02-14-tokenizer-cache-before.json`
- `docs/benchmarks/2026-02-14-tokenizer-cache-after.json`

Decode-path optimization benchmark (2026-02-14):

- Changes:
  - preallocated KV cache writes (`slice_update`) instead of repeated concatenation,
  - GQA attention path no longer explicitly repeats K/V heads before fused SDPA.
- Machine: Apple M4 Pro, macOS 26.2
- Model: `Qwen/Qwen3-ASR-0.6B`, dtype `float16`

Artifacts:

- `docs/benchmarks/2026-02-14-gqa-kvcache-short.json`
- `docs/benchmarks/2026-02-14-gqa-kvcache-10s.json`
- `docs/benchmarks/2026-02-14-gqa-kvcache-10s-repeat.json`

Quantized comparison artifacts (2026-02-14):

- Runtime:
  - `docs/benchmarks/2026-02-14-4bit-short.json`
  - `docs/benchmarks/2026-02-14-4bit-10s.json`
- Quality vs FP16 (LibriSpeech sample):
  - `docs/benchmarks/2026-02-14-librispeech-fp16-20.json`
  - `docs/benchmarks/2026-02-14-librispeech-4bit-20.json`

Generate with:

```bash
python scripts/benchmark_asr.py tests/fixtures/test_speech.wav \
  --model Qwen/Qwen3-ASR-0.6B \
  --runs 5 \
  --json-output docs/benchmarks/latest.json
```
