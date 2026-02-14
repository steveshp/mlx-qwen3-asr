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
  - preallocated KV cache writes instead of repeated concatenation,
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

Speaker-balanced LibriSpeech quality artifacts (2026-02-14):

- `docs/benchmarks/2026-02-14-librispeech-fp16-100-speaker-round-robin.json`
- `docs/benchmarks/2026-02-14-librispeech-8bit-g64-100-speaker-round-robin.json`
- `docs/benchmarks/2026-02-14-librispeech-4bit-g64-100-speaker-round-robin.json`

Quantization matrix sweep artifacts (2026-02-14):

- `docs/benchmarks/2026-02-14-quant-matrix.json`
- `docs/benchmarks/2026-02-14-quant-matrix.md`
- `docs/benchmarks/2026-02-14-quant-matrix-post-wavfast.json`
- `docs/benchmarks/2026-02-14-quant-matrix-post-wavfast.md`
- `docs/benchmarks/2026-02-14-quant-matrix-speaker100.json`
- `docs/benchmarks/2026-02-14-quant-matrix-speaker100.md`
- CI/manual workflow output names:
  - `docs/benchmarks/ci-quant-matrix.json`
  - `docs/benchmarks/ci-quant-matrix.md`

KV cache write-path follow-up (2026-02-14):

- `docs/benchmarks/2026-02-14-kvcache-write-path.json`
- `docs/benchmarks/2026-02-14-kvcache-write-path.md`

WAV fast-path loader experiment (2026-02-14):

- `docs/benchmarks/2026-02-14-wav-fastpath.json`
- `docs/benchmarks/2026-02-14-wav-fastpath.md`

Forced aligner backend smoke comparison (2026-02-14):

- `docs/benchmarks/2026-02-14-aligner-backend-smoke.json`
- `docs/benchmarks/2026-02-14-aligner-backend-smoke.md`

Forced aligner parity lane run (2026-02-14):

- `docs/benchmarks/2026-02-14-aligner-parity-10.json`
- `docs/benchmarks/2026-02-14-aligner-parity-10.md`
- `docs/benchmarks/2026-02-14-aligner-parity-50.json`
- `docs/benchmarks/2026-02-14-aligner-parity-50.md`
- `docs/benchmarks/2026-02-14-aligner-parity-10-ja-ko-tokenizer.json`
- `docs/benchmarks/2026-02-14-aligner-parity-10-ja-ko-tokenizer.md`

Tokenizer loader path benchmark (2026-02-14):

- `docs/benchmarks/2026-02-14-tokenizer-loader-path-benchmark.json`
- `docs/benchmarks/2026-02-14-tokenizer-loader-path-benchmark.md`

Tokenizer resolved-local-path benchmark (2026-02-14):

- `docs/benchmarks/2026-02-14-tokenizer-local-path-benchmark.json`
- `docs/benchmarks/2026-02-14-tokenizer-local-path-benchmark.md`

Encoder windowing threshold benchmark (2026-02-14):

- `docs/benchmarks/2026-02-14-encoder-windowing-threshold.json`
- `docs/benchmarks/2026-02-14-encoder-windowing-threshold.md`

Post-change short-clip sanity benchmark (2026-02-14):

- `docs/benchmarks/2026-02-14-short-after-windowed-hybrid.json`

Speculative decoding prototype benchmarks (2026-02-14):

- `docs/benchmarks/2026-02-14-speculative-1p7b-vs-0p6b-smoke.json`
- `docs/benchmarks/2026-02-14-speculative-1p7b-vs-0p6b.json`
- `docs/benchmarks/2026-02-14-speculative-1p7b-vs-0p6b-10s.json`
- `docs/benchmarks/2026-02-14-speculative-prototype.md`

Generate with:

```bash
python scripts/benchmark_asr.py tests/fixtures/test_speech.wav \
  --model Qwen/Qwen3-ASR-0.6B \
  --runs 5 \
  --json-output docs/benchmarks/latest.json
```
