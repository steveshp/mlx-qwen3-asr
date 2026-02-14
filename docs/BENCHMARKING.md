# Benchmarking Protocol

Use this protocol for all runtime optimization work.

## Goal

Hold quality constant while improving runtime on Apple Silicon.

## Command

```bash
python scripts/benchmark_asr.py tests/fixtures/test_speech.wav \
  --model Qwen/Qwen3-ASR-0.6B \
  --dtype float16 \
  --warmup-runs 1 \
  --runs 5 \
  --json-output docs/benchmarks/latest.json
```

Streaming benchmark:

```bash
python scripts/benchmark_streaming.py tests/fixtures/test_speech.wav \
  --model Qwen/Qwen3-ASR-0.6B \
  --chunk-size-sec 2.0 \
  --max-context-sec 30.0 \
  --runs 3 \
  --json-output docs/benchmarks/latest-streaming.json
```

Latest streaming baseline artifact:
- `docs/benchmarks/2026-02-14-streaming-rolling-baseline.json`
- `docs/benchmarks/2026-02-14-streaming-rolling-baseline.md`

Mel parity evaluation:

```bash
python scripts/eval_mel_parity.py \
  --json-output docs/benchmarks/latest-mel-parity.json
```

Latest mel parity artifacts:
- `docs/benchmarks/2026-02-14-mel-parity.json`
- `docs/benchmarks/2026-02-14-mel-parity.md`

## Metrics

- `latency_sec.mean`
- `latency_sec.median`
- `rtf` (real-time factor = mean_latency / audio_duration)

Lower is better for all three.

## Method

1. Run benchmark on baseline commit.
2. Run benchmark on candidate commit with same machine/settings.
3. Compare JSON outputs.
4. If speed improves but quality gate fails, reject the optimization.

## Recommended Scenarios

- Short clip (latency sensitivity): `tests/fixtures/test_speech.wav`
- Long-form clip (stability/throughput): add a fixed long fixture and track same metrics.

## Reporting Template

Include this in PRs that claim performance gains:

```text
Machine: <chip + RAM + macOS>
Model: <id>, dtype=<dtype>
Before: mean=<x>s median=<y>s rtf=<z>
After:  mean=<x2>s median=<y2>s rtf=<z2>
Quality Gate: fast=<pass/fail>, release=<pass/fail or not run>
```

## Latest Local Finding (2026-02-14)

- Change: tokenizer instance caching across repeated `transcribe()` calls.
- Machine: Apple M4 Pro, macOS 26.2.
- Workload: `tests/fixtures/test_speech.wav`, model `Qwen/Qwen3-ASR-0.6B`, dtype `float16`, warmup=1, runs=3.
- Before: mean latency `1.7217s`, RTF `0.6796`.
- After: mean latency `0.5464s`, RTF `0.2157`.
- Delta: `-68.3%` mean latency, `-68.3%` RTF.

Raw JSON artifacts are tracked under `docs/benchmarks/`.

## Additional Local Finding (2026-02-14)

- Change set:
  - KV-cache preallocation (remove per-step concat growth cost),
  - direct grouped-query fused attention (remove explicit K/V head repetition).
- Machine: Apple M4 Pro, macOS 26.2.
- Model: `Qwen/Qwen3-ASR-0.6B`, dtype `float16`.
- Short clip (`tests/fixtures/test_speech.wav`): mean latency `0.5303s`, RTF `0.2093`.
- 10-second clip (`/tmp/test_10s.wav`): mean latency `0.9420s`, RTF `0.0942` (repeat run `0.9546s`, RTF `0.0955`).

## Quantized Comparison (2026-02-14)

Quantized runtime path is now validated end-to-end (local converted 4-bit model loads and transcribes).

- Machine: Apple M4 Pro, macOS 26.2.
- Model family: `Qwen/Qwen3-ASR-0.6B`.

Runtime:
- FP16 short fixture (`2.53s`): mean `0.5303s`, RTF `0.2093`.
- 4-bit short fixture (`2.53s`): mean `0.1591s`, RTF `0.0628`.
- FP16 10s clip: mean `0.9420s`, RTF `0.0942`.
- 4-bit 10s clip: mean `0.2831s`, RTF `0.0283`.

Quality (LibriSpeech test-clean sample, 20 utterances):
- FP16: WER `0.007317`, CER `0.002195`, RTF `0.1384`.
- 4-bit: WER `0.007317`, CER `0.001647`, RTF `0.0432`.

This sample indicates no measurable WER regression while achieving roughly 3x throughput improvement.

## Quantization Matrix Sweep (2026-02-14)

A full sweep (`fp16`, `4bit-g64`, `4bit-g32`, `8bit-g64`) was run via:

```bash
python scripts/benchmark_quantization_matrix.py
```

Artifacts:
- `docs/benchmarks/2026-02-14-quant-matrix.json`
- `docs/benchmarks/2026-02-14-quant-matrix.md`
- Refreshed run after WAV fast-path:
  - `docs/benchmarks/2026-02-14-quant-matrix-post-wavfast.json`
  - `docs/benchmarks/2026-02-14-quant-matrix-post-wavfast.md`
- CI/manual workflow: `.github/workflows/quantization-matrix.yml`

Current recommended operating point for `Qwen/Qwen3-ASR-0.6B` on Apple Silicon:
- `4bit-g64` (best speed with no sampled WER regression vs fp16 in this run).

Latest refreshed matrix highlights (`2026-02-14-quant-matrix-post-wavfast.md`):
- fp16 long 10s: mean `0.9088s`, RTF `0.0909`
- 4bit-g64 long 10s: mean `0.2286s`, RTF `0.0229`
- 4bit-g64 long speedup vs fp16: `3.98x`
- WER delta vs fp16 on sampled LibriSpeech: `+0.000000`

## KV Cache Write-Path Follow-up (2026-02-14)

`KVCache.update()` preallocated writes were compared:
- baseline: `mx.slice_update`
- candidate: in-place slice assignment

Artifacts:
- `docs/benchmarks/2026-02-14-kvcache-write-path.json`
- `docs/benchmarks/2026-02-14-kvcache-write-path.md`

Result summary:
- fp16 short/long and q4 short improved in mean latency.
- q4 long median stayed effectively unchanged; one outlier inflated mean.
- Decision: keep in-place write path (neutral-to-better overall).

## WAV Loader Fast-Path (2026-02-14)

Added a native WAV parser (PCM + IEEE float) with ffmpeg fallback for
unsupported/non-WAV formats. This removes ffmpeg process startup overhead for
common `.wav` inputs.

Artifacts:
- `docs/benchmarks/2026-02-14-wav-fastpath.json`
- `docs/benchmarks/2026-02-14-wav-fastpath.md`

Measured impact (same-session A/B):
- fp16 short: ~`-7.23%`
- fp16 long 10s: ~`+0.40%` (effectively neutral)
- 4bit-g64 short: ~`-24.78%`
- 4bit-g64 long 10s: ~`-9.35%`

Short-clip latency benefits are strongest for fast quantized profiles where
audio loading overhead is a larger fraction of total runtime.

## Rejected/Neutral Experiments (2026-02-14)

To avoid rediscovering low-signal paths, these were tested and not kept:

- Decode-loop `eval_interval` default changes (`0`, `4`):
  - No consistent net gain across fp16/q4 short+long scenarios.
  - Final decision: keep `eval_interval=1`.
- One-step-ahead async decode scheduling (mlx-lm style adaptation):
  - Regressed in end-to-end A/B on this repo's benchmark scenarios.
  - Final decision: reverted.
- `compute_features()` attention-mask skip for `padding="do_not_pad"` (status update):
  - Early microbench was mixed/noisy, so it was initially reverted.
  - Current implementation now keeps this optimization with test coverage:
    no-pad path skips mask creation; padded modes still use mask-derived true length.

## Forced Aligner Backend Smoke (2026-02-14)

Experimental native MLX timestamp backend (`--aligner-backend mlx`) was
compared against `qwen_asr` on a short deterministic fixture.

Artifacts:
- `docs/benchmarks/2026-02-14-aligner-backend-smoke.json`
- `docs/benchmarks/2026-02-14-aligner-backend-smoke.md`

Result snapshot (`tests/fixtures/test_speech.wav`, 5 warm runs):
- `mlx` mean: `0.0414s`
- `qwen_asr` mean: `0.1957s`
- Relative mean speed (`qwen_asr / mlx`): `4.73x`
- Both backends produced identical first-word timestamp spans in this smoke case.

Scope note:
- This is a single-sample sanity comparison, not yet a full multilingual
  timestamp quality parity verdict.

## Forced Aligner Parity Lane (2026-02-14)

A deterministic LibriSpeech subset parity run was added for native aligner
validation (`scripts/eval_aligner_parity.py`).

Artifacts:
- `docs/benchmarks/2026-02-14-aligner-parity-10.json`
- `docs/benchmarks/2026-02-14-aligner-parity-10.md`
- `docs/benchmarks/2026-02-14-aligner-parity-50.json`
- `docs/benchmarks/2026-02-14-aligner-parity-50.md`

Latest expanded snapshot (`test-clean`, 50 samples):
- text-match rate (`mlx` vs `qwen_asr` word sequence): `1.0000`
- timing MAE (all boundaries): `5.6909 ms`
- mean latency: `mlx=0.2113s`, `qwen_asr=0.5570s`
- relative speed (`qwen_asr / mlx`): `2.64x`

Gate integration:
- Optional release gate lane:
  - `RUN_ALIGNER_PARITY=1 ALIGNER_PARITY_SAMPLES=10`
  - thresholds: text-match `>= 1.0`, timing MAE `<= 60ms`.
