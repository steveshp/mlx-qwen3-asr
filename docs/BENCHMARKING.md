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
  - KV-cache preallocation via `slice_update` (remove per-step concat growth cost),
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
