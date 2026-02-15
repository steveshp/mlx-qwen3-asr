# Streaming KV-Cache Benchmark (2026-02-15)

Compared against rolling re-transcription baseline (`2026-02-14-streaming-rolling-baseline.json`).

## Short Clip (tests/fixtures/test_speech.wav, 2.53s)

### Variant A: `enable_tail_refine=true` (accuracy-first finalization)

| Metric | Rolling Baseline | KV-Cache + Tail Refine | Delta |
|---|---:|---:|---:|
| `total_mean` | 1.1344s | 1.8301s | +61.3% |
| `chunk_mean_mean` | 0.2673s | 0.3050s | +14.1% |
| `finish_mean` | 0.5997s | 1.2201s | +103.5% |
| `rtf` | 0.4478 | 0.7224 | +61.3% |

### Variant B: `enable_tail_refine=false` (latency-first finalization)

| Metric | Rolling Baseline | KV-Cache (No Tail Refine) | Delta |
|---|---:|---:|---:|
| `total_mean` | 1.1344s | 1.0838s | -4.5% |
| `chunk_mean_mean` | 0.2673s | 0.3067s | +14.7% |
| `finish_mean` | 0.5997s | 0.4703s | -21.6% |
| `rtf` | 0.4478 | 0.4278 | -4.5% |

Quality note on this fixture:
- `enable_tail_refine=true` recovers the final word (`"...lazy dog."`).
- `enable_tail_refine=false` ends early (`"...lazy."`) but is faster.

## Long Clip Scaling (synthetic 30s, tail refine default)

- Audio: `/tmp/test_speech_30s.wav`
- Chunks: `15` @ 2.0s
- `total_mean`: `11.1569s`
- `chunk_mean_mean`: `0.7438s`
- `finish_mean`: `0.000002s`
- `rtf`: `0.3719`

Observation: finish overhead is near-zero when there is no pending tail chunk; chunk-level runtime remains bounded and avoids full-history re-transcription per chunk.
