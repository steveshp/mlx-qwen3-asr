# Tokenizer Local-Path Benchmark

- Suite: `tokenizer-local-path-benchmark-v1`
- Audio: `tests/fixtures/test_speech.wav`
- Model: `Qwen/Qwen3-ASR-0.6B`
- Dtype: `float16`
- Runs per mode: `5`

| Tokenizer path source | Mean latency (s) | Median (s) | Min (s) | Max (s) |
|---|---:|---:|---:|---:|
| Repo ID path | 2.6500 | 2.6383 | 2.5881 | 2.7318 |
| Resolved local snapshot path | 2.2958 | 2.2949 | 2.2783 | 2.3195 |

Relative speedup (repo-id / resolved-local): **1.15x**

Benchmark command pattern:
- Spawn fresh Python process per run.
- Compare end-to-end single `transcribe(...)` latency under two tokenizer init modes.
