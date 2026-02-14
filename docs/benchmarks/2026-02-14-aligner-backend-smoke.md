# Forced Aligner Backend Smoke Comparison

- Audio: `tests/fixtures/test_speech.wav` (2.53s)
- Runs per backend: `5` (after one warm run)

| Backend | Mean (s) | Median (s) | Min (s) | Max (s) |
|---|---:|---:|---:|---:|
| mlx | 0.0414 | 0.0410 | 0.0404 | 0.0434 |
| qwen_asr | 0.1957 | 0.1998 | 0.1760 | 0.2104 |

Relative mean speed (`qwen_asr / mlx`): `4.73x`
