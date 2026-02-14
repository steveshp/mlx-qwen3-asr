# Quantization Matrix

- Source model: `Qwen/Qwen3-ASR-0.6B`
- Eval subset: `test-clean`
- Eval samples: `100`
- Eval sampling: `speaker_round_robin`
- Benchmark runs: `5`
- Long clip length: `10s`

| Config | Short Mean (s) | Short RTF | Long Mean (s) | Long RTF | WER | CER | Eval RTF |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp16 | 0.4635 | 0.1830 | 0.8341 | 0.0834 | 0.022874 | 0.005865 | 0.1212 |
| 4bit-g64 | 0.1275 | 0.0503 | 0.1784 | 0.0178 | 0.027190 | 0.008798 | 0.0337 |
| 8bit-g64 | 0.1088 | 0.0429 | 0.2682 | 0.0268 | 0.023306 | 0.005865 | 0.0380 |

## Relative to fp16

- `4bit-g64` long-clip speedup vs fp16: `4.68x` (WER delta `+0.004316`)
- `8bit-g64` long-clip speedup vs fp16: `3.11x` (WER delta `+0.000432`)
