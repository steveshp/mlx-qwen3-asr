# Quantization Matrix

- Source model: `Qwen/Qwen3-ASR-0.6B`
- Eval subset: `test-other`
- Eval samples: `100`
- Eval sampling: `speaker_round_robin`
- Benchmark runs: `5`
- Long clip length: `10s`

| Config | Short Mean (s) | Short RTF | Long Mean (s) | Long RTF | WER | CER | Eval RTF |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp16 | 0.4004 | 0.1580 | 0.7097 | 0.0710 | 0.041954 | 0.020948 | 0.0994 |
| 4bit-g64 | 0.0886 | 0.0350 | 0.1624 | 0.0162 | 0.055762 | 0.027441 | 0.0229 |
| 8bit-g64 | 0.1056 | 0.0417 | 0.1939 | 0.0194 | 0.041423 | 0.020826 | 0.0272 |

## Relative to fp16

- `4bit-g64` long-clip speedup vs fp16: `4.37x` (WER delta `+0.013808`)
- `8bit-g64` long-clip speedup vs fp16: `3.66x` (WER delta `-0.000531`)
