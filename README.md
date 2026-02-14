# mlx-qwen3-asr

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://pypi.org/project/mlx-qwen3-asr/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

Run [Qwen3-ASR](https://huggingface.co/collections/Qwen/qwen3-asr) — the current best open-source speech recognition model — natively on Apple Silicon.

A ground-up reimplementation of the [official PyTorch model](https://github.com/QwenLM/Qwen3-ASR) using Apple's [MLX](https://github.com/ml-explore/mlx) framework. Same weights, same output quality, optimized for Mac GPUs via Metal. No PyTorch dependency for core transcription.

## Why this exists

Qwen3-ASR is the new state of the art in open-source ASR, beating Whisper-large-v3 across nearly every benchmark and supporting 30 languages plus 22 Chinese dialects. But the official implementation is **PyTorch + NVIDIA CUDA** — it doesn't use Apple GPUs.

This project rewrites every layer for MLX so the same model runs natively on M1/M2/M3/M4 hardware. Not a wrapper — a full reimplementation with correct interleaved MRoPE, per-chunk windowed encoder attention, and all the details that matter for output quality.

## Quick start

```bash
pip install mlx-qwen3-asr
```

```python
from mlx_qwen3_asr import transcribe

result = transcribe("audio.wav")
print(result.text)
```

```bash
mlx-qwen3-asr audio.wav
```

That's it. First run downloads model weights from HuggingFace (~1.2 GB for 0.6B).

## Performance on Apple Silicon

Measured on the 0.6B model. All numbers from controlled single-machine runs with benchmark artifacts committed to the repo.

| Configuration | Short clip latency | 10s clip latency | Real-time factor | vs fp16 |
|---|---|---|---|---|
| **fp16** (baseline) | 0.50s | 0.91s | 0.09x | — |
| **4-bit** (q4, group 64) | 0.12s | 0.23s | 0.02x | **3.98x faster** |
| **8-bit** (q8, group 64) | 0.14s | 0.26s | 0.03x | 3.46x faster |

4-bit quantization shows **zero measurable WER degradation** on LibriSpeech test-clean samples. Recommended for most use cases.

### Optimizations applied

- **Preallocated KV cache** with in-place writes (no per-token concatenation)
- **Direct grouped-query fused attention** (no explicit K/V head expansion)
- **Native WAV fast-path** — bypasses ffmpeg process startup for PCM/float WAV files
- **Cached model and tokenizer** — repeated `transcribe()` calls skip reload overhead
- **4-bit / 8-bit quantization** — 4x speed gain with no quality loss

## Model quality

Word error rates from the [Qwen3-ASR technical report](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) (lower is better):

| Benchmark | Qwen3-ASR 1.7B | Whisper-large-v3 |
|---|---|---|
| LibriSpeech test-clean | **1.51** | 2.02 |
| LibriSpeech test-other | **3.04** | 4.28 |
| WenetSpeech test-net | **4.97** | 9.68 |
| Fleurs (avg 30 langs) | **5.2** | 8.1 |

### Our measured results (0.6B, LibriSpeech test-clean, 20 samples)

| Configuration | WER | CER | Mean latency | RTF |
|---|---|---|---|---|
| fp16 | 0.73% | 0.22% | 1.13s | 0.14 |
| 4-bit (g64) | 0.73% | 0.16% | 0.39s | 0.05 |
| 8-bit (g64) | 0.73% | 0.22% | 0.45s | 0.06 |

4-bit quantization matches fp16 quality at 2.9x faster evaluation throughput. All benchmark JSON artifacts are committed to `docs/benchmarks/` for reproducibility.

## Models

| | Qwen3-ASR-0.6B (default) | Qwen3-ASR-1.7B |
|---|---|---|
| **Use case** | Fast local transcription | Maximum accuracy |
| **Encoder** | 18 layers, d=896 | 24 layers, d=1024 |
| **Decoder** | 28 layers, GQA 16/8 | 28 layers, GQA 16/8 |
| **Languages** | 30 + 22 Chinese dialects | 30 + 22 Chinese dialects |

```python
# Default: 0.6B (fast)
result = transcribe("audio.wav")

# Accuracy-first: 1.7B
result = transcribe("audio.wav", model="Qwen/Qwen3-ASR-1.7B")
```

## Timestamps

Word-level timestamps are available via forced alignment. Two backends:

```bash
# Native MLX backend (no extra dependencies)
mlx-qwen3-asr audio.wav --timestamps --aligner-backend mlx

# Official Qwen backend (requires: pip install qwen-asr)
mlx-qwen3-asr audio.wav --timestamps --aligner-backend qwen_asr

# Auto: try MLX first, fall back to official
mlx-qwen3-asr audio.wav --timestamps --aligner-backend auto
```

The native MLX backend is 2.6x faster than the PyTorch-backed official aligner with matching text output and <6ms timing deviation on tested samples.

## Quantization

Convert and run a 4-bit model for ~4x faster inference:

```bash
python scripts/convert.py \
  --model Qwen/Qwen3-ASR-0.6B \
  --quantize 4 --group-size 64 \
  --output-dir ./qwen3-asr-4bit

mlx-qwen3-asr audio.wav --model ./qwen3-asr-4bit
```

## Output formats

```bash
mlx-qwen3-asr audio.wav -f txt           # plain text
mlx-qwen3-asr audio.wav -f srt -o out/   # SRT subtitles
mlx-qwen3-asr audio.wav -f json          # structured JSON
mlx-qwen3-asr *.wav -f all -o out/       # all formats
```

Supported: `txt`, `json`, `srt`, `vtt`, `tsv`.

## Supported languages

30 languages: Arabic, Cantonese, Chinese, Czech, Danish, Dutch, English, Filipino, Finnish, French, German, Greek, Hindi, Hungarian, Indonesian, Italian, Japanese, Korean, Macedonian, Malay, Persian, Polish, Portuguese, Romanian, Russian, Spanish, Swedish, Thai, Turkish, Vietnamese.

Plus 22 Chinese dialects (Sichuan, Shanghai, Cantonese, etc.).

## Quality gates

This project enforces quality parity with the official PyTorch implementation before any optimization lands.

```bash
# Fast gate (CI enforces on every PR)
python scripts/quality_gate.py --mode fast

# Release gate (token-level parity with official PyTorch)
RUN_REFERENCE_PARITY=1 python scripts/quality_gate.py --mode release

# Golden evaluation on LibriSpeech
python scripts/eval_librispeech.py --subset test-clean --samples 100
```

Every performance optimization is committed with before/after benchmark JSON artifacts so regressions are caught and claims are verifiable.

## Development

```bash
git clone https://github.com/moona3k/mlx-qwen3-asr.git
cd mlx-qwen3-asr
pip install -e ".[dev]"
pytest -q
```

## Acknowledgments

- [Qwen team](https://github.com/QwenLM) at Alibaba for the Qwen3-ASR model
- [Apple MLX team](https://github.com/ml-explore/mlx) for the MLX framework
- [mlx-whisper](https://github.com/ml-explore/mlx-examples) for architecture patterns

## License

Apache 2.0. See [LICENSE](LICENSE) for details.
