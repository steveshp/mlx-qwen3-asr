# Research: Qwen3-ASR

Comprehensive research findings on Qwen3-ASR, the SOTA open-source ASR model from Alibaba's Qwen team.

## Model Family

| Model | Parameters | Use Case | Key Metric |
|-------|-----------|----------|------------|
| Qwen3-ASR-0.6B | ~600M | Throughput-optimized | 2000x throughput at concurrency 128, 92ms TTFT |
| Qwen3-ASR-1.7B | ~1.7B | Maximum accuracy (SOTA) | Best WER across nearly all benchmarks |
| Qwen3-ForcedAligner-0.6B | ~600M | Word-level timestamps | 42.9ms average alignment shift (AAS) |

## Benchmark Results

### English ASR (Word Error Rate — lower is better)

| Model | LibriSpeech clean | LibriSpeech other | CommonVoice-15 en |
|-------|-------------------|-------------------|-------------------|
| Qwen3-ASR-1.7B | **1.51** | — | — |
| Whisper-large-v3 | 1.63 | — | — |
| GPT-4o-Transcribe | 1.97 | — | — |
| Qwen3-ASR-0.6B | 1.72 | — | — |

### Chinese ASR (Character Error Rate — lower is better)

| Model | WenetSpeech | AISHELL-1 | AISHELL-2 |
|-------|-------------|-----------|-----------|
| Qwen3-ASR-1.7B | **4.97** | — | — |
| Whisper-large-v3 | 9.86 | — | — |
| Qwen3-ASR-0.6B | 5.38 | — | — |

### Forced Alignment (Average Alignment Shift — lower is better)

| Model | AAS (ms) |
|-------|----------|
| Qwen3-ForcedAligner-0.6B | **42.9** |
| NeMo Forced Aligner (NFA) | 129.8 |
| WhisperX | 133.2 |

## Supported Languages (30 + dialects)

### Primary Languages
Chinese, English, Cantonese, Arabic, German, French, Spanish, Portuguese, Indonesian, Italian, Korean, Russian, Thai, Vietnamese, Japanese, Turkish, Hindi, Malay, Dutch, Swedish, Danish, Finnish, Polish, Czech, Filipino, Persian, Greek, Romanian, Hungarian, Macedonian

### Chinese Dialects (22)
Sichuan dialect, Shanghai dialect, Cantonese, and 19 others covering major regional varieties.

## Architecture Summary

- **Type:** Encoder-decoder transformer
- **Audio encoder:** Conv2d stem (8x downsample) + transformer layers + projection
- **Text decoder:** Qwen3-style with MRoPE (Multi-dimensional Rotary Position Embedding)
- **Innovation:** Interleaved MRoPE for 3D position encoding (temporal, height, width)
- **Training:** Built on Qwen3-Omni foundation model with large-scale speech data

## Special Capabilities

- **Singing voice recognition:** Can transcribe singing, not just speech
- **Forced alignment:** Dedicated model for word-level timestamps with 42.9ms precision
- **Streaming support:** State-machine approach with prefix rollback
- **Long audio:** Up to 1200s per chunk with energy-based splitting

## Paper

- **Title:** Qwen3-ASR Technical Report
- **ArXiv:** [2601.21337](https://arxiv.org/abs/2601.21337)
- **Released:** January 2026

## Official Repository Quality Notes

Issues found in the official codebase:
- Bare `except:` clause (should be `except ImportError:`)
- `__all__ = ["__version__"]` but `__version__` never defined
- Stale `self.conv1` reference (should be `self.conv2d1`)
- pyproject.toml references `qwen_tts` instead of `qwen_asr`
- Dependencies pinned to exact versions (e.g., `transformers==4.57.6`, `vllm==0.14.0`)
