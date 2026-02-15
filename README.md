# mlx-qwen3-asr

[![PyPI version](https://img.shields.io/pypi/v/mlx-qwen3-asr.svg)](https://pypi.org/project/mlx-qwen3-asr/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://pypi.org/project/mlx-qwen3-asr/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

Run [Qwen3-ASR](https://huggingface.co/collections/Qwen/qwen3-asr) — one of the strongest open-source speech recognition models — natively on Apple Silicon.

A ground-up reimplementation of the [official PyTorch model](https://github.com/QwenLM/Qwen3-ASR) using Apple's [MLX](https://github.com/ml-explore/mlx) framework. Same weights, same output quality, optimized for Mac GPUs via Metal. No PyTorch dependency for core transcription.

## Why this exists

[Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) is one of the strongest open-source ASR models available, with benchmark results exceeding Whisper-large-v3 across multiple languages and datasets. It supports 30 languages plus 22 Chinese dialects. But the official implementation is **PyTorch + NVIDIA CUDA** — it doesn't use Apple GPUs.

This project rewrites every layer for MLX so the same model runs natively on M1/M2/M3/M4 hardware. Not a wrapper — a full reimplementation with correct interleaved MRoPE, per-chunk windowed encoder attention, and all the architectural details that matter for output quality.

### What's included

- **Full encoder-decoder pipeline** — audio encoder (Conv2d stem + windowed transformer) and text decoder (Qwen3-style with interleaved MRoPE), reimplemented from scratch for MLX
- **Whisper-parity mel frontend** — native log-mel spectrogram computation validated against HuggingFace WhisperFeatureExtractor, with cached filterbank and Hann window
- **Both model sizes** — 0.6B (fast, default) and 1.7B (higher accuracy)
- **Long audio support** — energy-based chunking up to 20 minutes per chunk, no 30-second feature truncation
- **Word-level timestamps** — native MLX forced aligner (default, 2.6x faster than PyTorch alternative) with O(n log n) LIS-based timestamp correction
- **4-bit and 8-bit quantization** — up to 4.7x speedup with measured quality reporting on 100 speaker-balanced samples
- **Multiple output formats** — txt, json, srt, vtt, tsv
- **Session API** — explicit model/tokenizer ownership with no hidden global state
- **Speculative decoding** — experimental opt-in path (0.6B drafts for 1.7B target), parity-verified
- **Streaming** — experimental rolling decode with bounded context window
- **Native WAV fast-path** — custom binary WAV parser bypasses ffmpeg for PCM/float WAV files
- **349 tests** — every optimization is benchmark-gated with committed JSON artifacts
- **Minimal dependencies** — mlx, numpy, huggingface-hub, transformers (tokenizer only)

## Requirements

- **Apple Silicon Mac** (M1/M2/M3/M4) — this is an MLX project, Metal GPU required
- **Python 3.10+**
- **ffmpeg** — required for non-WAV audio formats (mp3, m4a, flac, mp4, etc.). WAV files work without ffmpeg via the native fast-path loader
- **~1.2 GB memory** for 0.6B model (fp16), **~3.4 GB** for 1.7B

## Installation

Install from PyPI:

```bash
pip install mlx-qwen3-asr
```

Install with optional timestamp alignment extras (for Japanese/Korean tokenization parity):

```bash
pip install "mlx-qwen3-asr[aligner]"
```

For development:

```bash
git clone https://github.com/moona3k/mlx-qwen3-asr.git
cd mlx-qwen3-asr
pip install -e ".[dev]"
```

## Quick start

### Python API

```python
from mlx_qwen3_asr import transcribe

result = transcribe("audio.wav")
print(result.text)
print(result.language)
```

By default, `transcribe()` uses `Qwen/Qwen3-ASR-0.6B` for fast local usage on Mac. Use `Qwen/Qwen3-ASR-1.7B` when you want higher accuracy and can afford higher latency/memory.

With options:

```python
result = transcribe(
    "meeting.mp3",
    model="Qwen/Qwen3-ASR-1.7B",
    language="English",
    verbose=True,
)
print(result.text)
```

### Session API (recommended for repeated calls)

The `Session` object owns model and tokenizer state explicitly — no hidden globals, no cache surprises:

```python
from mlx_qwen3_asr import Session

session = Session(model="Qwen/Qwen3-ASR-0.6B")

# Fast repeated transcription — model stays loaded
for audio_file in audio_files:
    result = session.transcribe(audio_file)
    print(result.text)
```

### Loading models explicitly

```python
from mlx_qwen3_asr import load_model, load_audio, transcribe

model, config = load_model("Qwen/Qwen3-ASR-0.6B")
audio = load_audio("speech.wav")
result = transcribe(audio, model=model)
```

### CLI

```bash
mlx-qwen3-asr audio.wav
```

Specify model, language, and output format:

```bash
mlx-qwen3-asr recording.mp3 --model Qwen/Qwen3-ASR-0.6B --language English -f srt -o output/
```

Word-level timestamps:

```bash
mlx-qwen3-asr audio.wav --timestamps
```

Multiple files with all output formats:

```bash
mlx-qwen3-asr *.wav -f all -o transcripts/ --verbose
```

Run `mlx-qwen3-asr --help` for the full list of options.

## Performance on Apple Silicon

Measured on Apple M4 Pro, macOS 26.2. All numbers from controlled runs with benchmark JSON artifacts committed to the repo. See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for the full breakdown.

### Latency (0.6B)

| Configuration | Short clip (~2.5s) | 10s clip | Real-time factor | vs fp16 |
|---|---|---|---|---|
| **fp16** (baseline) | 0.46s | 0.83s | 0.08x | — |
| **8-bit** (q8, group 64) | 0.11s | 0.27s | 0.03x | 3.11x faster |
| **4-bit** (q4, group 64) | 0.13s | 0.18s | 0.02x | **4.68x faster** |

### Quality (0.6B, LibriSpeech test-clean, 100 speaker-balanced samples)

| Configuration | WER | CER | Mean eval latency | vs fp16 Speed |
|---|---|---|---|---|
| fp16 | 2.29% | 0.59% | 1.09s | — |
| 8-bit (g64) | 2.33% | 0.59% | 0.34s | 3.11x |
| 4-bit (g64) | 2.72% | 0.88% | 0.30s | 4.68x |

8-bit is near-fp16 quality (+0.04pp WER). 4-bit trades +0.43pp WER for maximum speed.

### Multilingual quality (FLEURS, 10 languages x 10 samples)

| Model | Primary Error Rate | Mean Latency | Best Languages | Weakest |
|---|---|---|---|---|
| **0.6B** fp16 | 9.37% | 1.35s | Spanish 3.0%, English 4.6%, Chinese 4.4% | Arabic 21.5%, French 18.2% |
| **1.7B** fp16 | **6.70%** | 3.86s | Spanish 0.7%, Japanese 3.6%, French 4.1% | Hindi 17.4%, Arabic 16.5% |

The 1.7B delivers a 28% relative improvement, with the biggest gains on French (-14.1pp), Japanese (-4.9pp), and Arabic (-5.0pp). The 1.7B runs ~2.86x slower.

### MLX vs PyTorch quality (0.6B, Multilingual-100)

| Metric | MLX | PyTorch | Delta |
|---|---:|---:|---:|
| Primary error rate | 9.54% | 10.34% | -0.81pp |
| WER | 16.00% | 16.69% | -0.70pp |
| CER | 5.43% | 5.64% | -0.21pp |

67% of samples produce identical text output. Remaining differences are minor lexical shifts, numeric surface forms (`10,000` vs `zehntausend`), or punctuation — not quality regressions.

On long-form audio (75-90s clips), **MLX is 4.19x faster** than PyTorch on the same machine.

### Optimizations applied

- **Preallocated KV cache** with in-place slice writes and rollback-safe trimming
- **Direct grouped-query fused attention** via `mx.fast.scaled_dot_product_attention` (no explicit K/V head expansion)
- **Hybrid encoder windowing** — dense block-diagonal mask for short audio, segmented per-window execution for long contexts (up to 4.2x faster on long audio)
- **Cached mel filterbank and Hann window** — computed once, reused across calls
- **Native WAV fast-path** — custom binary parser bypasses ffmpeg process startup for PCM/float WAV files (up to 25% faster on quantized short clips)
- **Direct Qwen2Tokenizer loading** — skips AutoTokenizer dynamic dispatch, reduces cold-start by ~1.7x
- **Cached model and tokenizer instances** — repeated `transcribe()` calls skip reload overhead
- **4-bit / 8-bit quantization** — up to 4.7x speed gain with explicit per-profile quality reporting

Full benchmark report: `docs/benchmarks/2026-02-14-quant-matrix-speaker100.md`. All 90+ benchmark artifacts are committed under `docs/benchmarks/` for reproducibility.

## Model quality

Word error rates from the [Qwen3-ASR technical report](https://arxiv.org/abs/2601.21337) compared against current open-source and proprietary leaders (lower is better):

### English benchmarks

| Benchmark | GPT-4o-Transcribe | Parakeet-TDT-0.6B | Whisper-large-v3 | Qwen3-ASR-0.6B | **Qwen3-ASR-1.7B** |
|---|---:|---:|---:|---:|---:|
| LibriSpeech test-clean | **1.39** | 1.93 | 1.51 | 2.11 | 1.63 |
| LibriSpeech test-other | 3.75 | 3.59 | 3.97 | 4.55 | **3.38** |
| FLEURS-en | 2.40 | 4.85 | 4.08 | 4.39 | **3.35** |
| GigaSpeech | 25.50 | — | 9.76 | 8.88 | **8.45** |

### Chinese + multilingual benchmarks

| Benchmark | GPT-4o-Transcribe | Whisper-large-v3 | Qwen3-ASR-0.6B | **Qwen3-ASR-1.7B** |
|---|---:|---:|---:|---:|
| WenetSpeech test-net | 15.30 | 9.86 | 5.97 | **4.97** |
| AISHELL-2 test | 4.24 | 5.06 | 3.15 | **2.71** |
| FLEURS (12-lang avg) | — | 5.27 | 7.57 | **4.90** |
| CommonVoice | — | 10.77 | 12.75 | **9.18** |

### Robustness benchmarks

| Benchmark | GPT-4o-Transcribe | Whisper-large-v3 | Qwen3-ASR-0.6B | **Qwen3-ASR-1.7B** |
|---|---:|---:|---:|---:|
| Accented English | 28.56 | 21.30 | 16.62 | **16.07** |
| Extreme Noise | 36.11 | 63.17 | 17.88 | **16.17** |
| Elders & Kids (Mandarin) | 14.27 | 10.61 | 4.48 | **3.81** |

GPT-4o-Transcribe leads on clean English read speech (1.39 WER). Parakeet-TDT-0.6B is strong on English. But Qwen3-ASR dominates on Chinese, multilingual, noisy, and accented speech — and is the only open-source model competitive across all categories.

*Parakeet numbers from [model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3). All other numbers from the Qwen3-ASR paper. Robustness benchmarks are Qwen3-ASR internal test sets.*

### Correctness validation

This implementation is validated against the official PyTorch model via multiple parity gates:

- **MLX vs PyTorch head-to-head** — on 100 multilingual samples, MLX matches or exceeds PyTorch quality (9.54% vs 10.34% primary error rate)
- **Token-level greedy parity** — 67% exact text match, 64% exact token match across 10 languages. Remaining differences are minor lexical/numeric surface form variations, not quality regressions
- **Expanded parity suite** — tested across LibriSpeech test-clean, test-other, synthetic long mixes, and noise variants (SNR 10dB, 5dB)
- **Long-form parity** — 10 multilingual clips (75-90s each) transcribed correctly with no chunking artifacts, 4.19x faster than PyTorch
- **Mel spectrogram parity** — custom MLX mel matches HuggingFace WhisperFeatureExtractor with MAE < 3e-7
- **Native aligner parity** — MLX forced aligner matches official `qwen-asr` backend with 100% text match rate, <6ms timing MAE, and 2.64x speed advantage on 50 LibriSpeech samples

## Model variants

| | Qwen3-ASR-0.6B (default) | Qwen3-ASR-1.7B |
|---|---|---|
| **Parameters** | 0.6B | 1.7B |
| **Audio encoder layers** | 18 | 24 |
| **Audio encoder dim** | 896 | 1024 |
| **Text decoder layers** | 28 | 28 |
| **Text hidden size** | 1024 | 2048 |
| **Text attention (Q/KV heads)** | GQA (16/8) | GQA (16/8) |
| **RoPE theta** | 1,000,000 | 1,000,000 |
| **HuggingFace** | `Qwen/Qwen3-ASR-0.6B` | `Qwen/Qwen3-ASR-1.7B` |

Both models use interleaved Multi-dimensional RoPE (MRoPE) with sections [24, 20, 20], 128-bin mel spectrograms, and the same tokenizer (vocabulary size 151,936).

```python
# Default: 0.6B (fast, ~1.2 GB memory)
result = transcribe("audio.wav")

# Accuracy-first: 1.7B (~3.4 GB memory)
result = transcribe("audio.wav", model="Qwen/Qwen3-ASR-1.7B")
```

## Timestamps

Word-level timestamps via forced alignment using a dedicated aligner model (`Qwen/Qwen3-ForcedAligner-0.6B`). This path is native MLX (no PyTorch backend bridge):

```bash
mlx-qwen3-asr audio.wav --timestamps
```

```python
result = transcribe("audio.wav", return_timestamps=True)
for segment in result.segments:
    print(f"{segment['start']:.2f}s - {segment['end']:.2f}s: {segment['text']}")
```

**Measured parity** (LibriSpeech test-clean, `n=50`):
| Metric | Value |
|---|---|
| Text match rate (MLX vs official) | 100% |
| Timing MAE (all word boundaries) | 5.69 ms |
| MLX aligner mean latency | 0.21s |
| Official backend mean latency | 0.56s |
| Relative speed | **2.64x faster** |

The aligner uses O(n log n) LIS-based timestamp correction (Fenwick tree) for monotonicity repair, validated against the legacy O(n^2) implementation via randomized parity tests.

For Japanese/Korean timestamp alignment, install the `[aligner]` extra so `nagisa`/`soynlp` tokenization matches the official path.

## Quantization

Convert and run a quantized model:

```bash
python scripts/convert.py \
  --model Qwen/Qwen3-ASR-0.6B \
  --quantize 4 --group-size 64 \
  --output-dir ./qwen3-asr-4bit

mlx-qwen3-asr audio.wav --model ./qwen3-asr-4bit
```

Recommended profiles:
- **Speed-first**: 4-bit, group_size=64 — 4.68x faster, +0.43 WER on 100 samples
- **Quality-first**: 8-bit, group_size=64 — 3.11x faster, +0.04 WER on 100 samples

Publish quantized models to HuggingFace:

```bash
HF_TOKEN=... python scripts/publish_quantized.py \
  --source-model Qwen/Qwen3-ASR-0.6B \
  --repo-id YOUR_USER/mlx-qwen3-asr-0.6b-4bit \
  --bits 4
```

## Output formats

```bash
mlx-qwen3-asr audio.wav -f txt           # plain text
mlx-qwen3-asr audio.wav -f srt -o out/   # SRT subtitles
mlx-qwen3-asr audio.wav -f json          # structured JSON
mlx-qwen3-asr audio.wav -f vtt -o out/   # WebVTT
mlx-qwen3-asr *.wav -f all -o out/       # all formats at once
```

Supported: `txt`, `json`, `srt`, `vtt`, `tsv`.

## Supported languages

Qwen3-ASR officially lists 30 core languages:

| | | | |
|---|---|---|---|
| Arabic | Cantonese | Chinese | Czech |
| Danish | Dutch | English | Filipino |
| Finnish | French | German | Greek |
| Hindi | Hungarian | Indonesian | Italian |
| Japanese | Korean | Macedonian | Malay |
| Persian | Polish | Portuguese | Romanian |
| Russian | Spanish | Swedish | Thai |
| Turkish | Vietnamese | | |

Plus 22 Chinese dialects (Sichuan, Shanghai, Cantonese, and others), for 52 total language/dialect variants.

## Experimental features

### Speculative decoding

Uses the 0.6B model as a draft to accelerate 1.7B inference. Currently parity-safe but slower on tested workloads due to draft audio encoder overhead:

```bash
mlx-qwen3-asr audio.wav \
  --model Qwen/Qwen3-ASR-1.7B \
  --draft-model Qwen/Qwen3-ASR-0.6B \
  --num-draft-tokens 4
```

```python
result = transcribe(
    "audio.wav",
    model="Qwen/Qwen3-ASR-1.7B",
    draft_model="Qwen/Qwen3-ASR-0.6B",
    num_draft_tokens=4,
)
```

Status: greedy parity verified, but 0.53-0.55x on short/10s clips. Not enabled by default until benchmark evidence shows net speed wins.

### Streaming

Rolling decode implementation for near-real-time transcription:

```python
from mlx_qwen3_asr.streaming import init_streaming, feed_audio, finish_streaming

state = init_streaming(chunk_size_sec=2.0, max_context_sec=30.0)
for chunk in audio_chunks:
    state = feed_audio(chunk, state)
    print(state.text)
state = finish_streaming(state)
```

CLI:
```bash
mlx-qwen3-asr --streaming --stream-finalization-mode accuracy audio.wav
```

- Ingests small PCM chunks (default 2s)
- Incremental decoder KV-cache reuse across chunk turns (avoids O(n²) re-transcription)
- Bounded context window (default 30s) for stable memory/runtime
- Prefix rollback controls (`unfixed_chunk_num`, `unfixed_token_num`)
- Configurable finalization policy: `finalization_mode="accuracy"` (default) or `"latency"`
- Backward-compatible override: `enable_tail_refine=True|False`
- Input validation: handles int16 PCM normalization, non-1D arrays, empty input

## API reference

### `transcribe(audio, *, model, draft_model, language, return_timestamps, forced_aligner, dtype, max_new_tokens, num_draft_tokens, verbose)`

Transcribe audio to text. Accepts a file path, numpy array, `mx.array`, or `(array, sample_rate)` tuple. Returns a `TranscriptionResult`.

### `Session(model, *, dtype, tokenizer_model)`

Explicit transcription session. Owns model and tokenizer state with no hidden globals.
- Offline: `session.transcribe(audio, ...)` with the same parameters as top-level `transcribe`.
- Streaming: `session.init_streaming(...)`, `session.feed_audio(pcm, state)`, `session.finish_streaming(state)`.

### `load_model(name_or_path, *, dtype)`

Load a Qwen3-ASR model and config from HuggingFace or local path. Returns `(model, config)`.

### `load_audio(path_or_url)`

Load and resample audio to mono 16 kHz. Returns an `mx.array`.

### `ForcedAligner(model_path, *, dtype, backend)`

Word-level forced aligner. Native backend: `mlx` (default).

### `TranscriptionResult`

Frozen dataclass:
- `text` (str) — transcribed text
- `language` (str) — detected or forced language
- `segments` (list[dict] | None) — word-level timestamps when requested: `[{"text": "hello", "start": 0.5, "end": 0.8}, ...]`

## Quality gates

This project enforces parity with the official PyTorch implementation. No optimization lands without passing quality gates and committing benchmark artifacts.

```bash
# Unit tests (349 tests, ~3s)
pytest -q

# Fast quality gate
python scripts/quality_gate.py --mode fast

# Release gate with token-level parity (downloads model weights)
RUN_REFERENCE_PARITY=1 python scripts/quality_gate.py --mode release

# Speaker-balanced WER evaluation (100 samples)
python scripts/eval_librispeech.py --subset test-clean --samples 100 --sampling speaker_round_robin

# Latency benchmark
python scripts/benchmark_asr.py tests/fixtures/test_speech.wav \
  --model Qwen/Qwen3-ASR-0.6B --runs 5 \
  --json-output docs/benchmarks/latest.json
```

Additional quality lanes available:
- **Aligner parity**: `RUN_ALIGNER_PARITY=1` — validates MLX aligner against official backend
- **Expanded parity suite**: `RUN_REFERENCE_PARITY_SUITE=1` — test-clean, test-other, long mixes, noise variants with Unicode-safe text comparison
- **Multilingual parity**: manifest-driven workflow via `scripts/build_multilingual_manifest.py` for cross-language validation

See `docs/QUALITY_GATE.md` for full documentation.
Evaluation coverage status and prioritized gaps are tracked in `docs/EVAL_GAPS.md`.

## Architecture overview

```
Audio (16kHz mono)
  → 128-bin log-mel spectrogram (native MLX, Whisper-compatible)
  → Conv2d stem (3 layers, stride 2 each → 8x downsample)
  → Sinusoidal position embeddings
  → Windowed transformer encoder (18 or 24 layers, hybrid dense/segmented attention)
  → LayerNorm + GELU projection → audio features

Chat-template prompt:
  <|im_start|>system\nYou are a helpful assistant.<|im_end|>
  <|im_start|>user\n<|audio_start|><|audio_pad|>*N<|audio_end|><|im_end|>
  <|im_start|>assistant\n

  → Token embedding (151,936 vocab)
  → Replace audio_pad positions with encoded audio features
  → Qwen3 text decoder (28 or 32 layers, interleaved MRoPE, SwiGLU, RMSNorm)
  → Autoregressive decode with preallocated KV cache
  → Parse output: "language English<asr_text>transcribed text here"
```

Key architectural details:
- **Interleaved MRoPE** — sections [24, 20, 20] with stride-3 frequency assignment across temporal, height, and width dimensions. This is the detail other MLX ports get wrong (using standard RoPE or chunked assignment).
- **Audio encoder uses LayerNorm + bias** — different from the text decoder which uses RMSNorm without bias.
- **Q/K norms** — RMSNorm applied per-head on queries and keys before attention (Qwen3 innovation).

## Project structure

```
mlx_qwen3_asr/           # 5,058 lines of source
├── audio.py              # Mel spectrogram + audio I/O (562 lines)
├── forced_aligner.py     # Forced alignment + LIS correction (511 lines)
├── encoder.py            # Audio encoder (504 lines)
├── decoder.py            # Text decoder + KV cache (464 lines)
├── generate.py           # Autoregressive + speculative decode (350 lines)
├── tokenizer.py          # Tokenizer wrapper + output parsing (347 lines)
├── model.py              # Top-level model + audio-text fusion (336 lines)
├── transcribe.py         # High-level pipeline (306 lines)
├── streaming.py          # Streaming state machine (288 lines)
├── load_models.py        # Model loading + caching (253 lines)
├── config.py             # Dataclass configs (228 lines)
├── cli.py                # CLI entry point (201 lines)
├── mrope.py              # Interleaved MRoPE (167 lines)
├── writers.py            # Output format writers (121 lines)
├── chunking.py           # Long audio splitting (104 lines)
├── session.py            # Session API (92 lines)
├── attention.py          # Attention utilities (67 lines)
└── convert.py            # Weight remapping (67 lines)

tests/                    # 4,873 lines, 349 tests
scripts/                  # Benchmarks, evaluation, conversion, publishing
docs/                     # Architecture, decisions, benchmarks, roadmap
docs/benchmarks/          # 90+ committed JSON artifacts for reproducibility
```

## Development

```bash
git clone https://github.com/moona3k/mlx-qwen3-asr.git
cd mlx-qwen3-asr
pip install -e ".[dev]"
pytest -q                 # 349 tests, ~3s
```

## Acknowledgments

- [Qwen team](https://github.com/QwenLM) at Alibaba for the Qwen3-ASR model
- [Apple MLX team](https://github.com/ml-explore/mlx) for the MLX framework
- [mlx-whisper](https://github.com/ml-explore/mlx-examples) for architecture patterns and inspiration

## License

Apache 2.0. See [LICENSE](LICENSE) for details.
