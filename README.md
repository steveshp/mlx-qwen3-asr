# mlx-qwen3-asr

[![PyPI version](https://img.shields.io/pypi/v/mlx-qwen3-asr.svg)](https://pypi.org/project/mlx-qwen3-asr/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://pypi.org/project/mlx-qwen3-asr/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

Run [Qwen3-ASR](https://huggingface.co/collections/Qwen/qwen3-asr) — one of the strongest open-source speech recognition models — natively on Apple Silicon.

A ground-up reimplementation of the [official PyTorch model](https://github.com/QwenLM/Qwen3-ASR) using Apple's [MLX](https://github.com/ml-explore/mlx) framework. Same weights, same output quality, optimized for Mac GPUs via Metal. No PyTorch dependency for core transcription.

## Why this exists

[Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) is positioned in its official report as SOTA among open-source ASR models, with strong benchmark results and support for 30 languages plus 22 Chinese dialects. But the official implementation is **PyTorch + NVIDIA CUDA** — it doesn't use Apple GPUs.

This project rewrites every layer for MLX so the same model runs natively on M1/M2/M3/M4 hardware. Not a wrapper — a full reimplementation with correct interleaved MRoPE, per-chunk windowed encoder attention, and all the details that matter for output quality.

### Key features

- Full audio encoder + text decoder with correct interleaved MRoPE
- Whisper-parity custom log-mel frontend (default path for `do_not_pad`)
- Supports both 1.7B and 0.6B model sizes
- Long audio chunking (up to 20 minutes per chunk) with no 30s feature truncation
- Word-level timestamps via official Qwen backend (default) or native MLX backend (experimental)
- Experimental rolling streaming ASR with bounded decode context
- Native fast-path WAV loader (PCM/float WAV) with ffmpeg fallback for other formats
- Multiple output formats: txt, json, srt, vtt, tsv
- Cached model/tokenizer instances for low repeated-call latency
- Decoder optimizations: preallocated KV cache + direct grouped-query fused attention
- 4-bit and 8-bit quantization — up to 4x speedup with explicit quality trade-off reporting
- Minimal dependencies: mlx, numpy, huggingface-hub, transformers

## Installation

Install from PyPI:

```bash
pip install mlx-qwen3-asr
```

Install with optional forced aligner support:

```bash
pip install "mlx-qwen3-asr[aligner]"
```

This extra installs `qwen-asr` (official reference backend) plus `nagisa` and
`soynlp` (JA/KO tokenization parity helpers for forced alignment).

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
    model="Qwen/Qwen3-ASR-1.7B",  # accuracy-first option
    language="English",
    verbose=True,
)
print(result.text)
```

### Loading models explicitly

```python
from mlx_qwen3_asr import load_model, load_audio, transcribe

model, config = load_model("Qwen/Qwen3-ASR-0.6B")
audio = load_audio("speech.wav")
result = transcribe(audio, model=model)
```

### Explicit session (no hidden globals)

```python
from mlx_qwen3_asr import Session

session = Session(model="Qwen/Qwen3-ASR-0.6B")
result = session.transcribe("audio.wav")
print(result.text)
```

### CLI

```bash
mlx-qwen3-asr audio.wav
```

Specify model, language, and output format:

```bash
mlx-qwen3-asr recording.mp3 --model Qwen/Qwen3-ASR-0.6B --language English -f srt -o output/
```

Word-level timestamps (default backend is native `mlx`):

```bash
mlx-qwen3-asr audio.wav --timestamps
```

Experimental speculative decoding (opt-in):

```bash
mlx-qwen3-asr audio.wav \
  --model Qwen/Qwen3-ASR-1.7B \
  --draft-model Qwen/Qwen3-ASR-0.6B \
  --num-draft-tokens 4
```

Current status: parity-safe experimental path; not enabled by default and may be
slower on short/medium clips.

Multiple files with all output formats:

```bash
mlx-qwen3-asr *.wav -f all -o transcripts/ --verbose
```

Run `mlx-qwen3-asr --help` for the full list of options.

## Performance on Apple Silicon

Measured on the 0.6B model. Numbers below reflect the latest speaker-balanced
matrix run (`docs/benchmarks/2026-02-14-quant-matrix-speaker100.md`).
Machine used for this run: Apple M4 Pro, macOS 26.2.

| Configuration | Short clip latency | 10s clip latency | Real-time factor | vs fp16 |
|---|---|---|---|---|
| **fp16** (baseline) | 0.46s | 0.83s | 0.08x | — |
| **4-bit** (q4, group 64) | 0.13s | 0.18s | 0.02x | **4.68x faster** |
| **8-bit** (q8, group 64) | 0.11s | 0.27s | 0.03x | 3.11x faster |

Quality on speaker-balanced LibriSpeech samples is reported below with `n` and sampling strategy.

### Optimizations applied

- **Preallocated KV cache** with in-place writes (no per-token concatenation)
- **Direct grouped-query fused attention** (no explicit K/V head expansion)
- **Native WAV fast-path** — bypasses ffmpeg process startup for PCM/float WAV files
- **Cached model and tokenizer** — repeated `transcribe()` calls skip reload overhead
- **4-bit / 8-bit quantization** — up to 4x speed gain (profile-dependent quality trade-off)

## Model quality

Official word error rates from the [Qwen3-ASR technical report](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) (lower is better):

| Benchmark | Qwen3-ASR 1.7B | Whisper-large-v3 |
|---|---|---|
| LibriSpeech test-clean | **1.51** | 2.02 |
| LibriSpeech test-other | **3.04** | 4.28 |
| WenetSpeech test-net | **4.97** | 9.68 |
| Fleurs (avg 30 langs) | **5.2** | 8.1 |

### Our measured results (0.6B, LibriSpeech test-clean, 100 samples, speaker round-robin)

| Configuration | WER | CER | Mean latency | RTF |
|---|---|---|---|---|
| fp16 | 2.29% | 0.59% | 1.09s | 0.121 |
| 8-bit (g64) | 2.33% | 0.59% | 0.34s | 0.038 |
| 4-bit (g64) | 2.72% | 0.88% | 0.30s | 0.034 |

On this speaker-balanced subset (`n=100`), 8-bit is near-fp16 quality
(+0.04 absolute WER points) while 4-bit is fastest but shows measurable
degradation (+0.43 absolute WER points). Benchmark artifacts are committed under
`docs/benchmarks/` for reproducibility.

## Model variants

| | Qwen3-ASR-0.6B (default) | Qwen3-ASR-1.7B |
|---|---|---|
| **Parameters** | 0.6B | 1.7B |
| **Audio encoder layers** | 18 | 24 |
| **Audio encoder dim (`d_model`)** | 896 | 1024 |
| **Text decoder layers** | 28 | 28 |
| **Text hidden size** | 1024 | 2048 |
| **Text attention (Q/KV heads)** | GQA (16/8) | GQA (16/8) |
| **RoPE theta** | 1,000,000 | 1,000,000 |
| **HuggingFace** | `Qwen/Qwen3-ASR-0.6B` | `Qwen/Qwen3-ASR-1.7B` |

Both models use Multi-dimensional RoPE (MRoPE), 128-bin mel spectrograms, and the same tokenizer with a vocabulary size of 151,936.

```python
# Default: 0.6B (fast)
result = transcribe("audio.wav")

# Accuracy-first: 1.7B
result = transcribe("audio.wav", model="Qwen/Qwen3-ASR-1.7B")
```

## Timestamps

Word-level timestamps are available via forced alignment. Two backends:

```bash
# Native MLX backend (default, no PyTorch dependency)
mlx-qwen3-asr audio.wav --timestamps --aligner-backend mlx

# Official Qwen backend (requires: pip install qwen-asr)
mlx-qwen3-asr audio.wav --timestamps --aligner-backend qwen_asr

# Auto: try MLX first, fall back to official
mlx-qwen3-asr audio.wav --timestamps --aligner-backend auto
```

Default backend is `mlx`. Use `qwen_asr` if you want the official reference
backend, or `auto` for MLX-first with automatic fallback.

Current measured parity snapshot (`test-clean`, English, `n=50`):
- text match rate (`mlx` vs `qwen_asr`): `1.0000`
- timing MAE (all boundaries): `5.6909 ms`
- mean latency: `mlx=0.2113s`, `qwen_asr=0.5570s` (`2.64x` relative speed)
- artifacts: `docs/benchmarks/2026-02-14-aligner-parity-50.{json,md}`

For Japanese/Korean timestamp alignment with the native MLX backend, install the `[aligner]` extra so `nagisa`/`soynlp` tokenization matches the official path.

## Streaming (experimental)

The streaming API is currently a **rolling decode** implementation:
- It ingests small PCM chunks (default 2s).
- It decodes with a bounded context window (default 30s) to keep per-chunk runtime stable.
- It applies prefix rollback controls (`unfixed_chunk_num`, `unfixed_token_num`) so only trailing units remain unstable.

It is not yet a full incremental decoder with persistent KV cache reuse across chunks.

## Current limitations

- Streaming is intentionally experimental and currently uses rolling re-decode.
- Speculative decoding is parity-safe but still experimental and currently slower
  on the tested short/10s lanes.
- Native MLX timestamp backend is validated on deterministic LibriSpeech
  subsets; multilingual parity coverage is still expanding.

## Quantization

Convert and run a 4-bit model for ~4x faster inference:

```bash
python scripts/convert.py \
  --model Qwen/Qwen3-ASR-0.6B \
  --quantize 4 --group-size 64 \
  --output-dir ./qwen3-asr-4bit

mlx-qwen3-asr audio.wav --model ./qwen3-asr-4bit
```

Recommended quantization profiles (Apple Silicon):
- Speed-first: `4-bit`, `group_size=64`
- Quality-first quantized: `8-bit`, `group_size=64`
- Latest matrix report: `docs/benchmarks/2026-02-14-quant-matrix-speaker100.md`
- Latest long-clip speedup vs fp16 (speaker-balanced run): **4.68x** (`4-bit`, `group_size=64`)
- 100-sample quality artifacts:
  - `docs/benchmarks/2026-02-14-librispeech-fp16-100-speaker-round-robin.json`
  - `docs/benchmarks/2026-02-14-librispeech-8bit-g64-100-speaker-round-robin.json`
  - `docs/benchmarks/2026-02-14-librispeech-4bit-g64-100-speaker-round-robin.json`

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
mlx-qwen3-asr *.wav -f all -o out/       # all formats
```

Supported: `txt`, `json`, `srt`, `vtt`, `tsv`.

## Supported languages

Qwen3-ASR officially lists 30 core languages:

| Language | Language | Language | Language |
|----------|----------|----------|----------|
| Arabic | Cantonese | Chinese | Czech |
| Danish | Dutch | English | Filipino |
| Finnish | French | German | Greek |
| Hindi | Hungarian | Indonesian | Italian |
| Japanese | Korean | Macedonian | Malay |
| Persian | Polish | Portuguese | Romanian |
| Russian | Spanish | Swedish | Thai |
| Turkish | Vietnamese | | |

Plus 22 Chinese dialects (Sichuan, Shanghai, Cantonese, and others),
for 52 total language/dialect variants in the official release.

## API reference

### `transcribe(audio, *, model, draft_model, language, return_timestamps, forced_aligner, dtype, max_new_tokens, num_draft_tokens, verbose)`

Transcribe audio to text. Accepts a file path, numpy array, `mx.array`, or `(array, sample_rate)` tuple. Returns a `TranscriptionResult`. Set `return_timestamps=True` to request word-level timestamps. You can pass a configured `ForcedAligner` instance for explicit backend control (`qwen_asr`, `mlx`, or `auto`). For experimental speculative decoding, pass `draft_model` and optionally tune `num_draft_tokens`.

### `load_model(name_or_path, *, dtype)`

Load a Qwen3-ASR model and its config from a HuggingFace repo or local path. Returns `(model, config)`.

### `load_audio(path_or_url)`

Load and resample audio to mono 16 kHz. Returns an `mx.array`.

### `TranscriptionResult`

Frozen dataclass with fields:
- `text` (str) — transcribed text
- `language` (str) — detected or forced language
- `segments` (list[dict] | None) — optional timestamped segments when requested

## Quality gates

This project enforces quality parity with the official PyTorch implementation before any optimization lands.

```bash
# Fast gate (CI enforces on every PR)
python scripts/quality_gate.py --mode fast

# Release gate (token-level parity with official PyTorch)
RUN_REFERENCE_PARITY=1 python scripts/quality_gate.py --mode release

# Optional native-aligner parity lane
RUN_REFERENCE_PARITY=1 RUN_ALIGNER_PARITY=1 ALIGNER_PARITY_SAMPLES=10 \
python scripts/quality_gate.py --mode release

# Optional broader parity suite (clean/other + long mixes + noise variants)
RUN_REFERENCE_PARITY=1 RUN_REFERENCE_PARITY_SUITE=1 \
REFERENCE_PARITY_SUITE_SUBSETS=test-clean,test-other \
REFERENCE_PARITY_SUITE_SAMPLES_PER_SUBSET=3 \
REFERENCE_PARITY_SUITE_INCLUDE_LONG_MIXES=1 \
REFERENCE_PARITY_SUITE_INCLUDE_NOISE_VARIANTS=1 \
REFERENCE_PARITY_SUITE_NOISE_SNRS_DB=10,5 \
python scripts/quality_gate.py --mode release

# Golden evaluation on LibriSpeech
python scripts/eval_librispeech.py --subset test-clean --samples 100 --sampling speaker_round_robin
```

Run unit tests directly:

```bash
pytest -q
```

Token-level greedy parity against official PyTorch (slow; downloads model weights):

```bash
RUN_REFERENCE_PARITY=1 REFERENCE_PARITY_MODEL=Qwen/Qwen3-ASR-0.6B pytest -q tests/test_reference_parity.py
```

Benchmark latency + RTF:

```bash
python scripts/benchmark_asr.py tests/fixtures/test_speech.wav \
  --model Qwen/Qwen3-ASR-0.6B \
  --runs 5 \
  --json-output docs/benchmarks/latest.json
```

Every performance optimization should include benchmark artifacts and pass
quality gates so claims remain auditable and reproducible.

`RUN_REFERENCE_PARITY_SUITE` remains an exploratory lane for broader coverage.
It reports both strict token parity and Unicode-safe normalized text parity.

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
- [mlx-whisper](https://github.com/ml-explore/mlx-examples) for architecture patterns and inspiration

## License

Apache 2.0. See [LICENSE](LICENSE) for details.
