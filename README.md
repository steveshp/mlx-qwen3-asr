# mlx-qwen3-asr

[![PyPI version](https://img.shields.io/pypi/v/mlx-qwen3-asr.svg)](https://pypi.org/project/mlx-qwen3-asr/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://pypi.org/project/mlx-qwen3-asr/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

Run [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) on Apple Silicon. A pure [MLX](https://github.com/ml-explore/mlx) reimplementation of the official PyTorch model — same weights, same output, optimized for Mac GPUs via Metal.

## What is this?

[Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) is the current state-of-the-art
open-source speech recognition model by the Alibaba Qwen team, beating
Whisper-large-v3 across nearly every benchmark. The official implementation runs
on **PyTorch + NVIDIA CUDA** — it doesn't use Apple GPUs.

**mlx-qwen3-asr** is a ground-up reimplementation in Apple's MLX framework so
the same model runs natively on Mac hardware (M1/M2/M3/M4). Not a wrapper — every
layer is rewritten for MLX's Metal backend while producing identical transcriptions.

### Key features

- Full audio encoder + text decoder with correct interleaved MRoPE
- Supports both 1.7B and 0.6B model sizes
- Long audio chunking (up to 20 minutes per chunk) with no 30s feature truncation
- Optional forced-alignment timestamps via official Qwen forced aligner backend (`qwen-asr`/PyTorch)
- Streaming ASR support
- Multiple output formats: txt, json, srt, vtt, tsv
- Cached model/tokenizer instances for low repeated-call latency in Python workflows
- Decoder optimizations: preallocated KV cache + direct grouped-query fused attention
- 4-bit and 8-bit quantization
- Minimal dependencies: mlx, numpy, huggingface-hub, transformers

## Installation

Install from PyPI:

```bash
pip install mlx-qwen3-asr
```

Install with optional forced aligner support (timestamps):

```bash
pip install "mlx-qwen3-asr[aligner]"
```

For development:

```bash
git clone https://github.com/moona3k/mlx-qwen3-asr.git
cd mlx-qwen3-asr
pip install -e ".[dev]"
```

## Quick Start

### Python API

```python
from mlx_qwen3_asr import transcribe

result = transcribe("audio.wav")
print(result.text)
print(result.language)
```

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

### Loading models explicitly

```python
from mlx_qwen3_asr import load_model, load_audio, transcribe

model, config = load_model("Qwen/Qwen3-ASR-0.6B")
audio = load_audio("speech.wav")
result = transcribe(audio, model=model)
```

## CLI Usage

Basic transcription:

```bash
mlx-qwen3-asr audio.wav
```

Specify model, language, and output format:

```bash
mlx-qwen3-asr recording.mp3 --model Qwen/Qwen3-ASR-0.6B --language English -f srt -o output/
```

Multiple files with all output formats:

```bash
mlx-qwen3-asr *.wav -f all -o transcripts/ --verbose
```

Run `mlx-qwen3-asr --help` for the full list of options.

For word-level timestamps:

```bash
pip install qwen-asr
mlx-qwen3-asr audio.wav --timestamps
```

Timestamps use the official forced-aligner backend (PyTorch) for now; core ASR
inference remains native MLX.

## API Reference

### `transcribe(audio, *, model, language, return_timestamps, forced_aligner, dtype, max_new_tokens, verbose)`

Transcribe audio to text. Accepts a file path, numpy array, `mx.array`, or
`(array, sample_rate)` tuple. Returns a `TranscriptionResult`.
Set `return_timestamps=True` to request word-level timestamps. This requires
the optional `qwen-asr` dependency (used as forced-aligner backend).

### `load_model(name_or_path, *, dtype)`

Load a Qwen3-ASR model and its config from a HuggingFace repo or local path.
Returns `(model, config)`.

### `load_audio(path_or_url)`

Load and resample audio to mono 16 kHz. Returns an `mx.array`.

### `TranscriptionResult`

Frozen dataclass with fields:
- `text` (str) -- transcribed text
- `language` (str) -- detected or forced language
- `segments` (list[dict] | None) -- optional timestamped segments when requested

## Supported Languages

Qwen3-ASR supports 52 languages and dialects in total. The table below lists
the 30 core languages from the official docs:

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

Chinese dialect support (22 dialects) is provided by the official model but is
not expanded in this table.

## Model Variants

| | Qwen3-ASR-1.7B | Qwen3-ASR-0.6B |
|---|---|---|
| **Parameters** | 1.7B | 0.6B |
| **Audio encoder layers** | 24 | 18 |
| **Text decoder layers** | 28 | 28 |
| **Audio encoder dim (`d_model`)** | 1024 | 896 |
| **Text hidden size** | 2048 | 1024 |
| **Text attention (Q/KV heads)** | GQA (16/8) | GQA (16/8) |
| **RoPE theta** | 1,000,000 | 1,000,000 |
| **Accuracy** | Higher | Slightly lower |
| **Speed** | Slower | Faster |
| **HuggingFace** | `Qwen/Qwen3-ASR-1.7B` | `Qwen/Qwen3-ASR-0.6B` |

Both models use Multi-dimensional RoPE (MRoPE), 128-bin mel spectrograms, and
the same tokenizer with a vocabulary size of 151,936.

## Benchmarks

Selected word error rates (WER, lower is better) on standard benchmarks:

| Benchmark | Qwen3-ASR-1.7B | Whisper-large-v3 |
|---|---|---|
| LibriSpeech test-clean | **1.51** | 2.02 |
| LibriSpeech test-other | **3.04** | 4.28 |
| WenetSpeech test-net | **4.97** | 9.68 |
| Fleurs (avg 30 langs) | **5.2** | 8.1 |

Numbers from the [Qwen3-ASR technical report](https://huggingface.co/Qwen/Qwen3-ASR-1.7B).

## Validation

Run the fast quality gate (what CI enforces on macOS):

```bash
python scripts/quality_gate.py --mode fast
```

Run release-quality gate (includes reference parity):

```bash
RUN_REFERENCE_PARITY=1 python scripts/quality_gate.py --mode release
```

Run golden quality evaluation (LibriSpeech deterministic subset):

```bash
python scripts/eval_librispeech.py \
  --subset test-clean \
  --samples 100 \
  --model Qwen/Qwen3-ASR-0.6B \
  --json-output docs/benchmarks/golden-librispeech.json
```

Run unit tests directly:

```bash
pytest -q
```

Optional: run token-level greedy parity against the official PyTorch
implementation (slow; downloads model weights):

```bash
RUN_REFERENCE_PARITY=1 REFERENCE_PARITY_MODEL=Qwen/Qwen3-ASR-0.6B pytest -q tests/test_reference_parity.py
```

Publish quantized models to HuggingFace:

```bash
HF_TOKEN=... python scripts/publish_quantized.py \
  --source-model Qwen/Qwen3-ASR-0.6B \
  --repo-id YOUR_USER/mlx-qwen3-asr-0.6b-4bit \
  --bits 4
```

Convert a local 4-bit model and run it:

```bash
python scripts/convert.py \
  --model Qwen/Qwen3-ASR-0.6B \
  --quantize 4 \
  --group-size 64 \
  --output-dir /tmp/qwen3-asr-0.6b-4bit

mlx-qwen3-asr tests/fixtures/test_speech.wav --model /tmp/qwen3-asr-0.6b-4bit
```

Benchmark latency + RTF (record JSON for before/after comparisons):

```bash
python scripts/benchmark_asr.py tests/fixtures/test_speech.wav \
  --model Qwen/Qwen3-ASR-0.6B \
  --runs 5 \
  --json-output docs/benchmarks/latest.json
```

Nightly regression workflow runs fast gate + golden quality sample + latency benchmark:

- `.github/workflows/nightly-regression.yml`

## Acknowledgments

- [Qwen team](https://github.com/QwenLM) at Alibaba for the Qwen3-ASR model
- [mlx-whisper](https://github.com/ml-explore/mlx-examples) for architecture patterns and inspiration
- [Apple MLX team](https://github.com/ml-explore/mlx) for the MLX framework

## License

Apache 2.0. See [LICENSE](LICENSE) for details.
