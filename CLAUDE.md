# CLAUDE.md

Agent instructions for working on mlx-qwen3-asr.

## North Star

**`pip install mlx-qwen3-asr` is the definitive way to run speech recognition on Apple Silicon.**

Not just transcription. The complete speech recognition experience:

- **Instant setup** — one pip install, no ffmpeg, no manual model downloads, no conversion scripts
- **Best-in-class speed** — pre-quantized models as default, Metal-optimized inference, sub-200ms latency on short clips
- **Full model coverage** — 0.6B for speed, 1.7B for accuracy, both tested and validated
- **Real-time streaming** — incremental transcription with KV cache reuse, not re-transcribe-everything
- **Word-level timestamps** — native MLX forced aligner, no PyTorch dependency
- **Zero unnecessary dependencies** — custom mel spectrogram, minimal tokenizer, no transformers at runtime
- **52 languages** — all validated, not just English
- **Production-grade** — proper error handling, memory management, batch support
- **Swift port** — once Python proves every decision, native Swift+MLX for apps and system integration

This is a ground-up reimplementation of the official PyTorch model using Apple's MLX framework. Same weights, same output, runs on Mac GPUs via Metal. Not a wrapper — every layer is rewritten for MLX.

### Execution order (how we get there)

Everything above matters. The order is about what unblocks what:

1. **Ship what works** — PyPI publish, pre-quantized models on HuggingFace, 1.7B validation
2. **Harden** — integration tests, bounds checks, dependency pins, debug cleanup
3. **Remove training wheels** — custom mel spectrogram (drop transformers for feature extraction), fully native forced aligner (drop PyTorch bridge)
4. **Real streaming** — incremental encoder, KV cache reuse across chunks, CJK-aware prefix rollback
5. **Zero external deps** — custom BPE tokenizer (drop transformers entirely), native audio decoding
6. **Swift port** — native Metal, system-level integration, app-embeddable framework

### What this is NOT

- Not a multi-model toolkit (that's mlx-audio)
- Not a training framework
- Not a server/API — just a library and CLI

## Project Context

This is an **MLX reimplementation of Qwen3-ASR** — the SOTA open-source ASR model — for Apple Silicon Macs. The official implementation is PyTorch + NVIDIA CUDA and doesn't use Apple GPUs. We rewrite every layer in Python + MLX so the same model runs natively on Mac hardware. Standalone package (not part of mlx-audio).

### Key Constraints

- **Python + MLX only** — all compute runs through MLX's Metal backend
- **Core ASR path has no PyTorch dependency at runtime**
- **Timestamps currently use optional `qwen-asr` (PyTorch) backend**
- **Minimal dependencies** — mlx, numpy, huggingface-hub, transformers (tokenizer only)
- **Correctness first** — proper MRoPE implementation (interleaved, not chunked)
- **Single-model focus** — only Qwen3-ASR, not a multi-model toolkit

### Architecture Overview

Qwen3-ASR is an encoder-decoder model:
1. Audio → mel spectrogram (128 bins) → Conv2d stem (8x downsample) → 24 transformer encoder layers → audio features
2. Audio features injected into text embedding sequence at placeholder positions
3. Text decoder (28 Qwen3-style layers with MRoPE) autoregressively generates transcription
4. Optional: Forced aligner (separate 0.6B model) provides word-level timestamps

## Documentation Map

| File | Purpose | When to Update |
|------|---------|----------------|
| `docs/RESEARCH.md` | Research findings, benchmarks, model analysis | New benchmark data, paper findings |
| `docs/ARCHITECTURE.md` | Qwen3-ASR architecture deep dive | Architecture understanding changes |
| `docs/DECISIONS.md` | Key decisions and rationale | Major technical choices |
| `docs/COMPARISON.md` | Comparison with alternatives | New competitors, feature changes |
| `docs/QUALITY_GATE.md` | Merge/release quality gates | Test policy changes |
| `docs/GOLDEN_DATASET.md` | Golden dataset policy and commands | Dataset/threshold policy changes |
| `docs/BENCHMARKING.md` | Runtime measurement protocol | Perf process changes |
| `docs/EXECUTION_TRACKER_2026-02-14.md` | Active optimization/refactor tracker | During this execution wave |
| `docs/ALGORITHMIC_MAXXING_2026-02-14.md` | Paper-backed algorithmic opportunities and applied findings | After deep paper/repo research passes |

## Code Conventions

### Python Style

- Python 3.10+ (f-strings, type hints, dataclasses)
- Use `mlx.nn.Module` for all model components (not torch)
- Type hints on all public functions
- Docstrings on all public classes and functions (Google style)
- No classes where functions suffice
- Prefer `@dataclass(frozen=True)` for output types

### File Organization

```
mlx_qwen3_asr/
├── __init__.py              # Public API only: transcribe(), load_model()
├── _version.py              # Single source of truth for version
├── config.py                # Dataclass configs (no MLX imports)
├── audio.py                 # Audio I/O + mel spectrogram (MLX)
├── mrope.py                 # Interleaved Multi-RoPE (critical correctness)
├── attention.py             # Shared SDPA helper (fused + fallback)
├── encoder.py               # Audio encoder modules + windowed attention helpers
├── decoder.py               # Text decoder modules + KV cache + causal mask
├── model.py                 # Qwen3ASRModel top-level glue + compatibility exports
├── convert.py               # Weight key remapping + Conv2d transpose
├── load_models.py           # HuggingFace download + model instantiation
├── generate.py              # Autoregressive decode loop + KV cache
├── transcribe.py            # High-level pipeline (the main entry point)
├── tokenizer.py             # Thin wrapper around HF Qwen2TokenizerFast
├── chunking.py              # Energy-based long audio splitting
├── forced_aligner.py        # Forced alignment model + LIS correction
├── streaming.py             # Streaming state machine
├── writers.py               # Output format writers (txt, srt, vtt, json, tsv)
├── cli.py                   # CLI entry point
└── assets/
    └── mel_filters.npz      # Pre-computed 128-bin Slaney mel filterbanks
```

### Naming Conventions

- Modules: `snake_case.py`
- Classes: `PascalCase` (e.g., `AudioEncoder`, `TextDecoder`)
- Functions: `snake_case` (e.g., `load_model`, `transcribe`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `SAMPLE_RATE`, `NUM_MEL_BINS`)
- Private: prefix with `_` (e.g., `_remap_key`, `_ModelHolder`)

## Git Commit Messages

This project uses **rich commit messages** optimized for AI-assisted development. Each commit captures enough context that a future agent (or human) can understand the full reasoning.

### Structure

```
<type>: <concise summary> (imperative mood)

## What Changed
Detailed breakdown of every file/section modified.

## Root Intent
Why this commit exists. The underlying problem or goal.

## Prompt That Would Produce This Diff
A detailed instruction that would recreate this work from scratch.
This is the "recipe" — if you gave this prompt to an AI agent with
access to the codebase, it should produce an equivalent diff.

## Files Changed
Summary with line counts for quick scanning.

Co-Authored-By: Claude <noreply@anthropic.com>
```

### Types

- `feat` — New feature
- `fix` — Bug fix
- `refactor` — Code restructuring
- `docs` — Documentation
- `test` — Adding tests
- `chore` — Maintenance, dependencies

### Example (Feature)

```
feat: implement interleaved MRoPE for text decoder

## What Changed
- mlx_qwen3_asr/mrope.py: New InterleavedMRoPE class with stride-3
  frequency assignment across sections [24,20,20]. Computes cos/sin
  embeddings for 3D position_ids (temporal, height, width).
- mlx_qwen3_asr/model.py: TextAttention now uses InterleavedMRoPE
  instead of nn.RoPE. Updated forward pass to pass position_ids.
- tests/test_mrope.py: Golden test comparing MLX output against
  PyTorch reference for known position_ids.

## Root Intent
The existing mlx-audio implementation uses standard nn.RoPE which
produces incorrect embeddings for Qwen3-ASR. The model uses Multi-
dimensional RoPE where frequency indices are interleaved across 3
spatial dimensions, not chunked. Without this, transcription quality
degrades significantly.

## Prompt That Would Produce This Diff
Implement interleaved MRoPE for Qwen3-ASR's text decoder in MLX.
Reference the official apply_interleaved_mrope() in the Qwen3-ASR
repo. Key details:
1. mrope_section = [24, 20, 20] (temporal, height, width)
2. Frequency assignment uses stride-3 interleaving: freq[0] → dim 0,
   freq[1] → dim 1, freq[2] → dim 2, freq[3] → dim 0, ...
3. position_ids shape is (batch, 3, seq_len) — one row per dimension
4. Output cos/sin shape is (batch, seq_len, head_dim)
Add a golden test comparing against PyTorch reference values.

## Files Changed
- mlx_qwen3_asr/mrope.py (+95)
- mlx_qwen3_asr/model.py (+12, ~8)
- tests/test_mrope.py (+48)

Co-Authored-By: Claude <noreply@anthropic.com>
```

### Example (Bug Fix)

```
fix: correct Conv2d weight transpose for audio encoder

## What Changed
- mlx_qwen3_asr/convert.py: Fixed transpose order for Conv2d weights.
  Was (out,in,kH,kW) → (out,kW,kH,in), now correctly (out,kH,kW,in).

## Root Intent
Audio encoder produced garbage features because Conv2d weights were
transposed with swapped height/width dimensions. PyTorch uses
(out_channels, in_channels, kH, kW) but MLX expects
(out_channels, kH, kW, in_channels).

## Prompt That Would Produce This Diff
Fix the Conv2d weight transpose in convert.py. The current code swaps
kH and kW during the PyTorch→MLX conversion. Correct order:
PyTorch (out,in,kH,kW) → MLX (out,kH,kW,in) via transpose(0,2,3,1).

## Files Changed
- mlx_qwen3_asr/convert.py (~3)

Co-Authored-By: Claude <noreply@anthropic.com>
```

### Why This Format

1. **Git history becomes documentation** — Rich context lives in version control, not lost chat logs
2. **Reproducible changes** — The prompt section is a recipe that could regenerate the diff
3. **Onboarding via archaeology** — New devs/agents understand decisions by reading commits
4. **Auditable reasoning** — The "why" is preserved alongside the "what"

### When to Use This Format

- **Always** for significant changes (multi-file, architectural, new modules)
- **Simplified** for trivial fixes (typos, single-line changes) — just type + summary + brief why

## Architecture Direction

These are known structural improvements to pursue as the codebase evolves. They represent the target architecture — implement them when touching the relevant code.

### Completed: Split model.py into encoder.py + decoder.py + model.py

Model modules are now separated by responsibility:
- `encoder.py` — `SinusoidalPositionEmbedding`, `AudioAttention`, `AudioEncoderLayer`, `AudioEncoder`, `_create_windowed_mask`
- `decoder.py` — `TextAttention`, `SwiGLU`, `TextDecoderLayer`, `TextDecoder`, `KVCache`, `_create_causal_mask`
- `model.py` — `Qwen3ASRModel` (top-level glue, audio injection, lm_head, compatibility re-exports)

### Add model.prefill() and model.step() methods

Currently `generate()` reaches into model internals: `model.model.embed_tokens()`, `model._inject_audio_features()`, `model.lm_head()`. Instead, add:
- `model.prefill(input_ids, audio_features, feature_lens, position_ids)` → returns logits + populated KV cache
- `model.step(token_id, position_ids, cache)` → returns logits + updated cache

Then `generate()` becomes a simple loop calling `step()` with no knowledge of model internals. This also enables real streaming (carry KV cache across chunks).

### Replace singletons with Session object

`_ModelHolder` and `_TokenizerHolder` are class-level mutable singletons that hide state, cause cache eviction bugs (aligner evicts ASR model), and make testing painful. Target API:

```python
# Power user path (explicit, testable, no hidden state)
session = mlx_qwen3_asr.Session(model="Qwen/Qwen3-ASR-0.6B")
result = session.transcribe("audio.wav")

# Convenience path (uses a default session internally)
result = mlx_qwen3_asr.transcribe("audio.wav")
```

The `Session` holds model, tokenizer, and optional aligner. Multiple sessions can coexist. The convenience `transcribe()` uses a module-level default session.

### Wire custom mel spectrogram to drop transformers for feature extraction

`log_mel_spectrogram()`, `stft()`, `_reflect_pad()`, `mel_filters()` exist in audio.py but are unused — the pipeline uses HF `WhisperFeatureExtractor` via `compute_features()`. Target:
1. Validate custom mel output matches `WhisperFeatureExtractor` output (bit-level parity on test fixtures)
2. Once proven, swap `compute_features()` to use custom implementation
3. `transformers` dependency becomes tokenizer-only (and eventually removable with custom BPE)

### Design streaming around resumable KV cache

Current streaming re-transcribes ALL accumulated audio every chunk — O(n²) and not real streaming. Target:
- `generate()` accepts and returns KV cache
- Streaming feeds new audio chunks through encoder, extends existing cache, decodes incrementally
- `_split_stable_unstable` must handle CJK (no whitespace splitting)

### Eliminate audio load round-trip

Currently: `load_audio()` → `mx.array` → `transcribe()` converts back to numpy → `compute_features()`. Unnecessary round-trip. Keep audio as numpy through the feature extraction pipeline, convert to `mx.array` only when entering the model.

## Working with This Codebase

### Before Making Changes

1. Read `docs/ARCHITECTURE.md` for model structure
2. Check the official Qwen3-ASR repo for reference implementation
3. Run tests: `python -m pytest tests/`

### After Making Changes

1. Run tests: `python -m pytest tests/`
2. Run linter: `ruff check mlx_qwen3_asr/`
3. Update docs if architecture understanding changed

### Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Language | Python + MLX | MLX has no Rust bindings; Metal compute is language-agnostic |
| Standalone vs mlx-audio | Standalone | mlx-audio cuts corners (no MRoPE), too many deps |
| Tokenizer | HF transformers (for now) | Qwen2TokenizerFast already exists; custom BPE is a future milestone |
| Audio loading | ffmpeg subprocess | Same as mlx-whisper, handles all formats |
| Weight format | safetensors | Standard for MLX ecosystem |
| RoPE | Custom interleaved MRoPE | MLX's nn.RoPE doesn't support 3D interleaved |
| Mel spectrogram | HF WhisperFeatureExtractor (for now) | Custom MLX implementation exists but needs parity validation before swapping in |

### Critical Correctness Rules

1. **MRoPE must be interleaved** — sections [24,20,20] with stride-3 interleaving, NOT chunked
2. **Audio encoder uses LayerNorm + bias** — different from text decoder
3. **Text decoder uses RMSNorm + no bias** — different from audio encoder
4. **Conv2d weight transpose** — PyTorch (out,in,kH,kW) → MLX (out,kH,kW,in)
5. **Sinusoidal pos embeddings are NOT in weights** — must be recomputed at init

### Known Bugs to Fix

1. **Audio injection bounds check** (model.py `_inject_audio_features`) — `cum_idx` can exceed `audio_features.shape[1]` if prompt token count doesn't match encoder output. Add assertion.
2. **Model cache eviction** (load_models.py `_ModelHolder`) — single-slot cache; loading aligner evicts ASR model. Change to dict keyed by `(path, dtype)` as interim fix before Session refactor.
3. **Bare except on fused attention** (model.py `_scaled_dot_product_attention`) — `except Exception` swallows real errors. Narrow to `(TypeError, ValueError)`.
4. **No language validation** (tokenizer.py `build_prompt_tokens`) — unsupported language strings silently degrade quality. Add warning against known language list.

## Verification Commands

### Install (development)

```bash
pip install -e ".[dev]"
```

### Quality gate (run before every PR)

```bash
python scripts/quality_gate.py --mode fast
```

### Release gate (run before tags/releases — requires `qwen-asr` for parity test)

```bash
RUN_REFERENCE_PARITY=1 python scripts/quality_gate.py --mode release
```

### Benchmark (run for any performance-related change)

```bash
python scripts/benchmark_asr.py tests/fixtures/test_speech.wav --runs 5 --json-output docs/benchmarks/latest.json
```

### Quick smoke test

```bash
python -c "import mlx_qwen3_asr; print(mlx_qwen3_asr.transcribe('tests/fixtures/test_audio.wav'))"
```

### CLI

```bash
mlx-qwen3-asr tests/fixtures/test_audio.wav --verbose
```

## Continuous Learning

This project uses a **self-improving workflow** where AI agents document what they learn.

### What to Document

After completing non-trivial work, record:
- **Gotchas discovered** — Things that wasted time or were surprising
- **Patterns that worked** — Approaches effective for MLX model porting
- **Failed approaches** — What didn't work and why

### Where to Document

| What | Where |
|------|-------|
| Cross-session learnings | `MEMORY.md` (auto-loaded, keep concise) |
| Codebase conventions | This file (CLAUDE.md) |
| Decision rationale | Commit messages + docs/DECISIONS.md |

### Self-Reflection Triggers

**After hitting an error or dead end:**
1. What was the root cause?
2. What did I try that didn't work?
3. Would a future agent hit the same wall? If yes, document it.

## Key References

- Official repo: https://github.com/QwenLM/Qwen3-ASR
- Paper: https://arxiv.org/abs/2601.21337
- mlx-whisper (pattern): https://github.com/ml-explore/mlx-examples/tree/main/whisper
- mlx-vlm MRoPE ref: https://github.com/Blaizzy/mlx-vlm
