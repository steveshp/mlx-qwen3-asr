# CLAUDE.md

Agent instructions for working on mlx-qwen3-asr.

## Project Context

This is an **MLX port of Qwen3-ASR** — the SOTA open-source ASR model — for Apple Silicon Macs. Written in Python, using Apple's MLX framework for Metal GPU acceleration. Standalone package (not part of mlx-audio).

### Key Constraints

- **Python + MLX only** — all compute runs through MLX's Metal backend
- **No PyTorch dependency at runtime** — weights are pre-converted or converted on-the-fly
- **Minimal dependencies** — mlx, numpy, huggingface-hub, transformers (tokenizer only)
- **Correctness first** — proper MRoPE implementation (interleaved, not chunked)
- **Single-model focus** — only Qwen3-ASR, not a multi-model toolkit

### Architecture Overview

Qwen3-ASR is an encoder-decoder model:
1. Audio → mel spectrogram (128 bins) → Conv2d stem (8x downsample) → 32 transformer encoder layers → audio features
2. Audio features injected into text embedding sequence at placeholder positions
3. Text decoder (32 Qwen3-style layers with MRoPE) autoregressively generates transcription
4. Optional: Forced aligner (separate 0.6B model) provides word-level timestamps

## Documentation Map

| File | Purpose | When to Update |
|------|---------|----------------|
| `docs/RESEARCH.md` | Research findings, benchmarks, model analysis | New benchmark data, paper findings |
| `docs/ARCHITECTURE.md` | Qwen3-ASR architecture deep dive | Architecture understanding changes |
| `docs/DECISIONS.md` | Key decisions and rationale | Major technical choices |
| `docs/COMPARISON.md` | Comparison with alternatives | New competitors, feature changes |

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
├── model.py                 # All nn.Module classes (~600 lines)
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
| Tokenizer | HF transformers | Qwen2TokenizerFast already exists, not worth reimplementing |
| Audio loading | ffmpeg subprocess | Same as mlx-whisper, handles all formats |
| Weight format | safetensors | Standard for MLX ecosystem |
| RoPE | Custom interleaved MRoPE | MLX's nn.RoPE doesn't support 3D interleaved |

### Critical Correctness Rules

1. **MRoPE must be interleaved** — sections [24,20,20] with stride-3 interleaving, NOT chunked
2. **Audio encoder uses LayerNorm + bias** — different from text decoder
3. **Text decoder uses RMSNorm + no bias** — different from audio encoder
4. **Conv2d weight transpose** — PyTorch (out,in,kH,kW) → MLX (out,kH,kW,in)
5. **Sinusoidal pos embeddings are NOT in weights** — must be recomputed at init

## Verification Commands

### Install (development)

```bash
pip install -e ".[dev]"
```

### Tests

```bash
python -m pytest tests/ -v
```

### Lint

```bash
ruff check mlx_qwen3_asr/
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
