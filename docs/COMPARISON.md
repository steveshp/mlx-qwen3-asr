# Comparison: MLX Qwen3-ASR Implementations

Feature comparison across available implementations for running Qwen3-ASR.

## Feature Matrix

| Feature | mlx-qwen3-asr | mlx-audio | qwen3-asr-swift | Official (PyTorch) |
|---------|---------------|-----------|-----------------|-------------------|
| MRoPE correct? | Yes (interleaved) | **No** (standard RoPE) | Yes | Yes |
| Streaming? | Yes | No | Yes | Yes |
| Forced aligner? | Yes | No | No | Yes |
| Long audio (>20min)? | Yes (energy split) | **No** (truncation bug) | Yes | Yes |
| Quantization? | Yes (4/8-bit) | Yes | No | No |
| 1.7B model correct? | Yes (MHA, 32 KV heads) | **No** (uses GQA/8 KV heads) | Yes | Yes |
| Install method | pip install | pip install | Build from source | pip install |
| Platform | macOS (Apple Silicon) | macOS (Apple Silicon) | macOS (Apple Silicon) | Linux (CUDA) |
| Compute backend | MLX (Metal) | MLX (Metal) | MLX (Metal) | PyTorch (CUDA) |

## Detailed Comparison

### mlx-qwen3-asr (this project)

- **Pros:**
  - Correct interleaved MRoPE implementation
  - Full feature set: streaming, forced alignment, long audio
  - Standalone package with minimal dependencies
  - Proper 1.7B config (MHA, not GQA)
  - Energy-based audio chunking for long files

- **Cons:**
  - New project, less battle-tested
  - Single-model focus (only Qwen3-ASR)

### mlx-audio (Blaizzy/mlx-audio)

- **Pros:**
  - Multi-model support (Whisper, MMS, Qwen, etc.)
  - Established project with community
  - pip-installable

- **Cons:**
  - Uses standard nn.RoPE instead of interleaved MRoPE -- incorrect for Qwen3-ASR
  - Issue #459: Long audio truncation
  - Applies 0.6B GQA config to 1.7B model (should be MHA)
  - Depends on bleeding-edge: transformers==5.0.0rc3, mlx-lm==0.30.5
  - No forced aligner support
  - No streaming support

### qwen3-asr-swift (ivan-digital/qwen3-asr-swift)

- **Pros:**
  - Full Swift + MLX implementation
  - Correct MRoPE
  - Fast: ~0.6s for 10s audio on M2 Max
  - Streaming support

- **Cons:**
  - No pip install -- must build from source
  - Swift-only (no Python API)
  - No forced aligner
  - Smaller community

### Official PyTorch (QwenLM/Qwen3-ASR)

- **Pros:**
  - Reference implementation (source of truth)
  - Full feature set
  - Best tested

- **Cons:**
  - CUDA only -- no Mac support
  - Requires PyTorch + GPU
  - Heavy dependencies (vllm==0.14.0)
  - Code quality issues (bare except, stale references)

## Performance Estimates

| Implementation | Hardware | Audio Length | Time |
|---------------|----------|-------------|------|
| qwen3-asr-swift | M2 Max | 10s | ~0.6s |
| mlx-qwen3-asr (est.) | M2 Max | 10s | ~1-2s |
| Official PyTorch | A100 | 10s | ~0.3s |

*Note: mlx-qwen3-asr estimates are preliminary. Actual performance depends on model size, quantization, and audio length.*

## Key Bugs in mlx-audio's Qwen3-ASR Implementation

1. **Standard RoPE instead of interleaved MRoPE:**
   - Uses `nn.RoPE` which doesn't support 3D frequency interleaving
   - Sections [24,20,20] with stride-3 pattern are not implemented
   - Results in degraded transcription quality

2. **Incorrect 1.7B model config:**
   - Applies 0.6B's GQA config (16 heads, 8 KV heads) to 1.7B model
   - 1.7B actually uses MHA (32 heads = 32 KV heads)
   - This changes attention computation significantly

3. **Long audio truncation (Issue #459):**
   - Audio longer than a threshold gets truncated instead of chunked
   - No energy-based splitting or overlap handling
