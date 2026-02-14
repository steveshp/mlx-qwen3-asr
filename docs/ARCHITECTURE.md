# Architecture: Qwen3-ASR

Deep dive into the Qwen3-ASR model architecture.

## High-Level Architecture

```
Audio (16kHz mono) → Mel Spectrogram (128 bins)
    → Conv2d stem (3 layers, stride 2 each → 8x downsample)
    → Sinusoidal position embeddings (NOT learned)
    → 32 Transformer encoder layers (bidirectional attention)
    → LayerNorm + GELU projection → audio features (3584-dim)

Text prompt: <system>...<user><audio_start><audio_pad>*N<audio_end>
    → Token embedding (151936 vocab)
    → Replace audio_pad positions with audio features
    → 32 Qwen3 decoder layers (MRoPE, SwiGLU, RMSNorm)
    → LM head → next token logits
```

## Audio Encoder

### Conv2d Stem (8x Downsample)

Three convolutional layers, each with stride 2:

| Layer | In Channels | Out Channels | Kernel | Stride | Padding |
|-------|------------|--------------|--------|--------|---------|
| conv2d1 | 1 | 480 | 3x3 | 2 | 1 |
| conv2d2 | 480 | 480 | 3x3 | 2 | 1 |
| conv2d3 | 480 | 480 | 3x3 | 2 | 1 |

Each followed by GELU activation.

**Input:** Mel spectrogram (batch, 1, n_frames, 128)
**After conv:** (batch, 480, n_frames/8, 128/8) = (batch, 480, time', 16)
**Reshape:** (batch, time', 480 x 16) = (batch, time', 7680)
**Linear projection:** (batch, time', 7680) -> (batch, time', 1280)

### Sinusoidal Position Embeddings

Fixed (not learned) sinusoidal embeddings added after the conv stem:
- PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
- PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
- Maximum positions: 1500
- NOT stored in weight files -- computed at initialization

### Transformer Encoder Layers

32 layers, each with:
- **Pre-norm:** LayerNorm (with bias) -- NOT RMSNorm
- **Self-attention:** Bidirectional MHA, 20 heads, head_dim=64, WITH bias on projections
- **FFN:** Linear(1280->5120) + GELU + Linear(5120->1280), WITH bias
- **Residual connections** around attention and FFN

### Output Projection

After the transformer stack:
1. LayerNorm (ln_post)
2. Linear(1280->1280) + GELU (proj1)
3. Linear(1280->3584) (proj2)

Output: (batch, n_tokens, 3584) -- projected to match text decoder hidden_size (4096 for 1.7B, 1024 for 0.6B... wait, 3584 is the output_dim, not hidden_size)

## Text Decoder

### Interleaved MRoPE

Multi-dimensional Rotary Position Embedding -- the critical correctness component.

**Sections:** [24, 20, 20] for temporal, height, width dimensions
**Frequency assignment:** Stride-3 interleaving (NOT chunking)

```
freq[0]  -> temporal (section 0)
freq[1]  -> height   (section 1)
freq[2]  -> width    (section 2)
freq[3]  -> temporal (section 0)
freq[4]  -> height   (section 1)
freq[5]  -> width    (section 2)
...
freq[63] -> width    (section 2)
```

**position_ids shape:** (batch, 3, seq_len) -- one row per spatial dimension
**Output cos/sin shape:** (batch, seq_len, head_dim=128)

### Q/K Norms

Qwen3 innovation: RMSNorm applied per-head on queries and keys before RoPE.
- q_norm: RMSNorm(head_dim=128)
- k_norm: RMSNorm(head_dim=128)

### SwiGLU MLP

```
gate = gate_proj(x)         # Linear(4096 -> 22016, no bias)
up = up_proj(x)             # Linear(4096 -> 22016, no bias)
hidden = silu(gate) * up    # Element-wise
output = down_proj(hidden)  # Linear(22016 -> 4096, no bias)
```

### Decoder Layer Structure

Pre-norm with RMSNorm (NOT LayerNorm):
```
h = x + self_attn(rms_norm(x), cos, sin, mask, cache)
h = h + mlp(rms_norm(h))
```

## Audio-Text Fusion

Audio features are injected into the text embedding sequence:

1. Text prompt contains `<|audio_pad|>` placeholder tokens (token_id=151646)
2. Audio encoder produces features of shape (batch, n_audio_tokens, 3584)
3. Text embeddings are computed for all tokens including placeholders
4. Placeholder positions are replaced with audio features
5. Combined sequence is processed by the text decoder

The replacement uses cumulative indexing:
- Find positions where input_ids == audio_token_id
- Map each position to the corresponding audio feature vector
- Use mx.where for efficient selection

## Model Configuration Comparison

### Audio Encoder

| Parameter | 1.7B | 0.6B |
|-----------|------|------|
| encoder_layers | 32 | 18 |
| encoder_attention_heads | 20 | 14 |
| encoder_ffn_dim | 5120 | 3584 |
| d_model | 1280 | 896 |
| head_dim | 64 | 64 |
| output_dim | 3584 | 1024 |
| n_window | 100 | 50 |
| n_window_infer | 400 | 800 |
| downsample_hidden_size | 480 | 480 |

### Text Decoder

| Parameter | 1.7B | 0.6B |
|-----------|------|------|
| vocab_size | 151936 | 151936 |
| hidden_size | 4096 | 1024 |
| intermediate_size | 22016 | 3072 |
| num_hidden_layers | 32 | 28 |
| num_attention_heads | 32 | 16 |
| num_key_value_heads | **32 (MHA)** | **8 (GQA)** |
| head_dim | 128 | 128 |
| max_position_embeddings | 128000 | 65536 |
| rope_theta | 5000000.0 | 1000000.0 |

**Key difference:** 1.7B uses MHA (32 heads = 32 KV heads), 0.6B uses GQA (16 heads, 8 KV heads).

## Forced Aligner Architecture

The aligner (Qwen3-ForcedAligner-0.6B) is a separate model:

- **Audio encoder:** 24 layers, 16 heads, d_model=1024
- **Text decoder:** Same architecture as ASR-0.6B (28 layers, GQA 16/8)
- **Classification head:** Non-autoregressive, classify_num=5000 time bins
- **Time resolution:** timestamp_segment_time=80ms per classification unit
- **LIS correction:** Longest Increasing Subsequence for monotonic timestamps

## Weight Key Mapping

```
HuggingFace key                              -> MLX key
thinker.audio_tower.conv2d1.weight           -> audio_tower.conv2d1.weight (+ transpose)
thinker.audio_tower.conv2d1.bias             -> audio_tower.conv2d1.bias
thinker.audio_tower.layers.0.self_attn.*     -> audio_tower.layers.0.self_attn.*
thinker.model.layers.0.self_attn.q_proj.*    -> model.layers.0.self_attn.q_proj.*
thinker.model.embed_tokens.*                 -> model.embed_tokens.*
thinker.lm_head.*                            -> lm_head.*
```

All keys: strip `thinker.` prefix.
Conv2d weights: transpose (out, in, kH, kW) -> (out, kH, kW, in) via transpose(0, 2, 3, 1).

## Prompt Template

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|audio_start|><|audio_pad|>...(N times)...<|audio_pad|><|audio_end|>
<|im_start|>assistant
```

Output format: `language {detected_language}<asr_text>{transcription text}`

## Key Constants

```python
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
NUM_MEL_BINS = 128
MAX_ASR_INPUT_SECONDS = 1200.0
MIN_ASR_INPUT_SECONDS = 0.5
REPETITION_THRESHOLD = 20
```
