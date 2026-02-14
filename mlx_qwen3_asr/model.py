"""Main neural network module for Qwen3-ASR in MLX.

Qwen3-ASR is an encoder-decoder model for automatic speech recognition:
  - Audio Encoder: Conv2d stem (8x downsample) -> sinusoidal pos -> transformer -> projection
  - Text Decoder: Embedding -> transformer with MRoPE -> LM head
  - Top-level: Audio features injected into text embeddings at placeholder positions

Weight naming follows the HuggingFace checkpoint layout so that
``mx.load`` / ``model.load_weights`` works without key remapping.
"""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .config import AudioEncoderConfig, Qwen3ASRConfig, TextDecoderConfig
from .mrope import InterleavedMRoPE, apply_rotary_pos_emb

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scaled_dot_product_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    mask: Optional[mx.array] = None,
    scale: Optional[float] = None,
) -> mx.array:
    """Compute scaled dot-product attention with an optional mask.

    Tries ``mx.fast.scaled_dot_product_attention`` first (fused kernel),
    falling back to a manual implementation when unavailable.

    Args:
        q: Query tensor, shape (B, H, L, D).
        k: Key tensor, shape (B, H, S, D).
        v: Value tensor, shape (B, H, S, D).
        mask: Optional attention mask broadcastable to (B, H, L, S).
        scale: Scaling factor; defaults to 1/sqrt(D).

    Returns:
        Attention output, shape (B, H, L, D).
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    if hasattr(mx.fast, "scaled_dot_product_attention"):
        return mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask=mask
        )

    # Manual fallback
    scores = (q @ k.transpose(0, 1, 3, 2)) * scale
    if mask is not None:
        scores = scores + mask
    weights = mx.softmax(scores, axis=-1)
    return weights @ v


# ---------------------------------------------------------------------------
# Audio Encoder Components
# ---------------------------------------------------------------------------


class SinusoidalPositionEmbedding(nn.Module):
    """Fixed (non-learned) sinusoidal position embeddings.

    Computed at init time and stored as a frozen buffer -- these are NOT
    part of the saved weights.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    Args:
        num_positions: Maximum number of positions to support.
        embedding_dim: Dimensionality of each position embedding vector.
    """

    def __init__(self, num_positions: int, embedding_dim: int):
        super().__init__()
        self.num_positions = num_positions
        self.embedding_dim = embedding_dim

        half_dim = embedding_dim // 2
        # 10000^(2i/d) computed as exp(2i/d * log(10000))
        inv_freq = 1.0 / (
            10000.0 ** (mx.arange(0, half_dim, dtype=mx.float32) / half_dim)
        )
        positions = mx.arange(0, num_positions, dtype=mx.float32)
        # Outer product: (num_positions, half_dim)
        angles = positions[:, None] * inv_freq[None, :]
        # Interleave sin and cos: (num_positions, embedding_dim)
        pe = mx.concatenate([mx.sin(angles), mx.cos(angles)], axis=-1)
        # If embedding_dim is odd, trim the last column
        if embedding_dim % 2 == 1:
            pe = pe[:, :embedding_dim]
        self._pe = pe
        self.freeze()

    def __call__(self, position_ids: mx.array) -> mx.array:
        """Look up position embeddings.

        Args:
            position_ids: Integer positions, shape (...,).

        Returns:
            Embeddings of shape (..., embedding_dim).
        """
        return self._pe[position_ids]


class AudioAttention(nn.Module):
    """Bidirectional multi-head attention for the audio encoder.

    Uses bias on all projections. No causal mask, no RoPE.

    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor, shape (B, L, D).

        Returns:
            Output tensor, shape (B, L, D).
        """
        B, L, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (B, L, H, Dh) then transpose to (B, H, L, Dh)
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Bidirectional attention -- no causal mask
        out = _scaled_dot_product_attention(q, k, v)

        # (B, H, L, Dh) -> (B, L, H, Dh) -> (B, L, D)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(out)


class AudioEncoderLayer(nn.Module):
    """Pre-norm transformer layer for the audio encoder.

    Uses LayerNorm (NOT RMSNorm) and GELU activation, following the
    Whisper / Qwen3-ASR audio encoder convention.

    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        encoder_ffn_dim: Feed-forward intermediate dimension.
    """

    def __init__(self, d_model: int, num_heads: int, encoder_ffn_dim: int):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.self_attn = AudioAttention(d_model, num_heads)
        self.final_layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, encoder_ffn_dim, bias=True)
        self.fc2 = nn.Linear(encoder_ffn_dim, d_model, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor, shape (B, L, D).

        Returns:
            Output tensor, shape (B, L, D).
        """
        # Self-attention block (pre-norm)
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x)
        x = residual + x

        # Feed-forward block (pre-norm)
        residual = x
        x = self.final_layer_norm(x)
        x = self.fc1(x)
        x = nn.gelu(x)
        x = self.fc2(x)
        x = residual + x

        return x


class AudioEncoder(nn.Module):
    """Full audio encoder: Conv2d stem -> sinusoidal pos -> transformer -> projection.

    The Conv2d stem applies three stride-2 convolutions for 8x temporal
    downsampling. The output is projected to the text decoder's hidden size
    via a 2-layer MLP.

    Args:
        config: AudioEncoderConfig with model hyperparameters.
    """

    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.config = config
        dhs = config.downsample_hidden_size  # 480 for 1.7B

        # Conv2d stem: 3 layers, each stride=2, total 8x temporal downsample
        self.conv2d1 = nn.Conv2d(1, dhs, kernel_size=3, stride=2, padding=1)
        self.conv2d2 = nn.Conv2d(dhs, dhs, kernel_size=3, stride=2, padding=1)
        self.conv2d3 = nn.Conv2d(dhs, dhs, kernel_size=3, stride=2, padding=1)

        # After 3x stride-2 on 128 mel bins: 128 -> 64 -> 32 -> 16
        freq_after_conv = config.num_mel_bins // 8
        self.conv_out = nn.Linear(dhs * freq_after_conv, config.d_model, bias=False)

        # Sinusoidal position embeddings (fixed, not learned)
        self.embed_positions = SinusoidalPositionEmbedding(
            config.max_source_positions, config.d_model
        )

        # Transformer encoder layers
        self.layers = [
            AudioEncoderLayer(
                config.d_model,
                config.encoder_attention_heads,
                config.encoder_ffn_dim,
            )
            for _ in range(config.encoder_layers)
        ]

        # Post-transformer layer norm
        self.ln_post = nn.LayerNorm(config.d_model)

        # Output projection MLP: d_model -> d_model -> output_dim
        self.proj1 = nn.Linear(config.d_model, config.d_model)
        self.proj2 = nn.Linear(config.d_model, config.output_dim)

    def _apply_conv_stem(self, x: mx.array) -> mx.array:
        """Apply the 3-layer Conv2d stem with GELU activations.

        Args:
            x: Input tensor, shape (B, 1, T, F) where T=time, F=freq.

        Returns:
            Output tensor, shape (B, dhs, T', F') after 8x downsampling.
        """
        x = nn.gelu(self.conv2d1(x))
        x = nn.gelu(self.conv2d2(x))
        x = nn.gelu(self.conv2d3(x))
        return x

    def get_output_lengths(self, input_lengths: mx.array) -> mx.array:
        """Compute output sequence lengths after the Conv2d stem.

        With stride=2, padding=1, kernel_size=3:
            L_out = floor((L_in + 2*1 - 3) / 2 + 1) = (L_in + 1) // 2

        Applied three times for three conv layers.

        Args:
            input_lengths: Frame counts, shape (B,).

        Returns:
            Token counts after conv stem, shape (B,).
        """
        lengths = input_lengths
        for _ in range(3):
            lengths = (lengths + 1) // 2
        return lengths

    def __call__(
        self,
        input_features: mx.array,
        feature_lens: mx.array,
    ) -> mx.array:
        """Encode audio mel spectrograms into feature vectors.

        Args:
            input_features: Mel spectrogram, shape (B, n_mels, n_frames).
            feature_lens: Frame counts per sample, shape (B,).

        Returns:
            Audio features, shape (B, n_tokens, output_dim).
        """
        B = input_features.shape[0]

        # (B, n_mels, n_frames) -> (B, n_frames, n_mels) -> (B, 1, n_frames, n_mels)
        # Note: MLX Conv2d expects (B, H, W, C) -- NHWC format
        x = input_features.transpose(0, 2, 1)  # (B, n_frames, n_mels)
        x = x[:, :, :, None]  # (B, n_frames, n_mels, 1) -- NHWC with C=1

        # Apply Conv2d stem: output is (B, T', F', dhs) in NHWC
        x = self._apply_conv_stem(x)

        # Reshape: (B, T', F', dhs) -> (B, T', F' * dhs)
        B, T_prime, F_prime, C = x.shape
        x = x.reshape(B, T_prime, F_prime * C)

        # Project to d_model
        x = self.conv_out(x)  # (B, T', d_model)

        # Add sinusoidal position embeddings
        positions = mx.arange(T_prime)
        x = x + self.embed_positions(positions)

        # Transformer encoder layers
        for layer in self.layers:
            x = layer(x)

        # Post-norm
        x = self.ln_post(x)

        # Output projection MLP
        x = nn.gelu(self.proj1(x))
        x = self.proj2(x)  # (B, T', output_dim)

        return x


# ---------------------------------------------------------------------------
# Text Decoder Components
# ---------------------------------------------------------------------------


class TextAttention(nn.Module):
    """Causal self-attention with MRoPE and Q/K norms for the text decoder.

    Uses RMSNorm on queries and keys (Qwen3 innovation). Supports grouped
    query attention (GQA) when num_kv_heads < num_heads, though the 1.7B
    model uses full MHA.

    Args:
        config: TextDecoderConfig with model hyperparameters.
    """

    def __init__(self, config: TextDecoderConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

        # Per-head RMSNorm on queries and keys (Qwen3 innovation)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
        layer_idx: int = 0,
    ) -> mx.array:
        """Forward pass.

        Args:
            x: Input hidden states, shape (B, L, D).
            cos: MRoPE cosine embeddings, shape (B, L, head_dim).
            sin: MRoPE sine embeddings, shape (B, L, head_dim).
            mask: Optional causal attention mask, broadcastable to (B, H, L, S).
            cache: Optional KV cache for autoregressive generation.
            layer_idx: Index of this layer in the decoder stack (for cache).

        Returns:
            Output tensor, shape (B, L, D).
        """
        B, L, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape: (B, L, num_heads * head_dim) -> (B, L, num_heads, head_dim)
        q = q.reshape(B, L, self.num_heads, self.head_dim)
        k = k.reshape(B, L, self.num_kv_heads, self.head_dim)
        v = v.reshape(B, L, self.num_kv_heads, self.head_dim)

        # Apply Q/K norms (operates on last dim)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Transpose to (B, H, L, D) for attention
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Apply MRoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Update KV cache if provided
        if cache is not None:
            k, v = cache.update(k, v, layer_idx)

        # GQA: repeat K, V if num_kv_heads < num_heads
        if self.num_kv_groups > 1:
            k = mx.repeat(k, self.num_kv_groups, axis=1)
            v = mx.repeat(v, self.num_kv_groups, axis=1)

        # Scaled dot-product attention
        out = _scaled_dot_product_attention(q, k, v, mask=mask)

        # (B, H, L, Dh) -> (B, L, H, Dh) -> (B, L, D)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network (Qwen3-style).

    Computes: down_proj(silu(gate_proj(x)) * up_proj(x))

    Args:
        hidden_size: Input/output dimension.
        intermediate_size: Intermediate (expanded) dimension.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor, shape (..., hidden_size).

        Returns:
            Output tensor, shape (..., hidden_size).
        """
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TextDecoderLayer(nn.Module):
    """Pre-norm transformer decoder layer with RMSNorm.

    Uses RMSNorm (NOT LayerNorm) as is standard for Qwen3/LLaMA-family models.

    Args:
        config: TextDecoderConfig with model hyperparameters.
    """

    def __init__(self, config: TextDecoderConfig):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = TextAttention(config)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
        layer_idx: int = 0,
    ) -> mx.array:
        """Forward pass.

        Args:
            x: Input hidden states, shape (B, L, D).
            cos: MRoPE cosine embeddings.
            sin: MRoPE sine embeddings.
            mask: Optional causal attention mask.
            cache: Optional KV cache.
            layer_idx: Index of this layer (for cache).

        Returns:
            Output tensor, shape (B, L, D).
        """
        # Self-attention block
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, cos, sin, mask=mask, cache=cache, layer_idx=layer_idx)
        x = residual + x

        # Feed-forward block
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x


class TextDecoder(nn.Module):
    """Full text decoder: embedding -> transformer with MRoPE -> norm.

    Note: The LM head is NOT included here; it lives in the top-level
    Qwen3ASRModel to match the HuggingFace weight layout.

    Args:
        config: TextDecoderConfig with model hyperparameters.
    """

    def __init__(self, config: TextDecoderConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            TextDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # MRoPE with interleaved frequency assignment
        self.rotary_emb = InterleavedMRoPE(
            head_dim=config.head_dim,
            base=config.rope_theta,
            mrope_section=[24, 20, 20],
        )

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        """Forward pass through the text decoder.

        Exactly one of ``input_ids`` or ``inputs_embeds`` must be provided.

        Args:
            input_ids: Token IDs, shape (B, L).
            inputs_embeds: Pre-computed embeddings, shape (B, L, D).
            position_ids: MRoPE position IDs, shape (B, 3, L).
            attention_mask: Causal mask, broadcastable to (B, H, L, S).
            cache: Optional KV cache for autoregressive generation.

        Returns:
            Hidden states, shape (B, L, D).
        """
        if inputs_embeds is not None:
            h = inputs_embeds
        elif input_ids is not None:
            h = self.embed_tokens(input_ids)
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        # Compute MRoPE cos/sin
        cos, sin = self.rotary_emb(position_ids, dtype=h.dtype)

        # Prepare causal mask if not provided
        if attention_mask is None:
            L = h.shape[1]
            if cache is not None and cache.offset > 0:
                # During generation, single-token step -- no mask needed
                attention_mask = None
            else:
                # Full causal mask
                attention_mask = _create_causal_mask(L, h.dtype)

        # Apply transformer layers
        for i, layer in enumerate(self.layers):
            h = layer(h, cos, sin, mask=attention_mask, cache=cache, layer_idx=i)

        # Final norm
        h = self.norm(h)
        return h


# ---------------------------------------------------------------------------
# KV Cache
# ---------------------------------------------------------------------------


class KVCache:
    """Per-layer key-value cache for autoregressive generation.

    Stores cached key and value tensors for each layer, concatenating new
    entries along the sequence dimension on each step.

    Args:
        num_layers: Number of decoder layers.
    """

    def __init__(self, num_layers: int):
        self.keys: list[Optional[mx.array]] = [None] * num_layers
        self.values: list[Optional[mx.array]] = [None] * num_layers
        self.offset: int = 0

    def update(
        self,
        key: mx.array,
        value: mx.array,
        layer_idx: int,
    ) -> tuple[mx.array, mx.array]:
        """Append new key/value entries to the cache.

        Args:
            key: New key tensor, shape (B, H, L_new, D).
            value: New value tensor, shape (B, H, L_new, D).
            layer_idx: Which layer this cache belongs to.

        Returns:
            Tuple of (full_keys, full_values) including cached history.
        """
        if self.keys[layer_idx] is not None:
            self.keys[layer_idx] = mx.concatenate(
                [self.keys[layer_idx], key], axis=2
            )
            self.values[layer_idx] = mx.concatenate(
                [self.values[layer_idx], value], axis=2
            )
        else:
            self.keys[layer_idx] = key
            self.values[layer_idx] = value

        # Update offset after the last layer processes its step
        if layer_idx == len(self.keys) - 1:
            self.offset += key.shape[2]

        return self.keys[layer_idx], self.values[layer_idx]

    @property
    def seq_len(self) -> int:
        """Return the current cached sequence length."""
        return self.offset


# ---------------------------------------------------------------------------
# Causal Mask Helper
# ---------------------------------------------------------------------------


def _create_causal_mask(seq_len: int, dtype: mx.Dtype = mx.float32) -> mx.array:
    """Create an additive causal mask for self-attention.

    Masked positions are filled with -inf (or a very large negative number);
    unmasked positions are 0.

    Args:
        seq_len: Sequence length.
        dtype: Output data type.

    Returns:
        Causal mask, shape (1, 1, seq_len, seq_len).
    """
    mask = mx.full((seq_len, seq_len), -1e9, dtype=dtype)
    mask = mx.triu(mask, k=1)  # zero on and below diagonal, -1e9 above
    return mask[None, None, :, :]  # (1, 1, L, L)


# ---------------------------------------------------------------------------
# Top-level Model
# ---------------------------------------------------------------------------


class Qwen3ASRModel(nn.Module):
    """Top-level Qwen3-ASR model: audio encoder + text decoder + LM head.

    During inference:
      1. Encode audio via the audio tower.
      2. Build text embeddings, injecting audio features at placeholder positions.
      3. Run the text decoder autoregressively.
      4. Project to vocabulary logits with the LM head.

    Weight key mapping (matches HuggingFace checkpoint):
      - ``audio_tower.*``  -- AudioEncoder
      - ``model.*``        -- TextDecoder (embed_tokens, layers, norm)
      - ``lm_head.*``      -- final linear projection

    Args:
        config: Qwen3ASRConfig with full model hyperparameters.
    """

    def __init__(self, config: Qwen3ASRConfig):
        super().__init__()
        self.config = config
        self.audio_token_id = config.audio_token_id

        self.audio_tower = AudioEncoder(config.audio_config)
        self.model = TextDecoder(config.text_config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
        )

        # Tie weights if configured
        if config.text_config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def __call__(
        self,
        input_ids: mx.array,
        input_features: Optional[mx.array] = None,
        feature_lens: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        """Forward pass through the full model.

        Args:
            input_ids: Token IDs including audio placeholders, shape (B, L).
            input_features: Mel spectrogram, shape (B, n_mels, n_frames).
                Required on the first call (prefill); None during generation.
            feature_lens: Audio frame counts, shape (B,).
                Required when input_features is provided.
            attention_mask: Optional causal mask.
            position_ids: MRoPE position IDs, shape (B, 3, L).
            cache: Optional KV cache for autoregressive generation.

        Returns:
            Logits, shape (B, L, vocab_size).
        """
        # Get text embeddings
        embeds = self.model.embed_tokens(input_ids)

        # Encode and inject audio features if audio is provided
        if input_features is not None:
            audio_features = self.audio_tower(input_features, feature_lens)
            audio_mask = input_ids == self.audio_token_id
            embeds = self._inject_audio_features(embeds, audio_features, audio_mask)

        # Run text decoder
        hidden = self.model(
            inputs_embeds=embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            cache=cache,
        )

        # Project to vocabulary logits
        logits = self.lm_head(hidden)
        return logits

    def _inject_audio_features(
        self,
        embeds: mx.array,
        audio_features: mx.array,
        audio_mask: mx.array,
    ) -> mx.array:
        """Replace audio placeholder positions with encoded audio features.

        Uses cumulative-sum indexing to map each placeholder position to the
        corresponding audio feature vector, then selects via ``mx.where``.

        Args:
            embeds: Text embeddings, shape (B, L, D). Positions matching
                audio_token_id contain placeholder embeddings to replace.
            audio_features: Encoded audio, shape (B, N_audio, D) where
                N_audio is the number of audio tokens produced by the encoder.
            audio_mask: Boolean mask, shape (B, L). True where
                input_ids == audio_token_id.

        Returns:
            Embeddings with audio features injected, shape (B, L, D).
        """
        # Build a cumulative index that maps each position to an audio feature index.
        # For positions where audio_mask is True, cum_idx increases by 1.
        # For non-audio positions, cum_idx stays the same (pointing to the last
        # valid audio feature, but those positions won't be selected anyway).
        cum_idx = mx.cumsum(audio_mask.astype(mx.int32), axis=1) - 1
        cum_idx = mx.maximum(cum_idx, 0)  # (B, L)

        # Gather audio features for every position using cum_idx.
        # audio_features[b, cum_idx[b]] gives shape (L, D) for each batch element.
        B = embeds.shape[0]
        audio_expanded_parts = []
        for b in range(B):
            expanded = audio_features[b, cum_idx[b]]  # (L, D)
            audio_expanded_parts.append(expanded)
        audio_expanded = mx.stack(audio_expanded_parts)  # (B, L, D)

        # Select audio features at masked positions, text embeds elsewhere
        mask_3d = audio_mask[:, :, None]  # (B, L, 1)
        result = mx.where(mask_3d, audio_expanded, embeds)
        return result

    def create_cache(self) -> KVCache:
        """Create a fresh KV cache for autoregressive generation.

        Returns:
            Empty KVCache with the correct number of layers.
        """
        return KVCache(self.config.text_config.num_hidden_layers)

    @property
    def num_audio_encoder_layers(self) -> int:
        """Return the number of audio encoder transformer layers."""
        return len(self.audio_tower.layers)

    @property
    def num_text_decoder_layers(self) -> int:
        """Return the number of text decoder transformer layers."""
        return len(self.model.layers)
