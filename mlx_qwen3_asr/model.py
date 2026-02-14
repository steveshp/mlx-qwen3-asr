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
        try:
            return mx.fast.scaled_dot_product_attention(
                q, k, v, scale=scale, mask=mask
            )
        except (TypeError, ValueError, RuntimeError):
            # Fall through to a compatibility path below.
            pass

    # Manual fallback
    if q.shape[1] != k.shape[1]:
        if q.shape[1] % k.shape[1] != 0:
            raise ValueError(
                f"Incompatible attention heads: q={q.shape[1]}, k={k.shape[1]}"
            )
        groups = q.shape[1] // k.shape[1]
        k = mx.repeat(k, groups, axis=1)
        v = mx.repeat(v, groups, axis=1)

    scores = (q @ k.transpose(0, 1, 3, 2)) * scale
    if mask is not None:
        scores = scores + mask
    weights = mx.softmax(scores, axis=-1)
    return weights @ v


# ---------------------------------------------------------------------------
# Audio Encoder Components
# ---------------------------------------------------------------------------

# Use segmented per-window execution only once we have enough windows that
# dense block-mask attention becomes a clear overhead.
_WINDOWED_SEGMENT_MIN_WINDOWS = 20


class SinusoidalPositionEmbedding(nn.Module):
    """Fixed (non-learned) sinusoidal position embeddings.

    Computed at init time and stored as a frozen buffer -- these are NOT
    part of the saved weights.

    Uses the official Qwen3-ASR formula matching ``SinusoidsPositionEmbedding``
    from the HuggingFace reference::

        log_timescale_increment = log(10000) / (channels // 2 - 1)
        inv_timescales = exp(-i * log_timescale_increment)

    This differs from the standard Transformer PE formula by using
    ``half_dim - 1`` in the denominator (not ``half_dim``).

    Args:
        num_positions: Maximum number of positions to support.
        embedding_dim: Dimensionality of each position embedding vector.
    """

    def __init__(self, num_positions: int, embedding_dim: int):
        super().__init__()
        self.num_positions = num_positions
        self.embedding_dim = embedding_dim

        half_dim = embedding_dim // 2
        # Match official: log_timescale_increment = log(10000) / (half_dim - 1)
        log_timescale_increment = math.log(10000.0) / (half_dim - 1)
        inv_timescales = mx.exp(
            -log_timescale_increment * mx.arange(half_dim, dtype=mx.float32)
        )
        positions = mx.arange(0, num_positions, dtype=mx.float32)
        # Outer product: (num_positions, half_dim)
        scaled_time = positions[:, None] * inv_timescales[None, :]
        # Concatenate sin and cos: (num_positions, embedding_dim)
        pe = mx.concatenate([mx.sin(scaled_time), mx.cos(scaled_time)], axis=-1)
        # If embedding_dim is odd, trim the last column
        if embedding_dim % 2 == 1:
            pe = pe[:, :embedding_dim]
        self._pe = pe
        self.freeze()

    def __call__(self, positions: int | mx.array) -> mx.array:
        """Return sinusoidal embeddings by length or explicit indices.

        Args:
            positions:
                - int: returns embeddings for 0..positions-1.
                - mx.array: returns embeddings indexed by that tensor.

        Returns:
            Embeddings of shape (positions, embedding_dim).
        """
        if isinstance(positions, int):
            return self._pe[:positions]
        return self._pe[positions]


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

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor, shape (B, L, D).
            mask: Optional attention mask, broadcastable to (B, H, L, S).
                Used to mask padded positions in the encoder.

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

        # Bidirectional attention with optional padding mask
        out = _scaled_dot_product_attention(q, k, v, mask=mask)

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

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor, shape (B, L, D).
            mask: Optional attention mask for padding, broadcastable to (B, H, L, S).

        Returns:
            Output tensor, shape (B, L, D).
        """
        # Self-attention block (pre-norm)
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, mask=mask)
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
            x: Input tensor in NHWC format, shape ``(B, H=mel, W=time, C)``.

        Returns:
            Output tensor in NHWC, shape ``(B, mel_down, time_down, dhs)``.
        """
        x = nn.gelu(self.conv2d1(x))
        x = nn.gelu(self.conv2d2(x))
        x = nn.gelu(self.conv2d3(x))
        return x

    def get_output_lengths(self, input_lengths: mx.array) -> mx.array:
        """Compute output token counts matching the official encoder formula.

        The encoder splits mel frames into chunks of ``n_window * 2``
        (typically 100) frames each.  Full chunks produce 13 tokens; the
        tail chunk (if any) produces fewer.  The formula matches
        ``_get_feat_extract_output_lengths`` from the official Qwen3-ASR
        codebase.

        Args:
            input_lengths: Mel frame counts, shape ``(B,)``.

        Returns:
            Output token counts, shape ``(B,)``.
        """
        chunk_size = self.config.n_window * 2  # e.g. 100
        # Tail chunk: frames not filling a full chunk
        tail_frames = input_lengths % chunk_size
        # Apply 3x stride-2 conv downsampling to tail
        tail_after_conv1 = (tail_frames - 1) // 2 + 1
        tail_after_conv2 = (tail_after_conv1 - 1) // 2 + 1
        tail_tokens = (tail_after_conv2 - 1) // 2 + 1
        # When tail_frames == 0 there is no tail chunk → 0 tokens.
        # (MLX integer // truncates towards zero for negatives, unlike
        # Python/PyTorch which floor towards -inf, so (0-1)//2 gives 0
        # not -1, producing an off-by-one without this guard.)
        tail_tokens = mx.where(
            tail_frames > 0, tail_tokens, mx.zeros_like(tail_tokens)
        )
        # Full chunks produce the conv-downsampled length of chunk_size.
        full_after_conv1 = (chunk_size - 1) // 2 + 1
        full_after_conv2 = (full_after_conv1 - 1) // 2 + 1
        full_chunk_tokens = (full_after_conv2 - 1) // 2 + 1
        n_full_chunks = input_lengths // chunk_size
        return tail_tokens + n_full_chunks * full_chunk_tokens

    def __call__(
        self,
        input_features: mx.array,
        feature_lens: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Encode audio mel spectrograms into feature vectors.

        Implements the official Qwen3-ASR encoder pipeline:

        1. Split mel into chunks of ``n_window * 2`` frames
        2. Per-chunk Conv2d stem with channel-major reshape
        3. Per-chunk sinusoidal position embeddings
        4. Windowed block-diagonal attention across transformer layers
        5. Post-processing: LayerNorm, GELU projection

        Args:
            input_features: Mel spectrogram, shape ``(B, n_mels, n_frames)``.
                May be padded (e.g. to 3000 frames / 30 seconds).
            feature_lens: Actual frame counts per sample (before padding),
                shape ``(B,)``.

        Returns:
            Tuple of ``(audio_features, output_lens)``:
                audio_features: shape ``(B, max_tokens, output_dim)``
                output_lens: shape ``(B,)``, valid token count per sample
        """
        B = input_features.shape[0]
        chunk_size = self.config.n_window * 2
        n_window_infer = self.config.n_window_infer

        all_features = []
        all_output_lens = []

        for b in range(B):
            feat_len = int(feature_lens[b].item())
            mel = input_features[b, :, :feat_len]  # (128, actual_frames)
            features = self._encode_single(mel, chunk_size, n_window_infer)
            all_features.append(features)
            all_output_lens.append(features.shape[0])

        # Pad to max length and stack into batch
        max_len = max(all_output_lens)
        padded = []
        for f in all_features:
            if f.shape[0] < max_len:
                pad_arr = mx.zeros(
                    (max_len - f.shape[0], f.shape[-1]), dtype=f.dtype
                )
                f = mx.concatenate([f, pad_arr], axis=0)
            padded.append(f)

        output = mx.stack(padded)  # (B, max_len, output_dim)
        output_lens = mx.array(all_output_lens)
        return output, output_lens

    def _encode_single(
        self,
        mel: mx.array,
        chunk_size: int,
        n_window_infer: int,
    ) -> mx.array:
        """Encode a single (unbatched) mel spectrogram.

        Implements the per-sample encoder forward pass matching the official
        ``Qwen3ASRAudioEncoder.forward()`` logic:

        - Split mel into chunks of ``chunk_size`` frames
        - Process each chunk through the Conv2d stem independently
        - Use channel-major reshape: ``(1, F', T', C) -> (1, T', C*F')``
        - Apply per-chunk sinusoidal position embeddings (each chunk starts
          from position 0)
        - Run transformer with windowed (block-diagonal) attention

        Args:
            mel: Mel spectrogram, shape ``(n_mels, n_frames)``, already
                trimmed to actual length (no padding).
            chunk_size: Number of mel frames per chunk (``n_window * 2``).
            n_window_infer: Number of mel frames per attention window.

        Returns:
            Encoded features, shape ``(n_tokens, output_dim)``.
        """
        total_frames = mel.shape[1]

        # --- Per-chunk Conv2d processing ---
        chunk_token_lens: list[int] = []
        chunk_conv_outputs: list[mx.array] = []

        for start in range(0, total_frames, chunk_size):
            end = min(start + chunk_size, total_frames)
            chunk_mel = mel[:, start:end]  # (n_mels, chunk_len)

            # MLX NHWC: (1, H=n_mels, W=chunk_len, C=1)
            x = chunk_mel[None, :, :, None]

            # Conv2d stem: output is (1, mel_down, time_down, dhs) in NHWC
            x = self._apply_conv_stem(x)

            # Channel-major reshape to match conv_out weight layout.
            # PyTorch does: (B,C,F,T) -> permute(0,3,1,2) -> (B,T,C,F) -> view(B,T,C*F)
            # MLX NHWC:    (1,F',T',C) -> transpose(0,2,3,1) -> (1,T',C,F')
            _, F_d, T_d, C_d = x.shape
            x = x.transpose(0, 2, 3, 1)  # (1, T', C=dhs, F'=freq_down)
            x = x.reshape(1, T_d, C_d * F_d)  # (1, T', dhs*freq_down)

            chunk_token_lens.append(T_d)
            chunk_conv_outputs.append(x[0])  # (T', dhs*freq_down)

        # Concatenate all chunks and project to d_model
        x = mx.concatenate(chunk_conv_outputs, axis=0)  # (total_tokens, dhs*freq_down)
        x = self.conv_out(x)  # (total_tokens, d_model)

        # --- Per-chunk sinusoidal position embeddings ---
        # Each chunk gets PE starting from position 0 (matching official)
        max_chunk_tokens = max(chunk_token_lens)
        pe = self.embed_positions(max_chunk_tokens)  # (max_chunk_tokens, d_model)

        pe_parts: list[mx.array] = []
        for ct in chunk_token_lens:
            pe_parts.append(pe[:ct])
        pe_full = mx.concatenate(pe_parts, axis=0)  # (total_tokens, d_model)
        x = x + pe_full

        # --- Windowed attention ---
        total_tokens = x.shape[0]
        tokens_per_full_chunk = chunk_token_lens[0]
        tokens_per_window = tokens_per_full_chunk * (n_window_infer // chunk_size)

        # Build cu_seqlens for attention windows
        cu_seqlens: list[int] = [0]
        pos = 0
        while pos < total_tokens:
            window_end = min(pos + tokens_per_window, total_tokens)
            cu_seqlens.append(window_end)
            pos = window_end

        num_windows = len(cu_seqlens) - 1
        # Add batch dim for transformer layers: (1, total_tokens, d_model)
        x = x[None, :, :]
        if num_windows >= _WINDOWED_SEGMENT_MIN_WINDOWS:
            x = _apply_windowed_encoder_layers(x, self.layers, cu_seqlens)
        else:
            mask = _create_windowed_mask(total_tokens, cu_seqlens, x.dtype)
            for layer in self.layers:
                x = layer(x, mask=mask)

        x = x[0]  # remove batch dim: (total_tokens, d_model)

        # --- Post-processing ---
        x = self.ln_post(x)
        x = nn.gelu(self.proj1(x))
        x = self.proj2(x)

        return x  # (total_tokens, output_dim)


def _create_windowed_mask(
    seq_len: int,
    cu_seqlens: list[int],
    dtype: mx.Dtype = mx.float32,
) -> Optional[mx.array]:
    """Create a block-diagonal attention mask for windowed encoder attention.

    Tokens within the same window can attend to each other; tokens in
    different windows cannot.  This matches the ``cu_seqlens``-based
    windowed attention in the official Qwen3-ASR encoder.

    Args:
        seq_len: Total sequence length.
        cu_seqlens: Cumulative sequence lengths defining window boundaries.
            E.g. ``[0, 104, 208, 260]`` means three windows: [0..103],
            [104..207], [208..259].
        dtype: Output dtype.

    Returns:
        Additive attention mask of shape ``(1, 1, seq_len, seq_len)`` where
        cross-window positions have ``-1e9``, or ``None`` if there is only
        one window (no masking needed).
    """
    # Single window — no mask needed
    if len(cu_seqlens) <= 2:
        return None

    # Assign each position to a window via boundary counting
    boundaries = mx.array(cu_seqlens[1:-1])  # internal boundaries
    positions = mx.arange(seq_len)

    # window_id[i] = number of internal boundaries <= position i
    window_ids = mx.sum(
        positions[:, None] >= boundaries[None, :], axis=1
    )  # (seq_len,)

    # Tokens attend iff they share a window
    same_window = window_ids[:, None] == window_ids[None, :]  # (L, L)
    mask = mx.where(
        same_window,
        mx.array(0.0, dtype=dtype),
        mx.array(-1e9, dtype=dtype),
    )
    return mask[None, None, :, :]  # (1, 1, L, L)


def _apply_windowed_encoder_layers(
    x: mx.array,
    layers: list[AudioEncoderLayer],
    cu_seqlens: list[int],
) -> mx.array:
    """Apply encoder layers independently per attention window.

    This is mathematically equivalent to a full-sequence forward pass with a
    block-diagonal additive mask derived from ``cu_seqlens``. It avoids
    materializing dense ``(L, L)`` masks and prevents cross-window attention
    compute for long sequences.

    Args:
        x: Encoder hidden states, shape ``(1, L, D)``.
        layers: Audio encoder layers to apply.
        cu_seqlens: Cumulative window boundaries (inclusive start, exclusive end).

    Returns:
        Updated hidden states, shape ``(1, L, D)``.
    """
    # Single window: keep fast-path identical to prior behavior.
    if len(cu_seqlens) <= 2:
        for layer in layers:
            x = layer(x, mask=None)
        return x

    for layer in layers:
        parts: list[mx.array] = []
        for s, e in zip(cu_seqlens[:-1], cu_seqlens[1:]):
            parts.append(layer(x[:, s:e, :], mask=None))
        x = mx.concatenate(parts, axis=1)
    return x


# ---------------------------------------------------------------------------
# Text Decoder Components
# ---------------------------------------------------------------------------


class TextAttention(nn.Module):
    """Causal self-attention with MRoPE and Q/K norms for the text decoder.

    Uses RMSNorm on queries and keys (Qwen3 innovation). Supports grouped
    query attention (GQA) when num_kv_heads < num_heads.

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
        mrope_section = [24, 20, 20]
        if isinstance(config.rope_scaling, dict):
            mrope_section = config.rope_scaling.get("mrope_section", mrope_section)

        self.rotary_emb = InterleavedMRoPE(
            head_dim=config.head_dim,
            base=config.rope_theta,
            mrope_section=list(mrope_section),
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

    def __init__(self, num_layers: int, max_seq_len: Optional[int] = None):
        self.keys: list[Optional[mx.array]] = [None] * num_layers
        self.values: list[Optional[mx.array]] = [None] * num_layers
        self.offset: int = 0
        self.max_seq_len = max_seq_len

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
        if self.max_seq_len is None:
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
            full_k = self.keys[layer_idx]
            full_v = self.values[layer_idx]
        else:
            # Preallocated cache mode: write new tokens into a fixed buffer.
            B, H, L_new, D = key.shape
            start = self.offset
            end = start + L_new
            if end > self.max_seq_len:
                raise ValueError(
                    f"KV cache overflow: end={end}, max_seq_len={self.max_seq_len}"
                )

            if self.keys[layer_idx] is None:
                self.keys[layer_idx] = mx.zeros(
                    (B, H, self.max_seq_len, D), dtype=key.dtype
                )
                self.values[layer_idx] = mx.zeros(
                    (B, H, self.max_seq_len, D), dtype=value.dtype
                )

            # Follow mlx-lm's cache pattern: in-place writes into preallocated
            # buffers avoid constructing a fresh array on each token step.
            self.keys[layer_idx][..., start:end, :] = key
            self.values[layer_idx][..., start:end, :] = value

            full_k = self.keys[layer_idx][:, :, :end, :]
            full_v = self.values[layer_idx][:, :, :end, :]

        # Update offset after the last layer processes its step
        if layer_idx == len(self.keys) - 1:
            self.offset += key.shape[2]

        return full_k, full_v

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
    cache_key = (seq_len, str(dtype))
    cached = _CAUSAL_MASK_CACHE.get(cache_key)
    if cached is not None:
        return cached

    mask = mx.full((seq_len, seq_len), -1e9, dtype=dtype)
    mask = mx.triu(mask, k=1)  # zero on and below diagonal, -1e9 above
    mask = mask[None, None, :, :]  # (1, 1, L, L)
    _CAUSAL_MASK_CACHE[cache_key] = mask
    return mask


_CAUSAL_MASK_CACHE: dict[tuple[int, str], mx.array] = {}


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
        output_size = config.classify_num or config.text_config.vocab_size

        self.audio_tower = AudioEncoder(config.audio_config)
        self.model = TextDecoder(config.text_config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            output_size,
            bias=False,
        )

        # Tie weights if configured
        if (
            config.text_config.tie_word_embeddings
            and output_size == config.text_config.vocab_size
        ):
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
            audio_features, _ = self.audio_tower(input_features, feature_lens)
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
        if (
            embeds.shape[0] != audio_features.shape[0]
            or embeds.shape[0] != audio_mask.shape[0]
            or embeds.shape[1] != audio_mask.shape[1]
        ):
            raise ValueError(
                "Audio injection shape mismatch: "
                f"embeds={embeds.shape}, "
                f"audio_features={audio_features.shape}, "
                f"audio_mask={audio_mask.shape}"
            )
        if embeds.shape[2] != audio_features.shape[2]:
            raise ValueError(
                "Audio injection channel mismatch: "
                f"embed_dim={embeds.shape[2]}, "
                f"audio_feature_dim={audio_features.shape[2]}"
            )

        audio_counts = mx.sum(audio_mask.astype(mx.int32), axis=1)
        max_count = int(mx.max(audio_counts).item())
        max_audio_tokens = int(audio_features.shape[1])
        if max_count > max_audio_tokens:
            raise ValueError(
                "Audio injection out of bounds: "
                f"max_audio_placeholders={max_count}, "
                f"audio_features_tokens={max_audio_tokens}"
            )

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

    def prefill(
        self,
        input_ids: mx.array,
        audio_features: mx.array,
        position_ids: mx.array,
        cache: KVCache,
    ) -> mx.array:
        """Prefill decode cache from prompt + injected audio features.

        Args:
            input_ids: Prompt token IDs including audio placeholders, shape (B, L).
            audio_features: Encoded audio features, shape (B, N_audio, D).
            position_ids: MRoPE position IDs for the prompt, shape (B, 3, L).
            cache: KV cache to populate in-place.

        Returns:
            Logits for the final prompt position, shape (B, 1, vocab_size).
        """
        embeds = self.model.embed_tokens(input_ids)
        audio_mask = input_ids == self.audio_token_id
        embeds = self._inject_audio_features(embeds, audio_features, audio_mask)

        hidden = self.model(
            inputs_embeds=embeds,
            position_ids=position_ids,
            attention_mask=None,
            cache=cache,
        )
        return self.lm_head(hidden[:, -1:, :])

    def step(
        self,
        input_ids: mx.array,
        position_ids: mx.array,
        cache: KVCache,
    ) -> mx.array:
        """Decode one autoregressive step with an existing cache.

        Args:
            input_ids: Next-token IDs, shape (B, 1).
            position_ids: Step MRoPE position IDs, shape (B, 3, 1).
            cache: KV cache to update in-place.

        Returns:
            Step logits, shape (B, 1, vocab_size).
        """
        embeds = self.model.embed_tokens(input_ids)
        hidden = self.model(
            inputs_embeds=embeds,
            position_ids=position_ids,
            attention_mask=None,
            cache=cache,
        )
        return self.lm_head(hidden)

    def create_cache(self, max_seq_len: Optional[int] = None) -> KVCache:
        """Create a fresh KV cache for autoregressive generation.

        Args:
            max_seq_len: Optional preallocation target for cache growth.

        Returns:
            Empty KVCache with the correct number of layers.
        """
        return KVCache(
            self.config.text_config.num_hidden_layers,
            max_seq_len=max_seq_len,
        )

    @property
    def num_audio_encoder_layers(self) -> int:
        """Return the number of audio encoder transformer layers."""
        return len(self.audio_tower.layers)

    @property
    def num_text_decoder_layers(self) -> int:
        """Return the number of text decoder transformer layers."""
        return len(self.model.layers)
