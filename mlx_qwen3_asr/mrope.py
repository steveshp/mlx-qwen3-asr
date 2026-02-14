"""Interleaved Multi-dimensional Rotary Position Embedding (MRoPE) for Qwen3-ASR.

Critical correctness module. The standard nn.RoPE does NOT work for Qwen3-ASR.
The model uses interleaved frequency assignment across 3 spatial dimensions
(temporal, height, width) with sections [24, 20, 20].

Frequency assignment uses STRIDE-3 INTERLEAVING (NOT chunking):
    - freq index 0 -> section 0 (temporal)
    - freq index 1 -> section 1 (height)
    - freq index 2 -> section 2 (width)
    - freq index 3 -> section 0 (temporal)
    - freq index 4 -> section 1 (height)
    - ... and so on

Total frequencies: (24 + 20 + 20) = 64 = head_dim // 2
Each frequency maps to 2 dimensions in the rotation, so 64 * 2 = 128 = head_dim.

Note: With sections [24, 20, 20] and stride-3 interleaving across 64 indices,
not all 64 positions are covered. Section 0 gets ceil(64/3)=22 indices (not 24),
leaving 2 frequency slots uncovered. These stay at zero (cos=0, sin=0), matching
the official PyTorch implementation which initializes with torch.zeros.

Reference: apply_interleaved_mrope() in the official Qwen3-ASR repo.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

MROPE_SECTION = [24, 20, 20]  # temporal, height, width


class InterleavedMRoPE:
    """Multi-dimensional RoPE with interleaved frequency assignment.

    Sections [24, 20, 20] assign frequencies to temporal/height/width dims
    using stride-3 interleaving (NOT chunking).

    Args:
        head_dim: Dimension per attention head (128 for Qwen3-ASR).
        base: RoPE base frequency (5000000.0 for Qwen3-ASR 1.7B).
        mrope_section: Frequency count per spatial dimension [24, 20, 20].
    """

    def __init__(
        self,
        head_dim: int,
        base: float = 5000000.0,
        mrope_section: list[int] | None = None,
    ):
        self.head_dim = head_dim
        self.base = base
        self.mrope_section = mrope_section or list(MROPE_SECTION)
        self.half_dim = head_dim // 2

        assert sum(self.mrope_section) == self.half_dim, (
            f"Sum of mrope_section {self.mrope_section} = {sum(self.mrope_section)} "
            f"must equal head_dim // 2 = {self.half_dim}"
        )

        # Precompute inverse frequencies: 1 / (base ^ (2i / head_dim))
        inv_freq = 1.0 / (
            base ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim)
        )
        self._inv_freq = inv_freq  # shape: (half_dim,)

        # Precompute interleaved frequency indices per section.
        # With stride-3 interleaving, section i gets indices [i, i+3, i+6, ...]
        # truncated to at most sec_size entries. The actual count may be less
        # than sec_size if there aren't enough stride-3 indices in [0, half_dim).
        total = sum(self.mrope_section)
        self._freq_indices: list[np.ndarray] = []
        self._actual_sizes: list[int] = []
        for i, sec_size in enumerate(self.mrope_section):
            indices = np.array(
                list(range(i, total, 3))[:sec_size], dtype=np.int32
            )
            self._freq_indices.append(indices)
            self._actual_sizes.append(len(indices))

        # Build the scatter mapping: for each section, which positions in the
        # concatenated [sec0, sec1, sec2] tensor map to which positions in the
        # full half_dim output. We precompute two arrays:
        #   _src_indices: positions in the concatenated tensor (length = n_covered)
        #   _dst_indices: positions in the half_dim output (length = n_covered)
        # Uncovered positions stay at zero.
        src_indices = []
        dst_indices = []
        offset = 0
        for i, indices in enumerate(self._freq_indices):
            for local_j, global_j in enumerate(indices):
                src_indices.append(offset + local_j)
                dst_indices.append(int(global_j))
            offset += len(indices)

        self._src_indices = np.array(src_indices, dtype=np.int32)
        self._dst_indices = np.array(dst_indices, dtype=np.int32)
        self._n_covered = len(src_indices)
        self._total_concat = offset  # sum of actual sizes

    def __call__(
        self,
        position_ids: mx.array,
        dtype: mx.Dtype = mx.float32,
    ) -> tuple[mx.array, mx.array]:
        """Compute interleaved MRoPE cos/sin embeddings.

        Args:
            position_ids: Shape (batch, 3, seq_len) -- one row per spatial
                dimension (temporal, height, width).
            dtype: Output dtype.

        Returns:
            Tuple of (cos, sin), each of shape (batch, seq_len, head_dim).
        """
        batch, n_dims, seq_len = position_ids.shape
        assert n_dims == 3, f"Expected 3 spatial dims, got {n_dims}"

        # Compute cos/sin for each section independently, then concatenate.
        all_cos: list[mx.array] = []
        all_sin: list[mx.array] = []

        for i, indices in enumerate(self._freq_indices):
            # Position values for this spatial dimension: (batch, seq_len)
            pos = position_ids[:, i, :]

            # Select inverse frequencies for this section
            freq_idx = mx.array(indices)
            inv_freq_sec = self._inv_freq[freq_idx]  # (actual_size,)

            # Compute angles: (batch, seq_len, actual_size)
            angles = pos[:, :, None].astype(mx.float32) * inv_freq_sec[None, None, :]

            all_cos.append(mx.cos(angles))
            all_sin.append(mx.sin(angles))

        # Concatenate: (batch, seq_len, total_concat)
        cos_cat = mx.concatenate(all_cos, axis=-1)
        sin_cat = mx.concatenate(all_sin, axis=-1)

        # Scatter into full half_dim output.
        # Build the output by selecting from cos_cat at src_indices and placing
        # at dst_indices. Since MLX doesn't have scatter, we construct the full
        # array by building a permutation array of size half_dim where covered
        # positions index into cos_cat and uncovered positions index into a
        # dummy zero column we append.

        # Append a zero column to cos_cat/sin_cat for uncovered positions
        zero_col = mx.zeros((batch, seq_len, 1), dtype=mx.float32)
        cos_cat_padded = mx.concatenate([cos_cat, zero_col], axis=-1)
        sin_cat_padded = mx.concatenate([sin_cat, zero_col], axis=-1)

        # Build gather indices: half_dim positions mapping into cos_cat_padded
        # Default to zero column (sentinel index) for uncovered positions
        gather = np.full(self.half_dim, self._total_concat, dtype=np.int32)
        for src, dst in zip(self._src_indices, self._dst_indices):
            gather[dst] = src

        gather_mx = mx.array(gather)
        cos_out = cos_cat_padded[:, :, gather_mx]  # (batch, seq_len, half_dim)
        sin_out = sin_cat_padded[:, :, gather_mx]

        # Duplicate for full head_dim: [cos, cos] for the rotation formula
        cos_full = mx.concatenate([cos_out, cos_out], axis=-1).astype(dtype)
        sin_full = mx.concatenate([sin_out, sin_out], axis=-1).astype(dtype)

        return cos_full, sin_full


def apply_rotary_pos_emb(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> tuple[mx.array, mx.array]:
    """Apply rotary position embeddings to query and key tensors.

    Uses the standard rotation formula:
        x_rot = x * cos + rotate_half(x) * sin

    where rotate_half splits the last dimension in half and applies:
        [-x2, x1] concatenation.

    Args:
        q: Query tensor, shape (batch, n_heads, seq_len, head_dim).
        k: Key tensor, shape (batch, n_heads, seq_len, head_dim).
        cos: Cosine embeddings from InterleavedMRoPE,
            shape (batch, seq_len, head_dim).
        sin: Sine embeddings from InterleavedMRoPE,
            shape (batch, seq_len, head_dim).

    Returns:
        Tuple of (q_embed, k_embed) with rotary embeddings applied,
        same shapes as inputs.
    """
    # Expand dims for multi-head broadcasting:
    # (batch, seq_len, head_dim) -> (batch, 1, seq_len, head_dim)
    if cos.ndim == 3:
        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]

    q_embed = q * cos + _rotate_half(q) * sin
    k_embed = k * cos + _rotate_half(k) * sin
    return q_embed, k_embed


def _rotate_half(x: mx.array) -> mx.array:
    """Rotate half of the hidden dims of the input.

    Splits the last dimension in half and returns [-x2, x1].

    Args:
        x: Input tensor, shape (..., head_dim).

    Returns:
        Rotated tensor, same shape as input.
    """
    mid = x.shape[-1] // 2
    x1 = x[..., :mid]
    x2 = x[..., mid:]
    return mx.concatenate([-x2, x1], axis=-1)
