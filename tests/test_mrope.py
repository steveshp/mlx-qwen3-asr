"""Tests for mlx_qwen3_asr/mrope.py."""

import pytest
import numpy as np
import mlx.core as mx

from mlx_qwen3_asr.mrope import (
    InterleavedMRoPE,
    apply_rotary_pos_emb,
    _rotate_half,
    MROPE_SECTION,
)


# ---------------------------------------------------------------------------
# InterleavedMRoPE constructor
# ---------------------------------------------------------------------------


class TestInterleavedMRoPEInit:
    """Test InterleavedMRoPE constructor validation."""

    def test_default_sections(self):
        mrope = InterleavedMRoPE(head_dim=128)
        assert mrope.mrope_section == [24, 20, 20]
        assert mrope.half_dim == 64

    def test_sections_must_sum_to_half_dim(self):
        """If sections don't sum to head_dim//2, assertion should fail."""
        with pytest.raises(AssertionError, match="must equal head_dim // 2"):
            InterleavedMRoPE(head_dim=128, mrope_section=[10, 10, 10])

    def test_custom_valid_sections(self):
        mrope = InterleavedMRoPE(head_dim=64, mrope_section=[12, 10, 10])
        assert mrope.half_dim == 32
        assert sum(mrope.mrope_section) == 32


# ---------------------------------------------------------------------------
# InterleavedMRoPE output shapes
# ---------------------------------------------------------------------------


class TestInterleavedMRoPEShapes:
    """Test output shapes of InterleavedMRoPE.__call__."""

    def test_output_shape(self):
        mrope = InterleavedMRoPE(head_dim=128)
        batch, seq_len = 2, 10
        position_ids = mx.zeros((batch, 3, seq_len), dtype=mx.int32)
        cos, sin = mrope(position_ids)
        assert cos.shape == (batch, seq_len, 128)
        assert sin.shape == (batch, seq_len, 128)

    def test_output_shape_single_batch(self):
        mrope = InterleavedMRoPE(head_dim=128)
        position_ids = mx.zeros((1, 3, 5), dtype=mx.int32)
        cos, sin = mrope(position_ids)
        assert cos.shape == (1, 5, 128)
        assert sin.shape == (1, 5, 128)

    def test_output_dtype(self):
        mrope = InterleavedMRoPE(head_dim=128)
        position_ids = mx.zeros((1, 3, 5), dtype=mx.int32)
        cos, sin = mrope(position_ids, dtype=mx.float32)
        assert cos.dtype == mx.float32
        assert sin.dtype == mx.float32


# ---------------------------------------------------------------------------
# Uncovered positions
# ---------------------------------------------------------------------------


class TestMRoPEUncoveredPositions:
    """Test that uncovered frequency indices produce zero cos/sin."""

    def test_uncovered_positions_are_zero(self):
        """With sections [24, 20, 20], stride-3 interleaving across 64 indices.
        Section 0 gets indices [0, 3, 6, ..., 63] -> that's ceil(64/3)=22 entries, not 24.
        Section 1 gets indices [1, 4, 7, ..., 61] -> 21 entries, not 20 (truncated to 20).
        Section 2 gets indices [2, 5, 8, ..., 62] -> 21 entries, not 20 (truncated to 20).

        The indices covered in half_dim (0..63) are:
          Section 0: 0,3,6,...,63 -> 22 indices (stops at 63)
          Section 1: 1,4,7,...,58 -> 20 indices (truncated from 21)
          Section 2: 2,5,8,...,59 -> 20 indices (truncated from 21)

        Total covered: 22+20+20 = 62 out of 64.
        Uncovered: indices 61, 62 in the half_dim space.

        Wait, let's compute precisely. Section 0 gets range(0, 64, 3)[:24]:
          range(0,64,3) = [0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,63]
          That's 22 elements. [:24] keeps all 22.
        Section 1 gets range(1, 64, 3)[:20]:
          range(1,64,3) = [1,4,7,10,13,16,19,22,25,28,31,34,37,40,43,46,49,52,55,58,61]
          That's 21 elements. [:20] keeps first 20: [1,4,...,58].
          So 61 is NOT covered.
        Section 2 gets range(2, 64, 3)[:20]:
          range(2,64,3) = [2,5,8,11,14,17,20,23,26,29,32,35,38,41,44,47,50,53,56,59,62]
          That's 21 elements. [:20] keeps first 20: [2,5,...,59].
          So 62 is NOT covered.

        Uncovered half_dim indices: 61, 62.
        In the full head_dim output, these map to positions 61 and 62 (first half)
        and 61+64=125 and 62+64=126 (second half, since cos/sin are duplicated).
        """
        mrope = InterleavedMRoPE(head_dim=128)
        # Use non-zero position ids to ensure covered positions are non-zero
        pos_ids = mx.ones((1, 3, 1), dtype=mx.int32) * 5
        cos, sin = mrope(pos_ids)
        mx.eval(cos, sin)

        cos_np = np.array(cos[0, 0, :])  # (128,)
        sin_np = np.array(sin[0, 0, :])

        # Positions 61 and 62 in the first half should be zero
        assert cos_np[61] == 0.0, f"cos[61] = {cos_np[61]}, expected 0.0"
        assert cos_np[62] == 0.0, f"cos[62] = {cos_np[62]}, expected 0.0"
        assert sin_np[61] == 0.0, f"sin[61] = {sin_np[61]}, expected 0.0"
        assert sin_np[62] == 0.0, f"sin[62] = {sin_np[62]}, expected 0.0"

        # The duplicated second half should also be zero at those positions
        assert cos_np[61 + 64] == 0.0
        assert cos_np[62 + 64] == 0.0
        assert sin_np[61 + 64] == 0.0
        assert sin_np[62 + 64] == 0.0

        # But a covered position (e.g., index 0) should be non-zero
        # cos(pos * inv_freq) for non-zero pos should not be exactly zero
        # (it's possible for specific values but extremely unlikely for pos=5)
        assert cos_np[0] != 0.0, "Covered position cos[0] should be non-zero"


# ---------------------------------------------------------------------------
# apply_rotary_pos_emb
# ---------------------------------------------------------------------------


class TestApplyRotaryPosEmb:
    """Test apply_rotary_pos_emb preserves tensor shapes."""

    def test_output_shapes(self):
        batch, n_heads, seq_len, head_dim = 2, 4, 10, 128
        q = mx.random.normal((batch, n_heads, seq_len, head_dim))
        k = mx.random.normal((batch, n_heads, seq_len, head_dim))
        cos = mx.random.normal((batch, seq_len, head_dim))
        sin = mx.random.normal((batch, seq_len, head_dim))

        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        assert q_rot.shape == (batch, n_heads, seq_len, head_dim)
        assert k_rot.shape == (batch, n_heads, seq_len, head_dim)

    def test_4d_cos_sin_no_expansion(self):
        """When cos/sin are already 4D, no unsqueeze should happen."""
        batch, n_heads, seq_len, head_dim = 1, 2, 5, 8
        q = mx.random.normal((batch, n_heads, seq_len, head_dim))
        k = mx.random.normal((batch, n_heads, seq_len, head_dim))
        cos = mx.random.normal((batch, 1, seq_len, head_dim))
        sin = mx.random.normal((batch, 1, seq_len, head_dim))

        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape


# ---------------------------------------------------------------------------
# _rotate_half
# ---------------------------------------------------------------------------


class TestRotateHalf:
    """Test _rotate_half with known input."""

    def test_known_values(self):
        x = mx.array([[1.0, 2.0, 3.0, 4.0]])
        result = _rotate_half(x)
        # Mid = 2: x1 = [1, 2], x2 = [3, 4]
        # Output = [-x2, x1] = [-3, -4, 1, 2]
        expected = [[-3.0, -4.0, 1.0, 2.0]]
        np.testing.assert_array_equal(np.array(result), expected)

    def test_preserves_shape(self):
        x = mx.random.normal((2, 4, 8))
        result = _rotate_half(x)
        assert result.shape == x.shape
