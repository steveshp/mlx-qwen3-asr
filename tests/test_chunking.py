"""Tests for mlx_qwen3_asr/chunking.py."""

import pytest
import numpy as np

from mlx_qwen3_asr.chunking import (
    split_audio_into_chunks,
    _find_split_point,
    MAX_CHUNK_SECONDS,
)


# ---------------------------------------------------------------------------
# split_audio_into_chunks
# ---------------------------------------------------------------------------


class TestSplitAudioIntoChunks:
    """Test split_audio_into_chunks() with various audio lengths."""

    def test_short_audio_returns_single_chunk(self):
        """Audio shorter than max_chunk_sec should return a single chunk."""
        sr = 16000
        # 10 seconds of audio, well below default max of 1200s
        audio = np.random.randn(10 * sr).astype(np.float32)
        chunks = split_audio_into_chunks(audio, sr=sr)
        assert len(chunks) == 1
        assert chunks[0][1] == 0.0  # offset
        np.testing.assert_array_equal(chunks[0][0], audio)

    def test_long_audio_returns_multiple_chunks(self):
        """Audio exceeding max_chunk_sec should be split."""
        sr = 16000
        max_sec = 5.0  # Use small max for testing
        # 12 seconds of audio
        audio = np.random.randn(int(12 * sr)).astype(np.float32)
        chunks = split_audio_into_chunks(audio, sr=sr, max_chunk_sec=max_sec)
        assert len(chunks) >= 2

    def test_offsets_are_non_negative(self):
        sr = 16000
        audio = np.random.randn(int(12 * sr)).astype(np.float32)
        chunks = split_audio_into_chunks(audio, sr=sr, max_chunk_sec=5.0)
        for _, offset in chunks:
            assert offset >= 0.0

    def test_offsets_are_monotonically_increasing(self):
        sr = 16000
        audio = np.random.randn(int(20 * sr)).astype(np.float32)
        chunks = split_audio_into_chunks(audio, sr=sr, max_chunk_sec=5.0)
        offsets = [offset for _, offset in chunks]
        for i in range(1, len(offsets)):
            assert offsets[i] > offsets[i - 1]

    def test_first_offset_is_zero(self):
        sr = 16000
        audio = np.random.randn(int(15 * sr)).astype(np.float32)
        chunks = split_audio_into_chunks(audio, sr=sr, max_chunk_sec=5.0)
        assert chunks[0][1] == 0.0

    def test_total_samples_preserved(self):
        """The sum of chunk lengths should equal the original audio length."""
        sr = 16000
        audio = np.random.randn(int(12 * sr)).astype(np.float32)
        chunks = split_audio_into_chunks(audio, sr=sr, max_chunk_sec=5.0)
        total_samples = sum(len(chunk) for chunk, _ in chunks)
        assert total_samples == len(audio)

    def test_chunks_are_non_overlapping(self):
        """Chunks should be contiguous and non-overlapping."""
        sr = 16000
        audio = np.random.randn(int(12 * sr)).astype(np.float32)
        chunks = split_audio_into_chunks(audio, sr=sr, max_chunk_sec=5.0)

        # Verify by checking offset + len = next offset
        for i in range(len(chunks) - 1):
            chunk, offset = chunks[i]
            next_offset = chunks[i + 1][1]
            expected_next = offset + len(chunk) / sr
            assert abs(expected_next - next_offset) < 1e-6

    def test_exact_max_returns_single_chunk(self):
        """Audio exactly at max_chunk_sec should return single chunk."""
        sr = 16000
        max_sec = 5.0
        audio = np.random.randn(int(max_sec * sr)).astype(np.float32)
        chunks = split_audio_into_chunks(audio, sr=sr, max_chunk_sec=max_sec)
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# _find_split_point
# ---------------------------------------------------------------------------


class TestFindSplitPoint:
    """Test _find_split_point() finds low-energy region."""

    def test_returns_valid_index(self):
        sr = 16000
        audio = np.random.randn(10 * sr).astype(np.float32)
        split_idx = _find_split_point(audio, sr)
        assert 0 <= split_idx < len(audio)

    def test_prefers_silence_region(self):
        """Split point should prefer a region with silence/low energy."""
        sr = 16000
        # Create audio with loud regions and a quiet gap near center
        audio = np.ones(10 * sr, dtype=np.float32) * 0.5
        # Insert silence near center (4.5s to 5.5s)
        silence_start = int(4.5 * sr)
        silence_end = int(5.5 * sr)
        audio[silence_start:silence_end] = 0.0

        split_idx = _find_split_point(audio, sr)
        # Split point should be within or near the silent region
        assert silence_start <= split_idx <= silence_end

    def test_near_midpoint(self):
        """Split point should be near the midpoint (within search_range)."""
        sr = 16000
        n_samples = 10 * sr
        audio = np.random.randn(n_samples).astype(np.float32) * 0.01  # quiet uniform
        split_idx = _find_split_point(audio, sr)
        # Default search_range=0.3 means searching 20%-80% of audio
        low = int(n_samples * 0.2)
        high = int(n_samples * 0.8)
        assert low <= split_idx <= high
