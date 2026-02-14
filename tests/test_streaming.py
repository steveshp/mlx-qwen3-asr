"""Tests for mlx_qwen3_asr/streaming.py."""

import pytest
import numpy as np

from mlx_qwen3_asr.streaming import (
    init_streaming,
    _split_stable_unstable,
    StreamingState,
    UNFIXED_TOKEN_NUM,
)


# ---------------------------------------------------------------------------
# init_streaming
# ---------------------------------------------------------------------------


class TestInitStreaming:
    """Test init_streaming() returns correct initial state."""

    def test_default_state(self):
        state = init_streaming()
        assert isinstance(state, StreamingState)
        assert state.chunk_size_samples == 32000  # 2.0 * 16000
        assert state._model_path == "Qwen/Qwen3-ASR-1.7B"
        assert state.text == ""
        assert state.language == "unknown"
        assert state.chunk_id == 0
        assert len(state.buffer) == 0
        assert len(state.audio_accum) == 0
        assert state.previous_tokens == []
        assert state.stable_text == ""

    def test_custom_chunk_size(self):
        state = init_streaming(chunk_size_sec=5.0)
        assert state.chunk_size_samples == 80000  # 5.0 * 16000

    def test_custom_sample_rate(self):
        state = init_streaming(chunk_size_sec=2.0, sample_rate=8000)
        assert state.chunk_size_samples == 16000  # 2.0 * 8000

    def test_custom_model(self):
        state = init_streaming(model="my/custom-model")
        assert state._model_path == "my/custom-model"


# ---------------------------------------------------------------------------
# StreamingState defaults
# ---------------------------------------------------------------------------


class TestStreamingStateDefaults:
    """Test StreamingState default values."""

    def test_default_buffer(self):
        state = StreamingState()
        assert isinstance(state.buffer, np.ndarray)
        assert len(state.buffer) == 0
        assert state.buffer.dtype == np.float32

    def test_default_audio_accum(self):
        state = StreamingState()
        assert isinstance(state.audio_accum, np.ndarray)
        assert len(state.audio_accum) == 0

    def test_default_text(self):
        state = StreamingState()
        assert state.text == ""

    def test_default_language(self):
        state = StreamingState()
        assert state.language == "unknown"

    def test_default_chunk_id(self):
        state = StreamingState()
        assert state.chunk_id == 0

    def test_default_chunk_size_samples(self):
        state = StreamingState()
        assert state.chunk_size_samples == 32000

    def test_default_previous_tokens(self):
        state = StreamingState()
        assert state.previous_tokens == []

    def test_default_stable_text(self):
        state = StreamingState()
        assert state.stable_text == ""


# ---------------------------------------------------------------------------
# _split_stable_unstable
# ---------------------------------------------------------------------------


class TestSplitStableUnstable:
    """Test _split_stable_unstable() with various inputs."""

    def test_short_text_all_unstable(self):
        """Text with fewer words than unfixed_tokens is all unstable."""
        stable, unstable = _split_stable_unstable("", "hello world")
        # "hello world" = 2 words, unfixed_tokens=5 by default
        assert stable == ""
        assert unstable == "hello world"

    def test_long_text_splits_correctly(self):
        """Text with more words than unfixed_tokens should split."""
        text = "one two three four five six seven eight"
        stable, unstable = _split_stable_unstable("", text)
        words = text.split()
        # 8 words, unfixed_tokens=5 -> stable = first 3, unstable = last 5
        assert stable == "one two three"
        assert unstable == "four five six seven eight"

    def test_preserves_previous_stable_text(self):
        """New stable text should be at least as long as previous stable."""
        prev_stable = "this is already stable text that is long"
        text = "hello world"  # Too short, all unstable
        stable, unstable = _split_stable_unstable(prev_stable, text)
        assert stable == prev_stable

    def test_empty_text(self):
        stable, unstable = _split_stable_unstable("", "")
        assert stable == ""
        assert unstable == ""

    def test_exactly_unfixed_tokens(self):
        """Text with exactly unfixed_tokens words should be all unstable."""
        words = ["word"] * UNFIXED_TOKEN_NUM
        text = " ".join(words)
        stable, unstable = _split_stable_unstable("", text)
        assert stable == ""
        assert unstable == text

    def test_one_more_than_unfixed(self):
        """Text with unfixed_tokens + 1 words: 1 stable, rest unstable."""
        words = [f"word{i}" for i in range(UNFIXED_TOKEN_NUM + 1)]
        text = " ".join(words)
        stable, unstable = _split_stable_unstable("", text)
        assert stable == words[0]
        assert unstable == " ".join(words[1:])

    def test_custom_unfixed_tokens(self):
        text = "a b c d e f g h"
        stable, unstable = _split_stable_unstable("", text, unfixed_tokens=3)
        assert stable == "a b c d e"
        assert unstable == "f g h"

    def test_stable_grows_monotonically(self):
        """Stable text should only grow, never shrink."""
        prev_stable = "one two three"
        # New transcription has more words
        text = "one two three four five six seven eight nine ten"
        stable, unstable = _split_stable_unstable(prev_stable, text)
        assert len(stable) >= len(prev_stable)
