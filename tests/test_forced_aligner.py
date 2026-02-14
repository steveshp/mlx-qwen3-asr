"""Tests for forced aligner text/timestamp processing utilities."""

from __future__ import annotations

import numpy as np
import pytest

import mlx_qwen3_asr.forced_aligner as famod
from mlx_qwen3_asr.forced_aligner import ForcedAlignTextProcessor


def test_clean_token_keeps_letters_digits_apostrophe():
    assert ForcedAlignTextProcessor.clean_token("can't!!! 123") == "can't123"


def test_tokenize_space_lang_splits_mixed_cjk():
    text = "hello 世界! test"
    assert ForcedAlignTextProcessor.tokenize_space_lang(text) == [
        "hello",
        "世",
        "界",
        "test",
    ]


def test_encode_timestamp_prompt_wraps_audio_and_timestamps():
    words, prompt = ForcedAlignTextProcessor.encode_timestamp_prompt("a b", "english")
    assert words == ["a", "b"]
    assert prompt.startswith("<|audio_start|><|audio_pad|><|audio_end|>")
    assert prompt.endswith("<timestamp><timestamp>")


def test_fix_timestamp_repairs_non_monotonic_sequence():
    raw = np.array([0, 80, 160, 120, 240, 320], dtype=np.int32)
    fixed = ForcedAlignTextProcessor.fix_timestamp(raw)
    assert len(fixed) == len(raw)
    assert all(fixed[i] <= fixed[i + 1] for i in range(len(fixed) - 1))


def test_parse_timestamp_ms_pairs_words():
    words = ["hello", "world"]
    timestamp_ms = np.array([0, 120, 130, 300], dtype=np.int32)
    aligned = ForcedAlignTextProcessor.parse_timestamp_ms(words, timestamp_ms)
    assert len(aligned) == 2
    assert aligned[0].text == "hello"
    assert aligned[0].start_time == 0.0
    assert aligned[0].end_time == 0.12
    assert aligned[1].text == "world"
    assert aligned[1].start_time == 0.13
    assert aligned[1].end_time == 0.3


def test_forced_aligner_rejects_invalid_backend():
    aligner = famod.ForcedAligner(backend="invalid")
    with pytest.raises(RuntimeError, match="Unsupported aligner backend"):
        aligner._ensure_loaded()


def test_forced_aligner_mlx_backend_reports_load_failure(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("mlx init failed")

    monkeypatch.setattr(famod, "_MLXForcedAlignerBackend", _boom)
    aligner = famod.ForcedAligner(backend=famod.ALIGNER_BACKEND_MLX)
    with pytest.raises(RuntimeError, match="Failed to load MLX aligner backend"):
        aligner._ensure_loaded()
