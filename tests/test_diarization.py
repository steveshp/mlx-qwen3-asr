"""Tests for diarization helpers."""

from __future__ import annotations

import numpy as np
import pytest

from mlx_qwen3_asr.diarization import (
    DEFAULT_SPEAKER_LABEL,
    build_speaker_segments_from_turns,
    diarize_chunk_items,
    diarize_word_segments,
    infer_speaker_turns,
    validate_diarization_config,
)


def test_validate_diarization_config_rejects_invalid_bounds():
    with pytest.raises(ValueError, match="diarization_max_speakers"):
        validate_diarization_config(
            num_speakers=None,
            min_speakers=3,
            max_speakers=2,
            window_sec=1.5,
            hop_sec=0.75,
        )


def test_diarize_word_segments_adds_speaker_labels():
    cfg = validate_diarization_config(
        num_speakers=None,
        min_speakers=1,
        max_speakers=8,
        window_sec=1.5,
        hop_sec=0.75,
    )
    words = [
        {"text": "hello", "start": 0.1, "end": 0.3},
        {"text": "world", "start": 0.35, "end": 0.6},
    ]
    labeled, speakers = diarize_word_segments(words, config=cfg)
    assert labeled[0]["speaker"] == DEFAULT_SPEAKER_LABEL
    assert speakers[0]["speaker"] == DEFAULT_SPEAKER_LABEL
    assert speakers[0]["text"] == "hello world"


def test_diarize_chunk_items_returns_fallback_speaker_segments():
    cfg = validate_diarization_config(
        num_speakers=None,
        min_speakers=1,
        max_speakers=8,
        window_sec=1.5,
        hop_sec=0.75,
    )
    chunks = [
        {"text": "hello", "start": 0.0, "end": 0.8},
        {"text": "world", "start": 1.0, "end": 1.5},
    ]
    speaker_segments = diarize_chunk_items(chunks, config=cfg)
    assert len(speaker_segments) == 1
    assert speaker_segments[0]["speaker"] == DEFAULT_SPEAKER_LABEL
    assert speaker_segments[0]["text"] == "hello world"


def test_infer_speaker_turns_splits_simple_two_tone_audio():
    sr = 16000
    dur = 1.0
    t = np.linspace(0.0, dur, int(sr * dur), endpoint=False, dtype=np.float32)
    low = 0.4 * np.sin(2.0 * np.pi * 180.0 * t)
    high = 0.4 * np.sin(2.0 * np.pi * 820.0 * t)
    audio = np.concatenate([low, high]).astype(np.float32)

    cfg = validate_diarization_config(
        num_speakers=2,
        min_speakers=1,
        max_speakers=4,
        window_sec=0.5,
        hop_sec=0.25,
    )
    turns = infer_speaker_turns(audio, sr=sr, config=cfg)
    speakers = {t["speaker"] for t in turns}
    assert len(speakers) == 2


def test_infer_speaker_turns_auto_mode_splits_simple_two_tone_audio():
    sr = 16000
    dur = 1.0
    t = np.linspace(0.0, dur, int(sr * dur), endpoint=False, dtype=np.float32)
    low = 0.4 * np.sin(2.0 * np.pi * 180.0 * t)
    high = 0.4 * np.sin(2.0 * np.pi * 820.0 * t)
    audio = np.concatenate([low, high]).astype(np.float32)

    cfg = validate_diarization_config(
        num_speakers=None,
        min_speakers=1,
        max_speakers=4,
        window_sec=0.5,
        hop_sec=0.25,
    )
    turns = infer_speaker_turns(audio, sr=sr, config=cfg)
    speakers = {t["speaker"] for t in turns}
    assert len(speakers) == 2


def test_diarize_word_segments_uses_turn_overlap():
    cfg = validate_diarization_config(
        num_speakers=2,
        min_speakers=1,
        max_speakers=4,
        window_sec=0.5,
        hop_sec=0.25,
    )
    turns = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0},
        {"speaker": "SPEAKER_01", "start": 1.0, "end": 2.0},
    ]
    words = [
        {"text": "hello", "start": 0.1, "end": 0.4},
        {"text": "world", "start": 1.2, "end": 1.6},
    ]
    labeled, speaker_segments = diarize_word_segments(
        words,
        config=cfg,
        speaker_turns=turns,
    )
    assert labeled[0]["speaker"] == "SPEAKER_00"
    assert labeled[1]["speaker"] == "SPEAKER_01"
    assert len(speaker_segments) == 2


def test_build_speaker_segments_from_turns_keeps_empty_turn_text():
    turns = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0},
        {"speaker": "SPEAKER_01", "start": 1.0, "end": 2.0},
    ]
    words = [{"text": "hello", "start": 0.1, "end": 0.4, "speaker": "SPEAKER_00"}]

    speaker_segments = build_speaker_segments_from_turns(
        speaker_turns=turns,
        word_segments=words,
    )

    assert len(speaker_segments) == 2
    assert speaker_segments[0]["speaker"] == "SPEAKER_00"
    assert speaker_segments[0]["text"] == "hello"
    assert speaker_segments[1]["speaker"] == "SPEAKER_01"
    assert speaker_segments[1]["text"] == ""
