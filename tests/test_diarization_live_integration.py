"""Optional live diarization integration test using pyannote backend.

Skipped by default. Enable with RUN_DIARIZATION_LIVE_INTEGRATION=1.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from mlx_qwen3_asr.diarization import infer_speaker_turns, validate_diarization_config


def _enabled() -> bool:
    return os.getenv("RUN_DIARIZATION_LIVE_INTEGRATION", "").strip() == "1"


@pytest.mark.skipif(
    not _enabled(),
    reason="Set RUN_DIARIZATION_LIVE_INTEGRATION=1 to enable.",
)
def test_pyannote_live_infer_smoke():
    pytest.importorskip("pyannote.audio")
    token = (
        os.getenv("PYANNOTE_AUTH_TOKEN")
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or ""
    )
    if not token:
        pytest.skip("Set PYANNOTE_AUTH_TOKEN (or HF_TOKEN) for live diarization test.")

    sr = 16000
    t = np.linspace(0.0, 1.0, sr, endpoint=False, dtype=np.float32)
    a = 0.25 * np.sin(2.0 * np.pi * 220.0 * t)
    b = 0.25 * np.sin(2.0 * np.pi * 660.0 * t)
    audio = np.concatenate([a, b]).astype(np.float32)

    cfg = validate_diarization_config(
        num_speakers=None,
        min_speakers=1,
        max_speakers=3,
    )
    turns = infer_speaker_turns(audio, sr=sr, config=cfg)

    assert isinstance(turns, list)
    if turns:
        assert {"speaker", "start", "end"}.issubset(set(turns[0].keys()))
