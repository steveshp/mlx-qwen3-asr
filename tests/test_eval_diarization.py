"""Unit tests for scripts/eval_diarization.py helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "eval_diarization.py"
    module_name = "eval_diarization_script"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_sanitize_segments_sorts_and_filters():
    mod = _load_module()
    cleaned = mod._sanitize_segments(  # noqa: SLF001
        [
            {"speaker": "B", "start": 3.0, "end": 2.0},
            {"speaker": "A", "start": 0.0, "end": 1.0},
            {"speaker": "", "start": 0.0, "end": 1.0},
            {"speaker": "A", "start": -1.0, "end": 0.5},
        ]
    )
    assert cleaned == [
        {"speaker": "A", "start": 0.0, "end": 0.5},
        {"speaker": "A", "start": 0.0, "end": 1.0},
    ]


def test_best_row_to_col_assignment_handles_unmatched():
    mod = _load_module()
    import numpy as np

    weights = np.array(
        [
            [5.0, 0.0],
            [4.0, 1.0],
        ]
    )
    assignment = mod._best_row_to_col_assignment(weights)  # noqa: SLF001
    assert assignment == [0, 1]

    weights2 = np.array(
        [
            [0.0],
            [2.0],
        ]
    )
    assignment2 = mod._best_row_to_col_assignment(weights2)  # noqa: SLF001
    assert assignment2 in ([-1, 0], [0, -1])


def test_diarization_metrics_perfect_under_label_permutation():
    mod = _load_module()
    ref = [
        {"speaker": "R1", "start": 0.0, "end": 1.0},
        {"speaker": "R2", "start": 1.0, "end": 2.0},
    ]
    hyp = [
        {"speaker": "X", "start": 0.0, "end": 1.0},
        {"speaker": "Y", "start": 1.0, "end": 2.0},
    ]
    metrics = mod.compute_sample_diarization_metrics(  # noqa: SLF001
        reference_segments=ref,
        hypothesis_segments=hyp,
        audio_duration_sec=2.0,
        frame_step_sec=0.01,
        collar_sec=0.0,
        ignore_overlap=False,
    )
    assert metrics["der"] == 0.0
    assert metrics["jer"] == 0.0


def test_diarization_metrics_detects_confusion_and_threshold_failures():
    mod = _load_module()
    ref = [
        {"speaker": "R1", "start": 0.0, "end": 1.0},
        {"speaker": "R2", "start": 1.0, "end": 2.0},
    ]
    hyp = [{"speaker": "X", "start": 0.0, "end": 2.0}]
    metrics = mod.compute_sample_diarization_metrics(  # noqa: SLF001
        reference_segments=ref,
        hypothesis_segments=hyp,
        audio_duration_sec=2.0,
        frame_step_sec=0.01,
        collar_sec=0.0,
        ignore_overlap=False,
    )
    assert metrics["der"] > 0.0
    assert metrics["jer"] > 0.0

    failures = mod._threshold_failures(  # noqa: SLF001
        der=metrics["der"],
        jer=metrics["jer"],
        fail_der_above=0.1,
        fail_jer_above=0.1,
    )
    assert len(failures) >= 1


def test_parse_manifest_requires_reference_speaker_segments(tmp_path: Path):
    mod = _load_module()
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"stub")
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        '{"sample_id":"s1","audio_path":"'
        + str(audio)
        + '","reference_speaker_segments":[]}\n',
        encoding="utf-8",
    )
    try:
        mod._parse_manifest(manifest)  # noqa: SLF001
    except ValueError as exc:
        assert "reference_speaker_segments" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for missing reference_speaker_segments")
