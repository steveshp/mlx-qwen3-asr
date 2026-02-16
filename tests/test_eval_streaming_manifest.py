"""Unit tests for scripts/eval_streaming_manifest.py helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "eval_streaming_manifest.py"
    module_name = "eval_streaming_manifest_script"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_split_modes_rejects_invalid_mode():
    mod = _load_script_module()
    with pytest.raises(ValueError, match="Unsupported endpointing mode"):
        mod._split_modes("fixed,vad")  # noqa: SLF001


def test_parse_manifest_requires_audio_path(tmp_path: Path):
    mod = _load_script_module()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"sample_id": "s1"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing audio_path"):
        mod._parse_manifest(manifest)  # noqa: SLF001


def test_main_emits_multifile_payload(monkeypatch, tmp_path: Path):
    mod = _load_script_module()

    a1 = tmp_path / "a1.wav"
    a2 = tmp_path / "a2.wav"
    a1.write_bytes(b"RIFF")
    a2.write_bytes(b"RIFF")

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"sample_id": "s1", "audio_path": str(a1), "subset": "set-a"}),
                json.dumps({"sample_id": "s2", "audio_path": str(a2), "subset": "set-b"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def _fake_load_model(_name, dtype=None):  # noqa: ANN001, ARG001
        return object(), {"dtype": str(dtype)}

    def _fake_load_audio(_path):  # noqa: ANN001
        return np.zeros(32000, dtype=np.float32)

    def _fake_init_streaming(**kwargs):  # noqa: ANN003
        return SimpleNamespace(
            chunk_id=0,
            text="",
            language="unknown",
            endpointing_mode=kwargs["endpointing_mode"],
        )

    def _fake_feed_audio(_chunk, state, model=None):  # noqa: ANN001, ANN002, ARG001
        state.chunk_id += 1

    def _fake_finish_streaming(state, model=None):  # noqa: ANN001, ARG001
        state.text = f"done-{state.endpointing_mode}"
        state.language = "English"

    def _fake_streaming_metrics(state):  # noqa: ANN001
        if state.endpointing_mode == "energy":
            return {
                "partial_stability": 0.94,
                "rewrite_rate": 0.04,
                "finalization_delta_chars": 2,
            }
        return {
            "partial_stability": 0.90,
            "rewrite_rate": 0.06,
            "finalization_delta_chars": 3,
        }

    monkeypatch.setattr(mod, "load_model", _fake_load_model)
    monkeypatch.setattr(mod, "load_audio", _fake_load_audio)
    monkeypatch.setattr(mod, "init_streaming", _fake_init_streaming)
    monkeypatch.setattr(mod, "feed_audio", _fake_feed_audio)
    monkeypatch.setattr(mod, "finish_streaming", _fake_finish_streaming)
    monkeypatch.setattr(mod, "streaming_metrics", _fake_streaming_metrics)

    output = tmp_path / "streaming_manifest.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_streaming_manifest.py",
            "--manifest-jsonl",
            str(manifest),
            "--endpointing-modes",
            "fixed,energy",
            "--json-output",
            str(output),
        ],
    )

    rc = mod.main()
    assert rc == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["suite"] == "streaming-manifest-quality-v1"
    assert payload["samples"] == 2
    assert payload["evaluations"] == 4
    assert set(payload["by_mode"].keys()) == {"fixed", "energy"}
    assert payload["aggregate"]["partial_stability_mean"] > 0.0


def test_main_fails_threshold(monkeypatch, tmp_path: Path):
    mod = _load_script_module()

    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"sample_id": "s1", "audio_path": str(audio)}) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(mod, "load_model", lambda _name, dtype=None: (object(), {}))
    monkeypatch.setattr(mod, "load_audio", lambda _path: np.zeros(16000, dtype=np.float32))
    monkeypatch.setattr(
        mod,
        "init_streaming",
        lambda **kwargs: SimpleNamespace(
            chunk_id=0,
            text="",
            language="unknown",
            endpointing_mode=kwargs["endpointing_mode"],
        ),
    )
    monkeypatch.setattr(
        mod,
        "feed_audio",
        lambda _chunk, state, model=None: setattr(state, "chunk_id", state.chunk_id + 1),
    )
    monkeypatch.setattr(mod, "finish_streaming", lambda state, model=None: None)
    monkeypatch.setattr(
        mod,
        "streaming_metrics",
        lambda _state: {
            "partial_stability": 0.10,
            "rewrite_rate": 0.80,
            "finalization_delta_chars": 100,
        },
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_streaming_manifest.py",
            "--manifest-jsonl",
            str(manifest),
            "--endpointing-modes",
            "fixed",
            "--fail-partial-stability-below",
            "0.85",
            "--fail-rewrite-rate-above",
            "0.30",
            "--fail-finalization-delta-chars-above",
            "32",
        ],
    )

    rc = mod.main()
    assert rc == 2
