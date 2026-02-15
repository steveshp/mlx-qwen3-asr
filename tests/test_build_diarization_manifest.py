"""Unit tests for scripts/build_diarization_manifest.py helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "build_diarization_manifest.py"
    )
    module_name = "build_diarization_manifest_script"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_safe_component():
    mod = _load_module()
    assert mod._safe_component("a/b:c d") == "a_b_c_d"  # noqa: SLF001


def test_build_mix_plan_is_deterministic_and_spread():
    mod = _load_module()
    row_type = mod.ManifestRow  # noqa: SLF001
    rows = []
    for speaker in ("S1", "S2", "S3", "S4"):
        for idx in range(3):
            rows.append(
                row_type(
                    sample_id=f"{speaker}-{idx}",
                    speaker_id=speaker,
                    language="English",
                    audio_path=Path("/tmp/fake.wav"),
                    reference_text="x",
                )
            )
    p1 = mod._build_mix_plan(  # noqa: SLF001
        rows,
        num_mixes=5,
        segments_per_mix=3,
        seed=7,
    )
    p2 = mod._build_mix_plan(  # noqa: SLF001
        rows,
        num_mixes=5,
        segments_per_mix=3,
        seed=7,
    )
    assert [[x.sample_id for x in mix] for mix in p1] == [
        [x.sample_id for x in mix] for mix in p2
    ]
    assert len(p1) == 5
    for mix in p1:
        assert len(mix) == 3
        assert len({x.speaker_id for x in mix}) >= 2


def test_parse_manifest_requires_speaker_id(tmp_path: Path):
    mod = _load_module()
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"stub")
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        '{"sample_id":"s1","audio_path":"'
        + str(audio)
        + '","speaker_id":""}\n',
        encoding="utf-8",
    )
    try:
        mod._parse_manifest(manifest)  # noqa: SLF001
    except ValueError as exc:
        assert "speaker_id" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for missing speaker_id")
