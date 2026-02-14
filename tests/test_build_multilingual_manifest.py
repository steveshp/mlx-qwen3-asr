"""Unit tests for scripts/build_multilingual_manifest.py helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = Path("scripts/build_multilingual_manifest.py")
    spec = importlib.util.spec_from_file_location("build_multilingual_manifest", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_language_label_known_and_fallback():
    mod = _load_module()
    assert mod._language_label_for_config("en_us") == "English"
    assert mod._language_label_for_config("xx_yy") == "xx-yy"


def test_select_indices_deterministic_and_unique():
    mod = _load_module()
    out1 = mod._select_indices(total=10, take=3, seed=7)
    out2 = mod._select_indices(total=10, take=3, seed=7)
    assert out1 == out2
    assert len(out1) == 3
    assert len(set(out1)) == 3


def test_select_indices_take_all_for_small_sets():
    mod = _load_module()
    assert mod._select_indices(total=3, take=10, seed=0) == [0, 1, 2]
    assert mod._select_indices(total=0, take=2, seed=0) == []


def test_sanitize_id_strips_unsafe_chars():
    mod = _load_module()
    assert mod._sanitize_id("abc/123:xy z") == "abc_123_xy_z"
    assert mod._sanitize_id("___") == "sample"


def test_parse_fleurs_tsv_without_header(tmp_path):
    mod = _load_module()
    tsv = tmp_path / "test.tsv"
    tsv.write_text(
        "42\tabc.wav\tHello world\thello world\th e l l o\t16000\tMALE\n",
        encoding="utf-8",
    )
    rows = mod._parse_fleurs_tsv(tsv)
    assert len(rows) == 1
    assert rows[0]["speaker_id"] == "42"
    assert rows[0]["audio_file"] == "abc.wav"
    assert rows[0]["transcription"] == "Hello world"


def test_resolve_language_configs_with_aliases():
    mod = _load_module()
    available = {"en_us", "cmn_hans_cn", "ja_jp"}
    out = mod._resolve_language_configs(["en", "zh_cn", "ja_jp"], available)
    assert out == ["en_us", "cmn_hans_cn", "ja_jp"]


def test_resolve_language_configs_rejects_unknown():
    mod = _load_module()
    available = {"en_us"}
    try:
        mod._resolve_language_configs(["xx_yy"], available)
    except ValueError as exc:
        assert "Unsupported FLEURS configs" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for unknown config")
