"""Unit tests for scripts/eval_nonnear_parity_matrix.py helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = Path("scripts/eval_nonnear_parity_matrix.py")
    spec = importlib.util.spec_from_file_location("eval_nonnear_parity_matrix", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_first_mismatch_handles_equal_content_and_length_delta():
    mod = _load_module()
    assert mod._first_mismatch([1, 2, 3], [1, 2, 3]) == -1
    assert mod._first_mismatch([1, 2, 3], [1, 9, 3]) == 1
    assert mod._first_mismatch([1, 2], [1, 2, 3]) == 2


def test_trim_eos_removes_only_trailing_tokens():
    mod = _load_module()
    eos = sorted(mod.EOS_IDS)[0]
    assert mod._trim_eos([1, eos, 2, eos, eos]) == [1, eos, 2]


def test_resolve_run_date_prefers_output_filename_prefix():
    mod = _load_module()
    assert (
        mod._resolve_run_date(Path("docs/benchmarks/2026-02-15-nonnear-parity-matrix.json"))
        == "2026-02-15"
    )


def test_resolve_run_date_falls_back_to_iso_date():
    mod = _load_module()
    out = mod._resolve_run_date(Path("nonnear-parity-matrix.json"))
    # YYYY-MM-DD
    assert len(out) == 10
    assert out[4] == "-"
    assert out[7] == "-"
