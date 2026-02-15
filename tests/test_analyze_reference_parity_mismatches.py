"""Unit tests for scripts/analyze_reference_parity_mismatches.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = Path("scripts/analyze_reference_parity_mismatches.py")
    spec = importlib.util.spec_from_file_location("analyze_reference_parity_mismatches", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_category_exact_and_punctuation():
    mod = _load_module()
    assert mod._category_for_row({"token_match": True}) == "exact_match"
    assert mod._category_for_row(
        {"token_match": False, "normalized_text_match": True}
    ) == "punctuation_or_tokenization"


def test_category_numeric_and_content_shift():
    mod = _load_module()
    numeric = mod._category_for_row(
        {
            "token_match": False,
            "normalized_text_match": False,
            "ref_text_normalized": "como 53 trat",
            "mlx_text_normalized": "gomo dreiundfünfzig trat",
        }
    )
    assert numeric == "numeric_surface_form"

    content = mod._category_for_row(
        {
            "token_match": False,
            "normalized_text_match": False,
            "ref_text_normalized": "hello world and this is long",
            "mlx_text_normalized": "completely unrelated phrase appears",
        }
    )
    assert content in {"minor_lexical_shift", "content_shift"}


def test_summarize_counts_by_language():
    mod = _load_module()
    payload = {
        "suite": "reference-parity-suite-v1",
        "model": "Qwen/Qwen3-ASR-0.6B",
        "samples": 3,
        "token_match_rate": 0.0,
        "text_match_rate": 0.0,
        "rows": [
            {
                "sample_id": "a",
                "language": "English",
                "token_match": True,
                "normalized_text_match": True,
                "ref_text_normalized": "a",
                "mlx_text_normalized": "a",
            },
            {
                "sample_id": "b",
                "language": "English",
                "token_match": False,
                "normalized_text_match": True,
                "ref_text_normalized": "hello world",
                "mlx_text_normalized": "hello world",
            },
            {
                "sample_id": "c",
                "language": "Japanese",
                "token_match": False,
                "normalized_text_match": False,
                "ref_text_normalized": "テストです",
                "mlx_text_normalized": "ショーです",
                "first_mismatch_index": 3,
            },
        ],
    }
    summary = mod._summarize(payload, top_k=5)
    assert summary["samples"] == 3
    assert summary["by_language"]["English"]["samples"] == 2
    assert summary["by_language"]["Japanese"]["samples"] == 1
    assert summary["category_counts"]["exact_match"] == 1
    assert len(summary["top_mismatches"]) == 2
