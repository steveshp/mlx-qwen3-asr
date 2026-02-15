#!/usr/bin/env python3
"""Analyze mismatch patterns in reference parity suite artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def _contains_digit(s: str) -> bool:
    return any(ch.isdigit() for ch in str(s))


def _category_for_row(row: dict[str, Any]) -> str:
    token_match = bool(row.get("token_match", row.get("text_match", False)))
    text_match = bool(
        row.get("normalized_text_match", row.get("text_match", token_match))
    )
    if token_match:
        return "exact_match"
    if text_match:
        return "punctuation_or_tokenization"

    ref_text = str(row.get("ref_text_normalized", ""))
    mlx_text = str(row.get("mlx_text_normalized", ""))
    if _contains_digit(ref_text) != _contains_digit(mlx_text):
        return "numeric_surface_form"

    dist = _edit_distance(ref_text, mlx_text)
    max_len = max(1, len(ref_text), len(mlx_text))
    ratio = dist / float(max_len)
    if ratio <= 0.12:
        return "minor_lexical_shift"
    return "content_shift"


def _bump(table: dict[str, int], key: str) -> None:
    table[key] = int(table.get(key, 0)) + 1


def _summarize(payload: dict[str, Any], top_k: int) -> dict[str, Any]:
    rows = list(payload.get("rows", []))
    by_language: dict[str, dict[str, Any]] = {}
    category_counts: dict[str, int] = {}
    mismatch_rows: list[dict[str, Any]] = []

    for row in rows:
        language = str(row.get("language", "unknown"))
        bucket = by_language.setdefault(
            language,
            {
                "samples": 0,
                "token_matches": 0,
                "text_matches": 0,
                "categories": {},
            },
        )
        bucket["samples"] += 1
        token_match = bool(row.get("token_match", row.get("text_match", False)))
        text_match = bool(
            row.get("normalized_text_match", row.get("text_match", token_match))
        )
        bucket["token_matches"] += 1 if token_match else 0
        bucket["text_matches"] += 1 if text_match else 0

        cat = _category_for_row(row)
        _bump(category_counts, cat)
        _bump(bucket["categories"], cat)

        if cat != "exact_match":
            ref_text = str(row.get("ref_text_normalized", ""))
            mlx_text = str(row.get("mlx_text_normalized", ""))
            dist = _edit_distance(ref_text, mlx_text)
            max_len = max(1, len(ref_text), len(mlx_text))
            mismatch_rows.append(
                {
                    "sample_id": row.get("sample_id"),
                    "subset": row.get("subset"),
                    "language": language,
                    "category": cat,
                    "first_mismatch_index": row.get("first_mismatch_index"),
                    "edit_distance": dist,
                    "edit_ratio": dist / float(max_len),
                    "ref_text_normalized": ref_text,
                    "mlx_text_normalized": mlx_text,
                }
            )

    for lang, stats in by_language.items():
        n = max(1, int(stats["samples"]))
        stats["token_match_rate"] = float(stats["token_matches"]) / float(n)
        stats["text_match_rate"] = float(stats["text_matches"]) / float(n)
        by_language[lang] = stats

    def _mismatch_sort_key(row: dict[str, Any]) -> tuple[float, int]:
        idx = row.get("first_mismatch_index")
        safe_idx = int(idx) if idx is not None else 0
        return (-float(row["edit_ratio"]), safe_idx)

    mismatch_rows.sort(key=_mismatch_sort_key)
    return {
        "suite": payload.get("suite"),
        "model": payload.get("model"),
        "samples": int(payload.get("samples", len(rows))),
        "token_match_rate": float(payload.get("token_match_rate", payload.get("match_rate", 0.0))),
        "text_match_rate": float(payload.get("text_match_rate", 0.0)),
        "category_counts": category_counts,
        "by_language": by_language,
        "top_mismatches": mismatch_rows[: max(0, top_k)],
    }


def _to_markdown(summary: dict[str, Any], source_path: str) -> str:
    lines: list[str] = []
    lines.append("# Reference Parity Mismatch Analysis")
    lines.append("")
    lines.append(f"- Source: `{source_path}`")
    lines.append(f"- Suite: `{summary.get('suite')}`")
    lines.append(f"- Model: `{summary.get('model')}`")
    lines.append(f"- Samples: `{summary.get('samples')}`")
    lines.append(
        f"- Token match rate: `{float(summary.get('token_match_rate', 0.0)):.4f}`"
    )
    lines.append(
        f"- Text match rate: `{float(summary.get('text_match_rate', 0.0)):.4f}`"
    )
    lines.append("")
    lines.append("## Categories")
    lines.append("")
    lines.append("| Category | Count |")
    lines.append("|---|---:|")
    for category, count in sorted(
        dict(summary.get("category_counts", {})).items(),
        key=lambda kv: (-int(kv[1]), kv[0]),
    ):
        lines.append(f"| {category} | {int(count)} |")

    lines.append("")
    lines.append("## By Language")
    lines.append("")
    lines.append("| Language | Samples | Token match rate | Text match rate |")
    lines.append("|---|---:|---:|---:|")
    for language, stats in sorted(dict(summary.get("by_language", {})).items()):
        lines.append(
            f"| {language} | {int(stats['samples'])} | "
            f"{float(stats['token_match_rate']):.4f} | "
            f"{float(stats['text_match_rate']):.4f} |"
        )

    top = list(summary.get("top_mismatches", []))
    if top:
        lines.append("")
        lines.append("## Top Mismatches")
        lines.append("")
        lines.append("| Sample | Language | Category | Edit ratio | First mismatch idx |")
        lines.append("|---|---|---|---:|---:|")
        for row in top:
            lines.append(
                f"| {row.get('sample_id')} | {row.get('language')} | {row.get('category')} | "
                f"{float(row.get('edit_ratio', 0.0)):.4f} | "
                f"{int(row.get('first_mismatch_index', -1))} |"
            )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze parity mismatch patterns.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--md-output", default=None)
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()

    input_path = Path(args.input_json).expanduser().resolve()
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    summary = _summarize(payload, top_k=args.top_k)

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.json_output:
        out = Path(args.json_output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.md_output:
        out = Path(args.md_output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_to_markdown(summary, source_path=str(input_path)), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
