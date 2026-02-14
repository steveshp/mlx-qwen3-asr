"""Documentation integrity checks for artifact-backed claims."""

from __future__ import annotations

import re
from pathlib import Path


def _expand_brace_suffix(path: str) -> list[str]:
    """Expand simple suffix braces like foo.{json,md}."""
    match = re.search(r"\{([^{}]+)\}", path)
    if match is None:
        return [path]
    opts = [opt.strip() for opt in match.group(1).split(",") if opt.strip()]
    return [path[: match.start()] + opt + path[match.end() :] for opt in opts]


def test_readme_dated_benchmark_artifact_refs_exist():
    readme = Path("README.md").read_text(encoding="utf-8")
    backticked = re.findall(r"(?<!`)`([^`\n]+)`(?!`)", readme)

    # Only validate concrete dated benchmark artifact references.
    refs = [p for p in backticked if p.startswith("docs/benchmarks/2026-")]
    assert refs, "No dated benchmark artifact references found in README.md"

    missing: list[str] = []
    for ref in refs:
        for expanded in _expand_brace_suffix(ref):
            if not Path(expanded).exists():
                missing.append(expanded)

    assert not missing, (
        "README references missing benchmark artifacts: "
        + ", ".join(sorted(set(missing)))
    )
