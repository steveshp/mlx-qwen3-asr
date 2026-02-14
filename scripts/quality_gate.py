#!/usr/bin/env python3
"""Quality gate runner for mlx-qwen3-asr.

Usage:
  python scripts/quality_gate.py --mode fast
  RUN_REFERENCE_PARITY=1 python scripts/quality_gate.py --mode release
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class StepResult:
    name: str
    cmd: str
    passed: bool
    duration_sec: float
    returncode: int
    note: str = ""


def _run(cmd: list[str], cwd: Path) -> StepResult:
    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    elapsed = time.perf_counter() - started
    return StepResult(
        name=cmd[0],
        cmd=" ".join(shlex.quote(c) for c in cmd),
        passed=proc.returncode == 0,
        duration_sec=elapsed,
        returncode=proc.returncode,
    )


def _tracked_py_files(repo: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def run_gate(mode: str, repo: Path, python_bin: str) -> tuple[list[StepResult], bool]:
    steps: list[StepResult] = []

    tracked = _tracked_py_files(repo)
    if not tracked:
        steps.append(
            StepResult(
                name="ruff",
                cmd="",
                passed=False,
                duration_sec=0.0,
                returncode=1,
                note="No tracked Python files found.",
            )
        )
        return steps, False

    steps.append(_run([python_bin, "-m", "ruff", "check", *tracked], repo))
    steps.append(_run([python_bin, "-m", "pytest", "-q"], repo))

    if mode == "release":
        if not (repo / "tests" / "test_reference_parity.py").exists():
            steps.append(
                StepResult(
                    name="reference-parity",
                    cmd="pytest -q tests/test_reference_parity.py",
                    passed=False,
                    duration_sec=0.0,
                    returncode=1,
                    note="Missing tests/test_reference_parity.py",
                )
            )
        else:
            steps.append(
                _run([python_bin, "-m", "pytest", "-q", "tests/test_reference_parity.py"], repo)
            )

        if os.environ.get("RUN_ALIGNER_PARITY") == "1":
            samples = os.environ.get("ALIGNER_PARITY_SAMPLES", "10")
            steps.append(
                _run(
                    [
                        python_bin,
                        str(repo / "scripts" / "eval_aligner_parity.py"),
                        "--subset",
                        "test-clean",
                        "--samples",
                        samples,
                        "--model",
                        "Qwen/Qwen3-ForcedAligner-0.6B",
                        "--fail-text-match-rate-below",
                        "1.0",
                        "--fail-timing-mae-ms-above",
                        "60.0",
                    ],
                    repo,
                )
            )

    ok = all(step.passed for step in steps)
    return steps, ok


def _resolve_python(repo: Path) -> str:
    venv_py = repo / ".venv" / "bin" / "python"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def main() -> int:
    parser = argparse.ArgumentParser(description="Run repository quality gate checks.")
    parser.add_argument(
        "--mode",
        choices=["fast", "release"],
        default="fast",
        help="fast: lint+tests, release: fast + reference parity test",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional path to write gate result JSON.",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    python_bin = _resolve_python(repo)

    started = time.perf_counter()
    steps, ok = run_gate(args.mode, repo, python_bin)
    total = time.perf_counter() - started

    print(f"\nQuality Gate ({args.mode})")
    print("=" * 32)
    for step in steps:
        status = "PASS" if step.passed else "FAIL"
        print(f"[{status}] {step.cmd} ({step.duration_sec:.2f}s)")
        if step.note:
            print(f"       note: {step.note}")
    print(f"Total: {total:.2f}s")
    print(f"Result: {'PASS' if ok else 'FAIL'}")

    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mode": args.mode,
            "passed": ok,
            "total_duration_sec": total,
            "steps": [asdict(s) for s in steps],
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
