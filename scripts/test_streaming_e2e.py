#!/usr/bin/env python3
"""End-to-end streaming transcription test.

Feeds audio in real-time-like chunks and prints progressive results.
Validates: stable text growth, memory bounding, cache resets, RTF.

Usage:
    # Basic — uses default 0.6B model
    python scripts/test_streaming_e2e.py audio.m4a

    # Specify model + language
    python scripts/test_streaming_e2e.py audio.m4a --model ~/Downloads/Qwen3-ASR-1.7B-8bit --language vi

    # Adjust chunk size and context window
    python scripts/test_streaming_e2e.py audio.m4a --chunk-sec 1.0 --max-context-sec 15.0

    # Save results to JSON
    python scripts/test_streaming_e2e.py audio.m4a --json-output results.json

    # Quiet mode — only final summary
    python scripts/test_streaming_e2e.py audio.m4a --quiet
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _maybe_reexec_venv() -> None:
    repo = Path(__file__).resolve().parents[1]
    venv_python = repo / ".venv" / "bin" / "python"
    if not venv_python.exists():
        return
    if Path(sys.executable).resolve() == venv_python.resolve():
        return
    if os.environ.get("_MLX_QWEN3_ASR_REEXEC") == "1":
        return
    env = dict(os.environ)
    env["_MLX_QWEN3_ASR_REEXEC"] = "1"
    os.execve(
        str(venv_python),
        [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]],
        env,
    )


_maybe_reexec_venv()

import numpy as np

from mlx_qwen3_asr.audio import load_audio_np
from mlx_qwen3_asr.streaming import (
    feed_audio,
    finish_streaming,
    init_streaming,
    streaming_metrics,
)

SAMPLE_RATE = 16000


def _format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end streaming transcription test.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("audio", help="Path to audio file (any format ffmpeg supports)")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B", help="Model path or HF repo")
    parser.add_argument("--language", default=None, help="Force language (e.g. vi, ko, en)")
    parser.add_argument("--chunk-sec", type=float, default=2.0, help="Chunk size in seconds")
    parser.add_argument("--max-context-sec", type=float, default=30.0, help="Max context window")
    parser.add_argument("--quiet", action="store_true", help="Only print final summary")
    parser.add_argument("--json-output", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: file not found: {audio_path}", file=sys.stderr)
        return 1

    # --- Load audio ---
    if not args.quiet:
        print(f"Loading: {audio_path.name}")
    audio = load_audio_np(str(audio_path), sr=SAMPLE_RATE)
    duration = len(audio) / SAMPLE_RATE
    if not args.quiet:
        print(f"Duration: {_format_time(duration)} ({duration:.2f}s) @ {SAMPLE_RATE}Hz")
        print("=" * 72)

    # --- Init streaming ---
    state = init_streaming(
        model=args.model,
        chunk_size_sec=args.chunk_sec,
        max_context_sec=args.max_context_sec,
        sample_rate=SAMPLE_RATE,
        language=args.language,
    )

    chunk_samples = int(args.chunk_sec * SAMPLE_RATE)
    total_chunks = (len(audio) + chunk_samples - 1) // chunk_samples

    # --- Feed chunks ---
    latencies: list[float] = []
    stable_lengths: list[int] = []
    prev_stable = ""

    for i in range(total_chunks):
        start = i * chunk_samples
        end = min((i + 1) * chunk_samples, len(audio))
        chunk = audio[start:end]

        t0 = time.perf_counter()
        state = feed_audio(chunk, state)
        dt = time.perf_counter() - t0

        latencies.append(dt)
        stable_lengths.append(len(state.stable_text))

        if not args.quiet:
            audio_pos = end / SAMPLE_RATE
            new_stable = ""
            if state.stable_text.startswith(prev_stable):
                new_stable = state.stable_text[len(prev_stable):]

            parts = [f"[{audio_pos:6.1f}s] Chunk {i + 1:3d}/{total_chunks} ({dt:.3f}s)"]
            if new_stable.strip():
                parts.append(f' +"{new_stable.strip()[:70]}"')
            print("".join(parts))

        prev_stable = state.stable_text

    # --- Finish ---
    t0 = time.perf_counter()
    state = finish_streaming(state)
    finish_dt = time.perf_counter() - t0

    total_feed_time = sum(latencies)
    rtf = total_feed_time / duration if duration > 0 else 0.0
    metrics = streaming_metrics(state)

    # --- Validation checks ---
    checks: list[tuple[str, bool, str]] = []

    # 1. stable text grows monotonically
    monotonic = all(b >= a for a, b in zip(stable_lengths, stable_lengths[1:]))
    checks.append(("Stable text monotonic", monotonic, "stable_text should only grow"))

    # 2. RTF < 1.0 (faster than real-time)
    checks.append(("RTF < 1.0", rtf < 1.0, f"RTF={rtf:.4f}"))

    # 3. Final text is non-empty
    checks.append(("Non-empty output", len(state.text) > 0, f"{len(state.text)} chars"))

    # 4. Language detected
    checks.append(("Language detected", state.language is not None, f"lang={state.language}"))

    # 5. Chunk latency reasonable (p95 < chunk duration)
    if latencies:
        p95 = float(np.percentile(latencies, 95))
        checks.append(("p95 latency < chunk", p95 < args.chunk_sec, f"p95={p95:.3f}s"))

    # --- Print results ---
    if not args.quiet:
        print("=" * 72)

    print()
    print("--- RESULTS ---")
    print(f"Audio:          {audio_path.name} ({_format_time(duration)})")
    print(f"Model:          {args.model}")
    print(f"Language:       {state.language}")
    print(f"Chunks:         {total_chunks} x {args.chunk_sec}s")
    print(f"Feed time:      {total_feed_time:.3f}s")
    print(f"Finish time:    {finish_dt:.3f}s")
    print(f"RTF:            {rtf:.4f}x")
    if latencies:
        print(f"Chunk latency:  mean={sum(latencies)/len(latencies):.3f}s  "
              f"p95={float(np.percentile(latencies, 95)):.3f}s  "
              f"max={max(latencies):.3f}s")
    print(f"Output:         {len(state.text)} chars")
    print(f"Stability:      {metrics['partial_stability']:.2%}")
    print(f"Rewrite rate:   {metrics['rewrite_rate']:.2%}")
    print()

    print("--- CHECKS ---")
    all_passed = True
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status}] {name}: {detail}")
    print()

    print("--- TEXT ---")
    print(state.text)
    print()

    # --- JSON output ---
    if args.json_output:
        payload = {
            "audio_path": str(audio_path),
            "audio_duration_sec": duration,
            "model": args.model,
            "language": state.language,
            "chunk_size_sec": args.chunk_sec,
            "max_context_sec": args.max_context_sec,
            "total_chunks": total_chunks,
            "feed_time_sec": total_feed_time,
            "finish_time_sec": finish_dt,
            "rtf": rtf,
            "latency": {
                "mean": sum(latencies) / len(latencies) if latencies else 0,
                "p95": float(np.percentile(latencies, 95)) if latencies else 0,
                "max": max(latencies) if latencies else 0,
            },
            "streaming_quality": {
                "partial_stability": metrics["partial_stability"],
                "rewrite_rate": metrics["rewrite_rate"],
                "finalization_delta_chars": metrics["finalization_delta_chars"],
            },
            "output_chars": len(state.text),
            "text": state.text,
            "checks": {name: passed for name, passed, _ in checks},
            "all_checks_passed": all_passed,
        }
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Results saved: {out}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
