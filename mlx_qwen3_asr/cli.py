"""Command-line interface for mlx-qwen3-asr."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ._version import __version__


def main():
    """CLI entry point for mlx-qwen3-asr."""
    parser = argparse.ArgumentParser(
        prog="mlx-qwen3-asr",
        description="Qwen3-ASR speech recognition on Apple Silicon via MLX",
    )

    parser.add_argument(
        "audio",
        nargs="+",
        help="Audio file(s) to transcribe",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-ASR-1.7B",
        help="Model name or path (default: Qwen/Qwen3-ASR-1.7B)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Force language (e.g., English, Chinese)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=".",
        help="Output directory (default: current directory)",
    )
    parser.add_argument(
        "--output-format", "-f",
        default="txt",
        choices=["txt", "json", "srt", "vtt", "tsv", "all"],
        help="Output format (default: txt)",
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Enable word-level timestamps via forced aligner",
    )
    parser.add_argument(
        "--forced-aligner",
        default="Qwen/Qwen3-ForcedAligner-0.6B",
        help="Forced aligner model (default: Qwen/Qwen3-ForcedAligner-0.6B)",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Model dtype (default: float16)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate (default: 1024)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress information",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    args = parser.parse_args()

    # Lazy imports for faster --help
    import mlx.core as mx

    from .transcribe import transcribe
    from .writers import get_writer

    # Parse dtype
    dtype_map = {
        "float16": mx.float16,
        "float32": mx.float32,
        "bfloat16": mx.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # Determine output formats
    if args.output_format == "all":
        formats = ["txt", "json", "srt", "vtt", "tsv"]
    else:
        formats = [args.output_format]

    # Process each audio file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for audio_path in args.audio:
        if not Path(audio_path).exists():
            print(f"Error: File not found: {audio_path}", file=sys.stderr)
            continue

        if args.verbose:
            print(f"\nTranscribing: {audio_path}")

        start_time = time.time()

        result = transcribe(
            audio=audio_path,
            model=args.model,
            language=args.language,
            return_timestamps=args.timestamps,
            forced_aligner=args.forced_aligner if args.timestamps else None,
            dtype=dtype,
            max_new_tokens=args.max_new_tokens,
            verbose=args.verbose,
        )

        elapsed = time.time() - start_time

        # Print result to stdout
        print(result.text)

        if args.verbose:
            print(f"\nLanguage: {result.language}")
            print(f"Time: {elapsed:.2f}s")

        # Write output files
        stem = Path(audio_path).stem
        for fmt in formats:
            out_path = output_dir / f"{stem}.{fmt}"
            writer = get_writer(fmt)
            writer(result, str(out_path))
            if args.verbose:
                print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
