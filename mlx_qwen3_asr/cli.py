"""Command-line interface for mlx-qwen3-asr."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from ._version import __version__
from .config import DEFAULT_MODEL_ID


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
        default=DEFAULT_MODEL_ID,
        help=f"Model name or path (default: {DEFAULT_MODEL_ID})",
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
        help="Request word-level timestamps via native MLX forced aligner.",
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
        "--draft-model",
        default=None,
        help="Optional draft model for speculative decoding (e.g., Qwen/Qwen3-ASR-0.6B)",
    )
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        default=4,
        help="Speculative decode draft width (default: 4)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress information",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Run experimental chunked streaming decode instead of offline transcribe",
    )
    parser.add_argument(
        "--stream-chunk-sec",
        type=float,
        default=2.0,
        help="Streaming chunk size in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--stream-max-context-sec",
        type=float,
        default=30.0,
        help="Streaming max context window in seconds (default: 30.0)",
    )
    parser.add_argument(
        "--stream-endpointing-mode",
        choices=["fixed", "energy"],
        default="fixed",
        help="Streaming chunk endpointing strategy (default: fixed)",
    )
    parser.add_argument(
        "--stream-finalization-mode",
        choices=["accuracy", "latency"],
        default="accuracy",
        help="Streaming finish policy (default: accuracy)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    args = parser.parse_args()

    if args.streaming and args.timestamps:
        print(
            "Error: --streaming does not support --timestamps.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if args.streaming and args.draft_model is not None:
        print(
            "Error: --streaming does not support --draft-model yet.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # Lazy imports for faster --help
    import mlx.core as mx
    import numpy as np

    from .audio import SAMPLE_RATE, load_audio
    from .forced_aligner import ForcedAligner
    from .streaming import feed_audio, finish_streaming, init_streaming
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
    had_error = False
    aligner = (
        ForcedAligner(
            model_path=args.forced_aligner,
            dtype=dtype,
            backend="mlx",
        )
        if args.timestamps
        else None
    )

    for audio_path in args.audio:
        if not Path(audio_path).exists():
            print(f"Error: File not found: {audio_path}", file=sys.stderr)
            had_error = True
            continue

        if args.verbose:
            print(f"\nTranscribing: {audio_path}")

        start_time = time.time()

        try:
            if not args.streaming:
                result = transcribe(
                    audio=audio_path,
                    model=args.model,
                    draft_model=args.draft_model,
                    language=args.language,
                    return_timestamps=args.timestamps,
                    forced_aligner=aligner,
                    dtype=dtype,
                    max_new_tokens=args.max_new_tokens,
                    num_draft_tokens=args.num_draft_tokens,
                    verbose=args.verbose,
                )
            else:
                audio_np = np.asarray(load_audio(audio_path), dtype=np.float32)
                chunk_samples = max(1, int(args.stream_chunk_sec * SAMPLE_RATE))
                state = init_streaming(
                    model=args.model,
                    chunk_size_sec=args.stream_chunk_sec,
                    max_context_sec=args.stream_max_context_sec,
                    sample_rate=SAMPLE_RATE,
                    dtype=dtype,
                    max_new_tokens=args.max_new_tokens,
                    endpointing_mode=args.stream_endpointing_mode,
                    finalization_mode=args.stream_finalization_mode,
                )
                for i in range(0, len(audio_np), chunk_samples):
                    state = feed_audio(audio_np[i : i + chunk_samples], state)
                state = finish_streaming(state)
                result = SimpleNamespace(
                    text=state.text,
                    language=state.language,
                    segments=None,
                )
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            had_error = True
            continue

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

    if had_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
