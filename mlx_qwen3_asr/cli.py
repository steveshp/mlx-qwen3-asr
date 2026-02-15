"""Command-line interface for mlx-qwen3-asr."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from ._version import __version__
from .config import DEFAULT_MODEL_ID


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0:
        return "--:--"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class _ChunkProgressPrinter:
    def __init__(self, *, enabled: bool, start_time: float):
        self._enabled = bool(enabled)
        self._start_time = float(start_time)

    def __call__(self, payload: dict) -> None:
        if not self._enabled:
            return
        event = str(payload.get("event", ""))
        if event not in {"chunk_completed", "completed"}:
            return

        elapsed = max(0.0, time.time() - self._start_time)
        progress = float(payload.get("progress", 0.0) or 0.0)
        total_chunks = int(payload.get("total_chunks", 0) or 0)
        chunk_index = int(payload.get("chunk_index", 0) or 0)
        total_audio_sec = float(payload.get("audio_duration_sec", 0.0) or 0.0)
        eta = None
        if progress > 0.0:
            eta = max(0.0, (elapsed / progress) - elapsed)

        if event == "completed":
            print(
                (
                    f"Progress: 100.0% ({_format_duration(total_audio_sec)} audio) "
                    f"in {_format_duration(elapsed)}"
                ),
                file=sys.stderr,
            )
            return

        print(
            (
                f"Progress: chunk {chunk_index}/{max(total_chunks, 1)} "
                f"({progress * 100.0:.1f}%) ETA {_format_duration(eta)}"
            ),
            file=sys.stderr,
        )


def _print_languages() -> None:
    from .tokenizer import known_language_aliases

    aliases = known_language_aliases()
    print("Supported language aliases:")
    for language, values in aliases.items():
        print(f"- {language}: {', '.join(values)}")


def _preflight_diarization_runtime() -> None:
    def _has_spec(name: str) -> bool:
        try:
            return importlib.util.find_spec(name) is not None
        except ModuleNotFoundError:
            return False

    if not _has_spec("pyannote.audio"):
        print(
            "Error: --diarize requires optional dependency 'pyannote.audio'.",
            file=sys.stderr,
        )
        print(
            'Install with: pip install "mlx-qwen3-asr[diarize]"',
            file=sys.stderr,
        )
        raise SystemExit(1)
    if not _has_spec("torch"):
        print(
            "Error: --diarize requires PyTorch via pyannote dependencies.",
            file=sys.stderr,
        )
        print(
            'Install with: pip install "mlx-qwen3-asr[diarize]"',
            file=sys.stderr,
        )
        raise SystemExit(1)

    token = (
        os.environ.get("PYANNOTE_AUTH_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or ""
    )
    if not token:
        print(
            (
                "Info: --diarize may require Hugging Face auth for gated models. "
                "Set PYANNOTE_AUTH_TOKEN (or HF_TOKEN) if model access fails."
            ),
            file=sys.stderr,
        )

    try:
        _ensure_diarization_backend_ready()
    except (ImportError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def _ensure_diarization_backend_ready() -> None:
    """Validate diarization backend/model access before transcription starts."""
    # Intentional private import: we want to fail fast before spending time
    # on ASR transcription when diarization backend access is invalid.
    from .diarization import _load_pyannote_pipeline

    _load_pyannote_pipeline()


def _emit_new_stable_text(
    state_text: str,
    emitted_text: str,
) -> str:
    current = str(state_text or "")
    emitted = str(emitted_text or "")
    if not current:
        return emitted
    if current.startswith(emitted):
        delta = current[len(emitted):]
        if delta:
            print(delta, end="", flush=True)
        return current
    # Fallback when stable text gets desynced unexpectedly.
    print(f"\n{current}", end="", flush=True)
    return current


def main():
    """CLI entry point for mlx-qwen3-asr."""
    parser = argparse.ArgumentParser(
        prog="mlx-qwen3-asr",
        description="Qwen3-ASR speech recognition on Apple Silicon via MLX",
    )

    parser.add_argument(
        "audio",
        nargs="*",
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
        "--diarize",
        action="store_true",
        help="Attach speaker labels to transcript output (offline only).",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Optional fixed speaker count override for --diarize.",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=1,
        help="Minimum speaker count for --diarize auto mode (default: 1).",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=8,
        help="Maximum speaker count for --diarize auto mode (default: 8).",
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
        "--no-progress",
        action="store_true",
        help="Disable non-verbose progress updates",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Run experimental chunked streaming decode instead of offline transcribe",
    )
    parser.add_argument(
        "--mic",
        action="store_true",
        help="Capture and transcribe live microphone audio (experimental)",
    )
    parser.add_argument(
        "--mic-device",
        default=None,
        help="Microphone device name/index for --mic",
    )
    parser.add_argument(
        "--mic-duration-sec",
        type=float,
        default=None,
        help="Optional microphone capture duration in seconds",
    )
    parser.add_argument(
        "--mic-sample-rate",
        type=int,
        default=16000,
        help="Microphone sample rate in Hz (default: 16000)",
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
        "--list-languages",
        action="store_true",
        help="Print known language aliases/codes and exit",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress transcription text on stdout",
    )
    output_group.add_argument(
        "--stdout-only",
        action="store_true",
        help="Print transcription to stdout without writing output files",
    )
    output_group.add_argument(
        "--no-output-file",
        action="store_true",
        dest="stdout_only",
        help="Alias for --stdout-only",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    args = parser.parse_args()

    if args.list_languages:
        _print_languages()
        return

    if args.mic and args.audio:
        print("Error: --mic cannot be used with audio file arguments.", file=sys.stderr)
        raise SystemExit(1)
    if not args.mic and not args.audio:
        parser.error("audio is required unless --mic or --list-languages is used")

    if args.streaming and args.timestamps:
        print(
            "Error: --streaming does not support --timestamps.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if args.streaming and args.diarize:
        print(
            "Error: --streaming does not support --diarize.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if args.mic and args.timestamps:
        print(
            "Error: --mic does not support --timestamps.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if args.mic and args.diarize:
        print(
            "Error: --mic does not support --diarize.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if args.mic and args.streaming:
        print(
            "Error: --mic implies streaming mode; do not pass --streaming.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if args.streaming and args.draft_model is not None:
        print(
            "Error: --streaming does not support --draft-model yet.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if args.mic and args.draft_model is not None:
        print(
            "Error: --mic does not support --draft-model.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if args.mic_sample_rate <= 0:
        print("Error: --mic-sample-rate must be > 0.", file=sys.stderr)
        raise SystemExit(1)
    if args.mic_duration_sec is not None and args.mic_duration_sec <= 0:
        print("Error: --mic-duration-sec must be > 0.", file=sys.stderr)
        raise SystemExit(1)
    if args.num_speakers is not None and args.num_speakers <= 0:
        print("Error: --num-speakers must be > 0.", file=sys.stderr)
        raise SystemExit(1)
    if args.min_speakers <= 0:
        print("Error: --min-speakers must be > 0.", file=sys.stderr)
        raise SystemExit(1)
    if args.max_speakers < args.min_speakers:
        print(
            "Error: --max-speakers must be >= --min-speakers.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if args.diarize and not args.streaming and not args.mic:
        _preflight_diarization_runtime()

    # Lazy imports for faster --help
    import mlx.core as mx
    import numpy as np

    from .audio import SAMPLE_RATE, load_audio
    from .forced_aligner import ForcedAligner
    from .streaming import feed_audio, finish_streaming, init_streaming
    from .transcribe import TranscriptionResult, transcribe
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
    subtitle_formats = {"srt", "vtt"}
    wants_subtitles = any(fmt in subtitle_formats for fmt in formats)
    if wants_subtitles and (args.streaming or args.mic):
        print(
            "Error: subtitle formats (srt/vtt) require offline transcription.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    effective_timestamps = bool(args.timestamps or wants_subtitles or args.diarize)
    auto_timestamp_reasons: list[str] = []
    if wants_subtitles and not args.timestamps:
        auto_timestamp_reasons.append("subtitle output was requested")
    if args.diarize and not args.timestamps:
        auto_timestamp_reasons.append("diarization was requested")
    if auto_timestamp_reasons:
        reasons = " and ".join(auto_timestamp_reasons)
        print(
            f"Info: auto-enabling timestamps because {reasons}.",
            file=sys.stderr,
        )
    if not Path(args.model).exists():
        print(
            (
                f"Model: {args.model}. "
                "First run may download model files from Hugging Face "
                "(~1.2 GB for 0.6B, larger for 1.7B)."
            ),
            file=sys.stderr,
        )

    # Process each audio file
    output_dir = Path(args.output_dir)
    write_output_files = not args.stdout_only
    if write_output_files:
        output_dir.mkdir(parents=True, exist_ok=True)
    had_error = False
    aligner = (
        ForcedAligner(
            model_path=args.forced_aligner,
            dtype=dtype,
            backend="mlx",
        )
        if effective_timestamps and not args.streaming and not args.mic
        else None
    )

    if args.mic:
        mic_started = time.time()
        try:
            import sounddevice as sd
        except ImportError as exc:
            print(
                "Error: --mic requires the optional dependency 'sounddevice'.",
                file=sys.stderr,
            )
            print(
                "Install with: pip install sounddevice",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc

        if args.verbose:
            print("Listening on microphone... Press Ctrl+C to stop.", file=sys.stderr)

        chunk_samples = max(1, int(args.stream_chunk_sec * args.mic_sample_rate))
        state = init_streaming(
            model=args.model,
            chunk_size_sec=args.stream_chunk_sec,
            max_context_sec=args.stream_max_context_sec,
            sample_rate=args.mic_sample_rate,
            dtype=dtype,
            max_new_tokens=args.max_new_tokens,
            endpointing_mode=args.stream_endpointing_mode,
            finalization_mode=args.stream_finalization_mode,
            language=args.language,
        )
        emitted_stable = ""
        try:
            with sd.InputStream(
                samplerate=args.mic_sample_rate,
                channels=1,
                dtype="float32",
                blocksize=chunk_samples,
                device=args.mic_device,
            ) as stream:
                while True:
                    if args.mic_duration_sec is not None:
                        elapsed = time.time() - mic_started
                        if elapsed >= args.mic_duration_sec:
                            break
                        remaining_sec = max(0.0, args.mic_duration_sec - elapsed)
                        frames = min(
                            chunk_samples,
                            max(1, int(remaining_sec * args.mic_sample_rate)),
                        )
                    else:
                        frames = chunk_samples

                    data, overflowed = stream.read(frames)
                    if overflowed and args.verbose:
                        print("Warning: microphone overflow detected.", file=sys.stderr)
                    chunk = np.asarray(data, dtype=np.float32).reshape(-1)
                    state = feed_audio(chunk, state)
                    if not args.quiet:
                        emitted_stable = _emit_new_stable_text(state.stable_text, emitted_stable)
        except KeyboardInterrupt:
            if args.verbose:
                print("\nStopping microphone capture...", file=sys.stderr)

        state = finish_streaming(state)
        result = TranscriptionResult(
            text=state.text,
            language=state.language,
            segments=None,
            chunks=None,
        )
        if not args.quiet:
            if result.text.startswith(emitted_stable):
                tail = result.text[len(emitted_stable):]
                if tail:
                    print(tail, end="", flush=True)
                print()
            else:
                print(f"\n{result.text}")
        elapsed = time.time() - mic_started
        if args.verbose:
            duration = elapsed
            rtf = (elapsed / duration) if duration > 0 else 0.0
            print(f"\nLanguage: {result.language}", file=sys.stderr)
            print(f"Time: {elapsed:.2f}s", file=sys.stderr)
            print(f"RTF: {rtf:.4f}x", file=sys.stderr)

        if write_output_files:
            stem = datetime.now().strftime("microphone-%Y%m%d-%H%M%S")
            for fmt in formats:
                out_path = output_dir / f"{stem}.{fmt}"
                writer = get_writer(fmt)
                try:
                    writer(result, str(out_path))
                except Exception as e:
                    print(f"Error: {e}", file=sys.stderr)
                    had_error = True
                    continue
                if args.verbose:
                    print(f"Written: {out_path}", file=sys.stderr)
    for audio_path in args.audio:
        if not Path(audio_path).exists():
            print(f"Error: File not found: {audio_path}", file=sys.stderr)
            had_error = True
            continue

        if args.verbose:
            print(f"\nTranscribing: {audio_path}")

        start_time = time.time()
        progress = _ChunkProgressPrinter(
            enabled=(not args.verbose and not args.no_progress),
            start_time=start_time,
        )

        try:
            if not args.streaming:
                result = transcribe(
                    audio=audio_path,
                    model=args.model,
                    draft_model=args.draft_model,
                    language=args.language,
                    return_timestamps=effective_timestamps,
                    diarize=args.diarize,
                    diarization_num_speakers=args.num_speakers,
                    diarization_min_speakers=args.min_speakers,
                    diarization_max_speakers=args.max_speakers,
                    return_chunks=True,
                    forced_aligner=aligner,
                    dtype=dtype,
                    max_new_tokens=args.max_new_tokens,
                    num_draft_tokens=args.num_draft_tokens,
                    verbose=args.verbose,
                    on_progress=progress,
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
                    language=args.language,
                )
                total_chunks = max(1, int(np.ceil(len(audio_np) / chunk_samples)))
                for i in range(0, len(audio_np), chunk_samples):
                    state = feed_audio(audio_np[i : i + chunk_samples], state)
                    if not args.verbose and not args.no_progress:
                        chunk_idx = (i // chunk_samples) + 1
                        progress_ratio = min(1.0, chunk_idx / total_chunks)
                        elapsed = max(0.0, time.time() - start_time)
                        eta = (elapsed / progress_ratio - elapsed) if progress_ratio > 0 else None
                        print(
                            (
                                f"Progress: chunk {chunk_idx}/{total_chunks} "
                                f"({progress_ratio * 100.0:.1f}%) ETA {_format_duration(eta)}"
                            ),
                            file=sys.stderr,
                        )
                state = finish_streaming(state)
                result = TranscriptionResult(
                    text=state.text,
                    language=state.language,
                    segments=None,
                    chunks=[
                        {
                            "text": state.text,
                            "start": 0.0,
                            "end": float(len(audio_np) / SAMPLE_RATE),
                            "chunk_index": 0,
                            "language": state.language,
                        }
                    ],
                )
                if not args.verbose and not args.no_progress:
                    audio_dur = _format_duration(len(audio_np) / SAMPLE_RATE)
                    elapsed_dur = _format_duration(time.time() - start_time)
                    print(
                        (
                            f"Progress: 100.0% ({audio_dur} audio) "
                            f"in {elapsed_dur}"
                        ),
                        file=sys.stderr,
                    )
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            had_error = True
            continue

        elapsed = time.time() - start_time

        # Print result to stdout
        if not args.quiet:
            print(result.text)

        if args.verbose:
            duration = None
            chunks = getattr(result, "chunks", None) or []
            if chunks:
                duration = float(chunks[-1].get("end", 0.0))
            rtf = (elapsed / duration) if duration and duration > 0 else None
            print(f"\nLanguage: {result.language}")
            print(f"Time: {elapsed:.2f}s")
            if rtf is not None:
                print(f"RTF: {rtf:.4f}x")

        # Write output files
        if write_output_files:
            stem = Path(audio_path).stem
            for fmt in formats:
                out_path = output_dir / f"{stem}.{fmt}"
                writer = get_writer(fmt)
                try:
                    writer(result, str(out_path))
                except Exception as e:
                    print(f"Error: {e}", file=sys.stderr)
                    had_error = True
                    continue
                if args.verbose:
                    print(f"Written: {out_path}")

    if had_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
