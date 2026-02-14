"""Output format writers for transcription results."""

from __future__ import annotations

import json
from typing import Callable

from .transcribe import TranscriptionResult


def write_txt(result: TranscriptionResult, output_path: str) -> None:
    """Write plain text transcription."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result.text)
        f.write("\n")


def write_json(result: TranscriptionResult, output_path: str) -> None:
    """Write JSON formatted transcription with metadata."""
    data = {
        "text": result.text,
        "language": result.language,
    }
    if result.segments:
        data["segments"] = result.segments

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_srt(result: TranscriptionResult, output_path: str) -> None:
    """Write SRT subtitle format. Requires segments with timestamps."""
    if not result.segments:
        # Fall back to single segment
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("1\n")
            f.write("00:00:00,000 --> 99:59:59,999\n")
            f.write(result.text)
            f.write("\n\n")
        return

    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(result.segments, 1):
            start = _format_timestamp_srt(seg["start"])
            end = _format_timestamp_srt(seg["end"])
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{seg['text']}\n\n")


def write_vtt(result: TranscriptionResult, output_path: str) -> None:
    """Write WebVTT subtitle format. Requires segments with timestamps."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")

        if not result.segments:
            f.write("00:00:00.000 --> 99:59:59.999\n")
            f.write(result.text)
            f.write("\n\n")
            return

        for seg in result.segments:
            start = _format_timestamp_vtt(seg["start"])
            end = _format_timestamp_vtt(seg["end"])
            f.write(f"{start} --> {end}\n")
            f.write(f"{seg['text']}\n\n")


def write_tsv(result: TranscriptionResult, output_path: str) -> None:
    """Write TSV format with start, end, text columns."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("start\tend\ttext\n")

        if not result.segments:
            f.write(f"0\t-1\t{result.text}\n")
            return

        for seg in result.segments:
            start_ms = int(seg["start"] * 1000)
            end_ms = int(seg["end"] * 1000)
            f.write(f"{start_ms}\t{end_ms}\t{seg['text']}\n")


def get_writer(fmt: str) -> Callable:
    """Get writer function by format name.

    Args:
        fmt: Format string - one of 'txt', 'json', 'srt', 'vtt', 'tsv'

    Returns:
        Writer function with signature (result, output_path) -> None
    """
    writers = {
        "txt": write_txt,
        "json": write_json,
        "srt": write_srt,
        "vtt": write_vtt,
        "tsv": write_tsv,
    }
    if fmt not in writers:
        raise ValueError(f"Unknown format '{fmt}'. Supported: {', '.join(writers.keys())}")
    return writers[fmt]


def _format_timestamp_srt(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    """Format seconds as VTT timestamp: HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
