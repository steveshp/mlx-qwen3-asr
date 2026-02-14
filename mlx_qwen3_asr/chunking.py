"""Energy-based audio chunking for long audio files."""

from __future__ import annotations

import numpy as np

MAX_CHUNK_SECONDS = 1200.0  # 20 minutes max per chunk
MIN_CHUNK_SECONDS = 0.5     # Minimum valid audio length


def split_audio_into_chunks(
    audio: np.ndarray,
    sr: int = 16000,
    max_chunk_sec: float = MAX_CHUNK_SECONDS,
) -> list[tuple[np.ndarray, float]]:
    """Split audio at low-energy boundaries if it exceeds max_chunk_sec.

    Uses RMS energy in sliding windows to find optimal split points.
    Recursively splits if chunks are still too long.

    Args:
        audio: Raw waveform as numpy array, shape (n_samples,)
        sr: Sample rate
        max_chunk_sec: Maximum chunk duration in seconds

    Returns:
        List of (chunk_waveform, offset_seconds) tuples
    """
    duration = len(audio) / sr

    if duration <= max_chunk_sec:
        return [(audio, 0.0)]

    # Find split point at lowest energy near the midpoint
    split_sample = _find_split_point(audio, sr)

    # Split
    left = audio[:split_sample]
    right = audio[split_sample:]
    right_offset = split_sample / sr

    # Recursively split if needed
    left_chunks = split_audio_into_chunks(left, sr, max_chunk_sec)
    right_chunks = split_audio_into_chunks(right, sr, max_chunk_sec)

    # Adjust offsets for right chunks
    result = left_chunks.copy()
    for chunk, offset in right_chunks:
        result.append((chunk, offset + right_offset))

    return result


def _find_split_point(
    audio: np.ndarray,
    sr: int,
    window_sec: float = 0.5,
    search_range: float = 0.3,
) -> int:
    """Find the lowest-energy point near the midpoint of the audio.

    Args:
        audio: Raw waveform
        sr: Sample rate
        window_sec: RMS window size in seconds
        search_range: Fraction of audio around midpoint to search (0.3 = middle 60%)

    Returns:
        Sample index for the split point
    """
    n_samples = len(audio)
    mid = n_samples // 2

    # Search window: middle 60% of audio
    search_start = int(n_samples * (0.5 - search_range))
    search_end = int(n_samples * (0.5 + search_range))

    # RMS energy in sliding windows
    window_samples = int(window_sec * sr)
    if window_samples < 1:
        window_samples = 1

    # Compute RMS for each position in search range
    min_energy = float("inf")
    best_pos = mid

    step = window_samples // 2 or 1
    for pos in range(search_start, search_end - window_samples, step):
        segment = audio[pos:pos + window_samples]
        rms = np.sqrt(np.mean(segment ** 2))
        if rms < min_energy:
            min_energy = rms
            best_pos = pos + window_samples // 2

    return best_pos
