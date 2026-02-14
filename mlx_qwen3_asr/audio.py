"""Audio loading and mel spectrogram computation for Qwen3-ASR.

Handles audio I/O via ffmpeg subprocess and computes log-mel spectrograms
using MLX for Metal-accelerated processing. Compatible with the Whisper-style
mel spectrogram pipeline used by Qwen3-ASR.

Key parameters:
    - Sample rate: 16000 Hz
    - FFT size: 400 (25ms window at 16kHz)
    - Hop length: 160 (10ms stride at 16kHz)
    - Mel bins: 128 (Slaney-normalized)
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Tuple, Union

import mlx.core as mx
import numpy as np

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
NUM_MEL_BINS = 128


def load_audio(
    source: Union[str, Path, np.ndarray, Tuple[np.ndarray, int]],
    sr: int = SAMPLE_RATE,
) -> mx.array:
    """Load audio from file path, numpy array, or (array, sample_rate) tuple.

    Returns mono float32 mx.array at the target sample rate. For file paths,
    uses ffmpeg subprocess (same pattern as mlx-whisper).

    Args:
        source: One of:
            - str or Path: path to an audio file (any format ffmpeg supports)
            - np.ndarray: raw waveform assumed to be at target sample rate
            - tuple of (np.ndarray, int): raw waveform with its sample rate
        sr: Target sample rate. Default 16000.

    Returns:
        Mono float32 mx.array of shape (n_samples,).

    Raises:
        RuntimeError: If ffmpeg fails or is not installed.
        ValueError: If source type is not recognized.
    """
    if isinstance(source, tuple):
        audio_np, orig_sr = source
        if not isinstance(audio_np, np.ndarray):
            audio_np = np.array(audio_np)
        if audio_np.size == 0:
            return mx.array(np.array([], dtype=np.float32))
        audio_np = audio_np.astype(np.float32)
        # Ensure mono
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=-1)
        if orig_sr != sr:
            audio_np = _resample_via_ffmpeg(audio_np, orig_sr, sr)
        return mx.array(audio_np)

    if isinstance(source, np.ndarray):
        if source.size == 0:
            return mx.array(np.array([], dtype=np.float32))
        audio_np = source.astype(np.float32)
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=-1)
        return mx.array(audio_np)

    if isinstance(source, (str, Path)):
        return _load_audio_file(str(source), sr)

    raise ValueError(
        f"Unsupported source type: {type(source)}. "
        "Expected str, Path, np.ndarray, or (np.ndarray, int)."
    )


def _load_audio_file(path: str, sr: int) -> mx.array:
    """Load audio file using ffmpeg, returning mono float32 at target sample rate."""
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        path,
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sr),
        "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Install with: brew install ffmpeg"
        )

    audio_np = np.frombuffer(result.stdout, np.int16).astype(np.float32) / 32768.0
    return mx.array(audio_np)


def _resample_via_ffmpeg(
    audio: np.ndarray, orig_sr: int, target_sr: int
) -> np.ndarray:
    """Resample audio array by piping through ffmpeg."""
    # Convert to s16le bytes
    audio_s16 = (audio * 32768.0).clip(-32768, 32767).astype(np.int16)
    input_bytes = audio_s16.tobytes()

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-f",
        "s16le",
        "-ac",
        "1",
        "-ar",
        str(orig_sr),
        "-i",
        "-",
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(target_sr),
        "-",
    ]
    try:
        result = subprocess.run(
            cmd, input=input_bytes, capture_output=True, check=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to resample audio: {e.stderr.decode()}") from e
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Install with: brew install ffmpeg"
        )

    return np.frombuffer(result.stdout, np.int16).astype(np.float32) / 32768.0


def mel_filters(n_mels: int = NUM_MEL_BINS) -> mx.array:
    """Load pre-computed mel filterbank from assets/mel_filters.npz.

    Args:
        n_mels: Number of mel bins. Must match a key in the .npz file.

    Returns:
        Mel filterbank matrix of shape (n_mels, n_fft // 2 + 1).

    Raises:
        FileNotFoundError: If mel_filters.npz is missing.
        KeyError: If the requested n_mels key is not in the file.
    """
    assets_dir = Path(__file__).parent / "assets"
    npz_path = assets_dir / "mel_filters.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Mel filterbank file not found: {npz_path}. "
            "Run: python scripts/generate_mel_filters.py"
        )
    filterbank = np.load(npz_path)
    key = f"mel_{n_mels}"
    if key not in filterbank:
        raise KeyError(
            f"Key '{key}' not found in {npz_path}. "
            f"Available keys: {list(filterbank.keys())}"
        )
    return mx.array(filterbank[key])


def _reflect_pad(x: mx.array, pad_len: int) -> mx.array:
    """Apply reflect padding to a 1-D array.

    Mimics np.pad(x, pad_len, mode='reflect'). For an input [a, b, c, d]
    with pad_len=2, the result is [c, b, a, b, c, d, c, b].

    Args:
        x: 1-D input array.
        pad_len: Number of samples to pad on each side.

    Returns:
        Padded 1-D array of length len(x) + 2 * pad_len.
    """
    if pad_len == 0:
        return x
    # Left pad: reversed slice x[1 : pad_len + 1]
    left = x[1 : pad_len + 1][::-1]
    # Right pad: reversed slice x[-(pad_len + 1) : -1]
    right = x[-(pad_len + 1) : -1][::-1]
    return mx.concatenate([left, x, right])


def stft(
    x: mx.array,
    window: mx.array,
    nperseg: int = N_FFT,
    noverlap: int | None = None,
) -> mx.array:
    """Compute Short-Time Fourier Transform using MLX.

    Uses mx.as_strided for efficient frame extraction and mx.fft.rfft for the
    FFT computation. Applies reflect padding of nperseg // 2 on each side,
    matching the Whisper/librosa convention.

    Args:
        x: Input waveform, shape (n_samples,).
        window: Window function, shape (nperseg,).
        nperseg: Segment length (FFT size). Default 400.
        noverlap: Overlap between segments. Default nperseg - HOP_LENGTH.

    Returns:
        Complex STFT matrix, shape (n_frames, n_fft // 2 + 1).
    """
    if noverlap is None:
        noverlap = nperseg - HOP_LENGTH

    hop = nperseg - noverlap
    pad_len = nperseg // 2

    # Reflect padding (mx.pad does not support mode="reflect", so do it manually)
    x = _reflect_pad(x, pad_len)

    # Frame extraction via as_strided
    # mx.as_strided uses element-based strides (not byte-based like numpy)
    num_frames = 1 + (x.shape[0] - nperseg) // hop
    frames = mx.as_strided(x, shape=(num_frames, nperseg), strides=(hop, 1))

    # Apply window and compute FFT
    windowed = frames * window
    return mx.fft.rfft(windowed)


def log_mel_spectrogram(
    audio: mx.array,
    n_mels: int = NUM_MEL_BINS,
) -> mx.array:
    """Compute log-mel spectrogram from raw audio waveform.

    Follows the Whisper-style pipeline:
    1. STFT with Hann window (400-sample FFT, 160-sample hop)
    2. Power spectrogram (magnitude squared)
    3. Mel filterbank projection
    4. Log10 with clamping and normalization

    Args:
        audio: Raw waveform, shape (n_samples,). Must be float32.
        n_mels: Number of mel bins (default 128).

    Returns:
        Log-mel spectrogram, shape (n_mels, n_frames). No zero-padding to
        a fixed length; the caller handles padding if needed.

    Raises:
        ValueError: If audio is empty.
    """
    if audio.size == 0:
        raise ValueError("Cannot compute mel spectrogram of empty audio.")

    # Hann window (using numpy for generation, then convert to mx)
    window = mx.array(np.hanning(N_FFT + 1)[:-1].astype(np.float32))

    # STFT
    freqs = stft(audio, window, nperseg=N_FFT, noverlap=N_FFT - HOP_LENGTH)
    magnitudes = mx.abs(freqs) ** 2  # (n_frames, n_fft // 2 + 1)

    # Mel filterbank projection
    filters = mel_filters(n_mels)  # (n_mels, n_fft // 2 + 1)
    mel_spec = filters @ magnitudes.T  # (n_mels, n_frames)

    # Log scale with clamping and normalization
    log_spec = mx.log10(mx.maximum(mel_spec, 1e-10))
    log_spec = mx.maximum(log_spec, mx.max(log_spec) - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    return log_spec
