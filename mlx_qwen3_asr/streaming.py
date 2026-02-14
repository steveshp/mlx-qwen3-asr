"""Streaming ASR with prefix rollback for real-time transcription."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .model import Qwen3ASRModel

# Streaming constants (from official repo)
UNFIXED_CHUNK_NUM = 2     # Trailing chunks considered unfixed
UNFIXED_TOKEN_NUM = 5     # Trailing tokens considered unfixed


@dataclass
class StreamingState:
    """State for streaming ASR session.

    Attributes:
        buffer: Pending audio samples not yet processed
        audio_accum: All accumulated audio so far
        text: Current best transcription
        language: Detected language
        chunk_id: Number of chunks processed
        chunk_size_samples: Samples per chunk
        previous_tokens: Tokens from previous transcription
        stable_text: Text considered stable (won't change)
    """
    buffer: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    audio_accum: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    text: str = ""
    language: str = "unknown"
    chunk_id: int = 0
    chunk_size_samples: int = 32000  # 2 seconds at 16kHz
    previous_tokens: list[int] = field(default_factory=list)
    stable_text: str = ""
    _model_path: str = "Qwen/Qwen3-ASR-1.7B"


def init_streaming(
    model: str = "Qwen/Qwen3-ASR-1.7B",
    chunk_size_sec: float = 2.0,
    sample_rate: int = 16000,
) -> StreamingState:
    """Initialize a streaming ASR session.

    Args:
        model: Model name or path
        chunk_size_sec: Audio chunk size in seconds
        sample_rate: Audio sample rate

    Returns:
        Initial streaming state
    """
    return StreamingState(
        chunk_size_samples=int(chunk_size_sec * sample_rate),
        _model_path=model,
    )


def feed_audio(
    pcm: np.ndarray,
    state: StreamingState,
    model: Optional[Qwen3ASRModel] = None,
) -> StreamingState:
    """Feed audio chunk to streaming ASR.

    Each chunk of audio:
    1. Accumulate in buffer
    2. When buffer >= chunk_size, process
    3. Transcribe accumulated audio
    4. Compare with previous transcription
    5. Apply prefix rollback for stability

    Args:
        pcm: Audio samples as float32 numpy array
        state: Current streaming state
        model: Pre-loaded model (if None, loads from state._model_path)

    Returns:
        Updated streaming state with new transcription
    """
    from .transcribe import transcribe

    # Accumulate audio
    state = StreamingState(
        buffer=np.concatenate([state.buffer, pcm]),
        audio_accum=np.concatenate([state.audio_accum, pcm]),
        text=state.text,
        language=state.language,
        chunk_id=state.chunk_id,
        chunk_size_samples=state.chunk_size_samples,
        previous_tokens=state.previous_tokens,
        stable_text=state.stable_text,
        _model_path=state._model_path,
    )

    # Check if we have enough audio for a new chunk
    if len(state.buffer) < state.chunk_size_samples:
        return state

    # Retain leftover samples beyond chunk_size for next iteration
    leftover = state.buffer[state.chunk_size_samples:]

    # Process: transcribe all accumulated audio
    result = transcribe(
        audio=state.audio_accum,
        model=state._model_path if model is None else model,
        verbose=False,
    )

    new_text = result.text
    new_language = result.language

    # Apply prefix rollback
    stable, unstable = _split_stable_unstable(
        state.stable_text,
        new_text,
        unfixed_tokens=UNFIXED_TOKEN_NUM,
    )

    return StreamingState(
        buffer=leftover,
        audio_accum=state.audio_accum,
        text=new_text,
        language=new_language,
        chunk_id=state.chunk_id + 1,
        chunk_size_samples=state.chunk_size_samples,
        previous_tokens=state.previous_tokens,
        stable_text=stable,
        _model_path=state._model_path,
    )


def finish_streaming(
    state: StreamingState,
    model: Optional[Qwen3ASRModel] = None,
) -> StreamingState:
    """Finalize streaming session, processing any remaining audio.

    Args:
        state: Current streaming state
        model: Pre-loaded model

    Returns:
        Final streaming state
    """
    if len(state.audio_accum) == 0:
        return state

    from .transcribe import transcribe

    # Final transcription of all audio
    result = transcribe(
        audio=state.audio_accum,
        model=state._model_path if model is None else model,
        verbose=False,
    )

    return StreamingState(
        buffer=np.array([], dtype=np.float32),
        audio_accum=state.audio_accum,
        text=result.text,
        language=result.language,
        chunk_id=state.chunk_id,
        chunk_size_samples=state.chunk_size_samples,
        previous_tokens=[],
        stable_text=result.text,
        _model_path=state._model_path,
    )


def _split_stable_unstable(
    prev_stable: str,
    new_text: str,
    unfixed_tokens: int = UNFIXED_TOKEN_NUM,
) -> tuple[str, str]:
    """Split transcription into stable and unstable parts.

    The last `unfixed_tokens` words are considered unstable and may change
    with future audio input.

    Args:
        prev_stable: Previously stable text
        new_text: New full transcription
        unfixed_tokens: Number of trailing tokens considered unstable

    Returns:
        Tuple of (stable_text, unstable_text)
    """
    words = new_text.split()

    if len(words) <= unfixed_tokens:
        return prev_stable, new_text

    stable_words = words[:-unfixed_tokens]
    unstable_words = words[-unfixed_tokens:]

    stable = " ".join(stable_words)
    unstable = " ".join(unstable_words)

    # Ensure new stable text is at least as long as previous
    if len(stable) < len(prev_stable):
        stable = prev_stable

    return stable, unstable
