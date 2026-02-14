"""High-level transcription pipeline for Qwen3-ASR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import mlx.core as mx
import numpy as np

from .audio import SAMPLE_RATE, compute_features, load_audio
from .chunking import split_audio_into_chunks
from .config import DEFAULT_MODEL_ID
from .forced_aligner import ForcedAligner
from .generate import GenerationConfig, generate
from .load_models import _ModelHolder
from .model import Qwen3ASRModel
from .tokenizer import Tokenizer, _TokenizerHolder, parse_asr_output

AudioInput = Union[str, Path, np.ndarray, mx.array, tuple[np.ndarray, int]]


@dataclass(frozen=True)
class TranscriptionResult:
    """Result of audio transcription.

    Attributes:
        text: The transcribed text
        language: Detected or forced language
        segments: Optional list of segments with timestamps [{text, start, end}, ...]
    """
    text: str
    language: str
    segments: Optional[list[dict]] = None


def transcribe(
    audio: AudioInput,
    *,
    model: Union[str, Qwen3ASRModel] = DEFAULT_MODEL_ID,
    language: Optional[str] = None,
    return_timestamps: bool = False,
    forced_aligner: Optional[Union[str, ForcedAligner]] = None,
    dtype: mx.Dtype = mx.float16,
    max_new_tokens: int = 1024,
    verbose: bool = False,
) -> TranscriptionResult:
    """Transcribe audio to text using Qwen3-ASR.

    Pipeline:
    1. Load audio -> mono 16kHz
    2. Split into chunks if > 1200s
    3. Per chunk: mel spectrogram -> encode -> build prompt -> decode -> parse
    4. Merge chunks
    5. Optional: forced aligner for word-level timestamps

    Args:
        audio: Audio source - file path, numpy array, mx.array, or (array, sr) tuple
        model: Model name/path or pre-loaded model instance
        language: Force language detection (e.g., "English", "Chinese")
        return_timestamps: Whether to return word-level timestamps.
        forced_aligner: Path/name of forced aligner model or prebuilt aligner object.
        dtype: Model dtype
        max_new_tokens: Maximum tokens to generate per chunk
        verbose: Print progress information

    Returns:
        TranscriptionResult with text, language, and optional segments
    """
    aligner = _resolve_aligner(return_timestamps, forced_aligner)

    # Load model
    if isinstance(model, str):
        model_obj, _ = _ModelHolder.get(model, dtype=dtype)
        model_path = _ModelHolder.get_resolved_path(model, dtype=dtype)
    else:
        model_obj = model
        model_path = DEFAULT_MODEL_ID

    # Load tokenizer (cached by model path)
    tokenizer = _TokenizerHolder.get(model_path)

    # Load audio
    audio_np = _to_audio_np(audio)

    if verbose:
        duration = len(audio_np) / SAMPLE_RATE
        print(f"Audio duration: {duration:.1f}s ({len(audio_np)} samples)")

    return _transcribe_loaded_components(
        audio_np=audio_np,
        model_obj=model_obj,
        tokenizer=tokenizer,
        dtype=dtype,
        language=language,
        aligner=aligner,
        return_timestamps=return_timestamps,
        max_new_tokens=max_new_tokens,
        verbose=verbose,
    )


def _to_audio_np(audio: AudioInput) -> np.ndarray:
    """Convert supported audio inputs to float32 numpy waveform at 16kHz."""
    if isinstance(audio, mx.array):
        return np.array(audio, dtype=np.float32)
    if isinstance(audio, np.ndarray):
        return audio.astype(np.float32)
    if isinstance(audio, tuple):
        return np.array(load_audio(audio), dtype=np.float32)
    return np.array(load_audio(audio), dtype=np.float32)


def _resolve_aligner(
    return_timestamps: bool,
    forced_aligner: Optional[Union[str, ForcedAligner]],
) -> Optional[ForcedAligner]:
    if not return_timestamps:
        return None
    if forced_aligner is None:
        return ForcedAligner()
    if isinstance(forced_aligner, str):
        return ForcedAligner(forced_aligner)
    return forced_aligner


def _transcribe_loaded_components(
    *,
    audio_np: np.ndarray,
    model_obj: Qwen3ASRModel,
    tokenizer: Tokenizer,
    dtype: mx.Dtype,
    language: Optional[str],
    aligner: Optional[ForcedAligner],
    return_timestamps: bool,
    max_new_tokens: int,
    verbose: bool,
) -> TranscriptionResult:
    """Transcribe using already-loaded model/tokenizer components."""
    chunks = split_audio_into_chunks(audio_np, sr=SAMPLE_RATE)

    if verbose and len(chunks) > 1:
        print(f"Split into {len(chunks)} chunks")

    all_texts = []
    all_segments: list[dict] = []
    detected_language = language or "unknown"

    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )

    for chunk_idx, (chunk_audio, offset) in enumerate(chunks):
        if verbose:
            print(
                f"Processing chunk {chunk_idx + 1}/{len(chunks)} "
                f"(offset={offset:.1f}s, duration={len(chunk_audio)/SAMPLE_RATE:.1f}s)"
            )

        mel, feature_lens = compute_features(chunk_audio)
        audio_features, _ = model_obj.audio_tower(mel.astype(dtype), feature_lens)
        n_audio_tokens = audio_features.shape[1]

        if verbose:
            print(f"  Audio features: {n_audio_tokens} tokens")

        prompt_tokens = tokenizer.build_prompt_tokens(
            n_audio_tokens=n_audio_tokens,
            language=language,
        )
        input_ids = mx.array([prompt_tokens])

        seq_len = input_ids.shape[1]
        positions = mx.arange(seq_len)[None, :]
        position_ids = mx.stack([positions, positions, positions], axis=1)

        output_tokens = generate(
            model=model_obj,
            input_ids=input_ids,
            audio_features=audio_features,
            position_ids=position_ids,
            config=gen_config,
        )

        raw_text = tokenizer.decode(output_tokens)
        lang, text = parse_asr_output(raw_text)

        if detected_language == "unknown":
            detected_language = lang

        all_texts.append(text)

        if return_timestamps and aligner is not None and text.strip():
            align_lang = language or lang
            if align_lang and align_lang != "unknown":
                aligned = aligner.align(chunk_audio, text, align_lang)
                for item in aligned:
                    all_segments.append(
                        {
                            "text": item.text,
                            "start": item.start_time + offset,
                            "end": item.end_time + offset,
                        }
                    )

        if verbose:
            print(f"  [{lang}] {text[:100]}{'...' if len(text) > 100 else ''}")

    return TranscriptionResult(
        text=" ".join(all_texts),
        language=detected_language,
        segments=all_segments if return_timestamps else None,
    )
