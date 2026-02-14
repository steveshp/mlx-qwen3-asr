"""High-level transcription pipeline for Qwen3-ASR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import mlx.core as mx
import numpy as np

from .audio import SAMPLE_RATE, load_audio, log_mel_spectrogram
from .chunking import split_audio_into_chunks
from .generate import GenerationConfig, generate
from .load_models import _ModelHolder
from .model import Qwen3ASRModel
from .tokenizer import Tokenizer, parse_asr_output


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
    audio: Union[str, Path, np.ndarray, mx.array, tuple[np.ndarray, int]],
    *,
    model: Union[str, Qwen3ASRModel] = "Qwen/Qwen3-ASR-1.7B",
    language: Optional[str] = None,
    return_timestamps: bool = False,
    forced_aligner: Optional[str] = None,
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
        return_timestamps: Whether to return word-level timestamps
        forced_aligner: Path/name of forced aligner model (requires return_timestamps)
        dtype: Model dtype
        max_new_tokens: Maximum tokens to generate per chunk
        verbose: Print progress information

    Returns:
        TranscriptionResult with text, language, and optional segments
    """
    # Load model
    if isinstance(model, str):
        model_obj, config = _ModelHolder.get(model, dtype=dtype)
        model_path = model
    else:
        model_obj = model
        model_path = "Qwen/Qwen3-ASR-1.7B"

    # Load tokenizer
    tokenizer = Tokenizer(model_path)

    # Load audio
    if isinstance(audio, mx.array):
        audio_np = np.array(audio)
    elif isinstance(audio, np.ndarray):
        audio_np = audio.astype(np.float32)
    elif isinstance(audio, tuple):
        audio_np = np.array(load_audio(audio))
    else:
        audio_np = np.array(load_audio(audio))

    if verbose:
        duration = len(audio_np) / SAMPLE_RATE
        print(f"Audio duration: {duration:.1f}s ({len(audio_np)} samples)")

    # Split into chunks if needed
    chunks = split_audio_into_chunks(audio_np, sr=SAMPLE_RATE)

    if verbose and len(chunks) > 1:
        print(f"Split into {len(chunks)} chunks")

    # Transcribe each chunk
    all_texts = []
    detected_language = language or "unknown"

    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )

    for chunk_idx, (chunk_audio, offset) in enumerate(chunks):
        if verbose:
            print(f"Processing chunk {chunk_idx + 1}/{len(chunks)} "
                  f"(offset={offset:.1f}s, duration={len(chunk_audio)/SAMPLE_RATE:.1f}s)")

        # Compute mel spectrogram
        audio_mx = mx.array(chunk_audio)
        mel = log_mel_spectrogram(audio_mx)  # (n_mels, n_frames)
        mel = mel[None, :, :]  # (1, n_mels, n_frames)

        # Encode audio
        feature_lens = mx.array([mel.shape[2]])
        audio_features = model_obj.audio_tower(mel, feature_lens)  # (1, n_tokens, dim)
        n_audio_tokens = audio_features.shape[1]

        if verbose:
            print(f"  Audio features: {n_audio_tokens} tokens")

        # Build prompt
        prompt_tokens = tokenizer.build_prompt_tokens(
            n_audio_tokens=n_audio_tokens,
            language=language,
        )
        input_ids = mx.array([prompt_tokens])

        # Build position_ids for MRoPE (3 spatial dims, all same for text)
        seq_len = input_ids.shape[1]
        positions = mx.arange(seq_len)[None, :]  # (1, seq_len)
        position_ids = mx.stack([positions, positions, positions], axis=1)  # (1, 3, seq_len)

        # Generate
        output_tokens = generate(
            model=model_obj,
            input_ids=input_ids,
            audio_features=audio_features,
            position_ids=position_ids,
            config=gen_config,
        )

        # Decode tokens to text
        raw_text = tokenizer.decode(output_tokens)
        lang, text = parse_asr_output(raw_text)

        if detected_language == "unknown":
            detected_language = lang

        all_texts.append(text)

        if verbose:
            print(f"  [{lang}] {text[:100]}{'...' if len(text) > 100 else ''}")

    # Merge chunks
    full_text = " ".join(all_texts)

    # Optional: forced alignment for timestamps
    segments = None
    if return_timestamps and forced_aligner:
        from .forced_aligner import ForcedAligner
        aligner = ForcedAligner(forced_aligner, dtype=dtype)
        aligned = aligner.align(
            audio=audio_np,
            text=full_text,
            language=detected_language,
        )
        segments = [
            {"text": w.text, "start": w.start_time, "end": w.end_time}
            for w in aligned
        ]

    return TranscriptionResult(
        text=full_text,
        language=detected_language,
        segments=segments,
    )
