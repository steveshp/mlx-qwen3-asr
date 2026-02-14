"""Explicit session API for model/tokenizer ownership."""

from __future__ import annotations

from typing import Optional, Union

import mlx.core as mx

from .config import DEFAULT_MODEL_ID
from .forced_aligner import ForcedAligner
from .load_models import _resolve_path, load_model
from .model import Qwen3ASRModel
from .tokenizer import Tokenizer
from .transcribe import (
    AudioInput,
    TranscriptionResult,
    _resolve_aligner,
    _to_audio_np,
    _transcribe_loaded_components,
)


class Session:
    """Explicit transcription session holding model and tokenizer state.

    This is the power-user path that avoids hidden process-global holders.
    """

    def __init__(
        self,
        model: Union[str, Qwen3ASRModel] = DEFAULT_MODEL_ID,
        *,
        dtype: mx.Dtype = mx.float16,
        tokenizer_model: Optional[str] = None,
    ) -> None:
        self.dtype = dtype

        if isinstance(model, str):
            self.model_id = model
            self.model, self.config = load_model(model, dtype=dtype)
            tok_path = tokenizer_model or str(_resolve_path(model))
        else:
            self.model_id = tokenizer_model or DEFAULT_MODEL_ID
            self.model = model
            self.config = None
            tok_path = tokenizer_model or DEFAULT_MODEL_ID

        self.tokenizer = Tokenizer(tok_path)

    def transcribe(
        self,
        audio: AudioInput,
        *,
        language: Optional[str] = None,
        return_timestamps: bool = False,
        forced_aligner: Optional[Union[str, ForcedAligner]] = None,
        max_new_tokens: int = 1024,
        verbose: bool = False,
    ) -> TranscriptionResult:
        """Transcribe audio using this session's loaded model/tokenizer."""
        aligner = _resolve_aligner(return_timestamps, forced_aligner)
        audio_np = _to_audio_np(audio)
        return _transcribe_loaded_components(
            audio_np=audio_np,
            model_obj=self.model,
            tokenizer=self.tokenizer,
            dtype=self.dtype,
            language=language,
            aligner=aligner,
            return_timestamps=return_timestamps,
            max_new_tokens=max_new_tokens,
            verbose=verbose,
        )
