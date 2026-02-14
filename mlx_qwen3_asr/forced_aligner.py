"""Forced alignment wrapper for word-level timestamps.

Supports:
- `qwen_asr` backend (official PyTorch forced aligner)
- `mlx` backend (native MLX prototype path)
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import numpy as np

from .audio import compute_features

DEFAULT_FORCED_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
ALIGNER_BACKEND_QWEN_ASR = "qwen_asr"
ALIGNER_BACKEND_MLX = "mlx"
ALIGNER_BACKEND_AUTO = "auto"


@dataclass(frozen=True)
class AlignedWord:
    """Word-level alignment item."""

    text: str
    start_time: float
    end_time: float


class ForcedAlignTextProcessor:
    """Text processor utilities for timestamp alignment input/output."""

    @staticmethod
    def is_kept_char(ch: str) -> bool:
        if ch == "'":
            return True
        cat = unicodedata.category(ch)
        return cat.startswith("L") or cat.startswith("N")

    @classmethod
    def clean_token(cls, token: str) -> str:
        return "".join(ch for ch in token if cls.is_kept_char(ch))

    @staticmethod
    def is_cjk_char(ch: str) -> bool:
        code = ord(ch)
        return (
            0x4E00 <= code <= 0x9FFF
            or 0x3400 <= code <= 0x4DBF
            or 0x20000 <= code <= 0x2A6DF
            or 0x2A700 <= code <= 0x2B73F
            or 0x2B740 <= code <= 0x2B81F
            or 0x2B820 <= code <= 0x2CEAF
            or 0xF900 <= code <= 0xFAFF
        )

    @classmethod
    def split_segment_with_cjk(cls, seg: str) -> list[str]:
        tokens: list[str] = []
        buf: list[str] = []

        def flush() -> None:
            nonlocal buf
            if buf:
                tokens.append("".join(buf))
                buf = []

        for ch in seg:
            if cls.is_cjk_char(ch):
                flush()
                tokens.append(ch)
            else:
                buf.append(ch)

        flush()
        return tokens

    @classmethod
    def tokenize_space_lang(cls, text: str) -> list[str]:
        tokens: list[str] = []
        for seg in text.split():
            cleaned = cls.clean_token(seg)
            if cleaned:
                tokens.extend(cls.split_segment_with_cjk(cleaned))
        return tokens

    @classmethod
    def tokenize_text(cls, text: str, language: str) -> list[str]:
        """Tokenize alignment text into word/character units.

        Current default path matches official behavior for space-delimited
        languages and mixed CJK text. Japanese/Korean specialized tokenizers
        are intentionally deferred to keep core dependencies minimal.
        """
        _ = language
        return cls.tokenize_space_lang(text)

    @classmethod
    def encode_timestamp_prompt(cls, text: str, language: str) -> tuple[list[str], str]:
        words = cls.tokenize_text(text, language)
        input_text = "<timestamp><timestamp>".join(words) + "<timestamp><timestamp>"
        input_text = "<|audio_start|><|audio_pad|><|audio_end|>" + input_text
        return words, input_text

    @staticmethod
    def fix_timestamp(data: np.ndarray) -> list[int]:
        """Repair non-monotonic timestamp sequence via LIS-based correction."""
        arr = data.tolist()
        n = len(arr)
        if n == 0:
            return []

        dp = [1] * n
        parent = [-1] * n

        for i in range(1, n):
            for j in range(i):
                if arr[j] <= arr[i] and dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    parent[i] = j

        max_len = max(dp)
        idx = dp.index(max_len)

        lis_indices: list[int] = []
        while idx != -1:
            lis_indices.append(idx)
            idx = parent[idx]
        lis_indices.reverse()

        is_normal = [False] * n
        for i in lis_indices:
            is_normal[i] = True

        out = arr.copy()
        i = 0
        while i < n:
            if is_normal[i]:
                i += 1
                continue

            j = i
            while j < n and not is_normal[j]:
                j += 1

            anomaly_count = j - i
            left_val = next((out[k] for k in range(i - 1, -1, -1) if is_normal[k]), None)
            right_val = next((out[k] for k in range(j, n) if is_normal[k]), None)

            if anomaly_count <= 2:
                for k in range(i, j):
                    if left_val is None:
                        out[k] = right_val
                    elif right_val is None:
                        out[k] = left_val
                    else:
                        out[k] = left_val if (k - (i - 1)) <= (j - k) else right_val
            else:
                if left_val is not None and right_val is not None:
                    step = (right_val - left_val) / (anomaly_count + 1)
                    for k in range(i, j):
                        out[k] = left_val + step * (k - i + 1)
                elif left_val is not None:
                    for k in range(i, j):
                        out[k] = left_val
                elif right_val is not None:
                    for k in range(i, j):
                        out[k] = right_val

            i = j

        return [int(v) for v in out]

    @classmethod
    def parse_timestamp_ms(cls, words: list[str], timestamp_ms: np.ndarray) -> list[AlignedWord]:
        fixed = cls.fix_timestamp(np.asarray(timestamp_ms))
        out: list[AlignedWord] = []
        for i, word in enumerate(words):
            start_i = i * 2
            end_i = start_i + 1
            if end_i >= len(fixed):
                break
            out.append(
                AlignedWord(
                    text=word,
                    start_time=round(float(fixed[start_i]) / 1000.0, 3),
                    end_time=round(float(fixed[end_i]) / 1000.0, 3),
                )
            )
        return out


class _MLXForcedAlignerBackend:
    """Native MLX forced aligner backend."""

    def __init__(self, model_path: str, dtype: mx.Dtype):
        from .load_models import _ModelHolder
        from .tokenizer import _TokenizerHolder

        self.dtype = dtype
        self.model, self.config = _ModelHolder.get(model_path, dtype=dtype)
        self.tokenizer = _TokenizerHolder.get(model_path)

        self.timestamp_token_id = self.config.timestamp_token_id
        self.timestamp_segment_time = self.config.timestamp_segment_time
        if self.config.classify_num is None:
            raise RuntimeError(
                "Model config missing classify_num required for forced alignment."
            )
        if self.timestamp_token_id is None or self.timestamp_segment_time is None:
            raise RuntimeError(
                "Model config missing timestamp_token_id/timestamp_segment_time."
            )

    @staticmethod
    def _build_prompt(words: list[str], n_audio_tokens: int) -> str:
        body = "<timestamp><timestamp>".join(words) + "<timestamp><timestamp>"
        return (
            "<|audio_start|>"
            + ("<|audio_pad|>" * n_audio_tokens)
            + "<|audio_end|>"
            + body
        )

    def align(self, audio: np.ndarray, text: str, language: str) -> list[AlignedWord]:
        if text.strip() == "":
            return []

        words = ForcedAlignTextProcessor.tokenize_text(text, language)
        if not words:
            return []

        mel, feature_lens = compute_features(audio.astype(np.float32))
        audio_features, _ = self.model.audio_tower(mel.astype(self.dtype), feature_lens)
        n_audio_tokens = int(audio_features.shape[1])

        prompt = self._build_prompt(words, n_audio_tokens)
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = mx.array([token_ids])

        seq_len = input_ids.shape[1]
        positions = mx.arange(seq_len)[None, :]
        position_ids = mx.stack([positions, positions, positions], axis=1)

        embeds = self.model.model.embed_tokens(input_ids)
        audio_mask = input_ids == self.model.audio_token_id
        embeds = self.model._inject_audio_features(
            embeds, audio_features.astype(embeds.dtype), audio_mask
        )
        hidden = self.model.model(
            inputs_embeds=embeds,
            position_ids=position_ids,
            attention_mask=None,
            cache=None,
        )
        logits = self.model.lm_head(hidden)

        pred_ids = np.array(mx.argmax(logits, axis=-1))[0]
        input_np = np.array(input_ids)[0]
        timestamp_ids = pred_ids[input_np == int(self.timestamp_token_id)]
        timestamp_ms = timestamp_ids.astype(np.float32) * float(self.timestamp_segment_time)
        return ForcedAlignTextProcessor.parse_timestamp_ms(words, timestamp_ms)


class _QwenASRForcedAlignerBackend:
    """Official qwen-asr forced aligner backend."""

    def __init__(self, model_path: str):
        try:
            from qwen_asr import Qwen3ForcedAligner  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "Timestamps require optional dependency `qwen-asr`. "
                "Install with: pip install qwen-asr"
            ) from e

        # CPU keeps this portable and avoids silently requiring CUDA.
        self._backend = Qwen3ForcedAligner.from_pretrained(
            model_path,
            device_map="cpu",
        )

    def align(self, audio: np.ndarray, text: str, language: str) -> list[AlignedWord]:
        if text.strip() == "":
            return []

        results = self._backend.align(
            audio=[(audio.astype(np.float32), 16000)],
            text=[text],
            language=[language],
        )
        if not results:
            return []

        first = results[0]
        items = getattr(first, "items", None)
        if items is None:
            items = first

        out: list[AlignedWord] = []
        for item in items:
            out.append(
                AlignedWord(
                    text=str(getattr(item, "text", "")),
                    start_time=float(getattr(item, "start_time", 0.0)),
                    end_time=float(getattr(item, "end_time", 0.0)),
                )
            )
        return out


class ForcedAligner:
    """Word-level forced aligner.

    Args:
        model_path: HF repo ID or local path for the forced aligner model.
        dtype: MLX dtype for native backend paths.
        backend: One of `qwen_asr`, `mlx`, `auto`.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_FORCED_ALIGNER_MODEL,
        dtype: mx.Dtype = mx.float16,
        backend: str = ALIGNER_BACKEND_QWEN_ASR,
    ):
        self.model_path = model_path
        self.dtype = dtype
        self.backend = backend
        self._backend: Optional[Any] = None

    def _ensure_loaded(self) -> None:
        if self._backend is not None:
            return

        if self.backend not in {
            ALIGNER_BACKEND_QWEN_ASR,
            ALIGNER_BACKEND_MLX,
            ALIGNER_BACKEND_AUTO,
        }:
            raise RuntimeError(
                f"Unsupported aligner backend '{self.backend}'. "
                f"Expected one of: {ALIGNER_BACKEND_QWEN_ASR}, "
                f"{ALIGNER_BACKEND_MLX}, {ALIGNER_BACKEND_AUTO}."
            )

        mlx_err: Optional[Exception] = None
        if self.backend in {ALIGNER_BACKEND_MLX, ALIGNER_BACKEND_AUTO}:
            try:
                self._backend = _MLXForcedAlignerBackend(self.model_path, self.dtype)
                return
            except Exception as e:  # pragma: no cover - exercised via runtime integration
                mlx_err = e
                if self.backend == ALIGNER_BACKEND_MLX:
                    raise RuntimeError(f"Failed to load MLX aligner backend: {e}") from e

        try:
            self._backend = _QwenASRForcedAlignerBackend(self.model_path)
            return
        except Exception as e:
            if mlx_err is not None:
                raise RuntimeError(
                    "Failed to load both MLX and qwen-asr aligner backends. "
                    f"mlx_error={mlx_err}; qwen_asr_error={e}"
                ) from e
            raise

    def align(
        self,
        audio: np.ndarray,
        text: str,
        language: str,
    ) -> list[AlignedWord]:
        """Align a transcript against audio and return word-level timestamps."""
        self._ensure_loaded()
        return self._backend.align(audio, text, language)  # type: ignore[union-attr]
