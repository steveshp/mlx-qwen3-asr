"""Forced alignment for word-level timestamps using Qwen3-ForcedAligner."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np


@dataclass(frozen=True)
class AlignedWord:
    """A word with its aligned timestamp.

    Attributes:
        text: The word text
        start_time: Start time in seconds
        end_time: End time in seconds
    """
    text: str
    start_time: float
    end_time: float


# Time resolution: 80ms per classification unit
TIMESTAMP_SEGMENT_TIME = 80  # ms
CLASSIFY_NUM = 5000


class ForcedAligner:
    """Forced alignment using Qwen3-ForcedAligner-0.6B.

    Uses non-autoregressive classification to assign timestamps to words.
    Each text position gets a classification logit over classify_num=5000 classes,
    where each class maps to a time position.

    Args:
        model_path: HuggingFace repo ID or local path
        dtype: Model dtype
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-ForcedAligner-0.6B",
        dtype: mx.Dtype = mx.float16,
    ):
        self.model_path = model_path
        self.dtype = dtype
        self._model = None
        self._config = None

    def _ensure_loaded(self):
        """Lazy-load the aligner model."""
        if self._model is not None:
            return

        from .load_models import load_model
        self._model, self._config = load_model(self.model_path, dtype=self.dtype)

    def align(
        self,
        audio: np.ndarray,
        text: str,
        language: str,
    ) -> list[AlignedWord]:
        """Align text to audio, producing word-level timestamps.

        Pipeline:
        1. Tokenize text into words (language-dependent splitting)
        2. Build aligner prompt with timestamp markers
        3. Run classification (non-autoregressive)
        4. Apply LIS correction for monotonic timestamps
        5. Convert classification indices to time

        Args:
            audio: Raw waveform numpy array
            text: Transcription text to align
            language: Language of the text

        Returns:
            List of AlignedWord with timestamps
        """
        # Tokenize into words
        words = self._tokenize_words(text, language)
        if not words:
            return []

        raise NotImplementedError(
            "Forced alignment is not yet implemented. "
            "The classify head for Qwen3-ForcedAligner requires additional work. "
            "See https://github.com/QwenLM/Qwen3-ASR for the reference implementation."
        )

    def _tokenize_words(self, text: str, language: str) -> list[str]:
        """Split text into words based on language.

        Language-specific tokenization:
        - CJK (Chinese, Japanese, Korean, Cantonese): character-level
        - Japanese with nagisa: morphological analysis (if available)
        - Korean with soynlp: L-tokenization (if available)
        - Others: whitespace splitting

        Args:
            text: Text to tokenize
            language: Language name

        Returns:
            List of word strings
        """
        text = text.strip()
        if not text:
            return []

        # Try language-specific tokenizers first (before falling back to char split)
        if language == "Japanese":
            try:
                import nagisa
                tokens = nagisa.tagging(text)
                return tokens.words
            except ImportError:
                # Fall through to character-level
                return [ch for ch in text if ch.strip()]

        if language == "Korean":
            try:
                from soynlp.tokenizer import LTokenizer
                tokenizer = LTokenizer()
                return tokenizer.tokenize(text)
            except ImportError:
                # Fall through to character-level
                return [ch for ch in text if ch.strip()]

        # Character-level splitting for other CJK languages
        cjk_languages = {"Chinese", "Cantonese"}
        if language in cjk_languages:
            return [ch for ch in text if ch.strip()]

        # Default: whitespace splitting
        return text.split()

    def _fix_timestamps_lis(self, predictions: list[int]) -> list[int]:
        """Apply Longest Increasing Subsequence correction.

        Raw predictions may be non-monotonic (word2 before word1).
        LIS finds the longest monotonic subsequence, then interpolates
        timestamps for non-monotonic predictions.

        O(n^2) implementation.

        Args:
            predictions: Raw classification predictions (time indices)

        Returns:
            Corrected monotonic predictions
        """
        n = len(predictions)
        if n <= 1:
            return predictions

        # Standard LIS with O(n^2) DP
        dp = [1] * n
        parent = [-1] * n

        for i in range(1, n):
            for j in range(i):
                if predictions[j] < predictions[i] and dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    parent[i] = j

        # Trace back LIS
        max_len = max(dp)
        max_idx = dp.index(max_len)

        lis_indices = set()
        idx = max_idx
        while idx != -1:
            lis_indices.add(idx)
            idx = parent[idx]

        # Interpolate non-LIS positions
        result = list(predictions)

        # Find LIS positions in order
        lis_positions = sorted(lis_indices)

        # Interpolate between LIS anchors using position-aware linear interpolation
        for i in range(n):
            if i in lis_indices:
                continue

            # Find surrounding LIS anchors (index and value)
            left_idx, left_val = -1, 0
            right_idx, right_val = n, predictions[-1] if predictions else 0

            for li in lis_positions:
                if li < i:
                    left_idx = li
                    left_val = predictions[li]
                elif li > i:
                    right_idx = li
                    right_val = predictions[li]
                    break

            # Position-aware linear interpolation
            span = right_idx - left_idx
            if span > 0:
                frac = (i - left_idx) / span
                result[i] = int(left_val + frac * (right_val - left_val))
            else:
                result[i] = left_val

        return result
