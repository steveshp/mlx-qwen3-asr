"""Tokenizer wrapper and prompt building for Qwen3-ASR."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def _from_pretrained_with_fix_flag(loader_cls, model_path: str):  # noqa: ANN001
    """Load a tokenizer class with optional mistral-regex compatibility flag."""
    common_kwargs = {"trust_remote_code": True}
    try:
        return loader_cls.from_pretrained(
            model_path,
            fix_mistral_regex=True,
            **common_kwargs,
        )
    except TypeError as e:
        if "fix_mistral_regex" not in str(e):
            raise
        return loader_cls.from_pretrained(model_path, **common_kwargs)


def _load_hf_tokenizer(model_path: str):
    """Load HF tokenizer with best-effort compatibility fixes.

    Some model/tokenizer bundles trigger a warning about Mistral regex behavior.
    Newer `transformers` versions support `fix_mistral_regex`; older versions
    may reject it, so we fall back cleanly.
    """
    # Prefer direct Qwen2 tokenizer import when available. This avoids the
    # heavier AutoTokenizer dynamic import path and improves cold-start latency.
    try:
        from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
    except Exception:
        Qwen2Tokenizer = None

    if Qwen2Tokenizer is not None:
        return _from_pretrained_with_fix_flag(Qwen2Tokenizer, model_path)

    from transformers import AutoTokenizer
    return _from_pretrained_with_fix_flag(AutoTokenizer, model_path)


class Tokenizer:
    """Thin wrapper around HuggingFace Qwen2TokenizerFast.

    Handles prompt template construction with audio placeholder tokens.
    """

    # Fallback special token IDs (looked up from tokenizer at init)
    IM_START_ID = 151644           # <|im_start|>
    IM_END_ID = 151645             # <|im_end|>
    EOS_TOKEN_IDS = [151643, 151645]

    def __init__(self, model_path: str):
        self._tokenizer = _load_hf_tokenizer(model_path)

        # Look up audio token IDs from the actual tokenizer vocab
        vocab = self._tokenizer.get_vocab()
        self.AUDIO_TOKEN_ID = vocab.get("<|audio_pad|>", 151676)
        self.AUDIO_START_TOKEN_ID = vocab.get("<|audio_start|>", 151669)
        self.AUDIO_END_TOKEN_ID = vocab.get("<|audio_end|>", 151670)
        self._system_tokens = self.encode("system\nYou are a helpful assistant.")
        self._user_tokens = self.encode("user\n")
        self._assistant_tokens = self.encode("assistant\n")
        self._newline_id = self.encode("\n")[0]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def build_prompt_tokens(
        self,
        n_audio_tokens: int,
        language: Optional[str] = None,
    ) -> list[int]:
        """Build chat-template prompt with audio placeholder tokens.

        Prompt template:
            <|im_start|>system
            You are a helpful assistant.<|im_end|>
            <|im_start|>user
            <|audio_start|><|audio_pad|>...(N times)...<|audio_pad|><|audio_end|><|im_end|>
            <|im_start|>assistant

        If language is specified, the assistant prefix becomes:
            <|im_start|>assistant
            language {lang}<asr_text>

        Args:
            n_audio_tokens: Number of audio feature tokens to placeholder
            language: Optional language to force (e.g., "English", "Chinese")

        Returns:
            List of token IDs forming the complete prompt
        """
        # System message
        tokens = [self.IM_START_ID]
        tokens.extend(self._system_tokens)
        tokens.append(self.IM_END_ID)
        tokens.append(self._newline_id)

        # User message with audio
        tokens.append(self.IM_START_ID)
        tokens.extend(self._user_tokens)
        tokens.append(self.AUDIO_START_TOKEN_ID)
        tokens.extend([self.AUDIO_TOKEN_ID] * n_audio_tokens)
        tokens.append(self.AUDIO_END_TOKEN_ID)
        tokens.append(self.IM_END_ID)
        tokens.append(self._newline_id)

        # Assistant prefix
        tokens.append(self.IM_START_ID)
        tokens.extend(self._assistant_tokens)

        # Optional language forcing
        if language:
            tokens.extend(self.encode(f"language {language}<asr_text>"))

        return tokens


class _TokenizerHolder:
    """Simple process-local cache for tokenizer instances."""

    _cache: dict[str, Tokenizer] = {}

    @staticmethod
    def _canonical_key(model_path: str) -> str:
        p = Path(model_path)
        if p.exists():
            return str(p.resolve())
        return model_path

    @classmethod
    def get(cls, model_path: str) -> Tokenizer:
        key = cls._canonical_key(model_path)
        tok = cls._cache.get(key)
        if tok is None:
            tok = Tokenizer(model_path)
            cls._cache[key] = tok
        return tok

    @classmethod
    def clear(cls) -> None:
        cls._cache.clear()


def parse_asr_output(text: str) -> tuple[str, str]:
    """Parse ASR model output into language and transcription text.

    Input format: "language English<asr_text>hello world"
    Output: ("English", "hello world")

    Args:
        text: Raw model output text

    Returns:
        Tuple of (detected_language, transcription_text)
    """
    if "<asr_text>" in text:
        parts = text.split("<asr_text>", 1)
        lang_part = parts[0].strip()
        transcript = parts[1].strip()

        # Extract language name from "language English"
        if lang_part.startswith("language "):
            lang = lang_part[len("language "):]
        else:
            lang = lang_part

        # Clean up any trailing special tokens
        for token in ["<|im_end|>", "<|endoftext|>"]:
            transcript = transcript.replace(token, "").strip()

        return lang, transcript

    # Fallback: no asr_text marker
    return "unknown", text.strip()
