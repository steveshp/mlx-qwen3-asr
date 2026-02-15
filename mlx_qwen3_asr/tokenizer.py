"""Tokenizer wrapper and prompt building for Qwen3-ASR."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

_LANGUAGE_CANONICAL: dict[str, str] = {
    # English
    "en": "English",
    "en-us": "English",
    "en-gb": "English",
    "english": "English",
    # Chinese
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-tw": "Chinese",
    "cmn": "Chinese",
    "mandarin": "Chinese",
    "chinese": "Chinese",
    # Japanese
    "ja": "Japanese",
    "ja-jp": "Japanese",
    "japanese": "Japanese",
    # Korean
    "ko": "Korean",
    "ko-kr": "Korean",
    "korean": "Korean",
    # German
    "de": "German",
    "de-de": "German",
    "german": "German",
    # French
    "fr": "French",
    "fr-fr": "French",
    "french": "French",
    # Spanish
    "es": "Spanish",
    "es-es": "Spanish",
    "es-419": "Spanish",
    "spanish": "Spanish",
    # Russian
    "ru": "Russian",
    "ru-ru": "Russian",
    "russian": "Russian",
    # Arabic
    "ar": "Arabic",
    "ar-eg": "Arabic",
    "arabic": "Arabic",
    # Hindi
    "hi": "Hindi",
    "hi-in": "Hindi",
    "hindi": "Hindi",
    # Common additional languages
    "it": "Italian",
    "it-it": "Italian",
    "italian": "Italian",
    "pt": "Portuguese",
    "pt-br": "Portuguese",
    "pt-pt": "Portuguese",
    "portuguese": "Portuguese",
    "tr": "Turkish",
    "tr-tr": "Turkish",
    "turkish": "Turkish",
    "nl": "Dutch",
    "nl-nl": "Dutch",
    "dutch": "Dutch",
}


_KNOWN_LANGUAGE_NAMES = tuple(sorted(set(_LANGUAGE_CANONICAL.values())))


def canonicalize_language(language: Optional[str]) -> Optional[str]:
    """Normalize common language aliases/codes to canonical prompt names."""
    if language is None:
        return None
    value = str(language).strip()
    if not value:
        return None
    key = value.casefold().replace("_", "-")
    return _LANGUAGE_CANONICAL.get(key, value)


def language_is_known(language: Optional[str]) -> bool:
    """Return True when a language maps through known aliases/codes."""
    if language is None:
        return False
    value = str(language).strip()
    if not value:
        return False
    key = value.casefold().replace("_", "-")
    return key in _LANGUAGE_CANONICAL


def known_language_names() -> tuple[str, ...]:
    """Return known canonical language names used for prompt forcing."""
    return _KNOWN_LANGUAGE_NAMES


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
    except ImportError:
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
            canon_lang = canonicalize_language(language)
            if canon_lang:
                tokens.extend(self.encode(f"language {canon_lang}<asr_text>"))

        return tokens

    def build_followup_prompt_tokens(
        self,
        n_audio_tokens: int,
        language: Optional[str] = None,
    ) -> list[int]:
        """Build a follow-up chat turn for incremental streaming.

        Template:
            <|im_end|>
            <|im_start|>user
            <|audio_start|><|audio_pad|>*N<|audio_end|><|im_end|>
            <|im_start|>assistant
        """
        tokens = [self.IM_END_ID, self._newline_id, self.IM_START_ID]
        tokens.extend(self._user_tokens)
        tokens.append(self.AUDIO_START_TOKEN_ID)
        tokens.extend([self.AUDIO_TOKEN_ID] * n_audio_tokens)
        tokens.append(self.AUDIO_END_TOKEN_ID)
        tokens.append(self.IM_END_ID)
        tokens.append(self._newline_id)
        tokens.append(self.IM_START_ID)
        tokens.extend(self._assistant_tokens)

        if language:
            canon_lang = canonicalize_language(language)
            if canon_lang:
                tokens.extend(self.encode(f"language {canon_lang}<asr_text>"))

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


def _detect_and_fix_repetitions(text: str, threshold: int = 20) -> str:
    """Collapse pathological repetitions in decoder output."""

    def _fix_char_runs(s: str) -> str:
        out: list[str] = []
        i = 0
        n = len(s)
        while i < n:
            j = i + 1
            while j < n and s[j] == s[i]:
                j += 1
            run = j - i
            if run > threshold:
                out.append(s[i])
            else:
                out.append(s[i:j])
            i = j
        return "".join(out)

    def _fix_pattern_runs(s: str, max_pattern_len: int = 20) -> str:
        n = len(s)
        if n < threshold * 2:
            return s
        i = 0
        out: list[str] = []
        while i <= n - threshold * 2:
            found = False
            for k in range(1, max_pattern_len + 1):
                if i + (k * threshold) > n:
                    break
                p = s[i : i + k]
                ok = True
                for rep in range(1, threshold):
                    start = i + rep * k
                    if s[start : start + k] != p:
                        ok = False
                        break
                if not ok:
                    continue
                end = i + threshold * k
                while end + k <= n and s[end : end + k] == p:
                    end += k
                out.append(p)
                out.append(_fix_pattern_runs(s[end:], max_pattern_len=max_pattern_len))
                i = n
                found = True
                break
            if found:
                break
            out.append(s[i])
            i += 1
        if i < n:
            out.append(s[i:])
        return "".join(out)

    return _fix_pattern_runs(_fix_char_runs(text))


def _strip_trailing_special_tokens(s: str) -> str:
    out = s.rstrip()
    trailing_tokens = ("<|im_end|>", "<|endoftext|>")
    changed = True
    while changed:
        changed = False
        for token in trailing_tokens:
            if out.endswith(token):
                out = out[: -len(token)].rstrip()
                changed = True
    return out.strip()


def parse_asr_output(
    text: str,
    user_language: Optional[str] = None,
) -> tuple[str, str]:
    """Parse ASR model output into language and transcription text.

    Input format: "language English<asr_text>hello world"
    Output: ("English", "hello world")

    Args:
        text: Raw model output text
        user_language: Optional forced language. If provided, return this
            language and treat model output as plain transcription text.

    Returns:
        Tuple of (detected_language, transcription_text)
    """
    if text is None:
        return "unknown", ""
    s = str(text).strip()
    if not s:
        return "unknown", ""

    s = _detect_and_fix_repetitions(s)

    if user_language:
        forced = canonicalize_language(user_language) or user_language
        if "<asr_text>" in s:
            transcript = _strip_trailing_special_tokens(s.split("<asr_text>", 1)[1])
            return forced, transcript
        return forced, _strip_trailing_special_tokens(s)

    if "<asr_text>" in s:
        parts = s.split("<asr_text>", 1)
        lang_part = parts[0].strip()
        transcript = _strip_trailing_special_tokens(parts[1])

        # Extract language name from "language English"
        if lang_part.startswith("language "):
            lang = lang_part[len("language "):]
        else:
            lang = lang_part

        if "language none" in lang_part.lower() and not transcript:
            return "", ""

        return lang, transcript

    # Fallback: no asr_text marker
    return "unknown", _strip_trailing_special_tokens(s)
