"""Tokenizer wrapper and prompt building for Qwen3-ASR."""

from __future__ import annotations

from typing import Optional


class Tokenizer:
    """Thin wrapper around HuggingFace Qwen2TokenizerFast.

    Handles prompt template construction with audio placeholder tokens.
    """

    # Special token IDs
    AUDIO_TOKEN_ID = 151646       # <|audio_pad|>
    AUDIO_START_TOKEN_ID = 151647  # <|audio_start|>
    AUDIO_END_TOKEN_ID = 151648    # <|audio_end|>
    IM_START_ID = 151644           # <|im_start|>
    IM_END_ID = 151645             # <|im_end|>
    EOS_TOKEN_IDS = [151643, 151645]

    def __init__(self, model_path: str):
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

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
        tokens.extend(self.encode("system\nYou are a helpful assistant."))
        tokens.append(self.IM_END_ID)
        tokens.append(self.encode("\n")[0])

        # User message with audio
        tokens.append(self.IM_START_ID)
        tokens.extend(self.encode("user\n"))
        tokens.append(self.AUDIO_START_TOKEN_ID)
        tokens.extend([self.AUDIO_TOKEN_ID] * n_audio_tokens)
        tokens.append(self.AUDIO_END_TOKEN_ID)
        tokens.append(self.IM_END_ID)
        tokens.append(self.encode("\n")[0])

        # Assistant prefix
        tokens.append(self.IM_START_ID)
        tokens.extend(self.encode("assistant\n"))

        # Optional language forcing
        if language:
            tokens.extend(self.encode(f"language {language}<asr_text>"))

        return tokens


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
