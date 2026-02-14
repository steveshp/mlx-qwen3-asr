"""Autoregressive text generation for Qwen3-ASR."""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx

from .model import Qwen3ASRModel

# Repetition detection constants (from official repo)
REPETITION_THRESHOLD = 20


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 1024
    temperature: float = 0.0  # greedy by default
    eos_token_ids: list[int] = field(default_factory=lambda: [151643, 151645])
    eval_interval: int = 1


def generate(
    model: Qwen3ASRModel,
    input_ids: mx.array,
    audio_features: mx.array,
    position_ids: mx.array,
    config: GenerationConfig = None,
) -> list[int]:
    """Generate text tokens autoregressively.

    Two phases:
    1. Prefill: Process full input sequence (text prompt + audio features)
    2. Decode: Generate one token at a time with KV cache

    Args:
        model: The Qwen3ASR model
        input_ids: Input token IDs with audio placeholders, shape (1, seq_len)
        audio_features: Encoded audio features from audio encoder, shape (1, n_audio, dim)
        position_ids: MRoPE position IDs, shape (1, 3, seq_len)
        config: Generation configuration

    Returns:
        List of generated token IDs (excluding the input)
    """
    if config is None:
        config = GenerationConfig()

    max_seq_len = int(input_ids.shape[1] + config.max_new_tokens)
    cache = model.create_cache(max_seq_len=max_seq_len)

    # Phase 1: Prefill prompt and populate cache.
    logits = model.prefill(
        input_ids=input_ids,
        audio_features=audio_features,
        position_ids=position_ids,
        cache=cache,
    )

    # Sample first token
    token = _sample(logits, config.temperature)
    generated = [token]

    # Phase 2: Autoregressive decode
    seq_len = input_ids.shape[1]
    if config.max_new_tokens > 1:
        next_pos_base = mx.arange(
            seq_len,
            seq_len + (config.max_new_tokens - 1),
            dtype=position_ids.dtype,
        )  # (T,)
        next_pos_3d = mx.stack([next_pos_base, next_pos_base, next_pos_base], axis=0)
        next_pos_3d = next_pos_3d[None, :, :]  # (1, 3, T)
    else:
        next_pos_3d = None

    for step in range(1, config.max_new_tokens):
        # Check EOS
        if token in config.eos_token_ids:
            break

        # Check repetition
        if _detect_repetition(generated):
            break

        # Next token input
        next_ids = mx.array([[token]])

        # Reuse precomputed decode positions to avoid per-step allocation.
        next_position_ids = next_pos_3d[:, :, step - 1 : step]

        logits = model.step(
            input_ids=next_ids,
            position_ids=next_position_ids,
            cache=cache,
        )
        token = _sample(logits, config.temperature)
        generated.append(token)

        # Materialize cache periodically to avoid graph buildup while
        # reducing per-step synchronization overhead.
        if config.eval_interval > 0 and (step % config.eval_interval == 0):
            if hasattr(cache, "keys") and hasattr(cache, "values"):
                cache_tensors = [c for c in cache.keys if c is not None] + [
                    c for c in cache.values if c is not None
                ]
                if cache_tensors:
                    mx.eval(cache_tensors)

    # Remove EOS token if present
    if generated and generated[-1] in config.eos_token_ids:
        generated = generated[:-1]

    return generated


def _sample(logits: mx.array, temperature: float) -> int:
    """Sample a token from logits.

    Args:
        logits: Shape (1, 1, vocab_size) or (1, vocab_size)
        temperature: 0.0 for greedy, > 0 for sampling

    Returns:
        Selected token ID
    """
    logits = logits.reshape(-1)  # (vocab_size,)

    if temperature <= 0.0:
        # Greedy
        return mx.argmax(logits).item()
    else:
        # Temperature sampling — pass logits directly (categorical expects log-probs)
        return mx.random.categorical(logits / temperature).item()


def _detect_repetition(tokens: list[int], threshold: int = REPETITION_THRESHOLD) -> bool:
    """Detect repetitive token patterns.

    Two-stage detection (from official repo):
    1. Single token repeated > threshold times consecutively
    2. Pattern of 2-10 tokens repeated > threshold/pattern_len times

    Returns:
        True if repetition detected
    """
    if len(tokens) < threshold:
        return False

    # Stage 1: Single token repetition
    count = 1
    for i in range(len(tokens) - 1, 0, -1):
        if tokens[i] == tokens[i - 1]:
            count += 1
            if count >= threshold:
                return True
        else:
            break

    # Stage 2: Pattern repetition (check patterns of length 2-10)
    for pattern_len in range(2, min(11, len(tokens) // 2 + 1)):
        pattern = tokens[-pattern_len:]
        repeat_count = 1
        pos = len(tokens) - pattern_len * 2
        while pos >= 0:
            if tokens[pos:pos + pattern_len] == pattern:
                repeat_count += 1
                pos -= pattern_len
            else:
                break
        if repeat_count >= max(2, threshold // pattern_len):
            return True

    return False
