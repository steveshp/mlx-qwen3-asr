"""Autoregressive text generation for Qwen3-ASR."""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx

from .model import KVCache, Qwen3ASRModel

# Repetition detection constants (from official repo)
REPETITION_THRESHOLD = 20


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 1024
    temperature: float = 0.0  # greedy by default
    eos_token_ids: list[int] = field(default_factory=lambda: [151643, 151645])


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

    num_layers = len(model.model.layers)
    cache = KVCache(num_layers)

    # Phase 1: Prefill
    # Get embeddings and inject audio features
    embeds = model.model.embed_tokens(input_ids)
    audio_mask = input_ids == model.audio_token_id
    embeds = model._inject_audio_features(embeds, audio_features, audio_mask)

    # Forward pass through decoder (attention_mask=None lets TextDecoder
    # create its own causal mask, which is required during prefill)
    hidden = model.model(
        inputs_embeds=embeds,
        position_ids=position_ids,
        attention_mask=None,
        cache=cache,
    )

    # Get logits for last position
    logits = model.lm_head(hidden[:, -1:, :])

    # Sample first token
    token = _sample(logits, config.temperature)
    generated = [token]

    # Phase 2: Autoregressive decode
    seq_len = input_ids.shape[1]
    for step in range(1, config.max_new_tokens):
        # Check EOS
        if token in config.eos_token_ids:
            break

        # Check repetition
        if _detect_repetition(generated):
            break

        # Next token input
        next_ids = mx.array([[token]])

        # Update position_ids: increment all 3 dimensions
        cur_pos = seq_len + step - 1
        next_position_ids = mx.array([[[cur_pos], [cur_pos], [cur_pos]]])

        # Forward with cache
        next_embeds = model.model.embed_tokens(next_ids)
        hidden = model.model(
            inputs_embeds=next_embeds,
            position_ids=next_position_ids,
            attention_mask=None,  # cache handles masking
            cache=cache,
        )

        logits = model.lm_head(hidden)
        token = _sample(logits, config.temperature)
        generated.append(token)

        # Evaluate to avoid graph buildup
        mx.eval(
            [c for c in cache.keys if c is not None]
            + [c for c in cache.values if c is not None]
        )

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
