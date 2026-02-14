"""Weight remapping and conversion for Qwen3-ASR HuggingFace -> MLX."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def remap_weights(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Remap HuggingFace weight keys to MLX model keys.

    Two transformations:
    1. Strip 'thinker.' prefix from all keys
       HF: thinker.audio_tower.conv2d1.weight -> audio_tower.conv2d1.weight
       HF: thinker.model.layers.0.self_attn.q_proj.weight -> model.layers.0.self_attn.q_proj.weight
       HF: thinker.lm_head.weight -> lm_head.weight

    2. Transpose Conv2d weights from PyTorch to MLX format
       PyTorch: (out_channels, in_channels, kH, kW)
       MLX:     (out_channels, kH, kW, in_channels)
       Transform: transpose(0, 2, 3, 1)

    Returns:
        Remapped weights dict ready for model.load_weights()
    """
    remapped = {}

    for key, value in weights.items():
        # Strip thinker. prefix
        new_key = key
        if new_key.startswith("thinker."):
            new_key = new_key[len("thinker."):]

        # Transpose Conv2d weights
        # Conv2d weight keys match pattern: audio_tower.conv2d{1,2,3}.weight
        if "conv2d" in new_key and new_key.endswith(".weight") and value.ndim == 4:
            # PyTorch (out, in, kH, kW) -> MLX (out, kH, kW, in)
            value = value.transpose(0, 2, 3, 1)

        remapped[new_key] = value

    return remapped


def quantize_model(
    model: nn.Module,
    bits: int = 4,
    group_size: int = 64,
) -> nn.Module:
    """Quantize model Linear and Embedding layers.

    Args:
        model: The model to quantize
        bits: Quantization bits (4 or 8)
        group_size: Quantization group size

    Returns:
        Quantized model (in-place modification)
    """
    nn.quantize(model, bits=bits, group_size=group_size)
    return model
