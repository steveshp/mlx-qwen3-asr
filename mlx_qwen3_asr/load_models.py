"""Model loading from HuggingFace Hub or local path."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.utils as mlx_utils

from .config import DEFAULT_MODEL_ID, Qwen3ASRConfig
from .convert import remap_weights
from .model import Qwen3ASRModel

logger = logging.getLogger(__name__)


class _ModelHolder:
    """Singleton cache so repeated transcribe() calls don't reload the model."""

    _model: Optional[Qwen3ASRModel] = None
    _config: Optional[Qwen3ASRConfig] = None
    _path: Optional[str] = None
    _resolved_path: Optional[str] = None
    _dtype: Optional[mx.Dtype] = None

    @classmethod
    def get(
        cls,
        path_or_hf_repo: str,
        dtype: mx.Dtype = mx.float16,
    ) -> tuple[Qwen3ASRModel, Qwen3ASRConfig]:
        if cls._model is not None and cls._path == path_or_hf_repo and cls._dtype == dtype:
            return cls._model, cls._config

        model, config, resolved_path = _load_model_with_resolved_path(path_or_hf_repo, dtype=dtype)
        cls._model = model
        cls._config = config
        cls._path = path_or_hf_repo
        cls._resolved_path = str(resolved_path)
        cls._dtype = dtype
        return model, config

    @classmethod
    def get_resolved_path(
        cls,
        path_or_hf_repo: str,
        dtype: mx.Dtype = mx.float16,
    ) -> str:
        """Return a local resolved model path for the cached model key."""
        cls.get(path_or_hf_repo, dtype=dtype)
        if cls._resolved_path is None:
            return path_or_hf_repo
        return cls._resolved_path

    @classmethod
    def clear(cls):
        cls._model = None
        cls._config = None
        cls._path = None
        cls._resolved_path = None
        cls._dtype = None


def load_model(
    path_or_hf_repo: str = DEFAULT_MODEL_ID,
    dtype: mx.Dtype = mx.float16,
) -> tuple[Qwen3ASRModel, Qwen3ASRConfig]:
    """Load Qwen3-ASR model from local path or HuggingFace Hub.

    Pipeline:
    1. snapshot_download() if not a local path
    2. Parse config.json -> Qwen3ASRConfig
    3. Load safetensors -> remap_weights()
    4. Instantiate model, load weights, set dtype
    5. Return (model, config)

    Args:
        path_or_hf_repo: Local directory or HuggingFace repo ID
        dtype: Model dtype (default float16 for memory efficiency)

    Returns:
        Tuple of (model, config)
    """
    model, config, _ = _load_model_with_resolved_path(path_or_hf_repo, dtype=dtype)
    return model, config


def _load_model_with_resolved_path(
    path_or_hf_repo: str,
    dtype: mx.Dtype,
) -> tuple[Qwen3ASRModel, Qwen3ASRConfig, Path]:
    """Load model and return both config and resolved local model path."""
    model_path = _resolve_path(path_or_hf_repo)

    # Load config
    config_path = model_path / "config.json"
    with open(config_path) as f:
        raw_config = json.load(f)
    config = Qwen3ASRConfig.from_dict(raw_config)

    # Load weights
    weights = _load_safetensors(model_path)
    weights = remap_weights(weights)

    # Instantiate model
    model = Qwen3ASRModel(config)

    quant_cfg = _read_quantization_config(model_path)
    quantized = _is_quantized_weights(weights)
    if quantized:
        if quant_cfg is not None:
            bits = int(quant_cfg.get("bits", 4))
            group_size = int(quant_cfg.get("group_size", 64))
        else:
            bits, group_size = _infer_quantization_params(weights, model)
        nn.quantize(model, bits=bits, group_size=group_size)

    # Load weights into model
    model.load_weights(list(weights.items()))

    # Cast to target dtype
    if dtype != mx.float32 and not quantized:
        params = _cast_tree_dtype(model.parameters(), dtype)
        model.load_weights(list(mlx_utils.tree_flatten(params)))

    mx.eval(model.parameters())
    model.eval()

    if quantized:
        logger.info(f"Loaded quantized model from {model_path}")
    else:
        logger.info(f"Loaded model from {model_path} with dtype {dtype}")
    return model, config, model_path


def _cast_tree_dtype(tree: dict, dtype: mx.Dtype) -> dict:
    """Recursively cast floating-point mx.array leaves in a parameter tree."""
    return mlx_utils.tree_map(
        lambda x: x.astype(dtype)
        if isinstance(x, mx.array) and mx.issubdtype(x.dtype, mx.floating)
        else x,
        tree,
    )


def _resolve_path(path_or_hf_repo: str) -> Path:
    """Resolve to local path, downloading from HF Hub if needed."""
    local = Path(path_or_hf_repo)
    if local.exists() and (local / "config.json").exists():
        return local

    # Download from HuggingFace Hub
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=path_or_hf_repo,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
    )
    return Path(path)


def _read_quantization_config(model_path: Path) -> Optional[dict]:
    """Read optional quantization metadata from model directory."""
    qconf = model_path / "quantization_config.json"
    if not qconf.exists():
        return None
    try:
        return json.loads(qconf.read_text(encoding="utf-8"))
    except Exception:
        return None


def _is_quantized_weights(weights: dict[str, mx.array]) -> bool:
    """Return True if weight dict appears to be MLX-quantized."""
    return any(k.endswith(".scales") for k in weights)


def _majority_or_default(values: list[int], default: int) -> int:
    if not values:
        return default
    return Counter(values).most_common(1)[0][0]


def _infer_quantization_params(
    weights: dict[str, mx.array],
    model: Qwen3ASRModel,
) -> tuple[int, int]:
    """Infer quantization (bits, group_size) from saved tensors."""
    ref_params = dict(mlx_utils.tree_flatten(model.parameters()))
    bit_candidates: list[int] = []
    group_candidates: list[int] = []

    for key, packed_weight in weights.items():
        if not key.endswith(".weight"):
            continue
        scales_key = key[:-7] + ".scales"
        if scales_key not in weights:
            continue

        ref = ref_params.get(key)
        if ref is None or ref.ndim < 2 or packed_weight.ndim < 2:
            continue

        input_dim = int(ref.shape[-1])
        packed_cols = int(packed_weight.shape[-1])
        if input_dim > 0 and (packed_cols * 32) % input_dim == 0:
            bits = (packed_cols * 32) // input_dim
            if bits in (4, 8):
                bit_candidates.append(bits)

        scale_cols = int(weights[scales_key].shape[-1])
        if scale_cols > 0 and input_dim % scale_cols == 0:
            group_size = input_dim // scale_cols
            if group_size in (32, 64, 128):
                group_candidates.append(group_size)

    bits = _majority_or_default(bit_candidates, 4)
    group_size = _majority_or_default(group_candidates, 64)
    return bits, group_size


def _load_safetensors(model_path: Path) -> dict[str, mx.array]:
    """Load all safetensors files from a directory."""
    weights = {}
    safetensor_files = sorted(model_path.glob("*.safetensors"))

    if not safetensor_files:
        raise FileNotFoundError(
            f"No safetensors files found in {model_path}. "
            "Ensure the model has been downloaded correctly."
        )

    for sf_path in safetensor_files:
        w = mx.load(str(sf_path))
        weights.update(w)

    logger.info(f"Loaded {len(weights)} weight tensors from {len(safetensor_files)} files")
    return weights
