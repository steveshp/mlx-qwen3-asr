"""Model loading from HuggingFace Hub or local path."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import mlx.core as mx

from .config import Qwen3ASRConfig
from .convert import remap_weights
from .model import Qwen3ASRModel

logger = logging.getLogger(__name__)


class _ModelHolder:
    """Singleton cache so repeated transcribe() calls don't reload the model."""

    _model: Optional[Qwen3ASRModel] = None
    _config: Optional[Qwen3ASRConfig] = None
    _path: Optional[str] = None
    _dtype: Optional[mx.Dtype] = None

    @classmethod
    def get(
        cls,
        path_or_hf_repo: str,
        dtype: mx.Dtype = mx.float16,
    ) -> tuple[Qwen3ASRModel, Qwen3ASRConfig]:
        if cls._model is not None and cls._path == path_or_hf_repo and cls._dtype == dtype:
            return cls._model, cls._config

        model, config = load_model(path_or_hf_repo, dtype=dtype)
        cls._model = model
        cls._config = config
        cls._path = path_or_hf_repo
        cls._dtype = dtype
        return model, config

    @classmethod
    def clear(cls):
        cls._model = None
        cls._config = None
        cls._path = None
        cls._dtype = None


def load_model(
    path_or_hf_repo: str = "Qwen/Qwen3-ASR-1.7B",
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

    # Load weights into model
    model.load_weights(list(weights.items()))

    # Cast to target dtype
    if dtype != mx.float32:
        params = {}
        for k, v in model.parameters().items():
            if isinstance(v, mx.array):
                params[k] = v.astype(dtype)
            elif isinstance(v, dict):
                params[k] = {kk: vv.astype(dtype) if isinstance(vv, mx.array) else vv
                             for kk, vv in v.items()}
            else:
                params[k] = v
        model.load_weights(list(mx.utils.tree_flatten(params)))

    mx.eval(model.parameters())
    model.eval()

    logger.info(f"Loaded model from {model_path} with dtype {dtype}")
    return model, config


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
