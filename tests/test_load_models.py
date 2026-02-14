"""Tests for mlx_qwen3_asr/load_models.py."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.utils as mlx_utils

from mlx_qwen3_asr.load_models import (
    _cast_tree_dtype,
    _infer_quantization_params,
    _is_quantized_weights,
    _read_quantization_config,
    _resolve_path,
)


class TestCastTreeDtype:
    """Test recursive dtype casting of parameter trees."""

    def test_casts_nested_arrays(self):
        tree = {
            "a": mx.ones((2, 2), dtype=mx.float32),
            "b": {
                "c": [mx.zeros((1,), dtype=mx.float32), {"d": mx.array([3.0])}],
            },
            "int_array": mx.array([1, 2, 3], dtype=mx.int32),
            "name": "keep-me",
        }

        casted = _cast_tree_dtype(tree, mx.float16)
        leaves = mlx_utils.tree_flatten(casted)

        for _, value in leaves:
            if isinstance(value, mx.array):
                if mx.issubdtype(value.dtype, mx.floating):
                    assert value.dtype == mx.float16

        assert casted["int_array"].dtype == mx.int32

        assert casted["name"] == "keep-me"


class TestResolvePath:
    """Test model path resolution logic."""

    def test_uses_local_path_when_config_exists(self, tmp_path: Path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")

        resolved = _resolve_path(str(model_dir))
        assert resolved == model_dir

    def test_downloads_from_hub_for_nonlocal_path(self, monkeypatch):
        expected = "/tmp/fake-model-dir"

        def fake_snapshot_download(repo_id, allow_patterns):  # noqa: ANN001
            assert repo_id == "Qwen/Qwen3-ASR-1.7B"
            assert "*.safetensors" in allow_patterns
            return expected

        monkeypatch.setattr(
            "huggingface_hub.snapshot_download",
            fake_snapshot_download,
        )

        resolved = _resolve_path("Qwen/Qwen3-ASR-1.7B")
        assert resolved == Path(expected)


class _FakeModel:
    def __init__(self):
        self._params = {
            "layer.weight": mx.zeros((4, 64), dtype=mx.float32),
            "other.weight": mx.zeros((8, 128), dtype=mx.float32),
        }

    def parameters(self):
        return self._params


class TestQuantizationHelpers:
    def test_is_quantized_weights(self):
        assert _is_quantized_weights({"a.weight": mx.zeros((1, 1))}) is False
        assert _is_quantized_weights({"a.scales": mx.zeros((1, 1))}) is True

    def test_infer_quantization_params(self):
        # layer.weight input_dim=64, packed_cols=8 -> bits=4
        # layer.scales cols=1 -> group_size=64
        weights = {
            "layer.weight": mx.zeros((4, 8), dtype=mx.uint32),
            "layer.scales": mx.zeros((4, 1), dtype=mx.float16),
            "layer.biases": mx.zeros((4, 1), dtype=mx.float16),
            # Add one noisy candidate that should be ignored for group-size mode.
            "other.weight": mx.zeros((8, 16), dtype=mx.uint32),
            "other.scales": mx.zeros((8, 16), dtype=mx.float16),
            "other.biases": mx.zeros((8, 16), dtype=mx.float16),
        }
        bits, group_size = _infer_quantization_params(weights, _FakeModel())
        assert bits == 4
        assert group_size == 64

    def test_read_quantization_config(self, tmp_path: Path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        cfg_path = model_dir / "quantization_config.json"
        cfg_path.write_text('{"bits": 4, "group_size": 64}', encoding="utf-8")
        cfg = _read_quantization_config(model_dir)
        assert cfg == {"bits": 4, "group_size": 64}
