"""Tests for mlx_qwen3_asr/load_models.py."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.utils as mlx_utils

from mlx_qwen3_asr.config import ACCURACY_MODEL_ID
from mlx_qwen3_asr.load_models import (
    _cast_tree_dtype,
    _infer_quantization_params,
    _is_quantized_weights,
    _ModelHolder,
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
            assert repo_id == ACCURACY_MODEL_ID
            assert "*.safetensors" in allow_patterns
            return expected

        monkeypatch.setattr(
            "huggingface_hub.snapshot_download",
            fake_snapshot_download,
        )

        resolved = _resolve_path(ACCURACY_MODEL_ID)
        assert resolved == Path(expected)


class TestModelHolder:
    def test_get_resolved_path_uses_cached_resolve(self, monkeypatch):
        _ModelHolder.clear()
        sentinel_model = object()
        sentinel_cfg = object()

        def fake_loader(path_or_hf_repo, dtype):  # noqa: ANN001
            assert path_or_hf_repo == ACCURACY_MODEL_ID
            return sentinel_model, sentinel_cfg, Path("/tmp/qwen3-resolved")

        monkeypatch.setattr(
            "mlx_qwen3_asr.load_models._load_model_with_resolved_path",
            fake_loader,
        )

        model, cfg = _ModelHolder.get(ACCURACY_MODEL_ID, dtype=mx.float16)
        assert model is sentinel_model
        assert cfg is sentinel_cfg
        assert _ModelHolder.get_resolved_path(ACCURACY_MODEL_ID, dtype=mx.float16) == (
            "/tmp/qwen3-resolved"
        )

        _ModelHolder.clear()

    def test_caches_multiple_models_without_eviction(self, monkeypatch):
        _ModelHolder.clear()
        calls: list[tuple[str, mx.Dtype]] = []
        store: dict[str, object] = {}

        def fake_loader(path_or_hf_repo, dtype):  # noqa: ANN001
            calls.append((path_or_hf_repo, dtype))
            model = store.setdefault(f"m:{path_or_hf_repo}", object())
            cfg = store.setdefault(f"c:{path_or_hf_repo}", object())
            return model, cfg, Path(f"/tmp/{path_or_hf_repo.replace('/', '_')}")

        monkeypatch.setattr(
            "mlx_qwen3_asr.load_models._load_model_with_resolved_path",
            fake_loader,
        )

        m_a_1, _ = _ModelHolder.get("Qwen/A", dtype=mx.float16)
        m_b_1, _ = _ModelHolder.get("Qwen/B", dtype=mx.float16)
        m_a_2, _ = _ModelHolder.get("Qwen/A", dtype=mx.float16)

        assert m_a_1 is m_a_2
        assert m_a_1 is not m_b_1
        assert calls == [("Qwen/A", mx.float16), ("Qwen/B", mx.float16)]
        assert _ModelHolder.get_resolved_path("Qwen/B", dtype=mx.float16) == "/tmp/Qwen_B"

        _ModelHolder.clear()

    def test_cache_key_isolated_by_dtype(self, monkeypatch):
        _ModelHolder.clear()
        calls: list[tuple[str, mx.Dtype]] = []
        store: dict[tuple[str, str], tuple[object, object]] = {}

        def fake_loader(path_or_hf_repo, dtype):  # noqa: ANN001
            calls.append((path_or_hf_repo, dtype))
            key = (path_or_hf_repo, str(dtype))
            model, cfg = store.setdefault(key, (object(), object()))
            return model, cfg, Path(f"/tmp/{path_or_hf_repo.replace('/', '_')}")

        monkeypatch.setattr(
            "mlx_qwen3_asr.load_models._load_model_with_resolved_path",
            fake_loader,
        )

        m_f16, _ = _ModelHolder.get("Qwen/A", dtype=mx.float16)
        m_f32, _ = _ModelHolder.get("Qwen/A", dtype=mx.float32)
        m_f16_2, _ = _ModelHolder.get("Qwen/A", dtype=mx.float16)

        assert m_f16 is m_f16_2
        assert m_f16 is not m_f32
        assert calls == [("Qwen/A", mx.float16), ("Qwen/A", mx.float32)]

        _ModelHolder.clear()


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

    def test_read_quantization_config_returns_none_on_invalid_json(self, tmp_path: Path, caplog):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        cfg_path = model_dir / "quantization_config.json"
        cfg_path.write_text("{not-json", encoding="utf-8")

        cfg = _read_quantization_config(model_dir)
        assert cfg is None
        assert "Failed to parse quantization metadata" in caplog.text
