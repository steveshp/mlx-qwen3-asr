"""Tests for mlx_qwen3_asr/config.py."""

import pytest

from mlx_qwen3_asr.config import (
    AudioEncoderConfig,
    Qwen3ASRConfig,
    TextDecoderConfig,
)


# ---------------------------------------------------------------------------
# AudioEncoderConfig
# ---------------------------------------------------------------------------


class TestAudioEncoderConfigDefaults:
    """Test default (1.7B) AudioEncoderConfig values."""

    def test_num_mel_bins(self):
        cfg = AudioEncoderConfig()
        assert cfg.num_mel_bins == 128

    def test_encoder_layers(self):
        cfg = AudioEncoderConfig()
        assert cfg.encoder_layers == 32

    def test_encoder_attention_heads(self):
        cfg = AudioEncoderConfig()
        assert cfg.encoder_attention_heads == 20

    def test_encoder_ffn_dim(self):
        cfg = AudioEncoderConfig()
        assert cfg.encoder_ffn_dim == 5120

    def test_d_model(self):
        cfg = AudioEncoderConfig()
        assert cfg.d_model == 1280

    def test_output_dim(self):
        cfg = AudioEncoderConfig()
        assert cfg.output_dim == 3584

    def test_max_source_positions(self):
        cfg = AudioEncoderConfig()
        assert cfg.max_source_positions == 1500

    def test_downsample_hidden_size(self):
        cfg = AudioEncoderConfig()
        assert cfg.downsample_hidden_size == 480

    def test_activation_function(self):
        cfg = AudioEncoderConfig()
        assert cfg.activation_function == "gelu"

    def test_dropout(self):
        cfg = AudioEncoderConfig()
        assert cfg.dropout == 0.0


class TestAudioEncoderConfigFor06B:
    """Test for_0_6b() class method (values from HuggingFace config)."""

    def test_encoder_layers(self):
        cfg = AudioEncoderConfig.for_0_6b()
        assert cfg.encoder_layers == 18

    def test_encoder_attention_heads(self):
        cfg = AudioEncoderConfig.for_0_6b()
        assert cfg.encoder_attention_heads == 14

    def test_encoder_ffn_dim(self):
        cfg = AudioEncoderConfig.for_0_6b()
        assert cfg.encoder_ffn_dim == 3584

    def test_d_model(self):
        cfg = AudioEncoderConfig.for_0_6b()
        assert cfg.d_model == 896

    def test_output_dim(self):
        cfg = AudioEncoderConfig.for_0_6b()
        assert cfg.output_dim == 1024

    def test_downsample_hidden_size(self):
        cfg = AudioEncoderConfig.for_0_6b()
        assert cfg.downsample_hidden_size == 480


class TestAudioEncoderConfigFromDict:
    """Test AudioEncoderConfig.from_dict()."""

    def test_ignores_unknown_keys(self):
        d = {"d_model": 512, "unknown_key": 999, "foo": "bar"}
        cfg = AudioEncoderConfig.from_dict(d)
        assert cfg.d_model == 512
        assert not hasattr(cfg, "unknown_key")

    def test_partial_override(self):
        d = {"encoder_layers": 8}
        cfg = AudioEncoderConfig.from_dict(d)
        assert cfg.encoder_layers == 8
        # Other fields should be defaults
        assert cfg.d_model == 1280


# ---------------------------------------------------------------------------
# TextDecoderConfig
# ---------------------------------------------------------------------------


class TestTextDecoderConfigDefaults:
    """Test default (1.7B) TextDecoderConfig values."""

    def test_vocab_size(self):
        cfg = TextDecoderConfig()
        assert cfg.vocab_size == 151936

    def test_hidden_size(self):
        cfg = TextDecoderConfig()
        assert cfg.hidden_size == 4096

    def test_intermediate_size(self):
        cfg = TextDecoderConfig()
        assert cfg.intermediate_size == 22016

    def test_num_hidden_layers(self):
        cfg = TextDecoderConfig()
        assert cfg.num_hidden_layers == 32

    def test_num_attention_heads(self):
        cfg = TextDecoderConfig()
        assert cfg.num_attention_heads == 32

    def test_num_key_value_heads(self):
        cfg = TextDecoderConfig()
        assert cfg.num_key_value_heads == 32

    def test_1_7b_uses_mha(self):
        """1.7B uses MHA: num_attention_heads == num_key_value_heads."""
        cfg = TextDecoderConfig()
        assert cfg.num_attention_heads == cfg.num_key_value_heads == 32

    def test_head_dim(self):
        cfg = TextDecoderConfig()
        assert cfg.head_dim == 128

    def test_hidden_act(self):
        cfg = TextDecoderConfig()
        assert cfg.hidden_act == "silu"

    def test_rope_theta(self):
        cfg = TextDecoderConfig()
        assert cfg.rope_theta == 5000000.0

    def test_rms_norm_eps(self):
        cfg = TextDecoderConfig()
        assert cfg.rms_norm_eps == 1e-6

    def test_attention_bias_false(self):
        cfg = TextDecoderConfig()
        assert cfg.attention_bias is False

    def test_tie_word_embeddings_false(self):
        cfg = TextDecoderConfig()
        assert cfg.tie_word_embeddings is False


class TestTextDecoderConfigFor06B:
    """Test for_0_6b() class method (values from HuggingFace config)."""

    def test_hidden_size(self):
        cfg = TextDecoderConfig.for_0_6b()
        assert cfg.hidden_size == 1024

    def test_intermediate_size(self):
        cfg = TextDecoderConfig.for_0_6b()
        assert cfg.intermediate_size == 3072

    def test_num_hidden_layers(self):
        cfg = TextDecoderConfig.for_0_6b()
        assert cfg.num_hidden_layers == 28

    def test_num_attention_heads(self):
        cfg = TextDecoderConfig.for_0_6b()
        assert cfg.num_attention_heads == 16

    def test_num_key_value_heads(self):
        cfg = TextDecoderConfig.for_0_6b()
        assert cfg.num_key_value_heads == 8

    def test_0_6b_uses_gqa(self):
        """0.6B uses GQA: 16 attention heads, 8 KV heads."""
        cfg = TextDecoderConfig.for_0_6b()
        assert cfg.num_attention_heads == 16
        assert cfg.num_key_value_heads == 8
        assert cfg.num_attention_heads != cfg.num_key_value_heads

    def test_tie_word_embeddings_true(self):
        cfg = TextDecoderConfig.for_0_6b()
        assert cfg.tie_word_embeddings is True


# ---------------------------------------------------------------------------
# Qwen3ASRConfig
# ---------------------------------------------------------------------------


class TestQwen3ASRConfigFromDict:
    """Test Qwen3ASRConfig.from_dict() with various formats."""

    def test_hf_nested_format(self):
        """HF config.json nests audio/text under thinker_config."""
        d = {
            "thinker_config": {
                "audio_config": {"d_model": 512, "encoder_layers": 8},
                "text_config": {"hidden_size": 2048, "num_hidden_layers": 16},
            },
            "audio_token_id": 99999,
        }
        cfg = Qwen3ASRConfig.from_dict(d)
        assert cfg.audio_config.d_model == 512
        assert cfg.audio_config.encoder_layers == 8
        assert cfg.text_config.hidden_size == 2048
        assert cfg.text_config.num_hidden_layers == 16
        assert cfg.audio_token_id == 99999

    def test_flat_format(self):
        """Flat format with audio_config/text_config at top level."""
        d = {
            "audio_config": {"d_model": 256},
            "text_config": {"hidden_size": 1024},
        }
        cfg = Qwen3ASRConfig.from_dict(d)
        assert cfg.audio_config.d_model == 256
        assert cfg.text_config.hidden_size == 1024

    def test_ignores_unknown_keys_in_nested_configs(self):
        d = {
            "audio_config": {"d_model": 512, "does_not_exist": True},
            "text_config": {"hidden_size": 2048, "also_fake": 42},
        }
        cfg = Qwen3ASRConfig.from_dict(d)
        assert cfg.audio_config.d_model == 512
        assert cfg.text_config.hidden_size == 2048

    def test_defaults_when_keys_missing(self):
        """from_dict with empty dict should use all defaults."""
        cfg = Qwen3ASRConfig.from_dict({})
        assert cfg.audio_token_id == 151646
        assert cfg.audio_start_token_id == 151647
        assert cfg.user_token_id == 872
        assert cfg.audio_config.d_model == 1280
        assert cfg.text_config.hidden_size == 4096

    def test_support_languages(self):
        d = {"support_languages": ["en", "zh"]}
        cfg = Qwen3ASRConfig.from_dict(d)
        assert cfg.support_languages == ["en", "zh"]

    def test_support_languages_default_none(self):
        cfg = Qwen3ASRConfig.from_dict({})
        assert cfg.support_languages is None


class TestQwen3ASRConfigClassMethods:
    """Test for_0_6b() and for_1_7b() class methods."""

    def test_for_1_7b_audio(self):
        cfg = Qwen3ASRConfig.for_1_7b()
        assert cfg.audio_config.d_model == 1280
        assert cfg.audio_config.encoder_layers == 32

    def test_for_1_7b_text(self):
        cfg = Qwen3ASRConfig.for_1_7b()
        assert cfg.text_config.hidden_size == 4096
        assert cfg.text_config.num_hidden_layers == 32

    def test_for_0_6b_audio(self):
        cfg = Qwen3ASRConfig.for_0_6b()
        assert cfg.audio_config.d_model == 896
        assert cfg.audio_config.encoder_layers == 18

    def test_for_0_6b_text(self):
        cfg = Qwen3ASRConfig.for_0_6b()
        assert cfg.text_config.hidden_size == 1024
        assert cfg.text_config.num_hidden_layers == 28
