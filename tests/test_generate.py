"""Tests for mlx_qwen3_asr/generate.py."""

import mlx.core as mx

from mlx_qwen3_asr.generate import (
    REPETITION_THRESHOLD,
    GenerationConfig,
    _detect_repetition,
    _sample,
    generate,
)

# ---------------------------------------------------------------------------
# GenerationConfig
# ---------------------------------------------------------------------------


class TestGenerationConfig:
    """Test GenerationConfig defaults."""

    def test_default_max_new_tokens(self):
        cfg = GenerationConfig()
        assert cfg.max_new_tokens == 1024

    def test_default_temperature(self):
        cfg = GenerationConfig()
        assert cfg.temperature == 0.0

    def test_default_eos_token_ids(self):
        cfg = GenerationConfig()
        assert cfg.eos_token_ids == [151643, 151645]

    def test_default_eval_interval(self):
        cfg = GenerationConfig()
        assert cfg.eval_interval == 1


# ---------------------------------------------------------------------------
# _detect_repetition
# ---------------------------------------------------------------------------


class TestDetectRepetition:
    """Test _detect_repetition() logic."""

    def test_short_sequence_returns_false(self):
        """Sequences shorter than threshold should return False."""
        tokens = [1, 2, 3, 4, 5]
        assert _detect_repetition(tokens) is False

    def test_no_repetition(self):
        """Varied tokens should not trigger repetition."""
        tokens = list(range(REPETITION_THRESHOLD + 10))
        assert _detect_repetition(tokens) is False

    def test_single_token_repeated(self):
        """A single token repeated >= threshold times should be detected."""
        tokens = [42] * (REPETITION_THRESHOLD + 5)
        assert _detect_repetition(tokens) is True

    def test_single_token_just_at_threshold(self):
        tokens = [42] * REPETITION_THRESHOLD
        assert _detect_repetition(tokens) is True

    def test_single_token_below_threshold(self):
        """Single token repeated fewer than threshold times (but sequence >= threshold).
        Must ensure no pattern detection triggers either."""
        # Use enough varied tokens, then repeat below both single and pattern thresholds.
        # Pattern threshold for len-2 is max(2, 20//2) = 10, so keep repeats < 10.
        varied = list(range(REPETITION_THRESHOLD))
        tokens = varied + [42] * 5  # 5 consecutive repeats, well below 20
        assert _detect_repetition(tokens) is False

    def test_pattern_repetition(self):
        """Pattern of 2+ tokens repeated many times should be detected."""
        pattern = [10, 20]
        # threshold // pattern_len = 20 // 2 = 10, need at least max(2, 10) = 10
        tokens = pattern * 12  # 12 repetitions, length 24 >= 20
        assert _detect_repetition(tokens) is True

    def test_pattern_3_tokens(self):
        """Pattern of 3 tokens repeated enough times."""
        pattern = [10, 20, 30]
        # threshold // 3 = 6 (rounded down), need >= max(2, 6) = 6
        # Use 8 repetitions, length = 24 >= 20
        tokens = pattern * 8
        assert _detect_repetition(tokens) is True

    def test_pattern_not_enough_repeats(self):
        """Pattern repeated fewer times than needed should return False."""
        pattern = [10, 20]
        # Only 3 repetitions, well below threshold
        varied = list(range(100, 120))
        tokens = varied + pattern * 3
        assert _detect_repetition(tokens) is False

    def test_empty_list(self):
        assert _detect_repetition([]) is False

    def test_exactly_threshold_length_no_repetition(self):
        tokens = list(range(REPETITION_THRESHOLD))
        assert _detect_repetition(tokens) is False


# ---------------------------------------------------------------------------
# _sample
# ---------------------------------------------------------------------------


class TestSample:
    """Test _sample() sampling logic."""

    def test_greedy_returns_argmax(self):
        """Temperature 0.0 (greedy) should return the argmax index."""
        logits = mx.array([[[0.1, 0.5, 0.3, 0.9, 0.2]]])
        token = _sample(logits, temperature=0.0)
        assert token == 3  # index of 0.9

    def test_greedy_with_negative_values(self):
        logits = mx.array([[[-10.0, -5.0, -1.0, -0.5, -2.0]]])
        token = _sample(logits, temperature=0.0)
        assert token == 3  # index of -0.5

    def test_greedy_with_2d_input(self):
        """_sample reshapes to 1D, so 2D input should work."""
        logits = mx.array([[0.1, 0.5, 0.9, 0.3]])
        token = _sample(logits, temperature=0.0)
        assert token == 2  # index of 0.9

    def test_temperature_sampling_returns_valid_index(self):
        """Temperature > 0 should return a valid index within vocab size."""
        vocab_size = 100
        logits = mx.random.normal((1, 1, vocab_size))
        token = _sample(logits, temperature=1.0)
        assert 0 <= token < vocab_size

    def test_negative_temperature_acts_as_greedy(self):
        """Negative temperature should act as greedy (temperature <= 0.0)."""
        logits = mx.array([[[0.1, 0.5, 0.9, 0.3]]])
        token = _sample(logits, temperature=-1.0)
        assert token == 2


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


class TestGenerate:
    """Test top-level generate() orchestration."""

    def test_generate_uses_model_prefill_and_step_interfaces(self):
        class _DummyModel:
            def __init__(self):
                self.calls = []
                self.cache_obj = object()

            def create_cache(self, max_seq_len=None):  # noqa: ANN001
                self.calls.append(("create_cache", max_seq_len))
                return self.cache_obj

            def prefill(self, input_ids, audio_features, position_ids, cache):  # noqa: ANN001
                self.calls.append(
                    (
                        "prefill",
                        tuple(input_ids.shape),
                        tuple(audio_features.shape),
                        tuple(position_ids.shape),
                        cache is self.cache_obj,
                    )
                )
                # greedy -> token 1
                return mx.array([[[0.0, 1.0, 0.0]]], dtype=mx.float32)

            def step(self, input_ids, position_ids, cache):  # noqa: ANN001
                self.calls.append(
                    (
                        "step",
                        tuple(input_ids.shape),
                        tuple(position_ids.shape),
                        cache is self.cache_obj,
                    )
                )
                # greedy -> eos token 2
                return mx.array([[[0.0, 0.0, 1.0]]], dtype=mx.float32)

        model = _DummyModel()
        input_ids = mx.array([[10, 20, 30, 40, 50]])
        audio_features = mx.zeros((1, 8, 4))
        position_ids = mx.zeros((1, 3, 5), dtype=mx.int32)
        config = GenerationConfig(max_new_tokens=3, temperature=0.0, eos_token_ids=[2])

        out = generate(
            model=model,
            input_ids=input_ids,
            audio_features=audio_features,
            position_ids=position_ids,
            config=config,
        )

        assert out == [1]
        assert model.calls[0] == ("create_cache", 8)
        assert model.calls[1][0] == "prefill"
        assert model.calls[2][0] == "step"
