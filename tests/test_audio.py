"""Tests for mlx_qwen3_asr/audio.py."""

import struct
import subprocess
import wave
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import mlx_qwen3_asr.audio as audio_mod
from mlx_qwen3_asr.audio import (
    N_FFT,
    SAMPLE_RATE,
    _reflect_pad,
    compute_features,
    load_audio,
    log_mel_spectrogram,
    mel_filters,
    stft,
)

# ---------------------------------------------------------------------------
# load_audio
# ---------------------------------------------------------------------------


class TestLoadAudioNumpy:
    """Test load_audio() with numpy array inputs."""

    def test_mono_array(self):
        arr = np.random.randn(16000).astype(np.float32)
        result = load_audio(arr)
        assert isinstance(result, mx.array)
        assert result.shape == (16000,)

    def test_stereo_to_mono(self):
        """Stereo (2-channel) should be averaged to mono."""
        stereo = np.random.randn(16000, 2).astype(np.float32)
        result = load_audio(stereo)
        assert result.ndim == 1
        assert result.shape == (16000,)
        # Check that it's the mean of the two channels
        expected = stereo.mean(axis=-1)
        np.testing.assert_allclose(np.array(result), expected, atol=1e-5)

    def test_empty_array(self):
        arr = np.array([], dtype=np.float32)
        result = load_audio(arr)
        assert result.size == 0

    def test_integer_array_cast_to_float(self):
        """Integer arrays should be cast to float32."""
        arr = np.array([100, 200, 300], dtype=np.int16)
        result = load_audio(arr)
        assert result.dtype == mx.float32


class TestLoadAudioTuple:
    """Test load_audio() with (array, sample_rate) tuple inputs."""

    def test_same_sample_rate(self):
        arr = np.random.randn(16000).astype(np.float32)
        result = load_audio((arr, SAMPLE_RATE))
        assert result.shape == (16000,)

    def test_empty_tuple(self):
        arr = np.array([], dtype=np.float32)
        result = load_audio((arr, SAMPLE_RATE))
        assert result.size == 0

    def test_stereo_tuple(self):
        stereo = np.random.randn(16000, 2).astype(np.float32)
        result = load_audio((stereo, SAMPLE_RATE))
        assert result.ndim == 1


class TestLoadAudioErrors:
    """Test load_audio() error handling."""

    def test_unsupported_type_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported source type"):
            load_audio(12345)

    def test_unsupported_type_list_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported source type"):
            load_audio([1, 2, 3])


class TestLoadAudioWavFastPath:
    """WAV loader fast-path behavior."""

    def _write_wav(
        self,
        path: Path,
        samples: np.ndarray,
        sr: int = SAMPLE_RATE,
        channels: int = 1,
    ) -> None:
        samples = np.asarray(samples, dtype=np.float32)
        if channels > 1:
            samples = samples.reshape(-1, channels)
            flat = samples.reshape(-1)
        else:
            flat = samples
        pcm = (np.clip(flat, -1.0, 1.0) * 32767.0).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())

    def _write_wav_float32(
        self,
        path: Path,
        samples: np.ndarray,
        sr: int = SAMPLE_RATE,
        channels: int = 1,
    ) -> None:
        samples = np.asarray(samples, dtype=np.float32)
        if channels > 1:
            samples = samples.reshape(-1, channels)
            flat = samples.reshape(-1)
        else:
            flat = samples
        payload = flat.astype("<f4").tobytes()
        byte_rate = sr * channels * 4
        block_align = channels * 4
        fmt_chunk = struct.pack(
            "<HHIIHH",
            3,  # IEEE float
            channels,
            sr,
            byte_rate,
            block_align,
            32,
        )
        riff_size = 4 + (8 + len(fmt_chunk)) + (8 + len(payload))
        with open(path, "wb") as f:
            f.write(b"RIFF")
            f.write(struct.pack("<I", riff_size))
            f.write(b"WAVE")
            f.write(b"fmt ")
            f.write(struct.pack("<I", len(fmt_chunk)))
            f.write(fmt_chunk)
            f.write(b"data")
            f.write(struct.pack("<I", len(payload)))
            f.write(payload)

    def test_wav_fast_path_avoids_ffmpeg(self, tmp_path: Path, monkeypatch):
        wav_path = tmp_path / "mono.wav"
        src = np.linspace(-0.5, 0.5, SAMPLE_RATE, dtype=np.float32)
        self._write_wav(wav_path, src, sr=SAMPLE_RATE, channels=1)

        def fail_run(*args, **kwargs):  # noqa: ANN001
            raise AssertionError("ffmpeg subprocess should not be used for supported WAV")

        monkeypatch.setattr(subprocess, "run", fail_run)

        audio = np.array(load_audio(str(wav_path)))
        assert audio.ndim == 1
        assert len(audio) == SAMPLE_RATE
        np.testing.assert_allclose(audio[:200], src[:200], atol=1e-3)

    def test_wav_fast_path_stereo_to_mono(self, tmp_path: Path):
        wav_path = tmp_path / "stereo.wav"
        left = np.linspace(-1.0, 1.0, SAMPLE_RATE, dtype=np.float32)
        right = np.linspace(1.0, -1.0, SAMPLE_RATE, dtype=np.float32)
        stereo = np.stack([left, right], axis=1)
        self._write_wav(wav_path, stereo, sr=SAMPLE_RATE, channels=2)

        audio = np.array(load_audio(str(wav_path)))
        expected = stereo.mean(axis=1)
        np.testing.assert_allclose(audio, expected, atol=1e-3)

    def test_wav_fast_path_resamples_when_needed(self, tmp_path: Path, monkeypatch):
        wav_path = tmp_path / "resample.wav"
        src = np.linspace(-0.5, 0.5, 8000, dtype=np.float32)
        self._write_wav(wav_path, src, sr=8000, channels=1)

        called = []

        def fake_resample(audio: np.ndarray, orig_sr: int, target_sr: int):  # noqa: ANN001
            called.append((len(audio), orig_sr, target_sr))
            return np.zeros(16000, dtype=np.float32)

        monkeypatch.setattr(audio_mod, "_resample_via_ffmpeg", fake_resample)
        out = np.array(load_audio(str(wav_path), sr=16000))
        assert called == [(8000, 8000, 16000)]
        assert out.shape == (16000,)

    def test_wav_float32_fast_path(self, tmp_path: Path, monkeypatch):
        wav_path = tmp_path / "float32.wav"
        src = np.linspace(-0.8, 0.8, SAMPLE_RATE, dtype=np.float32)
        self._write_wav_float32(wav_path, src, sr=SAMPLE_RATE, channels=1)

        def fail_run(*args, **kwargs):  # noqa: ANN001
            raise AssertionError("ffmpeg subprocess should not be used for float WAV")

        monkeypatch.setattr(subprocess, "run", fail_run)

        audio = np.array(load_audio(str(wav_path)))
        np.testing.assert_allclose(audio, src, atol=1e-6)


# ---------------------------------------------------------------------------
# _reflect_pad
# ---------------------------------------------------------------------------


class TestReflectPad:
    """Test _reflect_pad() with known inputs."""

    def test_known_values(self):
        x = mx.array([1.0, 2.0, 3.0, 4.0])
        result = _reflect_pad(x, pad_len=2)
        # Reflect: [3, 2, 1, 2, 3, 4, 3, 2]
        expected = [3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]
        np.testing.assert_array_equal(np.array(result), expected)

    def test_zero_pad(self):
        x = mx.array([1.0, 2.0, 3.0])
        result = _reflect_pad(x, pad_len=0)
        np.testing.assert_array_equal(np.array(result), [1.0, 2.0, 3.0])

    def test_pad_len_one(self):
        x = mx.array([10.0, 20.0, 30.0])
        result = _reflect_pad(x, pad_len=1)
        # Left pad: x[1:2][::-1] = [20]
        # Right pad: x[-2:-1][::-1] = [20]
        expected = [20.0, 10.0, 20.0, 30.0, 20.0]
        np.testing.assert_array_equal(np.array(result), expected)

    def test_output_length(self):
        x = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pad_len = 3
        result = _reflect_pad(x, pad_len)
        assert result.shape[0] == len(np.array(x)) + 2 * pad_len


# ---------------------------------------------------------------------------
# mel_filters
# ---------------------------------------------------------------------------


class TestMelFilters:
    """Test mel_filters() loads and returns correct shape."""

    def test_default_shape(self):
        filters = mel_filters()
        assert filters.shape == (128, 201)

    def test_explicit_128(self):
        filters = mel_filters(n_mels=128)
        assert filters.shape == (128, 201)

    def test_dtype_is_float(self):
        filters = mel_filters()
        assert filters.dtype == mx.float32


# ---------------------------------------------------------------------------
# stft
# ---------------------------------------------------------------------------


class TestSTFT:
    """Test stft() output shape for known input length."""

    def test_output_shape(self):
        # n_samples = 1600 (0.1s at 16kHz)
        audio = mx.zeros((1600,))
        window = mx.array(np.hanning(N_FFT + 1)[:-1].astype(np.float32))
        result = stft(audio, window, nperseg=N_FFT)
        # After reflect padding: 1600 + 2*(400//2) = 1600 + 400 = 2000
        # num_frames = 1 + (2000 - 400) // 160 = 1 + 1600//160 = 1 + 10 = 11
        assert result.shape[0] == 11
        # rfft output: N_FFT // 2 + 1 = 201
        assert result.shape[1] == N_FFT // 2 + 1

    def test_output_shape_longer_audio(self):
        audio = mx.zeros((16000,))  # 1 second
        window = mx.array(np.hanning(N_FFT + 1)[:-1].astype(np.float32))
        result = stft(audio, window, nperseg=N_FFT)
        # After padding: 16000 + 400 = 16400
        # num_frames = 1 + (16400 - 400) // 160 = 1 + 16000//160 = 1 + 100 = 101
        assert result.shape[0] == 101
        assert result.shape[1] == 201


# ---------------------------------------------------------------------------
# log_mel_spectrogram
# ---------------------------------------------------------------------------


class TestLogMelSpectrogram:
    """Test log_mel_spectrogram() output shape and error handling."""

    def test_output_shape_for_known_length(self):
        # 1 second of audio
        audio = mx.array(np.random.randn(16000).astype(np.float32))
        result = log_mel_spectrogram(audio)
        assert result.shape[0] == 128  # n_mels
        # STFT yields 101 frames, Whisper-compatible mel path trims final frame.
        assert result.shape[1] == 100

    def test_output_shape_short_audio(self):
        # 0.1 second of audio
        audio = mx.array(np.random.randn(1600).astype(np.float32))
        result = log_mel_spectrogram(audio)
        assert result.shape[0] == 128
        # STFT yields 11 frames, Whisper-compatible mel path trims final frame.
        assert result.shape[1] == 10

    def test_raises_for_empty_audio(self):
        audio = mx.array(np.array([], dtype=np.float32))
        with pytest.raises(ValueError, match="empty audio"):
            log_mel_spectrogram(audio)

    def test_output_dtype(self):
        audio = mx.array(np.random.randn(1600).astype(np.float32))
        result = log_mel_spectrogram(audio)
        assert result.dtype == mx.float32


# ---------------------------------------------------------------------------
# compute_features
# ---------------------------------------------------------------------------


class TestComputeFeatures:
    """Test compute_features() shape/length behavior."""

    def test_default_no_padding(self):
        audio = np.random.randn(2 * SAMPLE_RATE).astype(np.float32)
        mel, feature_lens = compute_features(audio)
        assert mel.shape == (1, 128, 200)
        assert int(feature_lens.item()) == 200

    def test_long_audio_not_truncated(self):
        # 31s should produce 3100 frames with 10ms hop, not 3000.
        audio = np.random.randn(31 * SAMPLE_RATE).astype(np.float32)
        mel, feature_lens = compute_features(audio)
        assert mel.shape == (1, 128, 3100)
        assert int(feature_lens.item()) == 3100

    def test_max_length_padding_still_keeps_long_audio(self):
        # With truncation disabled, padding="max_length" should not clamp >30s.
        audio = np.random.randn(31 * SAMPLE_RATE).astype(np.float32)
        mel, feature_lens = compute_features(audio, padding="max_length")
        assert mel.shape == (1, 128, 3100)
        assert int(feature_lens.item()) == 3100

    def test_max_length_padding_tracks_true_length_for_short_audio(self):
        # For short inputs, max_length pads mel width but feature_lens should
        # still report the true unpadded frame count.
        audio = np.random.randn(2 * SAMPLE_RATE).astype(np.float32)
        mel, feature_lens = compute_features(audio, padding="max_length")
        assert mel.shape == (1, 128, 3000)
        assert int(feature_lens.item()) == 200

    def test_do_not_pad_path_skips_attention_mask(self, monkeypatch):
        def fail_extractor(_sr):  # noqa: ANN001
            raise AssertionError("HF extractor should not be used for do_not_pad")

        monkeypatch.setattr(audio_mod, "_get_feature_extractor", fail_extractor)

        mel, feature_lens = compute_features(np.zeros(3200, dtype=np.float32))
        assert mel.shape == (1, 128, 20)
        assert int(feature_lens.item()) == 20

    def test_padded_path_uses_attention_mask_for_true_length(self, monkeypatch):
        seen = {}

        def fake_extractor(audio_np, **kwargs):  # noqa: ANN001
            seen.update(kwargs)
            attn = np.zeros((1, 3000), dtype=np.int64)
            attn[0, :200] = 1
            return {
                "input_features": np.zeros((1, 128, 3000), dtype=np.float32),
                "attention_mask": attn,
            }

        monkeypatch.setattr(audio_mod, "_get_feature_extractor", lambda _sr: fake_extractor)

        mel, feature_lens = compute_features(
            np.zeros(1600, dtype=np.float32),
            padding="max_length",
        )
        assert seen["return_attention_mask"] is True
        assert mel.shape == (1, 128, 3000)
        assert int(feature_lens.item()) == 200

    def test_custom_mel_parity_with_hf_reference(self):
        audio = np.random.randn(2 * SAMPLE_RATE).astype(np.float32)
        mel_custom, lens_custom = compute_features(audio, padding="do_not_pad")
        mel_hf, lens_hf = audio_mod._compute_features_hf(  # noqa: SLF001
            audio,
            SAMPLE_RATE,
            "do_not_pad",
        )

        assert int(lens_custom.item()) == int(lens_hf.item())
        np.testing.assert_allclose(
            np.array(mel_custom),
            np.array(mel_hf),
            atol=1e-5,
            rtol=1e-5,
        )
