"""Generate mel_filters.npz asset for native Whisper-compatible mel frontend.

Shape: (128, 201) by default (n_mels=128, n_freqs=n_fft//2 + 1).
Implements Slaney mel-scale + Slaney area normalization in pure numpy.

Usage:
    python scripts/generate_mel_filters.py
"""

import numpy as np


def _hz_to_mel_slaney(hz: np.ndarray) -> np.ndarray:
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp  # 15
    logstep = np.log(6.4) / 27.0
    out = hz / f_sp
    mask = hz >= min_log_hz
    out[mask] = min_log_mel + np.log(hz[mask] / min_log_hz) / logstep
    return out


def _mel_to_hz_slaney(mel: np.ndarray) -> np.ndarray:
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp  # 15
    logstep = np.log(6.4) / 27.0
    out = mel * f_sp
    mask = mel >= min_log_mel
    out[mask] = min_log_hz * np.exp(logstep * (mel[mask] - min_log_mel))
    return out


def mel_filterbank(
    n_mels: int = 128,
    n_fft: int = 400,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Create Whisper-compatible Slaney-normalized mel filterbank.

    Returns shape (n_mels, n_fft // 2 + 1), i.e. (128, 201) by default.
    """
    n_freqs = 1 + n_fft // 2
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_freqs, dtype=np.float64)

    min_mel = _hz_to_mel_slaney(np.array([0.0], dtype=np.float64))[0]
    max_mel = _hz_to_mel_slaney(np.array([sample_rate / 2.0], dtype=np.float64))[0]
    mel_points = np.linspace(min_mel, max_mel, n_mels + 2, dtype=np.float64)
    hz_points = _mel_to_hz_slaney(mel_points)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float64)
    for i in range(n_mels):
        lower = hz_points[i]
        center = hz_points[i + 1]
        upper = hz_points[i + 2]
        if center <= lower or upper <= center:
            continue

        left = (fft_freqs - lower) / (center - lower)
        right = (upper - fft_freqs) / (upper - center)
        fb[i] = np.maximum(0.0, np.minimum(left, right))

        # Slaney area normalization.
        fb[i] *= 2.0 / (upper - lower)

    return fb.astype(np.float32, copy=False)


if __name__ == "__main__":
    import os

    filters = mel_filterbank()
    print(f"Filterbank shape: {filters.shape}")
    assert filters.shape == (128, 201), f"Expected (128, 201), got {filters.shape}"

    out_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "mlx_qwen3_asr",
        "assets",
        "mel_filters.npz",
    )
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, mel_128=filters)
    print(f"Saved to {out_path}")

    # Verify
    loaded = np.load(out_path)
    assert "mel_128" in loaded, "Key 'mel_128' not found"
    assert loaded["mel_128"].shape == (128, 201), f"Verification failed: {loaded['mel_128'].shape}"
    print("Verification passed.")
