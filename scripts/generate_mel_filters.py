"""Generate mel_filters.npz asset with 128-bin Slaney-normalized mel filterbank.

Shape: (128, 201) — n_mels=128, n_freqs = n_fft//2 + 1 = 201
Matches librosa/whisper's mel filterbank with Slaney normalization.

Usage:
    python scripts/generate_mel_filters.py
"""

import numpy as np


def mel_filterbank(n_mels: int = 128, n_fft: int = 400, sample_rate: int = 16000) -> np.ndarray:
    """Create Slaney-normalized mel filterbank.

    Args:
        n_mels: Number of mel filter bins.
        n_fft: FFT size.
        sample_rate: Audio sample rate in Hz.

    Returns:
        Filterbank matrix of shape (n_mels, n_fft // 2 + 1).
    """

    def hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    n_freqs = n_fft // 2 + 1  # 201
    all_freqs = np.linspace(0, sample_rate / 2, n_freqs)

    min_mel = hz_to_mel(0)
    max_mel = hz_to_mel(sample_rate / 2)
    mel_points = np.linspace(min_mel, max_mel, n_mels + 2)
    freq_points = mel_to_hz(mel_points)

    filterbank = np.zeros((n_mels, n_freqs))
    for i in range(n_mels):
        lower = freq_points[i]
        center = freq_points[i + 1]
        upper = freq_points[i + 2]

        # Rising slope
        for j in range(n_freqs):
            if lower <= all_freqs[j] <= center:
                filterbank[i, j] = (
                    (all_freqs[j] - lower) / (center - lower) if center != lower else 0
                )
            elif center < all_freqs[j] <= upper:
                filterbank[i, j] = (
                    (upper - all_freqs[j]) / (upper - center) if upper != center else 0
                )

        # Slaney normalization (area normalization)
        enorm = 2.0 / (freq_points[i + 2] - freq_points[i])
        filterbank[i] *= enorm

    return filterbank


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
