"""Generate mel_filters.npz asset matching WhisperFeatureExtractor mel filters.

Shape: (128, 201) — n_mels=128, n_freqs = n_fft//2 + 1 = 201
Matches HuggingFace WhisperFeatureExtractor exactly:
  mel_filter_bank(..., norm="slaney", mel_scale="slaney")

Usage:
    python scripts/generate_mel_filters.py
"""

import numpy as np
from transformers.audio_utils import mel_filter_bank


def mel_filterbank(n_mels: int = 128, n_fft: int = 400, sample_rate: int = 16000) -> np.ndarray:
    """Create Whisper-compatible mel filterbank.

    Returns shape (n_mels, n_fft // 2 + 1), i.e. (128, 201) by default.
    """
    fbank = mel_filter_bank(
        num_frequency_bins=1 + n_fft // 2,
        num_mel_filters=n_mels,
        min_frequency=0.0,
        max_frequency=sample_rate / 2,
        sampling_rate=sample_rate,
        norm="slaney",
        mel_scale="slaney",
    )
    return fbank.T.astype(np.float32, copy=False)


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
