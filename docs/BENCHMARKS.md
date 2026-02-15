# Benchmarks

Comprehensive benchmark results for mlx-qwen3-asr on Apple Silicon. All numbers measured on Apple M4 Pro, macOS 26.2. Every result has a committed JSON artifact under `docs/benchmarks/` for reproducibility.

## Summary

| Metric | 0.6B fp16 | 0.6B 4-bit | 0.6B 8-bit | 1.7B fp16 |
|---|---|---|---|---|
| LibriSpeech test-clean WER | 2.29% | 2.72% | 2.33% | — |
| LibriSpeech test-other WER | 4.20% | — | — | — |
| Multilingual primary | 9.37% | — | — | 6.70% |
| Short clip latency (~2.5s) | 0.46s | 0.13s | 0.11s | — |
| 10s clip latency | 0.83s | 0.18s | 0.27s | — |
| Multilingual mean latency | 1.44s | — | — | 4.12s |
| MLX vs PyTorch speed (long-form) | 4.19x faster | — | — | — |

---

## English Quality (0.6B fp16, LibriSpeech, 100 samples/subset)

| Subset | WER | CER | Mean Latency | RTF |
|---|---:|---:|---:|---:|
| test-clean | 2.29% | 0.59% | 0.86s | 0.0957 |
| test-other | 4.20% | 2.09% | 0.71s | 0.0985 |

Artifacts:
- `2026-02-15-librispeech-test-clean-100.json`
- `2026-02-15-librispeech-test-other-100.json`

---

## Quantization Quality (0.6B, LibriSpeech test-clean)

100 speaker-balanced samples, round-robin sampling across speakers.

| Configuration | WER | CER | Mean Latency | Real-Time Factor | vs fp16 Speed |
|---|---:|---:|---:|---:|---:|
| fp16 (baseline) | 2.29% | 0.59% | 1.09s | 0.121 | — |
| 8-bit (g64) | 2.33% | 0.59% | 0.34s | 0.038 | **3.11x** |
| 4-bit (g64) | 2.72% | 0.88% | 0.30s | 0.034 | **4.68x** |

8-bit is near-fp16 quality (+0.04pp WER). 4-bit trades +0.43pp WER for maximum speed.

Artifact: `2026-02-14-quant-matrix-speaker100.md`

---

## Latency (0.6B)

| Configuration | Short (~2.5s) | 10s clip | Real-Time Factor |
|---|---:|---:|---:|
| fp16 | 0.46s | 0.83s | 0.08x |
| 8-bit (g64) | 0.11s | 0.27s | 0.03x |
| 4-bit (g64) | 0.13s | 0.18s | 0.02x |

Artifacts: `2026-02-14-quant-matrix-speaker100.json`, `2026-02-14-quant-matrix-post-wavfast.json`

---

## Multilingual Quality (FLEURS, 10 languages x 10 samples)

Primary metric rule: CER for Chinese/Japanese/Korean; WER for all others.

### 0.6B (fp16)

| Language | Samples | WER | CER | Primary | Latency |
|---|---:|---:|---:|---:|---:|
| Arabic | 10 | 21.5% | 7.0% | 21.5% | 1.30s |
| Chinese | 10 | 90.0% | 4.4% | 4.4% | 0.81s |
| English | 10 | 4.6% | 1.9% | 4.6% | 0.81s |
| French | 10 | 18.2% | 9.2% | 18.2% | 1.14s |
| German | 10 | 8.0% | 4.7% | 8.0% | 1.38s |
| Hindi | 10 | 16.7% | 9.8% | 16.7% | 3.80s |
| Japanese | 10 | 82.1% | 8.5% | 8.5% | 1.23s |
| Korean | 10 | 17.2% | 6.7% | 6.7% | 1.17s |
| Russian | 10 | 8.8% | 3.4% | 8.8% | 1.47s |
| Spanish | 10 | 3.0% | 0.6% | 3.0% | 1.31s |
| **Aggregate** | **100** | **15.9%** | **5.4%** | **9.37%** | **1.44s** |

Artifact: `2026-02-15-manifest-quality-multilingual100-0p6b-refresh.json`

### 1.7B (fp16)

| Language | Samples | Primary | Latency |
|---|---:|---:|---:|
| Arabic | 10 | 16.5% | 3.78s |
| Chinese | 10 | 8.5% | 2.38s |
| English | 10 | 4.2% | 2.22s |
| French | 10 | 4.1% | 3.25s |
| German | 10 | 5.8% | 3.74s |
| Hindi | 10 | 17.4% | 11.06s |
| Japanese | 10 | 3.6% | 3.43s |
| Korean | 10 | 5.5% | 3.25s |
| Russian | 10 | 4.9% | 4.23s |
| Spanish | 10 | 0.7% | 3.91s |
| **Aggregate** | **100** | **6.70%** | **4.12s** |

Artifact: `2026-02-15-manifest-quality-multilingual100-1p7b-refresh.json`

### 0.6B vs 1.7B Comparison

| Language | 0.6B Primary | 1.7B Primary | Delta | Latency Ratio |
|---|---:|---:|---:|---:|
| Arabic | 21.5% | 16.5% | -5.0pp | 2.90x |
| Chinese | 4.4% | 8.5% | +4.1pp* | 2.94x |
| English | 4.6% | 4.2% | -0.5pp | 2.74x |
| French | 18.2% | 4.1% | **-14.1pp** | 2.85x |
| German | 8.0% | 5.8% | -2.2pp | 2.70x |
| Hindi | 16.7% | 17.4% | +0.7pp | 2.91x |
| Japanese | 8.5% | 3.6% | -4.9pp | 2.78x |
| Korean | 6.7% | 5.5% | -1.2pp | 2.79x |
| Russian | 8.8% | 4.9% | -3.9pp | 2.88x |
| Spanish | 3.0% | 0.7% | -2.2pp | 2.98x |
| **Overall** | **9.37%** | **6.70%** | **-2.66pp** | **2.86x** |

*Chinese +4.1pp is a numeric surface form artifact — the 1.7B spells out numbers in Chinese characters (e.g., `二十九` instead of `29`) while ground truth uses Arabic numerals. Both are correct. A number-aware normalizer would eliminate this gap.

**Takeaway:** 1.7B is the quality choice (28% relative improvement). 0.6B is the speed choice (2.86x faster). The biggest 1.7B wins are on French (-14.1pp), Arabic (-5.0pp), and Japanese (-4.9pp).

---

## Long-Form Quality (FLEURS concatenated, ~75-90s per clip)

0.6B model, 10 clips (one per language).

| Language | WER | CER | Primary | Latency |
|---|---:|---:|---:|---:|
| Arabic | 28.1% | 10.0% | 28.1% | 14.4s |
| Chinese | 47.6% | 1.8% | 1.8% | 6.6s |
| English | 8.0% | 2.9% | 8.0% | 8.7s |
| French | 13.9% | 6.0% | 13.9% | 11.8s |
| German | 6.1% | 2.2% | 6.1% | 12.0s |
| Hindi | 32.4% | 28.0% | 32.4% | 25.8s |
| Japanese | 89.5% | 10.9% | 10.9% | 7.2s |
| Korean | 10.3% | 4.0% | 4.0% | 7.6s |
| Russian | 15.2% | 4.7% | 15.2% | 9.5s |
| Spanish | 3.7% | 0.7% | 3.7% | 7.8s |
| **Aggregate** | **16.7%** | **7.0%** | **11.6%** | **11.1s** |

Quality is consistent with short-clip results. No long-audio truncation or chunking artifacts.

Artifact: `2026-02-15-manifest-quality-longform10.md`

---

## MLX vs PyTorch Parity (0.6B, Multilingual-100)

Head-to-head quality comparison: same audio, same model weights, greedy decode.

| Metric | MLX | PyTorch | Delta (MLX - Ref) |
|---|---:|---:|---:|
| WER | 16.00% | 16.69% | -0.70pp |
| CER | 5.43% | 5.64% | -0.21pp |
| Primary | 9.54% | 10.34% | **-0.81pp** |

MLX slightly outperforms PyTorch on aggregate — likely due to minor floating-point path differences that happen to favor MLX on this sample set.

### Per-Language Breakdown

| Language | MLX Primary | PyTorch Primary | Delta |
|---|---:|---:|---:|
| Arabic | 21.5% | 24.0% | -2.5pp |
| Chinese | 5.0% | 5.5% | -0.6pp |
| English | 4.6% | 6.0% | -1.4pp |
| French | 17.3% | 15.5% | +1.8pp |
| German | 8.0% | 6.7% | +1.3pp |
| Hindi | 16.7% | 21.2% | -4.5pp |
| Japanese | 9.3% | 10.8% | -1.5pp |
| Korean | 6.8% | 6.5% | +0.2pp |
| Russian | 8.8% | 9.3% | -0.5pp |
| Spanish | 3.0% | 2.6% | +0.4pp |

No language shows a systematic quality gap. Differences are within expected noise for n=10 per language.

Artifact: `2026-02-15-quality-head2head-mlx-vs-pytorch-multilingual100.md`

### Token-Level Parity Analysis

| Category | Count |
|---|---:|
| Exact match | 64 |
| Minor lexical shift | 26 |
| Numeric surface form | 5 |
| Punctuation/tokenization | 3 |
| Content shift | 2 |

67% of samples produce identical text. The remaining 33% differ in minor ways — synonym choices, number formatting (`10,000` vs `zehntausend`), or punctuation. Only 2% show meaningful content differences.

Artifact: `2026-02-15-reference-parity-suite-multilingual100-post-refactor-analysis.md`

### Long-Form Speed

| Metric | MLX | PyTorch | Ratio |
|---|---:|---:|---:|
| Mean latency (75-90s clips) | 11.55s | 48.39s | **4.19x faster** |

MLX on Apple Silicon is over 4x faster than PyTorch on the same machine for long-form inference.

Artifact: `2026-02-15-reference-parity-suite-longform10.md`

---

## Forced Aligner Parity (LibriSpeech, English)

MLX native aligner vs official `qwen-asr` PyTorch backend, 50 samples.

| Metric | Value |
|---|---:|
| Text match rate | 100% |
| Timing MAE (all boundaries) | 5.69 ms |
| MLX mean latency | 0.21s |
| Official backend mean latency | 0.56s |
| Relative speed | **2.64x faster** |

The MLX aligner produces identical text and <6ms timing deviation while running 2.64x faster than the PyTorch backend.

Artifact: `2026-02-14-aligner-parity-50.md`

---

## Mel Spectrogram Parity

MLX custom mel vs HuggingFace `WhisperFeatureExtractor(128)`.

| Metric | Value |
|---|---:|
| Max MAE | 2.83e-07 |
| Max absolute diff | 1.13e-04 |
| Frame length match | Exact |

Artifact: `2026-02-14-mel-parity.md`

---

## Optimization Impact

### Encoder Windowing (hybrid dense/segmented)

| Sequence Length | Windows | Dense (s) | Windowed (s) | Speedup |
|---:|---:|---:|---:|---:|
| 832 | 8 | 0.009 | 0.013 | 0.74x |
| 2,080 | 20 | 0.055 | 0.047 | 1.17x |
| 3,120 | 30 | 0.110 | 0.074 | 1.48x |
| 8,320 | 80 | 0.799 | 0.191 | **4.17x** |

Threshold: segmented execution kicks in at 20+ windows. Short audio uses dense masks (no regression), long audio gets up to 4.2x speedup.

Artifact: `2026-02-14-encoder-windowing-threshold.md`

### WAV Fast-Path Loader

| Scenario | Before | After | Improvement |
|---|---:|---:|---:|
| fp16 short | 0.545s | 0.506s | 7.2% faster |
| 4-bit short | 0.157s | 0.118s | **24.8% faster** |
| 4-bit 10s | 0.249s | 0.226s | 9.4% faster |

Biggest impact on quantized short clips where audio loading is a larger fraction of total time.

Artifact: `2026-02-14-wav-fastpath.md`

### Streaming (Rolling Decode)

| Metric | Value |
|---|---:|
| Total latency (2.53s audio) | 1.13s |
| Per-chunk mean latency | 0.27s |
| Per-chunk p95 latency | 0.51s |
| Real-time factor | 0.45 |

Artifact: `2026-02-14-streaming-rolling-baseline.md`

### Speculative Decoding (Experimental)

0.6B draft → 1.7B target, fp16.

| Clip | Baseline | Speculative | Relative |
|---|---:|---:|---:|
| Short (~2.5s) | 1.45s | 2.72s | 0.53x (slower) |
| 10s | 2.68s | 4.90s | 0.55x (slower) |

Greedy parity verified, but currently slower due to draft audio encoder overhead. Kept as experimental opt-in.

Artifact: `2026-02-14-speculative-prototype.md`

---

## Artifact Index

All benchmark artifacts are committed under `docs/benchmarks/`. Key files:

| Artifact | Description |
|---|---|
| `2026-02-14-quant-matrix-speaker100.md` | Quantization quality + latency matrix |
| `2026-02-15-manifest-quality-multilingual100.md` | 0.6B multilingual quality |
| `2026-02-15-manifest-quality-multilingual100-compare-0p6b-vs-1p7b.md` | 0.6B vs 1.7B comparison |
| `2026-02-15-quality-head2head-mlx-vs-pytorch-multilingual100.md` | MLX vs PyTorch parity |
| `2026-02-15-manifest-quality-longform10.md` | Long-form quality |
| `2026-02-15-reference-parity-suite-longform10.md` | Long-form speed (MLX vs PyTorch) |
| `2026-02-14-aligner-parity-50.md` | Forced aligner parity |
| `2026-02-14-mel-parity.md` | Mel spectrogram parity |
| `2026-02-14-encoder-windowing-threshold.md` | Encoder windowing optimization |
| `2026-02-14-wav-fastpath.md` | WAV fast-path optimization |
| `2026-02-14-speculative-prototype.md` | Speculative decoding prototype |
| `2026-02-14-streaming-rolling-baseline.md` | Streaming baseline |

90+ total JSON + markdown artifacts. See `docs/benchmarks/README.md` for the full chronological index.
