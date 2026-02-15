# Quality Matrix Refresh (2026-02-15)

Machine: Apple M4 Pro

## Commands

```bash
uv run python scripts/eval_librispeech.py --model Qwen/Qwen3-ASR-0.6B --dtype float16 --subset test-clean --samples 100 --sampling speaker_round_robin --max-new-tokens 256 --json-output docs/benchmarks/2026-02-15-librispeech-test-clean-100.json
uv run python scripts/eval_librispeech.py --model Qwen/Qwen3-ASR-0.6B --dtype float16 --subset test-other --samples 100 --sampling speaker_round_robin --max-new-tokens 256 --json-output docs/benchmarks/2026-02-15-librispeech-test-other-100.json
uv run python scripts/eval_manifest_quality.py --manifest-jsonl docs/benchmarks/2026-02-14-fleurs-multilingual-100-manifest.jsonl --model Qwen/Qwen3-ASR-0.6B --dtype float16 --max-new-tokens 1024 --json-output docs/benchmarks/2026-02-15-manifest-quality-multilingual100-0p6b-refresh.json
uv run python scripts/eval_manifest_quality.py --manifest-jsonl docs/benchmarks/2026-02-14-fleurs-multilingual-100-manifest.jsonl --model Qwen/Qwen3-ASR-1.7B --dtype float16 --max-new-tokens 1024 --json-output docs/benchmarks/2026-02-15-manifest-quality-multilingual100-1p7b-refresh.json
```

## English Quality (0.6B)

| Subset | Samples | Speakers | WER | CER | Mean latency (s) | RTF |
|---|---:|---:|---:|---:|---:|---:|
| test-clean | 100 | 40 | 2.29% | 0.59% | 0.8576 | 0.0957 |
| test-other | 100 | 33 | 4.20% | 2.09% | 0.7079 | 0.0985 |

Artifacts:
- `docs/benchmarks/2026-02-15-librispeech-test-clean-100.json`
- `docs/benchmarks/2026-02-15-librispeech-test-other-100.json`

## Multilingual Quality (FLEURS manifest, 100 samples)

| Model | Primary error | WER | CER | Mean latency (s) |
|---|---:|---:|---:|---:|
| Qwen3-ASR-0.6B | 9.37% | 15.89% | 5.41% | 1.4418 |
| Qwen3-ASR-1.7B | 6.70% | 16.27% | 3.22% | 4.1231 |

Derived:
- 1.7B relative primary-error improvement vs 0.6B: **28.42%**
- 1.7B latency ratio vs 0.6B: **2.86x** slower

Best/worst language buckets (primary metric):
- 0.6B best: Spanish (3.0%), Chinese (4.4%), English (4.6%)
- 0.6B worst: Hindi (16.7%), French (18.2%), Arabic (21.5%)
- 1.7B best: Spanish (0.7%), Japanese (3.6%), French (4.1%)
- 1.7B worst: Chinese (8.5%), Arabic (16.5%), Hindi (17.4%)

Artifacts:
- `docs/benchmarks/2026-02-15-manifest-quality-multilingual100-0p6b-refresh.json`
- `docs/benchmarks/2026-02-15-manifest-quality-multilingual100-1p7b-refresh.json`

## Notes

- This refresh extends the English lane from `test-clean` only to `test-clean + test-other`.
- These numbers should be treated as benchmark checkpoints for this commit state, not universal claims across all domains.
