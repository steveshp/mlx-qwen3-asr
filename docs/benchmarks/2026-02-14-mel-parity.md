# Custom Mel Parity (2026-02-14)

Comparison target: HF `WhisperFeatureExtractor` (`padding=do_not_pad`, `truncation=False`).

- Script: `scripts/eval_mel_parity.py`
- Seed: `0`
- Cases: random `{1,2,5,10,31}s` + `tests/fixtures/test_speech.wav`
- Thresholds: `MAE <= 1e-5`, `max_abs <= 2e-4`

## Result

- `PASS`
- Max MAE across all cases: `2.8314e-07`
- Max absolute difference across all cases: `1.1265e-04`
- Frame lengths: exact match in every case.

Raw artifact: `docs/benchmarks/2026-02-14-mel-parity.json`
