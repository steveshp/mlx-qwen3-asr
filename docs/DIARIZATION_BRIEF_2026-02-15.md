# Diarization Brief (2026-02-15)

Single-entry summary for diarization work completed in this pass.

## What is documented

1. Review + recommendation + risks
- `docs/DIARIZATION_NATIVE_PLAN_2026-02-15.md`

2. External research (GitHub + papers) with source links
- `docs/DIARIZATION_RESEARCH_2026-02-15.md`

3. Concrete implementation plan (API/modules/tests/eval/milestones)
- `docs/DIARIZATION_IMPLEMENTATION_SPEC_2026-02-15.md`

4. Project policy and roadmap integration
- `docs/DECISIONS.md` (Decision 21)
- `docs/ROADMAP.md` (native diarization lane)
- `docs/USER_FEEDBACK_ROI_PLAN_2026-02-15.md` (ROI entry + docs bundle)

5. Implementation landing status
- `mlx_qwen3_asr/diarization.py` (native baseline diarization runtime)
- `mlx_qwen3_asr/transcribe.py` / `mlx_qwen3_asr/session.py` / `mlx_qwen3_asr/cli.py` (pipeline + API/CLI integration)
- `tests/test_diarization.py` and diarization integration tests across CLI/session/transcribe/writers
- `scripts/eval_diarization.py` + quality-gate hook (`RUN_DIARIZATION_QUALITY_EVAL=1`) for DER/JER measurement
- Baseline benchmark artifacts:
  - `docs/benchmarks/2026-02-15-diarization-manifest-20.jsonl`
  - `docs/benchmarks/2026-02-15-diarization-quality-20.json`
  - `docs/benchmarks/2026-02-15-diarization-quality-20.md`
  - `docs/benchmarks/2026-02-15-diarization-quality-20-v2.json` (regressed experiment; rejected)
  - `docs/benchmarks/2026-02-15-diarization-quality-20-v3.json` (post-rollback validation)
  - `docs/benchmarks/2026-02-15-diarization-quality-20-v4.json` (accepted improvement)
  - `docs/benchmarks/2026-02-15-diarization-quality-20-v5.json` (accepted improvement)

## Final position

- Ship diarization only if native MLX.
- Keep it optional (`mlx-qwen3-asr[diarize]`) to preserve lean core install.
- Treat it as quality-gated infrastructure (DER/JER + latency), not a loose
  feature toggle.
- Reject runtime tweaks that worsen DER/JER, even if they reduce latency.
- Keep changes only when both quality and practicality improve (v5: DER `0.6762` -> `0.5051`, JER `0.6065` -> `0.4646`).
