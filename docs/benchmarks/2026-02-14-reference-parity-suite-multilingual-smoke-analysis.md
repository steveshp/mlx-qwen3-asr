# Reference Parity Mismatch Analysis

- Source: `/Users/dmoon/code/mlx-qwen3-asr/docs/benchmarks/2026-02-14-reference-parity-suite-multilingual-smoke.json`
- Suite: `reference-parity-suite-v1`
- Model: `Qwen/Qwen3-ASR-0.6B`
- Samples: `4`
- Token match rate: `0.5000`
- Text match rate: `0.5000`

## Categories

| Category | Count |
|---|---:|
| exact_match | 2 |
| minor_lexical_shift | 1 |
| numeric_surface_form | 1 |

## By Language

| Language | Samples | Token match rate | Text match rate |
|---|---:|---:|---:|
| Chinese | 1 | 1.0000 | 1.0000 |
| English | 1 | 1.0000 | 1.0000 |
| German | 1 | 0.0000 | 0.0000 |
| Japanese | 1 | 0.0000 | 0.0000 |

## Top Mismatches

| Sample | Language | Category | Edit ratio | First mismatch idx |
|---|---|---|---:|---:|
| de_de-18173364472777163131 | German | numeric_surface_form | 0.0824 | 0 |
| ja_jp-2377479830513507913 | Japanese | minor_lexical_shift | 0.0750 | 9 |
