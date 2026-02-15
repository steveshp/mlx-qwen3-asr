# Reference Parity Mismatch Analysis

- Source: `/Users/dmoon/code/mlx-qwen3-asr/docs/benchmarks/2026-02-14-reference-parity-suite-multilingual-20.json`
- Suite: `reference-parity-suite-v1`
- Model: `Qwen/Qwen3-ASR-0.6B`
- Samples: `20`
- Token match rate: `0.5500`
- Text match rate: `0.5500`

## Categories

| Category | Count |
|---|---:|
| exact_match | 11 |
| minor_lexical_shift | 8 |
| content_shift | 1 |

## By Language

| Language | Samples | Token match rate | Text match rate |
|---|---:|---:|---:|
| Arabic | 2 | 0.5000 | 0.5000 |
| Chinese | 2 | 0.5000 | 0.5000 |
| English | 2 | 1.0000 | 1.0000 |
| French | 2 | 0.0000 | 0.0000 |
| German | 2 | 1.0000 | 1.0000 |
| Hindi | 2 | 0.0000 | 0.0000 |
| Japanese | 2 | 0.5000 | 0.5000 |
| Korean | 2 | 0.5000 | 0.5000 |
| Russian | 2 | 1.0000 | 1.0000 |
| Spanish | 2 | 0.5000 | 0.5000 |

## Top Mismatches

| Sample | Language | Category | Edit ratio | First mismatch idx |
|---|---|---|---:|---:|
| hi_in-10245365873747218908 | Hindi | content_shift | 0.3841 | 33 |
| ja_jp-2377479830513507913 | Japanese | minor_lexical_shift | 0.0750 | 9 |
| ar_eg-16702308927396950177 | Arabic | minor_lexical_shift | 0.0696 | 19 |
| fr_fr-13789433312930016037 | French | minor_lexical_shift | 0.0462 | 33 |
| cmn_hans_cn-8061505693744764604 | Chinese | minor_lexical_shift | 0.0294 | 38 |
| ko_kr-737801410327834892 | Korean | minor_lexical_shift | 0.0260 | 3 |
| hi_in-14793234393688530328 | Hindi | minor_lexical_shift | 0.0244 | 3 |
| fr_fr-2262832907340477841 | French | minor_lexical_shift | 0.0103 | 25 |
| es_419-6295972223351365462 | Spanish | minor_lexical_shift | 0.0056 | 45 |
