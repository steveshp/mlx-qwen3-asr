# Reference Parity Mismatch Analysis

- Source: `/Users/dmoon/code/mlx-qwen3-asr/docs/benchmarks/2026-02-14-reference-parity-suite-multilingual-100.json`
- Suite: `reference-parity-suite-v1`
- Model: `Qwen/Qwen3-ASR-0.6B`
- Samples: `100`
- Token match rate: `0.6400`
- Text match rate: `0.6700`

## Categories

| Category | Count |
|---|---:|
| exact_match | 64 |
| minor_lexical_shift | 25 |
| numeric_surface_form | 5 |
| content_shift | 3 |
| punctuation_or_tokenization | 3 |

## By Language

| Language | Samples | Token match rate | Text match rate |
|---|---:|---:|---:|
| Arabic | 10 | 0.4000 | 0.5000 |
| Chinese | 10 | 0.7000 | 0.8000 |
| English | 10 | 0.8000 | 0.9000 |
| French | 10 | 0.4000 | 0.4000 |
| German | 10 | 0.6000 | 0.6000 |
| Hindi | 10 | 0.4000 | 0.4000 |
| Japanese | 10 | 0.6000 | 0.6000 |
| Korean | 10 | 0.9000 | 0.9000 |
| Russian | 10 | 0.9000 | 0.9000 |
| Spanish | 10 | 0.7000 | 0.7000 |

## Top Mismatches

| Sample | Language | Category | Edit ratio | First mismatch idx |
|---|---|---|---:|---:|
| hi_in-12919174378536885221 | Hindi | content_shift | 0.3235 | 7 |
| fr_fr-691450202103973076 | French | numeric_surface_form | 0.2515 | 12 |
| ja_jp-8922261396806111795 | Japanese | content_shift | 0.1613 | 7 |
| en_us-394135166243682296 | English | numeric_surface_form | 0.1385 | 17 |
| hi_in-16674359696770797961 | Hindi | content_shift | 0.1223 | 66 |
| cmn_hans_cn-15052299507121145030 | Chinese | numeric_surface_form | 0.1143 | 1 |
| de_de-4963712857525084701 | German | numeric_surface_form | 0.1068 | 12 |
| ar_eg-16702308927396950177 | Arabic | minor_lexical_shift | 0.0696 | 19 |
| de_de-11906634980733046933 | German | minor_lexical_shift | 0.0670 | 29 |
| ar_eg-6322491731206225587 | Arabic | minor_lexical_shift | 0.0526 | 0 |
| fr_fr-10929892609691631011 | French | minor_lexical_shift | 0.0430 | 5 |
| cmn_hans_cn-17232158003733395215 | Chinese | minor_lexical_shift | 0.0408 | 8 |
| ja_jp-4376267553407989455 | Japanese | minor_lexical_shift | 0.0405 | 6 |
| ja_jp-3483199886532772858 | Japanese | numeric_surface_form | 0.0400 | 2 |
| ar_eg-14219101187915533421 | Arabic | minor_lexical_shift | 0.0377 | 29 |
| hi_in-17876469696694013955 | Hindi | minor_lexical_shift | 0.0376 | 61 |
| fr_fr-16949862966091892699 | French | minor_lexical_shift | 0.0364 | 12 |
| ja_jp-12946582089204475428 | Japanese | minor_lexical_shift | 0.0333 | 23 |
| fr_fr-10143749263702132703 | French | minor_lexical_shift | 0.0292 | 9 |
| hi_in-10847487616323903195 | Hindi | minor_lexical_shift | 0.0214 | 105 |
| de_de-8970395757894156280 | German | minor_lexical_shift | 0.0205 | 30 |
| ar_eg-13121245795898949537 | Arabic | minor_lexical_shift | 0.0164 | 31 |
| fr_fr-17031409196685305809 | French | minor_lexical_shift | 0.0152 | 25 |
| hi_in-12193749519479711646 | Hindi | minor_lexical_shift | 0.0152 | 35 |
| ru_ru-14538972329666110700 | Russian | minor_lexical_shift | 0.0132 | 40 |
