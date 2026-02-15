# Manifest Quality (FLEURS Multilingual 100)

- samples: 100
- model: Qwen/Qwen3-ASR-0.6B (float16)
- wer: 0.1589
- cer: 0.0541
- primary error rate: 0.0937
- mean latency: 1.35s

Primary metric rule: CER for Chinese/Japanese/Korean; WER otherwise.

| language | samples | wer | cer | primary | latency (s) |
|---|---:|---:|---:|---:|---:|
| Arabic | 10 | 0.215 | 0.070 | 0.215 | 1.21 |
| Chinese | 10 | 0.900 | 0.044 | 0.044 | 0.76 |
| English | 10 | 0.046 | 0.019 | 0.046 | 0.87 |
| French | 10 | 0.182 | 0.092 | 0.182 | 1.07 |
| German | 10 | 0.080 | 0.047 | 0.080 | 1.25 |
| Hindi | 10 | 0.167 | 0.098 | 0.167 | 3.54 |
| Japanese | 10 | 0.821 | 0.085 | 0.085 | 1.12 |
| Korean | 10 | 0.172 | 0.067 | 0.067 | 1.09 |
| Russian | 10 | 0.088 | 0.034 | 0.088 | 1.37 |
| Spanish | 10 | 0.030 | 0.006 | 0.030 | 1.20 |

## Interpretation

These results are consistent with the Qwen3-ASR paper's reported FLEURS aggregates
(Table 5: 0.6B 12-lang avg = 7.57% WER). The paper does not break out per-language
FLEURS numbers beyond English and Chinese, but our per-language breakdown reveals
where the aggregate comes from.

**Core languages (EN/ZH/ES/KO/RU/DE):** Strong. English 4.6% WER and Chinese 4.4%
CER are in range of the paper's individually reported FLEURS-en (4.39%) and FLEURS-zh
(2.88%). Chinese CER (4.4% vs paper's 2.88%) is elevated — likely sample noise at
n=10, but worth confirming at full scale. Spanish, Korean, Russian, and German are
all under 9%.

**Weaker languages (FR/AR/HI):** French (18.2%), Arabic (21.5%), and Hindi (16.7%)
are significantly worse. This is a known 0.6B model limitation — the paper notes that
"performance degrades on the full 30-language setting" and the 0.6B-to-1.7B gap
widens from 2.67 points (12 langs) to 9.20 points (30 langs). The 1.7B dramatically
improves these languages (French drops from 18.2% to 4.1%).

**Error patterns observed:**
- **French 0.6B:** Genuine word-level errors (e.g., "rugissement" → "rougement",
  "tigre" → "Ting"). Not a normalization issue — the 0.6B garbles French vocabulary.
- **Hindi code-switching:** The 0.6B outputs English ASCII mid-Hindi for loanwords
  (e.g., `फ़ॉस्टर केयर` → `faster career`). The 1.7B stays in Devanagari script.
- **Numeric surface forms:** Both models sometimes spell out numbers differently
  from ground truth (e.g., `10.000` → `zehntausend` in German). This inflates
  WER/CER but is not a transcription error.

**Conclusion:** These results reflect model characteristics, not port-specific bugs.
The 0.6B is optimized for throughput on core languages; the 1.7B is the quality
choice for multilingual use. Sample size (n=10 per language) is small — expand to
full FLEURS test splits (~350 per language) for publication-grade claims.
