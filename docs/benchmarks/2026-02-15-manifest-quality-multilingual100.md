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
