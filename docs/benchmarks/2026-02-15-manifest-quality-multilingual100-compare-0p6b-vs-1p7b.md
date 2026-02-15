# Manifest Quality Comparison (0.6B vs 1.7B, Multilingual-100)

- WER: 0.6B=0.1589, 1.7B=0.1627, delta=+0.0037
- CER: 0.6B=0.0541, 1.7B=0.0322, delta=-0.0219
- Primary: 0.6B=0.0937, 1.7B=0.0670, delta=-0.0266
- Mean latency: 0.6B=1.35s, 1.7B=3.86s, ratio=2.86x

| language | primary 0.6B | primary 1.7B | delta | latency ratio |
|---|---:|---:|---:|---:|
| Arabic | 0.215 | 0.165 | -0.050 | 2.93x |
| Chinese | 0.044 | 0.085 | +0.041 | 2.83x |
| English | 0.046 | 0.042 | -0.005 | 2.58x |
| French | 0.182 | 0.041 | -0.141 | 2.87x |
| German | 0.080 | 0.058 | -0.022 | 2.84x |
| Hindi | 0.167 | 0.174 | +0.007 | 2.97x |
| Japanese | 0.085 | 0.036 | -0.049 | 2.79x |
| Korean | 0.067 | 0.055 | -0.012 | 2.85x |
| Russian | 0.088 | 0.049 | -0.039 | 2.84x |
| Spanish | 0.030 | 0.007 | -0.022 | 2.84x |

## Notes

**Chinese CER regression is a numeric surface form artifact, not a quality issue.**
The 1.7B model spells out numbers in Chinese characters (e.g., `二十九` instead of
`29`, `二零零六` instead of `2006`), while FLEURS ground truth uses Arabic numerals.
Both are correct transcriptions of the spoken audio. All 3 Chinese samples where 1.7B
CER exceeds 0.6B CER are explained by this pattern. A number-aware text normalizer
would eliminate the gap.

**Hindi near-parity (+0.007) on 10 samples is not statistically meaningful.**
Both models struggle similarly on Hindi FLEURS at this sample size.
