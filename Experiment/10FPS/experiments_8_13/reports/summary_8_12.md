# 10FPS Experiments 8--12 (no model inference)

All score-based analyses use the frozen `strict_v2` point labels. Experiment 11 is reported separately because it performs resumable sibling-rubric Judge calls.

## Experiment 8: sequential chain completion

| Model | Eligible chains | Soft chain | Strict all-stage | Source-cluster 95% CI |
|---|---:|---:|---:|---:|
| MiniCPM-o 4.5 | 350 | 0.031 | 0.3% | [0.014, 0.042] |
| Venus | 350 | 0.084 | 0.9% | [0.052, 0.094] |
| Gander | 350 | 0.038 | 0.3% | [0.019, 0.047] |
| MOSS* | 350 | 0.171 | 2.9% | [0.120, 0.175] |

## Experiment 9: counterfactual joint completion

| Model | Soft CJCS | Strict PairSuccess | Both / left-only / right-only / neither |
|---|---:|---:|---:|
| MiniCPM-o 4.5 | 0.024 | 0.0% | 0 / 4 / 63 / 383 |
| Venus | 0.063 | 1.1% | 5 / 15 / 48 / 382 |
| Gander | 0.025 | 0.4% | 2 / 4 / 48 / 396 |
| MOSS* | 0.128 | 2.2% | 10 / 26 / 61 / 353 |

## Experiment 10: interaction-control calibration

| Model | Macro action F1 | Action accuracy | Monitor keep | Cancel specificity | Both | Post-cancel substantive alert |
|---|---:|---:|---:|---:|---:|---:|
| MiniCPM-o 4.5 | 0.232 | 25.7% | 20.0% | 58.0% | 8.0% | 42.0% |
| Venus | 0.265 | 34.1% | 58.0% | 42.0% | 22.0% | 58.0% |
| Gander | 0.183 | 24.5% | 42.0% | 54.0% | 14.0% | 46.0% |
| MOSS* | 0.310 | 44.3% | 46.0% | 68.0% | 18.0% | 32.0% |

## Experiment 12: quality-weighted temporal alignment

| Model | QT precision | QT recall | QT-F1 | Tight timing F1 | Medium timing F1 | Formal timing F1 |
|---|---:|---:|---:|---:|---:|---:|
| MiniCPM-o 4.5 | 0.073 | 0.085 | 0.079 | 0.237 | 0.406 | 0.603 |
| Venus | 0.082 | 0.104 | 0.092 | 0.206 | 0.358 | 0.538 |
| Gander | 0.092 | 0.076 | 0.083 | 0.139 | 0.254 | 0.380 |
| MOSS* | 0.333 | 0.209 | 0.257 | 0.416 | 0.490 | 0.555 |

## Interpretation boundary

For an in-window required response with normalized quality q, the soft counts are TP=q, FP=1-q, FN=1-q. Silence adds FN=1; early/late attempts add FP=1 and FN=1; substantive WAIT violations and unmatched turns add FP=1.

The quality-time score is a diagnostic soft F1. It does not replace the separately reported content, coverage, waiting, or timing measurements. Tight/medium windows reuse the formal response labels and only test timing sensitivity; they do not re-Judge truncated utterances.
