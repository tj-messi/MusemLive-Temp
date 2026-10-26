# 1FPS Experiments 8--12 (no model inference)

All score-based analyses use the frozen `strict_v2` point labels. Experiment 11 is reported separately because it performs resumable sibling-rubric Judge calls.

## Experiment 8: sequential chain completion

| Model | Eligible chains | Soft chain | Strict all-stage | Source-cluster 95% CI |
|---|---:|---:|---:|---:|
| MiniCPM-o 4.5 | 350 | 0.158 | 2.9% | [0.115, 0.167] |
| Venus | 350 | 0.076 | 0.9% | [0.049, 0.090] |
| Gander | 350 | 0.068 | 0.9% | [0.038, 0.075] |
| MOSS* | 350 | 0.214 | 4.9% | [0.183, 0.261] |

## Experiment 9: counterfactual joint completion

| Model | Soft CJCS | Strict PairSuccess | Both / left-only / right-only / neither |
|---|---:|---:|---:|
| MiniCPM-o 4.5 | 0.109 | 1.1% | 5 / 25 / 71 / 349 |
| Venus | 0.044 | 0.0% | 0 / 14 / 67 / 369 |
| Gander | 0.061 | 0.7% | 3 / 24 / 67 / 356 |
| MOSS* | 0.206 | 3.3% | 15 / 63 / 72 / 300 |

## Experiment 10: interaction-control calibration

| Model | Macro action F1 | Action accuracy | Monitor keep | Cancel specificity | Both | Post-cancel substantive alert |
|---|---:|---:|---:|---:|---:|---:|
| MiniCPM-o 4.5 | 0.384 | 55.6% | 42.0% | 68.0% | 16.0% | 32.0% |
| Venus | 0.255 | 36.6% | 48.0% | 54.0% | 12.0% | 46.0% |
| Gander | 0.277 | 39.1% | 44.0% | 54.0% | 4.0% | 46.0% |
| MOSS* | 0.354 | 62.7% | 66.0% | 32.0% | 6.0% | 68.0% |

## Experiment 12: quality-weighted temporal alignment

| Model | QT precision | QT recall | QT-F1 | Tight timing F1 | Medium timing F1 | Formal timing F1 |
|---|---:|---:|---:|---:|---:|---:|
| MiniCPM-o 4.5 | 0.226 | 0.210 | 0.218 | 0.348 | 0.604 | 0.768 |
| Venus | 0.230 | 0.128 | 0.165 | 0.302 | 0.467 | 0.587 |
| Gander | 0.309 | 0.172 | 0.221 | 0.272 | 0.520 | 0.625 |
| MOSS* | 0.224 | 0.315 | 0.262 | 0.353 | 0.478 | 0.643 |

## Interpretation boundary

For an in-window required response with normalized quality q, the soft counts are TP=q, FP=1-q, FN=1-q. Silence adds FN=1; early/late attempts add FP=1 and FN=1; substantive WAIT violations and unmatched turns add FP=1.

The quality-time score is a diagnostic soft F1. It does not replace the separately reported content, coverage, waiting, or timing measurements. Tight/medium windows reuse the formal response labels and only test timing sensitivity; they do not re-Judge truncated utterances.
