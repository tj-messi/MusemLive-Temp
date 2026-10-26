# Experiment 11: counterfactual branch sensitivity (1 FPS)

Each response is scored once against its own branch target and once, in the
opposite direction, against the sibling branch target. `SwapGap` is
`((own_A + own_B) - (cross_A(y_B) + cross_B(y_A))) / 2` on the 0--5
strict-v2 scale. Positive values indicate branch-sensitive behavior.

| Model | Pairs | Own target | Swapped target | SwapGap | 95% source CI | Positive / tie / negative | Exact same | Token Jaccard | Char Jaccard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MiniCPM-o 4.5 | 300 | 1.384 | 1.315 | +0.069 | [-0.069, 0.218] | 33.3% / 34.7% / 32.0% | 17.0% | 0.360 | 0.421 |
| Venus | 300 | 0.808 | 0.854 | -0.045 | [-0.167, 0.062] | 20.0% / 54.0% / 26.0% | 39.3% | 0.484 | 0.498 |
| Gander | 300 | 1.024 | 0.941 | +0.082 | [-0.009, 0.171] | 22.3% / 58.7% / 19.0% | 43.7% | 0.592 | 0.636 |
| MOSS* | 300 | 2.016 | 1.332 | +0.684 | [0.502, 0.841] | 51.0% / 27.3% / 21.7% | 15.7% | 0.458 | 0.529 |

## Capability breakdown

### constraint_update

| Model | Own target | Swapped target | SwapGap | Positive rate | Exact same |
|---|---:|---:|---:|---:|---:|
| MiniCPM-o 4.5 | 1.223 | 1.245 | -0.022 | 34.0% | 16.0% |
| Venus | 0.752 | 0.816 | -0.064 | 22.0% | 28.0% |
| Gander | 0.997 | 0.838 | +0.159 | 28.0% | 36.0% |
| MOSS* | 2.457 | 1.308 | +1.149 | 58.0% | 12.0% |

### context_correction

| Model | Own target | Swapped target | SwapGap | Positive rate | Exact same |
|---|---:|---:|---:|---:|---:|
| MiniCPM-o 4.5 | 1.299 | 1.377 | -0.078 | 36.0% | 18.0% |
| Venus | 0.363 | 0.320 | +0.043 | 14.0% | 68.0% |
| Gander | 0.610 | 0.657 | -0.047 | 14.0% | 54.0% |
| MOSS* | 1.543 | 1.570 | -0.027 | 32.0% | 18.0% |

### historical_reference

| Model | Own target | Swapped target | SwapGap | Positive rate | Exact same |
|---|---:|---:|---:|---:|---:|
| MiniCPM-o 4.5 | 0.978 | 1.009 | -0.031 | 38.0% | 6.0% |
| Venus | 0.628 | 0.717 | -0.089 | 24.0% | 32.0% |
| Gander | 0.863 | 0.778 | +0.085 | 30.0% | 34.0% |
| MOSS* | 1.611 | 1.339 | +0.272 | 54.0% | 2.0% |

### monitor_cancel

| Model | Own target | Swapped target | SwapGap | Positive rate | Exact same |
|---|---:|---:|---:|---:|---:|
| MiniCPM-o 4.5 | 2.443 | 1.994 | +0.449 | 16.0% | 52.0% |
| Venus | 2.213 | 2.002 | +0.211 | 20.0% | 38.0% |
| Gander | 2.082 | 2.120 | -0.038 | 12.0% | 52.0% |
| MOSS* | 2.049 | 2.130 | -0.081 | 20.0% | 36.0% |

### task_resume

| Model | Own target | Swapped target | SwapGap | Positive rate | Exact same |
|---|---:|---:|---:|---:|---:|
| MiniCPM-o 4.5 | 1.247 | 1.057 | +0.190 | 46.0% | 2.0% |
| Venus | 0.294 | 0.388 | -0.094 | 14.0% | 46.0% |
| Gander | 0.825 | 0.591 | +0.234 | 28.0% | 48.0% |
| MOSS* | 2.037 | 1.292 | +0.745 | 58.0% | 12.0% |

### selective_attention

| Model | Own target | Swapped target | SwapGap | Positive rate | Exact same |
|---|---:|---:|---:|---:|---:|
| MiniCPM-o 4.5 | 1.112 | 1.208 | -0.096 | 30.0% | 8.0% |
| Venus | 0.601 | 0.881 | -0.280 | 26.0% | 24.0% |
| Gander | 0.764 | 0.663 | +0.101 | 22.0% | 38.0% |
| MOSS* | 2.397 | 0.351 | +2.046 | 84.0% | 14.0% |

## Interpretation boundary

The cross-score is a rubric-sensitivity diagnostic, not a replacement for
the original leaderboard. It is restricted to six target-incompatible paired
final slots. Temporal-update pairs and asymmetric proactive chains are excluded.
Raw cross-Judge labels are stored only in the external resumable cache.
