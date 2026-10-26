# Experiment 11: counterfactual branch sensitivity (10 FPS)

Each response is scored once against its own branch target and once, in the
opposite direction, against the sibling branch target. `SwapGap` is
`((own_A + own_B) - (cross_A(y_B) + cross_B(y_A))) / 2` on the 0--5
strict-v2 scale. Positive values indicate branch-sensitive behavior.

| Model | Pairs | Own target | Swapped target | SwapGap | 95% source CI | Positive / tie / negative | Exact same | Token Jaccard | Char Jaccard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MiniCPM-o 4.5 | 300 | 0.556 | 0.614 | -0.059 | [-0.169, 0.060] | 21.7% / 56.3% / 22.0% | 25.0% | 0.293 | 0.304 |
| Venus | 300 | 0.866 | 0.773 | +0.093 | [-0.047, 0.236] | 33.7% / 37.0% / 29.3% | 13.3% | 0.236 | 0.240 |
| Gander | 300 | 0.535 | 0.528 | +0.007 | [-0.111, 0.126] | 13.7% / 71.7% / 14.7% | 56.7% | 0.625 | 0.633 |
| MOSS* | 300 | 1.292 | 0.730 | +0.562 | [0.412, 0.712] | 34.3% / 55.3% / 10.3% | 50.7% | 0.614 | 0.646 |

## Capability breakdown

### constraint_update

| Model | Own target | Swapped target | SwapGap | Positive rate | Exact same |
|---|---:|---:|---:|---:|---:|
| MiniCPM-o 4.5 | 0.356 | 0.311 | +0.045 | 22.0% | 14.0% |
| Venus | 0.678 | 0.690 | -0.012 | 26.0% | 18.0% |
| Gander | 0.451 | 0.430 | +0.021 | 16.0% | 60.0% |
| MOSS* | 1.047 | 0.527 | +0.520 | 36.0% | 52.0% |

### context_correction

| Model | Own target | Swapped target | SwapGap | Positive rate | Exact same |
|---|---:|---:|---:|---:|---:|
| MiniCPM-o 4.5 | 0.256 | 0.261 | -0.005 | 26.0% | 30.0% |
| Venus | 0.811 | 0.679 | +0.132 | 32.0% | 10.0% |
| Gander | 0.185 | 0.218 | -0.033 | 6.0% | 72.0% |
| MOSS* | 1.204 | 1.158 | +0.046 | 16.0% | 60.0% |

### historical_reference

| Model | Own target | Swapped target | SwapGap | Positive rate | Exact same |
|---|---:|---:|---:|---:|---:|
| MiniCPM-o 4.5 | 0.136 | 0.085 | +0.051 | 18.0% | 28.0% |
| Venus | 0.266 | 0.332 | -0.066 | 24.0% | 20.0% |
| Gander | 0.098 | 0.117 | -0.019 | 8.0% | 42.0% |
| MOSS* | 0.328 | 0.201 | +0.127 | 18.0% | 70.0% |

### monitor_cancel

| Model | Own target | Swapped target | SwapGap | Positive rate | Exact same |
|---|---:|---:|---:|---:|---:|
| MiniCPM-o 4.5 | 1.836 | 2.219 | -0.383 | 20.0% | 40.0% |
| Venus | 2.000 | 1.518 | +0.482 | 46.0% | 18.0% |
| Gander | 1.946 | 1.909 | +0.037 | 28.0% | 38.0% |
| MOSS* | 2.521 | 1.914 | +0.607 | 26.0% | 64.0% |

### task_resume

| Model | Own target | Swapped target | SwapGap | Positive rate | Exact same |
|---|---:|---:|---:|---:|---:|
| MiniCPM-o 4.5 | 0.374 | 0.261 | +0.113 | 22.0% | 26.0% |
| Venus | 0.706 | 0.570 | +0.136 | 38.0% | 10.0% |
| Gander | 0.053 | 0.075 | -0.022 | 4.0% | 76.0% |
| MOSS* | 0.866 | 0.440 | +0.426 | 30.0% | 40.0% |

### selective_attention

| Model | Own target | Swapped target | SwapGap | Positive rate | Exact same |
|---|---:|---:|---:|---:|---:|
| MiniCPM-o 4.5 | 0.376 | 0.548 | -0.172 | 22.0% | 12.0% |
| Venus | 0.736 | 0.849 | -0.113 | 36.0% | 4.0% |
| Gander | 0.478 | 0.420 | +0.058 | 20.0% | 52.0% |
| MOSS* | 1.785 | 0.141 | +1.644 | 80.0% | 18.0% |

## Interpretation boundary

The cross-score is a rubric-sensitivity diagnostic, not a replacement for
the original leaderboard. It is restricted to six target-incompatible paired
final slots. Temporal-update pairs and asymmetric proactive chains are excluded.
Raw cross-Judge labels are stored only in the external resumable cache.
