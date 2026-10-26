# Experiment 13: strict-v2 cross-FPS sensitivity

Paired deltas are 10 FPS minus 1 FPS. Confidence intervals resample source videos and retain all tasks sharing each sampled source.
`task_macro_score` first averages strict-v2 slot scores within each task, then averages the 450 paired task values; it is not the slot-weighted Experiment 1 leaderboard mean.

## task_macro_score

| Model | 1 FPS | 10 FPS | Paired delta | Source-cluster 95% CI |
|---|---:|---:|---:|---:|
| MiniCPM-o 4.5 | 1.545 | 0.827 | -0.718 | [-0.838, -0.611] |
| Venus | 1.106 | 1.046 | -0.060 | [-0.174, 0.065] |
| Gander | 1.202 | 0.766 | -0.435 | [-0.546, -0.337] |
| MOSS* | 1.940 | 1.391 | -0.549 | [-0.678, -0.425] |

## soft_chain

| Model | 1 FPS | 10 FPS | Paired delta | Source-cluster 95% CI |
|---|---:|---:|---:|---:|
| MiniCPM-o 4.5 | 0.140 | 0.027 | -0.112 | [-0.140, -0.088] |
| Venus | 0.068 | 0.073 | 0.005 | [-0.024, 0.032] |
| Gander | 0.056 | 0.033 | -0.023 | [-0.043, -0.004] |
| MOSS* | 0.220 | 0.147 | -0.073 | [-0.120, -0.027] |

## soft_cjcs

| Model | 1 FPS | 10 FPS | Paired delta | Source-cluster 95% CI |
|---|---:|---:|---:|---:|
| MiniCPM-o 4.5 | 0.109 | 0.024 | -0.086 | [-0.108, -0.064] |
| Venus | 0.044 | 0.063 | 0.019 | [-0.002, 0.042] |
| Gander | 0.061 | 0.025 | -0.036 | [-0.056, -0.016] |
| MOSS* | 0.206 | 0.128 | -0.078 | [-0.111, -0.046] |

## strict_pair

| Model | 1 FPS | 10 FPS | Paired delta | Source-cluster 95% CI |
|---|---:|---:|---:|---:|
| MiniCPM-o 4.5 | 0.011 | 0.000 | -0.011 | [-0.022, -0.002] |
| Venus | 0.000 | 0.011 | 0.011 | [0.002, 0.021] |
| Gander | 0.007 | 0.004 | -0.002 | [-0.012, 0.008] |
| MOSS* | 0.033 | 0.022 | -0.011 | [-0.031, 0.010] |

## action_accuracy

| Model | 1 FPS | 10 FPS | Paired delta | Source-cluster 95% CI |
|---|---:|---:|---:|---:|
| MiniCPM-o 4.5 | 0.554 | 0.246 | -0.308 | [-0.351, -0.264] |
| Venus | 0.361 | 0.329 | -0.032 | [-0.066, 0.006] |
| Gander | 0.394 | 0.238 | -0.156 | [-0.195, -0.121] |
| MOSS* | 0.635 | 0.428 | -0.207 | [-0.245, -0.171] |
