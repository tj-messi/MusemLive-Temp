# Final 10 FPS results

All summaries below use the fixed `lenient_v3` point-Judge scoring profile and
report the expected slot count with zero API failures. Raw model generations,
per-slot Judge caches, media, weights, controlled task data, and credentials
remain outside Git.

## Main and intervention runs

| Run | Experiment | Condition | Mean / 5 | Summary |
|---|---:|---|---:|---|
| `main_minicpm` | 1–4 | MiniCPM-o 4.5 full input | 0.97 | [`summary.json`](lenient_v3/main_minicpm/summary.json) |
| `main_venus` | 1–4 | Venus full input | 1.22 | [`summary.json`](lenient_v3/main_venus/summary.json) |
| `main_gander` | 1–4 | Gander full input | 0.89 | [`summary.json`](lenient_v3/main_gander/summary.json) |
| `main_moss` | 1–4 | MOSS vision + ideal-ASR text | 1.61 | [`summary.json`](lenient_v3/main_moss/summary.json) |
| `force_venus` | 5 | Venus forced answer | 1.30 | [`summary.json`](lenient_v3/force_venus/summary.json) |
| `force_gander` | 5 | Gander forced answer | 1.59 | [`summary.json`](lenient_v3/force_gander/summary.json) |
| `force_minicpm` | 5 | MiniCPM forced answer | 1.04 | [`summary.json`](lenient_v3/force_minicpm/summary.json) |
| `mute_minicpm` | 6 | MiniCPM ambient audio muted | 1.13 | [`summary.json`](lenient_v3/mute_minicpm/summary.json) |
| `black_minicpm` | 6 | MiniCPM black video | 0.91 | [`summary.json`](lenient_v3/black_minicpm/summary.json) |
| `notext_moss` | 6 | MOSS without scene transcript | 1.40 | [`summary.json`](lenient_v3/notext_moss/summary.json) |
| `black_moss` | 6 | MOSS transcript only / black video | 0.64 | [`summary.json`](lenient_v3/black_moss/summary.json) |
| `blind_minicpm` | 7 | MiniCPM 3-second blind context | 1.44 | [`summary.json`](lenient_v3/blind_minicpm/summary.json) |
| `blind_moss` | 7 | MOSS 3-second blind context | 1.20 | [`summary.json`](lenient_v3/blind_moss/summary.json) |
| `recent30_minicpm` | 7 | MiniCPM recent 30 seconds | 0.68 | [`summary.json`](lenient_v3/recent30_minicpm/summary.json) |
| `recent30_moss` | 7 | MOSS recent 30 seconds | 1.30 | [`summary.json`](lenient_v3/recent30_moss/summary.json) |
| `offline_qwen_omni` | 7 | Offline Qwen-Omni upper bound | 3.28 overall / 3.35 user-query | [`summary.json`](lenient_v3/offline_qwen_omni/summary.json) |

## Alternate Judges

| Judge | MiniCPM | Venus | Gander | MOSS |
|---|---:|---:|---:|---:|
| HY3 | [0.77](alternate_judges/tencent_hy3/minicpm/summary.json) | [0.92](alternate_judges/tencent_hy3/venus/summary.json) | [0.74](alternate_judges/tencent_hy3/gander/summary.json) | [1.36](alternate_judges/tencent_hy3/moss/summary.json) |
| DeepSeek-v4-pro | [0.58](alternate_judges/deepseek_deepseek-v4-pro/minicpm/summary.json) | [0.60](alternate_judges/deepseek_deepseek-v4-pro/venus/summary.json) | [0.62](alternate_judges/deepseek_deepseek-v4-pro/gander/summary.json) | [1.08](alternate_judges/deepseek_deepseek-v4-pro/moss/summary.json) |
| GLM-5.2 | [0.85](alternate_judges/z-ai_glm-5.2/minicpm/summary.json) | [0.99](alternate_judges/z-ai_glm-5.2/venus/summary.json) | [0.76](alternate_judges/z-ai_glm-5.2/gander/summary.json) | [1.39](alternate_judges/z-ai_glm-5.2/moss/summary.json) |
| Kimi-k2.6 | [0.74](alternate_judges/moonshotai_kimi-k2.6/minicpm/summary.json) | [0.87](alternate_judges/moonshotai_kimi-k2.6/venus/summary.json) | [0.72](alternate_judges/moonshotai_kimi-k2.6/gander/summary.json) | [1.32](alternate_judges/moonshotai_kimi-k2.6/moss/summary.json) |

The Qwen, HY3, GLM, and Kimi rankings agree:
`MOSS > Venus > MiniCPM > Gander`. DeepSeek ranks
`MOSS > Gander > Venus > MiniCPM`; see the bootstrap and Kendall analysis in
[`../reports/analysis_1_4.md`](../reports/analysis_1_4.md).

## Aggregate artifacts

- [`summary.json`](summary.json): complete machine-readable Experiment 1–7 report.
- [`analysis_1_4.json`](analysis_1_4.json): response, timing, strata, Judge, and bootstrap results.
- [`analysis_5_7.json`](analysis_5_7.json): interventions, ablations, and context baselines.
- [`provenance/`](provenance/): provider fallback, recent-30 reuse, and offline-cache records.

The finalized 1 FPS track uses a different historical Judge profile. Raw 1 FPS
and 10 FPS scores are therefore not a controlled FPS-only comparison.
