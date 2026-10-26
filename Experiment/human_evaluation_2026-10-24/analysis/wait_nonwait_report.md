# Four-model five-judge WAIT and non-WAIT analysis

## Coverage

- Row-level scores: `2,500` (4 models x 125 slots x 5 judges).
- Successful judges: `5/5`; failed row-level calls: `0`.
- Score source mix: `{'api': 1665, 'deterministic': 835}`.
- Slot scopes: `125` overall, `15` WAIT, `110` non-WAIT, `103` SPEAK, and `7` CLARIFY.
- Assisted Human Baseline values average B01-B03; model values are saved candidate responses under the same frozen protocol.

## Five-judge consensus by expected action

The WAIT/non-WAIT difference is descriptive and must not be interpreted as a causal model effect.

| Entity | Overall | Task-balanced | WAIT | Non-WAIT | Non-WAIT minus WAIT | SPEAK | CLARIFY |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Assisted Human Baseline | 2.9547 | 3.0568 | 4.8889 | 2.6910 | -2.1979 | 2.7595 | 1.6819 |
| MOSS | 2.0949 | 2.1464 | 1.6667 | 2.1533 | +0.4866 | 2.2373 | 0.9171 |
| MiniCPM | 1.6611 | 1.6474 | 3.3333 | 1.4331 | -1.9002 | 1.5212 | 0.1371 |
| Gander | 1.2365 | 1.2400 | 2.6667 | 1.0415 | -1.6252 | 1.0967 | 0.2286 |
| Venus | 1.3190 | 1.2395 | 3.1333 | 1.0716 | -2.0617 | 1.1390 | 0.0800 |

## Per-judge results

| Entity | Judge | Overall | WAIT | Non-WAIT | Non-WAIT minus WAIT |
| --- | --- | ---: | ---: | ---: | ---: |
| Assisted Human Baseline | Qwen (primary) | 2.8048 | 4.8889 | 2.5206 | -2.3683 |
| Assisted Human Baseline | HY3 | 2.9835 | 4.8889 | 2.7236 | -2.1653 |
| Assisted Human Baseline | DeepSeek | 2.9427 | 4.8889 | 2.6773 | -2.2116 |
| Assisted Human Baseline | GLM | 3.0397 | 4.8889 | 2.7876 | -2.1013 |
| Assisted Human Baseline | Kimi | 3.0029 | 4.8889 | 2.7458 | -2.1431 |
| MOSS | Qwen (primary) | 1.8424 | 1.6667 | 1.8664 | +0.1997 |
| MOSS | HY3 | 2.2096 | 1.6667 | 2.2836 | +0.6169 |
| MOSS | DeepSeek | 2.1384 | 1.6667 | 2.2027 | +0.5360 |
| MOSS | GLM | 2.2056 | 1.6667 | 2.2791 | +0.6124 |
| MOSS | Kimi | 2.0784 | 1.6667 | 2.1345 | +0.4678 |
| MiniCPM | Qwen (primary) | 1.5280 | 3.3333 | 1.2818 | -2.0515 |
| MiniCPM | HY3 | 1.7320 | 3.3333 | 1.5136 | -1.8197 |
| MiniCPM | DeepSeek | 1.7008 | 3.3333 | 1.4782 | -1.8551 |
| MiniCPM | GLM | 1.7088 | 3.3333 | 1.4873 | -1.8460 |
| MiniCPM | Kimi | 1.6360 | 3.3333 | 1.4045 | -1.9288 |
| Gander | Qwen (primary) | 1.1728 | 2.6667 | 0.9691 | -1.6976 |
| Gander | HY3 | 1.2112 | 2.6667 | 1.0127 | -1.6540 |
| Gander | DeepSeek | 1.2688 | 2.6667 | 1.0782 | -1.5885 |
| Gander | GLM | 1.2704 | 2.6667 | 1.0800 | -1.5867 |
| Gander | Kimi | 1.2592 | 2.6667 | 1.0673 | -1.5994 |
| Venus | Qwen (primary) | 1.2448 | 3.3333 | 0.9600 | -2.3733 |
| Venus | HY3 | 1.3280 | 3.3333 | 1.0545 | -2.2788 |
| Venus | DeepSeek | 1.3928 | 3.3333 | 1.1282 | -2.2051 |
| Venus | GLM | 1.3008 | 2.3333 | 1.1600 | -1.1733 |
| Venus | Kimi | 1.3288 | 3.3333 | 1.0555 | -2.2778 |

## Judge sensitivity

| Entity | Overall range across judges | WAIT range | Non-WAIT range | Overall spread |
| --- | ---: | ---: | ---: | ---: |
| Assisted Human Baseline | 2.8048-3.0397 | 4.8889-4.8889 | 2.5206-2.7876 | 0.2349 |
| MOSS | 1.8424-2.2096 | 1.6667-1.6667 | 1.8664-2.2836 | 0.3672 |
| MiniCPM | 1.5280-1.7320 | 3.3333-3.3333 | 1.2818-1.5136 | 0.2040 |
| Gander | 1.1728-1.2704 | 2.6667-2.6667 | 0.9691-1.0800 | 0.0976 |
| Venus | 1.2448-1.3928 | 2.3333-3.3333 | 0.9600-1.1600 | 0.1480 |

## Interpretation

- Assisted Human Baseline scores `4.8889` on WAIT and `2.6910` on non-WAIT; the `-2.1979` gap reflects the frozen action and content rubric, not a claim that all Baseline answers are uniformly strong.
- Among the four models, `MiniCPM` has the highest consensus WAIT mean (`3.3333`), while `MOSS` has the highest non-WAIT mean (`2.1533`). WAIT rankings therefore cannot stand in for overall response quality.
- The largest overall spread across judge families is for `MOSS`: `1.8424` to `2.2096`. Judge-specific tables and row-level evidence should accompany any single-score ranking.
- These are AI-judge results on an assisted Baseline condition. They are not independent human validation; retain the Baseline and Preference provenance warnings in the archive README.

## Source files

- `source/model_judge_scores.json`: full 2,500-row normalized judge output with per-point evidence and forbidden-claim evidence.
- `source/model_judge_record_scores.json`: 500 records with all five judge scores and consensus values.
- `source/model_judge_comparison.json` and `.md`: frozen model/Human comparison analysis.
- `source/model_judge_summary.json`: source-run summary.
- `source/model_judge_manifest.json`: path-redacted run manifest.
- The raw API cache is intentionally excluded; it can contain provider payloads and is not needed to audit normalized scores.
