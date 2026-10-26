# MuseumLive experiments: isolated 1 FPS and 10 FPS tracks

This directory records the completed Experiment 1–7 evaluation suites run on
the 450-question manually selected MuseumLive benchmark set. The 1 FPS snapshot
was finalized on 2026-10-19 and the independent 10 FPS snapshot on 2026-10-21.

> **FPS boundary:** the detailed tables below belong to the finalized **1 FPS**
> track. The finalized **10 FPS** track is under [`10FPS/`](10FPS/), with its
> own [summary](10FPS/reports/summary.md), machine-readable results, figures,
> and manifest. The two snapshots preserve different Judge profiles and must
> not be presented as a controlled FPS-only comparison without harmonized
> rescoring. See [`EVALUATION_LOOP.md`](EVALUATION_LOOP.md).

The provenance-audited human-evaluation archive is documented in
[`human_evaluation_2026-10-24/`](human_evaluation_2026-10-24/README.md). It
contains 375 GroundTruth-assisted Baseline records, 625 batch-calibrated
Preference pseudo-ratings, seven-candidate A–G mappings, five-judge Baseline
scores, provenance, and SHA-256 checksums. The Preference records are
quarantined from human-validation analysis.

## Evaluation scope

- 450 tasks and 1,250 scored slots.
- 1,000 user-query slots, 100 visual-event `SPEAK` slots, and 150
  visual-event `WAIT` slots.
- Main models: MiniCPM-o 4.5, Venus, Gander, and MOSS.
- Main score: fixed point-based judge, 0–5.
- MOSS receives visual input plus ideal-ASR scene transcripts rather than raw
  scene audio; its modality setting is therefore not identical to the native
  audio models.

## Headline results

| Model | Overall | User-query | User-query response rate | Quality when answered | Timing F1 |
|---|---:|---:|---:|---:|---:|
| MOSS | **1.94** | **2.01** | **92.9%** | **2.16** | 0.64 |
| MiniCPM-o 4.5 | **1.59** | 1.31 | 81.3% | 1.61 | **0.77** |
| Gander | 1.24 | 0.91 | 47.4% | 1.92 | 0.62 |
| Venus | 1.16 | 0.79 | 48.5% | 1.63 | 0.59 |

All five judges produced the same ranking:

> **MOSS > MiniCPM-o 4.5 > Gander > Venus**

The ranking is stable, but the models fail in different ways. MOSS responds
most often and scores highest, but over-speaks: its false-alarm rate on slots
that require silence is 77%. MiniCPM provides the best timing balance. Gander
and Venus are limited primarily by deciding not to answer.

## Experiment summary

### 1. Response rate, conditional quality, and joint score

User-query response-rate / conditional-quality / joint-score:

- MiniCPM: 81% / 1.61 / 1.31.
- Venus: 48% / 1.63 / 0.79.
- Gander: 47% / 1.92 / 0.91.
- MOSS: 93% / 2.16 / 2.01.

The reported scores use the final response-matching rule: a new reply starts
after user speech ends and before the next user event. For MOSS, the question
text must also have been delivered. Responses outside this window are not used
as answers to the preceding query. MOSS remains grouped separately because its
question interface is text mediated rather than native audio.

### 2. Timing quality

- MiniCPM: event recall 0.82, timing precision 0.72, F1 **0.77**.
- Venus: event recall 0.53, timing precision 0.66, F1 0.59.
- Gander: event recall 0.51, timing precision **0.81**, F1 0.62.
- MOSS: event recall **0.91**, timing precision 0.50, F1 0.64.
- The always-speaking baseline has F1 0.54.

MOSS is fast and high-recall but talks too frequently. Gander is conservative
and precise but misses many events. MiniCPM achieves the best balance.

### 3. Evidence-distance and data stratification

Scores generally decline as relevant evidence moves farther from the query,
but the `>=30 s` bucket has only two samples and is not statistically useful.
The source clips are usually 10–15 minutes long, while the evaluated session
provided to a model is normally only tens of seconds (median about 23 s).
These results therefore measure short-session streaming interaction over long
video sources, not hour-long continuous-context processing.

### 4. Judge robustness and bootstrap

The Qwen, HY3, DeepSeek, GLM, and Kimi judges all produce the same model order,
with Kendall tau 1.0 against the main judge. Answered-slot correlation with the
main judge is 0.80–0.86. Cluster bootstrap also separates MOSS from MiniCPM and
MiniCPM from Gander/Venus. HY3 gives its related Gander model a small relative
advantage of approximately +0.08, insufficient to change the ranking.

### 5. Forced-answer intervention

| Model | Silent user-query slots | Mean after forcing on silent slots | >=2 score | Original user-query | Forced user-query |
|---|---:|---:|---:|---:|---:|
| Venus | 515 | 1.21 | 28% | 0.79 | **1.36** |
| Gander | 526 | **1.50** | **36%** | 0.91 | **1.65** |
| MiniCPM | 187 | 1.06 | 24% | 1.31 | **1.42** |

Paired improvements are +0.565 for Venus, +0.742 for Gander, and +0.110 for
MiniCPM; all three 95% confidence intervals exclude zero. A substantial part
of the Venus/Gander deficit comes from the listen/speak decision rather than
answer generation alone, although 43–54% of forced silent cases still score
zero.

### 6. Modality ablation

| Condition | Overall | User-query | Paired delta |
|---|---:|---:|---:|
| MiniCPM full | 1.59 | 1.31 | - |
| MiniCPM scene audio muted | **1.79** | **1.56** | +0.20 [0.09, 0.31] |
| MiniCPM black frames | 1.28 | 0.86 | -0.32 [-0.42, -0.22] |
| MOSS visual + transcript | **1.94** | **2.01** | - |
| MOSS without scene transcript | 1.77 | 1.77 | -0.17 [-0.29, -0.03] |
| MOSS transcript only / black frames | 1.52 | 1.21 | -0.42 [-0.57, -0.26] |

MiniCPM clearly uses visual evidence, but the current raw scene-audio channel
acts as a net distraction. MOSS benefits from both vision and ideal-ASR scene
transcripts, with vision contributing the larger effect.

### 7. Context shortcuts and offline upper bound

| Condition | MiniCPM | MOSS |
|---|---:|---:|
| Full streaming context | 1.31 | 2.01 |
| Most recent 30 seconds | 1.27 | 1.93 |
| Most recent 3 seconds | 1.18 | 1.79 |

The paired drops are statistically non-zero, but modest. Historical reference
is the most context-sensitive capability: MiniCPM falls 0.98 -> 0.54 -> 0.46,
and MOSS falls 1.61 -> 0.68 -> 0.68. Other capabilities often retain local
visual or linguistic shortcuts. The offline Qwen-Omni upper bound is 3.15,
showing a large remaining gap attributable to online timing, response policy,
context maintenance, and multimodal fusion.

## Additional Experiments 8–13

The same frozen runs also support six stateful-interaction diagnostics under a
harmonized `strict_v2` Judge profile. No evaluated-model inference is added.
Experiment 11 performs only resumable sibling-rubric Judge calls; its raw API
cache remains outside Git.

- **8:** sequential multi-stage chain completion.
- **9:** counterfactual joint completion and strict PairSuccess.
- **10:** `SPEAK`/`WAIT`/`CLARIFY` action calibration and named dual conditions.
- **11:** counterfactual branch swap sensitivity.
- **12:** quality-weighted temporal alignment and timing-window sensitivity.
- **13:** source-cluster uncertainty and paired 1 FPS ↔ 10 FPS sensitivity.

Results and reproduction details:

- [`1FPS/experiments_8_13/`](1FPS/experiments_8_13/): 1 FPS aggregates and reports.
- [`10FPS/experiments_8_13/`](10FPS/experiments_8_13/): 10 FPS aggregates and reports.
- [`experiments_8_13/`](experiments_8_13/): shared scripts, Experiment 13, provenance, validation, and manifests.

## Contents

- [`1FPS/experiments_1_7/reports/summary.md`](1FPS/experiments_1_7/reports/summary.md): concise final report.
- [`1FPS/experiments_1_7/reports/analysis_1_4.md`](1FPS/experiments_1_7/reports/analysis_1_4.md): complete Experiments 1–4 tables.
- [`1FPS/experiments_1_7/reports/analysis_5_7.md`](1FPS/experiments_1_7/reports/analysis_5_7.md): complete Experiments 5–7 tables.
- [`1FPS/experiments_1_7/results/`](1FPS/experiments_1_7/results): machine-readable analysis and validation summaries.
- [`1FPS/experiments_1_7/figures/`](1FPS/experiments_1_7/figures): six SVG figures.
- [`1FPS/experiments_1_7/scripts/`](1FPS/experiments_1_7/scripts): analysis, runner, evaluator, and judge snapshots used for this result set.
- [`1FPS/experiments_1_7/REPRODUCIBILITY.md`](1FPS/experiments_1_7/REPRODUCIBILITY.md): dependencies, commands, and external-artifact boundary.
- [`1FPS/experiments_1_7/manifests/`](1FPS/experiments_1_7/manifests): completeness evidence and SHA-256 checksums.
- [`10FPS/`](10FPS/): finalized 10 FPS contract, ordered matrix, status snapshot, reports, figures, results, and validation manifests.
- [`experiments_8_13/`](experiments_8_13/): cross-track Experiments 8–13 definitions, scripts, paired-FPS analysis, and validation.

Raw media, model weights, API caches, and per-slot model generations are not
committed, following the repository data policy. The committed JSON files
contain the complete aggregated numerical results and completeness checks.
