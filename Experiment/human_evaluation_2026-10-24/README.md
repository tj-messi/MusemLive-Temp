# Assisted Human Baseline and batch-calibrated Preference archive

**Provenance warning:** this is not an independent human-validation release.
The 375 Baseline answers were collected with GroundTruth visible, so they are an
assisted-reference condition. The 625 Preference records are batch-generated,
GT-calibrated pseudo-ratings rather than independent rating sessions. Every derived
Preference row is marked `analysis_eligible=false` and must not be used as human
preference evidence without recollection and revalidation.

This archive preserves the complete 45-task, 125-slot source snapshot for audit and
reuse. All annotator identifiers are pseudonymous. No media, credentials, names,
email addresses, model weights, or API caches are included.

## Snapshot

- Annotation record revision: `4723`.
- Assisted Human Baseline: `375` records from B01-B03, with GroundTruth visible.
- Batch-calibrated Preference: `625` pseudo-ratings from P01-P05; not independent human sessions.
- Five judges: Qwen primary, HY3, DeepSeek, GLM, and Kimi.
- Preference ratings: seven candidates scored from 0 to 5 by every rater.
- Candidate snapshot generated: `2026-10-24T10:00:36.936Z`.
- Preference submissions: `2026-10-24T14:21:49.215Z` to `2026-10-24T14:21:50.463Z`.
- Batch audit: `1.248` s submission span, `560` placeholder `12000 ms` durations, `624` GT-calibration notes.

## Five-judge Human-vs-model comparison

Same 45 tasks and 125 slots. Assisted Human Baseline averages B01-B03 after
taking the five-judge consensus for each slot.

| Entity | Overall | Task-balanced | WAIT | Non-WAIT | SPEAK | CLARIFY |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Assisted Human Baseline | 2.9547 | 3.0568 | 4.8889 | 2.6910 | 2.7595 | 1.6819 |
| MOSS | 2.0949 | 2.1464 | 1.6667 | 2.1533 | 2.2373 | 0.9171 |
| MiniCPM | 1.6611 | 1.6474 | 3.3333 | 1.4331 | 1.5212 | 0.1371 |
| Gander | 1.2365 | 1.2400 | 2.6667 | 1.0415 | 1.0967 | 0.2286 |
| Venus | 1.3190 | 1.2395 | 3.1333 | 1.0716 | 1.1390 | 0.0800 |

The score distribution is strongly bimodal. Assisted Human Baseline is `4.89` on
`WAIT` slots but `2.69` on slots that require speech or clarification.

## Four-model five-judge detail

The release now includes all `2,500` row-level scores
and `500` five-judge record summaries.
No model judge call failed. `data/judge_scope_analysis.csv` separates five-judge
consensus and each judge family across overall, WAIT, non-WAIT, SPEAK, and CLARIFY.

| Entity | Judge overall range | Judge WAIT range | Judge non-WAIT range |
| --- | ---: | ---: | ---: |
| Assisted Human Baseline | 2.8048-3.0397 | 4.8889-4.8889 | 2.5206-2.7876 |
| MOSS | 1.8424-2.2096 | 1.6667-1.6667 | 1.8664-2.2836 |
| MiniCPM | 1.5280-1.7320 | 3.3333-3.3333 | 1.2818-1.5136 |
| Gander | 1.1728-1.2704 | 2.6667-2.6667 | 0.9691-1.0800 |
| Venus | 1.2448-1.3928 | 2.3333-3.3333 | 0.9600-1.1600 |

WAIT and non-WAIT are reported separately because the frozen rubric rewards a
correct silent action differently from content completeness. The detailed report
also shows every judge family and the score spread around each consensus mean.

**Scoring-snapshot caveat:** the Baseline judge summary records annotation revision
`3538`, while the packaged final annotation snapshot is revision `4723`. Judge rows are
joined to Baseline records by pseudonymous profile, slot UID, and submission time. The
earlier raw annotation snapshot is not bundled, so the archive supports row-level score
audit but not a byte-for-byte reconstruction from that earlier revision.

## Fingerprint integrity note

Preference scores are joined to candidates by `uid + candidate_id`; the resolved
candidate text, observed state, action, response time, recorded fingerprint, and
current fingerprint are all retained in `preference_assignments.csv`.

Of 4,375 candidate assignments, 2,135 recorded fingerprints equal the current `v1` serializer and 2,240 do not. All mismatches are model assignments (`560` per model); all 1,875 Human Baseline assignments match. The mismatches come from the batch score migration rather than a later candidate update: the frozen candidate snapshot was generated before the earliest Preference submission. All 2,240 mismatches are explained by historical response-time serialization: the current resolved response time is empty for those assignments, while candidate text, observed state, and action still match. Raw fingerprints and scores are preserved rather than rewritten, and the boolean `fingerprint_matches_current_candidate` marks the distinction explicitly.

## Batch-calibrated Preference diagnostic (not human validation)

These raw score means are included only for provenance and debugging. They are not
valid Human Preference results; all rows have `analysis_eligible=false`.

| Candidate | Overall | WAIT | Non-WAIT | SPEAK | CLARIFY |
| --- | ---: | ---: | ---: | ---: | ---: |
| MOSS | 2.37 | 1.67 | 2.47 | 2.53 | 1.54 |
| MiniCPM | 1.71 | 3.05 | 1.53 | 1.56 | 1.11 |
| Gander | 1.40 | 2.52 | 1.24 | 1.26 | 1.00 |
| Venus | 1.34 | 2.36 | 1.20 | 1.23 | 0.77 |
| Human B01 | 3.15 | 4.13 | 3.01 | 3.05 | 2.49 |
| Human B02 | 3.14 | 3.99 | 3.03 | 3.06 | 2.51 |
| Human B03 | 3.05 | 4.17 | 2.90 | 2.91 | 2.66 |

## Files

- `data/items.jsonl`: 125 slots, ground truth, causal events, and seven candidate responses.
- `data/baseline_records.jsonl` and `.csv`: 375 assisted Human Baseline answers.
- `data/preference_records.jsonl`: 625 batch-calibrated pseudo-ratings with A-G mappings and seven scores.
- `data/preference_assignments.csv`: long-form 4,375 quarantined rows with A-G mappings, scores, resolved candidate responses, fingerprint-match audit flags, and analysis blockers.
- `data/baseline_judge_details.jsonl`: 1,875 normalized five-judge scores with semantic evidence.
- `data/model_judge_details.jsonl`: 2,500 normalized four-model five-judge scores with per-point evidence.
- `data/model_judge_record_scores.csv`: 500 model records with five-judge consensus scores.
- `data/judge_scope_analysis.csv`: entity x judge/consensus x scope aggregate analysis.
- `data/model_comparison.csv`: five-judge Human-vs-model table and capability breakdown.
- `data/preference_summary.csv`: diagnostic batch-score aggregates, explicitly marked ineligible for human-validation analysis.
- `analysis/wait_nonwait_report.md`: WAIT/non-WAIT findings, per-judge results, and judge sensitivity.
- `analysis/model_judge_comparison.md` and `.json`: frozen source comparison analysis.
- `source/`: immutable source snapshots and selected upstream score artifacts.
- `manifests/`: provenance, completeness checks, and SHA-256 checksums.

## Reproduce packaged artifacts

From the repository root:

```bash
python Experiment/human_evaluation_2026-10-24/scripts/build_release.py
python Experiment/human_evaluation_2026-10-24/scripts/verify_release.py
```

The build script reads only checked-in snapshots and standard-library Python. It rebuilds
the packaged tables from normalized source rows; it does not call judge APIs. Re-running
the judges or changing the annotation source requires the frozen scoring harness and
rubric outside this archive.
