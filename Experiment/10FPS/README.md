# Finalized 10 FPS Experiment 1–7 track

This track repeats the MuseumLive Experiment 1–7 schedule with ten ordered
visual frames per one-second decision tick. Audio/text alignment, task
definitions, branch independence, and the 1 Hz decision cadence remain fixed.

The snapshot was finalized on 2026-10-21 after all formal runs, recent-30
baselines, alternate-Judge panels, aggregate analyses, reports, figures, and
provenance checks passed.

## Completion

- 13/13 scheduled runs and 11,900/11,900 branches complete.
- Every main summary has the expected slot count and `api_failed = 0`.
- MiniCPM and MOSS recent-30 runs each have 1,000/1,000 scored slots.
- Four alternate Judges × four main models = 16/16 complete panels.
- The offline Qwen-Omni upper bound, six figures, aggregate JSON/Markdown
  reports, SHA-256 manifest, and validation record are complete.

The complete matrix and final status are in [`run_matrix.json`](run_matrix.json)
and [`status.json`](status.json).

## Results

- [`reports/summary.md`](reports/summary.md): concise Experiment 1–7 report.
- [`reports/analysis_1_4.md`](reports/analysis_1_4.md): response, timing,
  stratification, alternate-Judge, and bootstrap analyses.
- [`reports/analysis_5_7.md`](reports/analysis_5_7.md): forced-answer,
  modality-ablation, context-baseline, and offline-upper-bound analyses.
- [`results/`](results/): aggregate JSON, all main/recent-30 summaries,
  16 alternate-Judge summaries, and integrity provenance.
- [`figures/`](figures/): six SVG figures.
- [`manifests/`](manifests/): artifact inventory, checksums, and validation.

The main `lenient_v3` scores are MiniCPM 0.97, Venus 1.22, Gander 0.89,
and MOSS 1.61. These are not a controlled 1 FPS versus 10 FPS comparison: the
finalized 1 FPS snapshot preserves its historical Judge profile, whereas this
track uses `lenient_v3`. Compare model behavior within a track unless both
tracks are rescored with the same frozen match outputs/profile.

## Integrity disclosures

- One forced-Gander slot was blocked by provider data inspection and was
  completed by an offline fixed-rubric review without changing the model
  response.
- Alternate-Judge provider JSON truncation affected 22 slots. Retrying the
  same Judge model through a compatible provider and, for 17 slots, a compact
  JSON schema completed the labels. Model outputs, rubric, and deterministic
  scoring were unchanged.
- Recent-30 reused 810 formally identical outputs/judgments per model and
  reran/rejudged the 190 slots whose context actually changed.
- The offline upper bound reused a validated cache; 999 successful timing
  pairs matched and one original API error remained an empty response.

See [`results/provenance/`](results/provenance/) for machine-readable records.

## Reproduction

The executable, environment-configurable generation/Eval/Judge snapshot is in
[`../../pipelines/10-20/`](../../pipelines/10-20/). Run the public metadata and
final external-artifact checks with:

```bash
python3 Experiment/validate_fps.py

EXPERIMENT_OUTPUT_ROOT=/path/to/external/10fps-run \
JUDGE_DIR_NAME=judge_points_lenient_v3 \
python3 pipelines/10-20/postprocessing/verify_1_7.py
```

Raw generations, per-slot Judge caches, media, model weights, synthesized
audio, controlled task data, and credentials remain outside Git under the
repository artifact policy.

## Additional Experiments 8–13

The frozen 10 FPS outputs were also analyzed under the harmonized `strict_v2`
profile without new evaluated-model inference. See
[`experiments_8_13/reports/summary_8_12.md`](experiments_8_13/reports/summary_8_12.md)
for Experiments 8–10 and 12, and
[`experiments_8_13/reports/experiment_11_swap.md`](experiments_8_13/reports/experiment_11_swap.md)
for counterfactual branch sensitivity. Paired cross-FPS Experiment 13 and the
reproduction scripts are under [`../experiments_8_13/`](../experiments_8_13/).
