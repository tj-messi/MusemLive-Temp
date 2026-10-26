# Experiments 8--13: stateful interaction diagnostics

This package extends the finalized Experiment 1--7 snapshots without running
new model inference. It consumes the frozen task file, existing `strict_v2`
per-slot Judge labels, and saved streaming timelines from both FPS tracks.

## Experiment definitions

8. **Sequential chain completion.** Multi-stage branches are evaluated as a
   chain rather than as independent slots. The soft chain score is the
   geometric mean of normalized slot scores. The report also includes strict
   all-stage completion, local first/final-stage quality, missed-final rates,
   and final-stage failure conditional on first-stage success.
9. **Counterfactual joint completion.** The two independently executed
   branches of each task must both follow their branch-specific state. We
   report the existing strict `PairSuccess` rule, soft Counterfactual Joint
   Completion Score (CJCS), and the both/left-only/right-only/neither table.
10. **Interaction-control calibration.** Expected `SPEAK`, `WAIT`, and
    `CLARIFY` actions are separated from factual content. Reports include the
    action confusion matrix, macro action F1, and named dual-condition tests
    such as keep-monitor versus cancel-monitor.
11. **Counterfactual branch sensitivity.** For target-incompatible paired
    slots, responses are cross-scored against the sibling branch rubric using
    the same point-matching Judge and deterministic `strict_v2` score rule.
    Exact-output invariance and lexical similarity are reported alongside the
    own-target versus swapped-target score gap. Raw API caches remain outside
    Git.
12. **Quality-weighted temporal alignment.** The formal timing window is
   combined with normalized answer quality as a soft event F1. A separate
   timing-only sensitivity analysis uses tight, medium, and formal windows;
   it does not reassign or re-Judge response content.
13. **Uncertainty and FPS sensitivity.** Metrics are bootstrapped by source
    video, preserving all tasks that share a source. Cross-FPS comparisons use
   paired task values and the same `strict_v2` scoring profile.

For Experiment 12, a required in-window response with normalized quality
`q=score/5` contributes `TP=q`, `FP=1-q`, and `FN=1-q`. Silence contributes
`FN=1`; an early/late attempt contributes `FP=1,FN=1`; a substantive response
in a `WAIT` slot and each unmatched turn contribute `FP=1`. The formal user
window is from 0.5 seconds before TTS end through 6 seconds after TTS end; the
formal visual window ends 3 seconds after the annotated slot window.

Experiment 13's `task_macro_score` first averages slot scores within each task
and then averages the 450 paired task values. It is deliberately named
differently from the slot-weighted Experiment 1 leaderboard mean.

## Important boundaries

- These experiments add diagnostics; they do not change the Experiment 1--7
  leaderboards.
- The 1 FPS and 10 FPS comparisons use `judge_points/`, not the 10 FPS
  `judge_points_lenient_v3/` summaries.
- MOSS remains a transcript-assisted visual reference and is not input-matched
  to the native-audio systems.
- Experiment 11 makes Judge API calls but no evaluated-model inference calls.
- One 10 FPS Gander cross-direction was rejected by the sole Qwen endpoint's
  input inspection. The identical text was resubmitted in reversible Unicode
  escape notation to the same model; the rubric, response, prompt, labels, and
  deterministic score rule were unchanged. The audit record is
  [`results/experiment_11_fallback.json`](results/experiment_11_fallback.json).
- Matched prefix-fork readout probes and pre/post event probes require new
  model inference and are deliberately excluded.

## Reproduction

The aggregate analysis is intentionally parameterized by external artifact
roots:

```bash
python3 Experiment/experiments_8_13/scripts/analyze_existing_runs.py \
  --tasks /path/to/tasks_final450.json \
  --one-fps-root /path/to/model_compare_20261018 \
  --ten-fps-root /path/to/exp_10fps_20261019/runs \
  --repo-root .
```

Experiment 11 is resumable and writes raw Judge responses outside the
repository:

```bash
OPENROUTER_API_KEY=... \
python3 Experiment/experiments_8_13/scripts/run_swap_judge.py \
  --tasks /path/to/tasks_final450.json \
  --one-fps-root /path/to/model_compare_20261018 \
  --ten-fps-root /path/to/exp_10fps_20261019/runs \
  --cache-root /path/to/external/swap-cache \
  --repo-root .
```

Finally run:

```bash
python3 Experiment/experiments_8_13/scripts/verify_8_13.py --repo-root .
```

To additionally verify the committed provenance hashes against the external
frozen inputs, provide the same roots used for analysis:

```bash
python3 Experiment/experiments_8_13/scripts/verify_8_13.py \
  --repo-root . \
  --check-external \
  --tasks /path/to/tasks_final450.json \
  --one-fps-root /path/to/model_compare_20261018 \
  --ten-fps-root /path/to/exp_10fps_20261019/runs
```
