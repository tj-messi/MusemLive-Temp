# Reproducibility and artifact boundary

## Input-rate contract

This is the finalized **1 FPS** snapshot. Model decisions occur once per
second. At each decision tick the native-audio evaluators receive one visual
frame and the causally aligned one-second audio block; MOSS receives one frame
plus the aligned ideal-ASR text used by the original experiment. No future
frames or future events are sent.

The finalized 10 FPS rerun is a separate artifact family with ten ordered
frames per one-second decision tick. Its outputs must not be copied into this
directory or combined with these aggregate JSON files. The cross-track rules
and output boundaries are documented in
[`../../EVALUATION_LOOP.md`](../../EVALUATION_LOOP.md).

## Included snapshot

This directory contains the exact report, aggregate JSON, figures, analysis
scripts, orchestration scripts, evaluator variants, and point-based judge used
to finalize Experiments 1–7 on 2026-10-19.

The completed validation covered:

- 12 experiment runs, each with the expected result count and zero API failures.
- 16 alternate-judge runs: four judges by four models, each 1,250/1,250.
- Six SVG figures parsed successfully.
- The recent-30-second runs each contain 1,000 scored user-query slots. Of
  these, 810 have the same start point, model input, output, judge ID, and
  rubric as the formal run; 190 were rerun and re-judged. This reuse is recorded
  in the corresponding source summaries and provenance files.

## Rebuild aggregate analyses

The analysis scripts expect the original run tree used by the project. With
that external tree restored, run:

```bash
python3 scripts/analysis/analysis_a.py
python3 scripts/analysis/analysis_b.py
python3 scripts/analysis/build_report_1_7.py
python3 scripts/analysis/verify_1_7.py
```

The scripts are preserved as an exact execution snapshot and currently contain
paths relative to the original `work/` experiment layout. They are not intended
to silently download controlled data or model outputs.

## Runner snapshots

- `orchestrate.py`: forced-answer, blind-context, and modality-ablation runs.
- `offline_upper.py`: offline Qwen-Omni upper-bound generation.
- `blind_tasks.py`: 3-second blind-context task construction.
- `recent30_tasks.py` and `run_recent30.py`: recent-30-second task construction,
  optimized inference, judgment reuse, and result merge.

The `scripts/evaluators/` files preserve the experiment switches used for
forced answering, input ablation, transcript removal, and multi-judge scoring.

## Intentionally external artifacts

The following are excluded under the repository policy:

- source video and audio;
- model weights and local model caches;
- synthesized TTS audio and extracted frames;
- full per-slot model generations and judge raw-response caches;
- controlled task/media maps containing local storage paths.

The aggregate result JSON and reports are sufficient to inspect every reported
metric, confidence interval, model ranking, ablation, and completeness check.
