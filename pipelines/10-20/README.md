# MuseumLive 10-20 pipeline: source video to QA, Eval, and Judge

This directory freezes the end-to-end protocol that produced the current QA
snapshot and the code path used for the 2026-10-20 MuseumLive 10 FPS rerun.
Start with [`PIPELINE_OVERVIEW.md`](PIPELINE_OVERVIEW.md): it follows the real
lineage from source filtering, 25 FPS visual/audio annotation, QA generation
and human freezing through causal Eval, Judge Prompt, deterministic scoring,
Experiments 1–7, and final reports.

The finalized 10 FPS run is intentionally separate from the finalized 1 FPS
snapshot under [`../../Experiment/1FPS/`](../../Experiment/1FPS/) and writes
every runtime artifact to an external `MUSEUMLIVE_10FPS_RUN_ROOT`. Its public
aggregate snapshot is under [`../../Experiment/10FPS/`](../../Experiment/10FPS/).

## Fixed FPS contract

| Property | Finalized 1 FPS | This 10 FPS rerun |
|---|---:|---:|
| Decision cadence | 1 Hz | 1 Hz |
| Ordered visual frames per tick | 1 | 10 |
| Audio/text interval | causal 1 second | same causal 1 second |
| Future evidence | never | never |
| Formal tasks | 450 | same frozen 450 |
| Formal branches per run | 900 | 900 |
| Blind branches per run | 1,000 | 1,000 |

Ten frames are sampled uniformly from each frozen 25 FPS source second. The
action clock is not accelerated: each model still makes one listen/speak
decision per causal second. See
[`../../Experiment/EVALUATION_LOOP.md`](../../Experiment/EVALUATION_LOOP.md)
for the full task/branch/slot accounting and completion states.

## Loop implemented here

```text
frozen task JSON + media manifest
  -> seven persistent shards per condition
  -> strict-priority scheduler on GPUs 2,3,4,5,6,7
  -> 10 ordered frames + aligned audio/text per 1-second tick
  -> resume missing branch keys, or rebalance a long shard tail
  -> atomic merge into the original shard
  -> lenient_v3 point judge and slot-count/API-failure checks
  -> alternate judges + context/offline runs
  -> aggregate JSON, reports, figures, and validation
```

The scheduler never starts a later condition while an earlier condition has
unfinished branches. Temporary tail-worker outputs live under `rebalance/`,
outside the `runs/<condition>/shard*/...` judge glob, so merged branches cannot
be scored twice.

## Files

- `PIPELINE_OVERVIEW.md`: authoritative end-to-end loop, executed counts,
  artifact boundary, 1 FPS/10 FPS contract, and completion gates.
- `PROVENANCE.md`: source hashes and the exact publication-only
  sanitizations applied to executed generation code.
- `generation/`: executed visual annotation plus QA generation, verification,
  post-processing, selection, and frozen-task validation snapshots.
- `prompts/`: exact visual/audio annotation, QA authoring/verification, and
  point-Judge prompts.
- `protocols/`: task schema and measured QA-generation protocol history.
- `runners/orchestrate_10fps.py`: ordered 13-run scheduler, resume logic,
  branch progress, judging, and GPU 2–7 placement.
- `runners/rebalance_tail.py`: distributes only missing branch keys, retries
  workers, atomically merges results, validates completeness, and resumes the
  scheduler. MiniCPM, Venus, Gander, and MOSS are supported.
- `evaluators/`: the exact 10 FPS-capable MiniCPM/Venus, Gander, MOSS, shared
  media/TTS, and point-judge snapshots. Local paths are supplied by
  environment variables rather than embedded in source.
- `postprocessing/postprocess_10fps.sh`: waits for all 13 evaluated-and-judged
  runs, launches alternate judges, context/offline baselines, and rebuilds the
  Experiment 1–7 analyses.
- `postprocessing/*.py`: environment-configurable analysis, report, offline
  upper-bound, and final completeness checks used by the final stage.
- `env.example`: names every required external path, including the evaluator
  Python dependency and Qwen-TTS index roots, without storing secrets.

The ordered run definitions and expected counts are machine-readable in
[`../../Experiment/10FPS/run_matrix.json`](../../Experiment/10FPS/run_matrix.json).

## Run

Set the values shown in `env.example` in a private shell or job environment.
Do not put API keys in that file or in Git.

The frozen tasks and rubric remain external controlled benchmark artifacts;
provide them with `MUSEUMLIVE_TASKS_FILE` and `MUSEUMLIVE_RUBRIC_FILE`.
Evaluator-only Python packages and Qwen-TTS indexes remain external too; set
`MUSEUMLIVE_EVAL_PYLIB`, `MUSEUMLIVE_TTS_INDEX_ROOT`, and `MOSS_PYLIB_ROOT`
to the validated checkout paths. The repository publishes expected hashes,
schema validation, code, and prompts without committing private task data or
machine-local paths.

```bash
set -a
source /path/to/private/museumlive-10fps.env
set +a

python3 pipelines/10-20/runners/orchestrate_10fps.py
python3 pipelines/10-20/runners/orchestrate_10fps.py status
bash pipelines/10-20/postprocessing/postprocess_10fps.sh
EXPERIMENT_OUTPUT_ROOT="$MUSEUMLIVE_10FPS_RUN_ROOT" \
  python3 pipelines/10-20/postprocessing/verify_1_7.py
```

For a single long-tail `shard6`, first stop the regular scheduler process so
it cannot relaunch the same shard, then run for the matching model/condition:

```bash
kill -STOP ORCHESTRATOR_PID
python3 pipelines/10-20/runners/rebalance_tail.py \
  --orchestrator-pid ORCHESTRATOR_PID \
  --run-name force_venus --model venus
```

The helper sends `SIGCONT` only after the original shard contains every
expected branch. If the scheduler no longer exists, use
`--restart-orchestrator` instead.

## Publication boundary

The scheduled 10 FPS judge profile is `lenient_v3`, while the finalized 1 FPS
snapshot preserves its original fixed judge configuration. Until the judge
configuration is harmonized, raw 1 FPS and 10 FPS scores are not a controlled
FPS-only comparison. `Experiment/10FPS/status.json` and the adjacent reports,
figures, summaries, provenance, and manifest form the finalized 10 FPS
publication snapshot.

The external run and public Experiment snapshot intentionally use different
layouts: `report_1_7.md` links to `figures_1_7/` beside the runtime report,
whereas `Experiment/10FPS/reports/summary.md` links to `../figures/` and the
renamed public analysis/result files. After staging the figures, reports, and
results, publish the runtime report and rebuild the checksums with:

```bash
python3 pipelines/10-20/postprocessing/build_public_manifest.py \
  --root Experiment/10FPS \
  --finalized-at '2026-10-22T10:24:00+08:00' \
  --runtime-report "$MUSEUMLIVE_10FPS_RUN_ROOT/report_1_7.md"
```

The command translates those known layout differences, rejects any remaining
broken local Markdown link, and only then writes the artifact manifest and
`SHA256SUMS`.
