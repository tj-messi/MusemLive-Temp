# MuseumLive evaluation loop and FPS protocol

This document is the boundary between the finalized 1 FPS result snapshot and
the clean 10 FPS rerun. A number without an explicit FPS label is ambiguous and
must not be copied into a paper table, issue, or report.

The upstream source-video → annotation → QA → frozen-task lineage and the exact
generation/Judge prompts are documented in
[`../pipelines/10-20/PIPELINE_OVERVIEW.md`](../pipelines/10-20/PIPELINE_OVERVIEW.md).
This file focuses on the evaluation half of that larger loop.

## Two isolated tracks

| Contract | Finalized 1 FPS | Clean 10 FPS rerun |
|---|---|---|
| Decision cadence | 1 Hz | 1 Hz |
| Visual input per tick | 1 frame | 10 ordered frames sampled uniformly from the frozen 25 FPS release frames |
| Audio input per tick | causally aligned 1-second block | the same causally aligned 1-second block |
| Future data | never sent | never sent |
| Main task set | 450 tasks | the same frozen 450 tasks |
| Formal trajectory count | 900 branches | 900 branches per formal run |
| Formal judge count | 1,250 scored slots | 1,250 scored slots per formal run |
| Blind-context count | separate 1,000-slot runs | 1,000 branches / 1,000 scored slots per blind run |
| Result state | finalized on 2026-10-19 | finalized on 2026-10-21 |
| Repository location | [`1FPS/`](1FPS/) | [`10FPS/`](10FPS/) |

Changing visual FPS does **not** change the one-second action clock. It changes
only the amount of ordered visual evidence available inside each causal tick.

The finalized 10 FPS point judge uses the `lenient_v3` profile. The
1 FPS snapshot preserves its original fixed point-based judge configuration.
Raw scores across the tracks are therefore not a controlled FPS comparison
until the same judging configuration has been applied to both result sets.

## Unit accounting

Three counts appear in logs and should not be conflated:

- **Task:** a benchmark scenario. The formal set contains 450 tasks.
- **Branch:** an independently executed counterfactual trajectory. The formal
  set expands to 900 branches, and this is the scheduler's progress unit.
- **Scored slot:** a judgeable response or silence decision. The formal set
  contains 1,250 scored slots after branch outputs are expanded for judging.

## End-to-end loop

```text
frozen tasks + media map
          |
          v
create persistent task shards
          |
          v
strict-priority GPU scheduler (one run/condition at a time)
          |
          v
causal streaming evaluation at the selected visual FPS
          |
          +---- worker exits early? ----> resume only missing branches
          |
          +---- one long tail shard? ---> pause scheduler
          |                               split missing branches over GPUs
          |                               atomically merge into original shard
          |                               resume scheduler
          v
all expected branches present
          |
          v
point-based judge -> completeness checks -> alternate judges / context runs
          |
          v
aggregate JSON -> reports -> figures -> validation manifest
```

### 1. Freeze the input contract

The task file, media map, branch definitions, one-second decision clock, audio
alignment, and causal-prefix policy are frozen before a run starts. `VISUAL_FPS`
selects the visual track; output roots are FPS-specific so an old result cannot
be silently resumed into a new track.

### 2. Create persistent shards

Tasks are assigned by stable round-robin slicing. A shard keeps the same
`tasks.json` across restarts. Progress is computed from the number of branch
records in `eval/evaluation_results.json`, not from log lines or wall time.

The 10 FPS scheduler uses seven persistent shards and GPUs 2–7. Six shards run
concurrently; the seventh starts when a GPU becomes free. Runs follow the order
in [`10FPS/run_matrix.json`](10FPS/run_matrix.json), so a
later condition cannot consume GPUs while an earlier condition is unfinished.

### 3. Evaluate causally

Each branch starts from its own causal prefix. At tick `t`, the evaluator sends
only the visual frames and audio/text aligned to that tick, then records model
speech spans and timing metadata. Branches are independent; no model state or
future evidence is shared across counterfactual branches.

### 4. Resume or rebalance without double scoring

Normal recovery relaunches an incomplete persistent shard and skips branch keys
already present in its result JSON.

For a long final shard, the scheduler is paused and only the missing branch
keys are expanded into one-branch tasks distributed across GPUs 2–7. Temporary
worker outputs live outside the judge glob. A coordinator repeatedly performs
an atomic merge into the original shard result, preserving the original branch
order. The scheduler resumes only after the merged result contains every
expected key. This prevents duplicate trajectories and duplicate judge rows.

### 5. Judge and validate

A run becomes **evaluated** when every expected branch is present. It becomes
**complete** only after its judge summary exists, the expected slot count is
present, and `api_failed == 0`. Post-processing starts only after all scheduled
runs are complete, then generates alternate-judge results, context baselines,
the offline upper bound, aggregate analyses, reports, figures, and validation
records.

## Status vocabulary

- `queued`: no branch worker has started.
- `evaluating`: branch outputs are still being produced or merged.
- `judging`: all branches exist and the point judge is running.
- `complete`: branch and judge completeness checks pass.
- `finalized`: aggregate reports and validation artifacts are frozen for use.

Both the 1 FPS and 10 FPS tracks are finalized. The 10 FPS track reached this
state after all 13 scheduled runs, two recent-30 panels, 16 alternate-Judge
panels, aggregate reports, six figures, provenance records, and the final
validation check completed on 2026-10-21.

The executable 10 FPS implementation is versioned separately at
[`../pipelines/10-20/`](../pipelines/10-20/). Experiment directories record
protocols and evidence; pipeline directories record the code that creates and
checks that evidence.
