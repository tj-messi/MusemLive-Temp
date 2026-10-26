# MuseumLive benchmark snapshot

This directory contains the frozen 450-task MuseumLive experimental benchmark
snapshot, its source manifest, sample selections, and archived model outputs.
The corresponding Judge is under `judge/`, while the evaluated code and
postprocessing snapshots are under `pipelines/` and `Experiment/`.

This snapshot supports four levels of inspection:

1. validate the task, branch, slot, rubric, and source accounting;
2. inspect every archived response, point-level Judge decision, and score;
3. rerun the frozen Judge on compatible evaluation results;
4. restore authorized media and use the published pipeline to run a model.

Source videos, extracted frames, and audio are not redistributed in this Git
repository. See [Media restoration](#media-restoration) for the boundary.

## Snapshot

- 450 tasks: nine capabilities x 50 tasks.
- 900 counterfactual branches and 1,250 scored slots.
- 1,028 `SPEAK`, 72 `CLARIFY`, and 150 `WAIT` slots.
- 384 Chinese tasks and 66 English tasks.
- 129 source identifiers and 4,048 rubric entries.
- A 25 FPS, 640x360 JPEG source timeline and 16 kHz mono scene audio in the
  authorized media archive.
- Ten ordered frames sampled uniformly from each causal second for the 10 FPS
  track; the finalized 1 FPS track uses one frame per decision tick.

The records are an experimental snapshot. Independent benchmark-label review
and media-redistribution review remain separate release gates. Do not describe
the media as redistributable merely because a source URL is present.

## Files

```text
all_tasks.json                    frozen 450-task benchmark
schema.json                       task-level schema and media contract
summary.json                      capability, language, and pool counts
media_manifest.json               relative media paths, counts, and hashes
sources.tsv                       public source metadata and rights status
sample_10p_task_ids.json          reproducible 45-task sample
sample_10p_distinct_videos.json   source-disjoint sample metadata
candidates/*.jsonl                1,250 archived post-Judge records/model
```

The four `candidates/*.jsonl` files contain MiniCPM-o 4.5, Venus, Gander, and
MOSS records. Each row includes the task and slot identifiers, expected action,
model response, Judge action and point statuses, and the deterministic 0--5
score. These are post-Judge outputs, not raw model traces. See
[`candidates/README.md`](candidates/README.md).

The frozen hashes are:

```text
benchmark/all_tasks.json  5e37614d18d900c8c680891ea691911486fabb6d2a4dd575f23604b07648479b
judge/rubric.json         3f998b202dc885c433315cf5bc1401177ea6600ebb0fcda76e27da00764d017b
```

## Validate

The release validator needs only Python's standard library:

```bash
python tools/validate_release.py
```

It checks the frozen hashes, task accounting, language and capability counts,
129-row source manifest, candidate-row counts, local Markdown links, and common
credential or machine-path leakage patterns.

## Judge archived responses

Install the Judge dependency and provide an API key through the environment:

```bash
python -m pip install -r requirements.txt
export OPENROUTER_API_KEY="..."
python judge/run_judge.py \
  --eval runs/your_model/evaluation_results.json \
  --rubric judge/rubric.json \
  --out runs/your_model/judge
```

The LLM matcher labels action and rubric-point status only. Python computes the
final score. Silent expected-speech slots and silent `WAIT` slots are resolved
without an API call. The exact formula, Judge profile, and validity rules are
in [`../judge/JUDGE.md`](../judge/JUDGE.md).

MOSS receives video and scene transcripts with question text rather than native
user audio. Its response match starts after speech ends, after the question text
is delivered, within six seconds, and before the next user event. Complete MOSS
runtime logs can be converted into delivery-aware candidates with
[`../tools/rematch_moss.py`](../tools/rematch_moss.py); changed candidates must
be scored by the frozen Judge.

## Run a model

The end-to-end generation, evaluation, GPU scheduling, Judge, and reporting
pipeline is documented in
[`../pipelines/10-20/README.md`](../pipelines/10-20/README.md) and
[`../pipelines/10-20/PIPELINE_OVERVIEW.md`](../pipelines/10-20/PIPELINE_OVERVIEW.md).
The task record format is in
[`../pipelines/10-20/protocols/task_format.md`](../pipelines/10-20/protocols/task_format.md).

Full model execution additionally requires the model-specific evaluator
environment, model weights, authorized media, and the Qwen-TTS index. Paths are
provided through the variables in
[`../pipelines/10-20/env.example`](../pipelines/10-20/env.example).

## Reproduce reported results

The public 1 FPS and 10 FPS aggregates, figures, diagnostics, and provenance are
under [`../Experiment/`](../Experiment/). Start with:

- [`../Experiment/README.md`](../Experiment/README.md)
- [`../Experiment/EVALUATION_LOOP.md`](../Experiment/EVALUATION_LOOP.md)
- [`../Experiment/1FPS/experiments_1_7/REPRODUCIBILITY.md`](../Experiment/1FPS/experiments_1_7/REPRODUCIBILITY.md)

The 1 FPS and 10 FPS result families remain separate. Raw scores from different
Judge profiles must not be interpreted as an FPS-only comparison.

## Media restoration

`media_manifest.json` gives each source's relative frame directory, audio file,
FPS, counts, duration, and available checksums. `sources.tsv` gives the source
identifier, public URL when available, storage kind, task and capability counts,
frame/audio sizes, and `redistribution_status`.

An authorized copy should be restored under an external run root matching those
relative paths. The original videos, 25 FPS extracted frames, mixed audio, TTS
files, model weights, raw runtime traces, and API caches are intentionally
external because of size, credentials, or unreviewed redistribution rights.

## Citation and license boundary

This anonymous snapshot does not grant rights to third-party videos or model
weights. Cite the associated paper when it becomes available and retain source
attribution supplied in `sources.tsv`. A root license for original benchmark
records and code will be added only after the project's release review; absence
of a license is not a grant of additional rights.
