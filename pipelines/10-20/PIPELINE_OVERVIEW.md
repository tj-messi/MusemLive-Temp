# MuseumLive 10-20 end-to-end pipeline

This is the versioned description of the loop that produced the current
MuseumLive QA snapshot and evaluates it at isolated 1 FPS and 10 FPS visual
input rates.

## One-line loop

~~~text
source videos
  -> eligibility filtering and pool isolation
  -> 25 FPS frame freeze + 16 kHz scene audio
  -> per-second visual annotation + time-aligned audio annotation
  -> candidate mining and three QA authoring lines
  -> deterministic checks + leakage checks + multimodal verification
  -> balanced candidate pools + timeline review
  -> frozen 450 tasks / 900 counterfactual branches / 1,250 scored slots
  -> causal 1 Hz evaluation with either 1 or 10 visual frames per tick
  -> fixed rubric + LLM point matching + deterministic score
  -> completeness gates, Experiments 1–7, reports, figures, and manifests
~~~

## Evidence labels used below

- **Executed:** code was run and corresponding local artifacts/logs exist.
- **Human-frozen:** a person selected or approved the resulting items.
- **External controlled artifact:** required for exact reproduction but not
  committed because it is media, private data, a model, a large output, or a
  credential.
- **Protocol:** documented behavior or validation rule; rerunning it is not
  claimed unless an executed result is explicitly cited.

## 1. Source videos and pool isolation

Before captioning or QA generation, candidate media is separated from already
completed/reserved clips and invalid sources. The executed selection policy
kept ffprobe-readable first-person museum/exhibition clips, excluded prior
10+40-video work and recorded exclusions, and capped duration at 1,020 seconds.

The resulting intermediate pool contained 450 remaining clips. The generation
batch used the 316 clips longer than 60 seconds; it was combined with a legacy
50-video batch, for 366 videos available to the QA-generation stage.

This is an experimental benchmark snapshot, not a claim that every source can
be publicly redistributed. The final 450 tasks use 129 unique sources: 19 from
the legacy batch and 110 from the 316-video batch. Source licensing,
redistribution rights, and privacy review remain external release gates.

State: **Executed** selection and manifests; compliance review is an
**External controlled artifact / open release gate**.

## 2. Frozen visual timeline

The executed visual pipeline is
[`generation/annotate_visual_seconds.py`](generation/annotate_visual_seconds.py).

For every selected source it:

1. decodes with `ffmpeg -vf fps=25,scale=640:-2`;
2. numbers frames from `F000001.jpg`;
3. groups each complete 25-frame block into one source-time second;
4. excludes an incomplete final second;
5. gives the annotator all 25 ordered frames with frame IDs;
6. validates that every cited evidence range uses supplied frame IDs; and
7. writes one JSON annotation per second or records a gap.

The one-second annotation contains `summary`, grounded `events`,
`objects`, OCR, uncertainty, exact frame IDs, and the 25 FPS source contract.
The exact visual prompt is in
[`prompts/annotations.md`](prompts/annotations.md).

The recorded 316-video visual run reached 316/316 videos with no unresolved
window gaps after the explicit gap-fill pass.

State: **Executed**.

## 3. Scene audio and transcript alignment

The audio job extracts 15-second, mono 16 kHz PCM chunks, requests literal
speech and sound-event JSON, converts relative times to source times, and
intersects those spans with each one-second visual window. The previous
chunk's final transcript can be supplied only as continuity context; it is not
re-emitted as evidence.

The exact audio system/user prompts and captured counts are in
[`prompts/annotations.md`](prompts/annotations.md). Provider refusals,
missing audio, empty output, and parse failures remain explicit gaps; the
pipeline does not invent inaudible text.

State: **Executed**. Audio files, merged annotations, and API responses are
**External controlled artifacts**.

## 4. QA and counterfactual task generation

The executed, publication-sanitized snapshot is under
[`generation/qa_gen/`](generation/qa_gen/). Full prompts are exported in
[`prompts/qa_generation.md`](prompts/qa_generation.md), and the schema is in
[`protocols/task_format.md`](protocols/task_format.md).

Three generation lines share the same per-second visual/audio evidence:

| Line | Capabilities | Main gates |
|---|---|---|
| `gen_v3` | temporal update, selective attention, constraint update, context correction, task resume, monitor/cancel, proactive clarification | candidate-window ranking, time selection, frame-grounded authoring, deterministic structure checks, answer-leakage check, visual/audio verification, t0 visibility gate |
| `gen_hr` | historical reference | 30–60 s memory gap, appearance-leakage check, referent/attribute verification, absence near query |
| `gen_pg` | proactive viewpoint guidance | information hidden at t, visible at t2, same object, real camera-motion direction, control object visible |

Each accepted task contains two independent counterfactual branches. Branches
share source media and the pre-intervention prefix but differ in user state
changes such as keep/cancel, correct/no-correction, ambiguous/specific, or
resume/restart. Slots specify a causal time window, trigger, expected action
(`SPEAK`, `WAIT`, or `CLARIFY`), reference answer, atomic key facts, and
forbidden claims.

Failed checks feed concrete reasons into the next rewrite attempt. A candidate
is not accepted merely because an author model returned valid JSON.

Before pool selection, two executed text-normalization passes enforce the
first-person camera-wearer perspective and match task language to the scene
language. They preserve IDs, times, facts, branches, and slot structure, and
validate the returned structure before replacing text. Their exact prompts are
in [`prompts/qa_postprocessing.md`](prompts/qa_postprocessing.md).

Executed pool sizes:

- audio-aware legacy batch `50a`: 333 accepted pool tasks;
- audio-aware `316a` batch: 1,422 accepted pool tasks;
- combined timeline-review pool: 1,755 tasks.

State: **Executed** generation and automatic gates.

## 5. Human review and frozen 450-task snapshot

The timeline review UI displayed user events, source time, frames/audio
evidence, target answers, slot-level point-Judge results, and selection state.
The review export selected exactly 50 tasks for each of the nine capabilities.

Frozen contract:

| Item | Value |
|---|---:|
| tasks | 450 |
| capabilities | 9 × 50 |
| counterfactual branches | 900 |
| scored slots | 1,250 |
| source videos | 129 |
| languages | 384 Chinese / 66 English |
| expected actions | 1,028 SPEAK / 150 WAIT / 72 CLARIFY |
| task SHA-256 | `5e37614d18d900c8c680891ea691911486fabb6d2a4dd575f23604b07648479b` |
| frozen rubric entries | 4,048 |
| rubric SHA-256 | `3f998b202dc885c433315cf5bc1401177ea6600ebb0fcda76e27da00764d017b` |

Run [`generation/validate_frozen_tasks.py`](generation/validate_frozen_tasks.py)
against the external task file before any evaluation.

State: selection is **Human-frozen** for the experiment snapshot. Reference
answers/evidence and redistribution rights still require final independent
release review, so the repository does not call this a fully released,
independently human-labeled public test set.

## 6. The 1 FPS and 10 FPS evaluation contracts

Both tracks run the same 450 tasks, 900 branches, user-event times, scene
audio/transcript intervals, one-second decision clock, and causal-prefix rule.
At model inference, visual FPS is the only intended input-rate difference.
The currently published result tracks still have the Judge-profile confound
described below, so their raw final scores are not yet an FPS-only comparison.

| Contract | 1 FPS track | 10 FPS track |
|---|---:|---:|
| decision cadence | 1 Hz | 1 Hz |
| frozen source timeline | 25 FPS | 25 FPS |
| ordered frames per one-second tick | 1 | 10 |
| sampling | center of the source second | centers of 10 equal-width bins in the source second |
| scene audio/text interval | causal 1 s | causal 1 s |
| future frames/events | never | never |
| output root | `Experiment/1FPS` / external 1 FPS runs | `Experiment/10FPS` / external 10 FPS runs |

The 10 FPS frame offsets are computed as
`floor((i + 0.5) * 25 / 10)` for `i=0..9`, clipped to `0..24`. Therefore
the ten frames cover the second in order instead of taking its first ten
frames. A model still receives exactly one decision opportunity per second.

Every branch resets model state and carries its own causal history. The runner
never sends a future frame, future user event, reference answer, or sibling
branch state.

Model input distinctions are explicit:

- MiniCPM-o, Venus, and Gander receive visual frames plus mixed scene audio and
  user TTS under their duplex adapters;
- MOSS receives visual frames plus the time-aligned ideal-ASR scene transcript,
  not native scene audio;
- ablations mute scene audio, replace video with black frames, or remove the
  MOSS transcript as named in the run matrix.

The formal contract and accounting are also documented in
[`../../Experiment/EVALUATION_LOOP.md`](../../Experiment/EVALUATION_LOOP.md).

## 7. GPU scheduling, recovery, and completion

[`runners/orchestrate_10fps.py`](runners/orchestrate_10fps.py) executes the
13-run Experiment 1–7 matrix in strict priority on GPUs 2–7. Each formal run
has seven persistent shards. Six shards start concurrently; the seventh takes
the first available GPU.

Progress is the number of unique branch records in
`eval/evaluation_results.json`, not log lines. An incomplete shard is
relaunched and skips completed branch keys. A long final shard may be paused,
split by missing branch key across GPUs 2–7, atomically merged back into the
original shard, validated, and only then resumed. Temporary tail outputs stay
outside Judge globs to avoid double scoring.

A run is complete only if:

1. all expected branches are present;
2. the expected slot count is present;
3. `scorable_ok` equals the expected slot count; and
4. `api_failed == 0`.

State: the 10 FPS execution is **Executed / finalized**. All 13 scheduled runs
and their main Judge summaries are complete, with the final status and results
published under `Experiment/10FPS/`.

## 8. Judge Prompt, rubric, and score

The exact Prompt and formulas are in
[`prompts/judge_points.md`](prompts/judge_points.md), and executable code is
[`evaluators/judge_points.py`](evaluators/judge_points.py).

The Judge has three separable jobs:

1. freeze or load atomic rubric points independent of the evaluated model;
2. use an LLM only to label action correctness, point
   hit/partial/miss/contradict, forbidden-claim matches, and the special
   context-correction state;
3. compute the 0–5 score deterministically in Python.

Required silence is not free-form graded: a WAIT slot is scored from action
correctness. A required SPEAK/CLARIFY slot with no answer is exactly zero.
Explicit forbidden claims and contradictions remain penalized.

The scheduled 10 FPS profile is `lenient_v3`. The finalized 1 FPS snapshot
preserves its historical fixed profile. Consequently, unharmonized raw 1 FPS
and 10 FPS scores must not be described as an FPS-only controlled comparison.
A controlled table requires both tracks to be rescored with the same frozen
rubric, match outputs/profile, and completeness checks.

## 9. Experiments 1–7 and report loop

The matrix maps to:

1. response rate, conditional answer quality, and joint score;
2. temporal/event quality;
3. evidence distance and strata;
4. alternate Judges and cluster bootstrap;
5. forced-answer interventions;
6. modality ablations;
7. blind context, recent-30, and offline upper bound.

After all 13 base runs pass their completion gate,
[`postprocessing/postprocess_10fps.sh`](postprocessing/postprocess_10fps.sh)
launches alternate Judges and context/offline baselines, then rebuilds
machine-readable analyses, reports, figures, and validation records. The
published 2026-10-21 snapshot passed this gate: 16/16 alternate-Judge panels,
both recent-30 panels, the offline upper bound, reports, figures, provenance,
and validation are complete.

## 10. Reproduction boundary

Committed:

- executable generation/evaluation/Judge/analysis snapshots;
- exact Prompt text;
- schemas, run matrix, status snapshots, aggregate results, and checks;
- artifact hashes and path/environment contracts.

Not committed:

- source video, frames, audio, transcripts, TTS WAVs, model weights;
- candidate/task JSON containing controlled benchmark data;
- raw model responses, request caches, and per-slot generations;
- API credentials, cookies, local absolute paths, or provider headers.

Use [`env.example`](env.example) as the public path contract. Exact historical
regeneration additionally requires the same external media, frozen task/rubric
artifacts, model/provider versions, and recorded manual selection.
