# MuseumLive anonymous artifact

This repository contains the frozen evaluation materials, aggregate results,
prompts, and executable analysis code for the MuseumLive streaming
multimodal-interaction benchmark. It is prepared for anonymous review.

The benchmark evaluates whether a model can make causal listen/speak decisions
while viewing a museum walkthrough, hearing scene audio and user speech, and
maintaining state across multi-turn interaction. The frozen snapshot contains:

- 450 tasks balanced across nine interaction capabilities;
- 900 counterfactual branches;
- 1,250 scored decision slots;
- 129 source videos and 4,048 rubric entries;
- two isolated visual-input tracks: 1 FPS and 10 FPS.

## Contents

- `benchmark/`: frozen task records, schema, source/media manifests, sample
  selections, and recorded model outputs.
- `judge/`: the `lenient_v3` point-based Judge, its exact prompt, rubric, and
  deterministic scoring implementation.
- `Experiment/`: aggregate 1 FPS and 10 FPS results, reports, figures,
  completeness evidence, diagnostics, and the assisted-human-study archive.
- `pipelines/10-20/`: end-to-end generation, annotation, evaluation,
  orchestration, postprocessing, and reporting code.
- `pipelines/benchmark_judge/`: earlier holistic-judge code and result
  snapshots retained for provenance.
- `tools/validate_release.py`: dependency-free validation of task counts,
  frozen hashes, rubric shape, and anonymous-release hygiene.
- `tools/rematch_moss.py`: delivery-aware response matching for MOSS runtime
  logs, with regression tests under `tools/tests/`.

## Validate this checkout

```bash
python tools/validate_release.py
```

Expected frozen hashes:

```text
benchmark/all_tasks.json  5e37614d18d900c8c680891ea691911486fabb6d2a4dd575f23604b07648479b
judge/rubric.json         3f998b202dc885c433315cf5bc1401177ea6600ebb0fcda76e27da00764d017b
```

## Judge standardized outputs

The Judge separates semantic matching from scoring. An LLM matcher labels
action correctness, rubric-point status, forbidden claims, and contradictions;
Python computes the final 0--5 score.

```bash
python -m pip install -r requirements.txt
export OPENROUTER_API_KEY="..."
python judge/run_judge.py \
  --eval /path/to/evaluation_results.json \
  --rubric judge/rubric.json \
  --out runs/model-judge
```

Silent output in required-speech slots and silence in `WAIT` slots are scored
without an API call. For the main experiments the matcher is
`qwen/qwen3.8-omni-flash` at temperature zero.

## Reproduce reported aggregates

The committed JSON files under `Experiment/` contain the aggregate values used
for the paper's tables and figures. The exact scripts that produced the 1 FPS,
10 FPS, alternate-Judge, context, and state-diagnostic analyses are included.
They expect the external run tree described in
`Experiment/1FPS/experiments_1_7/REPRODUCIBILITY.md` and
`pipelines/10-20/env.example`.

The end-to-end pipeline covers source selection, 25 FPS annotation, audio
alignment, QA generation, leakage and structure checks, timeline review,
causal evaluation, point judging, GPU scheduling, recovery, and report
generation. Machine-specific paths are supplied through environment variables;
credentials are never read from repository files.

## MOSS protocol

MOSS receives video frames with time-aligned scene transcripts and user-query
text rather than native user audio. A response is matched only when it starts
after user speech ends, after the question text is delivered, within six
seconds, and before the next user event. Its results remain grouped under the
text-mediated input condition.

## External artifacts and release scope

Source videos, extracted 25 FPS frames, mixed audio/TTS files, model weights,
bulk runtime traces, raw Judge caches, and API credentials are not committed.
The 1,250 post-Judge candidate records per evaluated model are included under
`benchmark/candidates/`. The media manifest specifies relative destination
paths, frame and audio counts, source IDs, and available hashes so an authorized
copy can be validated without exposing machine-local storage.

The 450-task release remains an experimental benchmark snapshot. Independent
label review and media-redistribution review are separate release gates. The
source table records `redistribution_status=unreviewed`; inclusion of a source
identifier or upstream URL does not grant rights to redistribute the media.
