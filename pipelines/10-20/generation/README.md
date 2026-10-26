# Generation snapshot

This directory records the code path that produced the QA candidates used by
the frozen MuseumLive-Bench-450 snapshot. It is separate from the 10 FPS
evaluation code: QA generation uses frozen 25 FPS evidence, while evaluation
chooses either 1 or 10 ordered frames from each one-second evidence window.

## What is here

- `annotate_visual_seconds.py`: executed visual preprocessing/annotation
  implementation. It decodes source video at 25 FPS, groups exactly 25 frames
  per complete second, asks the visual annotator for grounded JSON, validates
  frame evidence, and records gaps instead of inventing missing annotations.
- `qa_gen/`: the executed 10-16 Omni QA generation, verification,
  first-person/language normalization, post-processing, pool selection, and
  cross-batch selection implementation. The publication change is limited to
  reading credentials and external paths from environment variables.
- `config.py`: environment-only path contract for the public snapshot.
- `validate_frozen_tasks.py`: deterministic check for the final 450-task
  contract and optional SHA-256 pin.

The audio annotation prompt and the exact 15-second-chunk to one-second-window
merge protocol are documented in `../prompts/annotations.md`. Raw video,
frames, audio, model responses, API caches, and candidate pools remain outside
Git under the repository artifact policy.

## QA generation stages

```text
per-second visual+audio annotations
  -> gen_v3: seven interaction/state capabilities
  -> gen_hr: historical visual reference
  -> gen_pg: proactive viewpoint guidance
  -> deterministic checks + leakage checks + image verification
  -> v3b post-processing gates
  -> per-batch balanced pools
  -> 1,755-candidate Omni review pool
  -> manual timeline review and exactly 50 tasks per capability
  -> frozen 450 tasks / 900 branches / 1,250 scored slots
```

The full generation, leakage, verification, and post-processing prompts are
in `../prompts/qa_generation.md`; the task schema is in
`../protocols/task_format.md`.

## Environment and commands

Set the generation variables from `../env.example` in a private shell. Never
write `OPENROUTER_API_KEY` to a repository file.

```bash
python3 generation/annotate_visual_seconds.py \
  --selection "$MUSEUMLIVE_VIDEO_SELECTION" \
  --output "$MUSEUMLIVE_25FPS_ROOT" --workers 8

python3 generation/qa_gen/run_gen.py 316a v3 --per-video 2 --budget 15 --workers 20
python3 generation/qa_gen/run_gen.py 316a hr --seeds 380 --budget 5 --workers 8
python3 generation/qa_gen/run_gen.py 316a pg --per-video 3 --target 190 --budget 5
python3 generation/qa_gen/run_gen.py 316a v3b audit
python3 generation/qa_gen/fix_first_person.py --batch 316a --workers 12
python3 generation/qa_gen/fix_language.py --batch 316a --workers 12
python3 generation/qa_gen/run_gen.py 316a pool --cap 175

python3 generation/validate_frozen_tasks.py "$MUSEUMLIVE_TASKS_FILE" \
  --sha256 5e37614d18d900c8c680891ea691911486fabb6d2a4dd575f23604b07648479b
```

These commands describe the executable snapshot. Regenerating the exact
historical candidate pool also requires the frozen model/provider versions,
the private source media, and the original raw model responses.
