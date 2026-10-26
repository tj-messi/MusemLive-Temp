# Archived model outputs

Each JSONL file contains 1,250 post-Judge slot records:

- `minicpm-results.jsonl`
- `venus-results.jsonl`
- `gander-results.jsonl`
- `moss-results.jsonl`

Rows include `task_id`, `branch_id`, `slot_id`, `capability`,
`expected_action`, the archived `response`, semantic-match fields, point-level
statuses and evidence, and the deterministic `score`.

These files preserve scored outputs for inspection and aggregate regeneration.
They do not contain complete per-second model traces, hidden state, media, model
weights, raw Judge API responses, or credentials. Full reruns require the
authorized media and the model-specific environments documented in the root
`REPRODUCIBILITY.md` and `pipelines/10-20/`.

MOSS uses video with text-delivered user questions rather than native user
audio. Its scores follow the delivery-aware matching rule described in the root
`README.md`. When complete runtime logs are available, `tools/rematch_moss.py`
rebuilds eligible response matches; any changed candidate must be rescored with
the frozen Judge.
