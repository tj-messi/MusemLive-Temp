# Point-based Judge (`lenient_v3`)

The Judge separates semantic matching from numeric scoring.

For each slot, an LLM matcher labels:

- action: `correct`, `partial`, or `wrong`;
- each fixed point: `hit`, `partial`, `miss`, or `contradict`;
- each forbidden claim: violated or not.

The matcher does not output the final score. Scores are deterministic:

```text
credit(hit)=1.0, credit(partial)=0.8, credit(miss/contradict)=0
action(correct)=1.0, action(partial)=0.7, action(wrong)=0

score = 5 * (0.40 * action + 0.60 * point_coverage)
        - 0.75 * forbidden_hits
        - 0.25 * contradictions
```

The score is clipped to `[0, 5]` and rounded to one decimal place. When the
rubric labels support points, coverage is `0.8 * core + 0.2 * support`.

For `context_correction`, correction behavior is the main component:

```text
score = 5 * (0.40 * correction + 0.25 * action + 0.35 * point_coverage)
        - 0.50 * reinforced_error
        - 0.75 * forbidden_hits
        - 0.25 * contradictions
```

Silent expected-speech slots still score 0 without an API call. Silent `WAIT`
slots score 5 without an API call. Explicit contradictions and forbidden
claims remain penalized. The matcher prompt is embedded in
`judge/run_judge.py`; fixed core/support labels are in `judge/rubric.json`.

`lenient_v3` only increases credit for a relevant but incomplete response. It
does not add a score floor and does not award silence. On the frozen 10 FPS
calibration set (MiniCPM, Venus, and Gander; 3,750 slots), the combined mean
changes from 0.92 under `strict_v2` to 1.03.

The default matcher is `qwen/qwen3.8-omni-flash` through OpenRouter with
temperature 0 and JSON output. Override only when explicitly running a
cross-Judge robustness experiment; do not mix alternate judges into the main
benchmark score. Report `judge_profile=lenient_v3` with every result table.
