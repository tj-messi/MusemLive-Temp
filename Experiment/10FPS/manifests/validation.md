# Validation record

Finalized: 2026-10-21

```text
COMPLETE: 16 run summaries, 16 alternate-judge panels, analyses, reports, figures and provenance are valid
FPS metadata validation passed: 13 runs, 11,900 branches
```

Additional checks:

- All 13 scheduled runs contain their expected branches and scored slots.
- Every required main, intervention, blind, recent-30, offline, and alternate-Judge summary reports `api_failed = 0`.
- Recent-30 MiniCPM and MOSS each contain 1,000/1,000 scored slots, with 810 identical formal judgments reused and 190 changed-context slots rejudged.
- All 16 alternate-Judge panels contain 1,250/1,250 scored slots.
- Six committed SVG figures parse as valid XML.
- Forced-Gander, recent-30, offline-upper-bound, and alternate-Judge fallback provenance records passed consistency checks.
- The public artifact manifest and `SHA256SUMS` were rebuilt from the finalized snapshot.
