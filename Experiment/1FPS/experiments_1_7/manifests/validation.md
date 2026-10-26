# Validation record

Finalized: 2026-10-19

```text
COMPLETE: experiments 1–7, all required slots, analyses, reports and figures are present
AUDIT OK: 12 experiment runs, 16 alternate-judge runs, 6 valid SVG figures
```

Additional checks:

- Every experiment result-file row count matches its `summary.json` slot count.
- All required experiment summaries report `api_failed = 0`.
- All 16 alternate-judge result sets contain 1,250 scored slots and zero API failures.
- Recent-30-second MiniCPM: 1,000/1,000, mean 1.27.
- Recent-30-second MOSS: 1,000/1,000, mean 1.93.
- The six committed SVG files parse as valid XML.
