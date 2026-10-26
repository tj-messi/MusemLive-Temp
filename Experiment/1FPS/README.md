# Finalized 1 FPS track

The finalized Experiment 1–7 snapshot is stored in
[`experiments_1_7/`](experiments_1_7/).

At each one-second decision tick, the native-audio evaluators receive one frame
and one causally aligned second of audio. MOSS receives one frame and the
aligned ideal-ASR text used by the original experiment. All published tables,
figures, aggregate JSON files, and validation records currently committed under
`Experiment/1FPS/experiments_1_7/` belong to this track.

Do not place any 10 FPS artifacts in the finalized 1 FPS directory. See
[`../EVALUATION_LOOP.md`](../EVALUATION_LOOP.md) for the cross-track
contract.

## Additional Experiments 8–13

The frozen 1 FPS outputs were reused for stateful diagnostics without new
evaluated-model inference. See
[`experiments_8_13/reports/summary_8_12.md`](experiments_8_13/reports/summary_8_12.md)
for Experiments 8–10 and 12, and
[`experiments_8_13/reports/experiment_11_swap.md`](experiments_8_13/reports/experiment_11_swap.md)
for counterfactual branch sensitivity. Shared Experiment 13 and reproduction
scripts are under [`../experiments_8_13/`](../experiments_8_13/).
