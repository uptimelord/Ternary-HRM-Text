# Experiment 27 - H256 Frozen Generalization Gate

## Question

Does the `h256` 2-bit attention candidate still hold up on the frozen
arithmetic file and a second seed?

This is an answer-token loss check, not generation accuracy. The local
experiment harness trains token-ID models directly, so this gate tokenizes the
frozen prompts and answers, masks the prompt span, and scores answer-token loss.

## Decision Rule

Promote if `combo_2bit_attention` at `hidden_size=256` and `steps=2000` passes seeds `1,2` with train eval gaps and frozen answer-loss gaps both within the `+/- 0.0203` noise floor, saves at least `3.0 MB` versus `combo_baseline`, and keeps quality per packed MB above baseline.

Kill if the second seed or the frozen answer-loss check exceeds the noise floor, or if the packed-size saving falls below `3.0 MB`.

## Command

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File `
  "experiments/Experiment 27 - H256 Frozen Generalization Gate/start_exp27_frozen_generalization.ps1"
```

## Results

Not run yet.
