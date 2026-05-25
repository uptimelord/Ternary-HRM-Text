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

Command:

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File `
  "experiments/Experiment 27 - H256 Frozen Generalization Gate/run_exp27_frozen_generalization.ps1"
```

Full result file: [`results_2000_seeds12_h256.md`](results_2000_seeds12_h256.md).

| Variant | Seed | Eval gap +/- noise floor | Frozen loss gap +/- noise floor | Quality/MB | Packed size | Size delta |
|---|---:|---|---|---:|---:|---:|
| `combo_baseline` | 1 | +0.0000 +/- 0.0203 (at noise floor) | +0.0000 +/- 0.0203 (at noise floor) | 0.01384 | 13.82 MB | +0.00 MB |
| `combo_2bit_attention` | 1 | +0.0134 +/- 0.0203 (at noise floor) | -0.2388 +/- 0.0203 (above noise floor) | 0.02085 | 9.15 MB | -4.67 MB |
| `combo_baseline` | 2 | +0.0000 +/- 0.0203 (at noise floor) | +0.0000 +/- 0.0203 (at noise floor) | 0.01393 | 13.82 MB | +0.00 MB |
| `combo_2bit_attention` | 2 | +0.0410 +/- 0.0203 (above noise floor) | -0.0057 +/- 0.0203 (at noise floor) | 0.02087 | 9.15 MB | -4.67 MB |

Summary:

| Variant | Runs | Mean eval gap +/- noise floor | Mean frozen gap +/- noise floor | Mean quality/MB | Mean packed size | Mean size delta |
|---|---:|---|---|---:|---:|---:|
| `combo_baseline` | 2 | +0.0000 +/- 0.0203 (at noise floor) | +0.0000 +/- 0.0203 (at noise floor) | 0.01389 | 13.82 MB | +0.00 MB |
| `combo_2bit_attention` | 2 | +0.0272 +/- 0.0203 (above noise floor) | -0.1223 +/- 0.0203 (above noise floor) | 0.02086 | 9.15 MB | -4.67 MB |

## Decision

Do not widen deploy for `combo_2bit_attention`.

The frozen answer-loss check did not expose a quality problem; if anything, it
favored the compressed candidate. The blocker is the second seed on normal eval:
seed 2 landed at `+0.0410 +/- 0.0203`, so the mean eval gap also rose above the
noise floor. Keep `combo_2bit_attention` as an export-clean h256 compression
candidate, but do not replace the h256 deploy path with it yet.
