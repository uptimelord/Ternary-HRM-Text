# Experiment 28 - H256 GQKV Frozen Gate

## Question

Does the less aggressive `h256` 2-bit attention candidate hold up better than
full 2-bit attention when checked with the frozen arithmetic file and a second
seed?

Candidate:

```text
combo_2bit_attention_gqkv
```

This is the same answer-token loss gate used in Exp 27. It tokenizes the frozen
prompts and answers, masks the prompt span, and scores answer-token loss.

## Decision Rule

Promote if `combo_2bit_attention_gqkv` at `hidden_size=256` and `steps=2000`
passes seeds `1,2` with train eval gaps and frozen answer-loss gaps both within
the `+/- 0.0203` noise floor, saves at least `3.0 MB` versus `combo_baseline`,
and keeps quality per packed MB above baseline.

Kill if either seed exceeds the noise floor on normal eval or frozen
answer-loss, or if the packed-size saving falls below `3.0 MB`.

## Command

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File `
  "experiments/Experiment 28 - H256 GQKV Frozen Gate/start_exp28_gqkv_frozen_gate.ps1"
```

## Results

Command:

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File `
  "experiments/Experiment 28 - H256 GQKV Frozen Gate/run_exp28_gqkv_frozen_gate.ps1"
```

Full result file: [`results_2000_seeds12_h256.md`](results_2000_seeds12_h256.md).

| Variant | Seed | Eval gap +/- noise floor | Frozen loss gap +/- noise floor | Quality/MB | Packed size | Size delta |
|---|---:|---|---|---:|---:|---:|
| `combo_baseline` | 1 | +0.0000 +/- 0.0203 (at noise floor) | +0.0000 +/- 0.0203 (at noise floor) | 0.01384 | 13.82 MB | +0.00 MB |
| `combo_2bit_attention_gqkv` | 1 | +0.0059 +/- 0.0203 (at noise floor) | +0.5914 +/- 0.0203 (above noise floor) | 0.01895 | 10.09 MB | -3.73 MB |

## Decision

Do not promote `combo_2bit_attention_gqkv`.

The normal eval gap looked fine on seed 1, but the frozen answer-loss check
failed hard: `+0.5914 +/- 0.0203`. That triggers the pre-registered kill rule,
so seed 2 was not run. The frozen benchmark did exactly what it was added for:
it caught a candidate that looked acceptable on the normal eval slice.
