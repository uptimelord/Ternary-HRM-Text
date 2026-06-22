# Experiment 124 - Atlas-Guided FPRM Repair

## Question

Can a gradient sensitivity atlas guide exact ternary state edits after Exp123
SFT, improving the hard-export FPRM without more normal SFT?

## Method

```text
trained Exp123 checkpoint
  -> strict baseline eval
  -> one gradient map on training rows only
  -> exact -1 <-> 0 <-> +1 candidate edits
  -> hard-export accept/reject
  -> strict final eval
```

The atlas ranks exact one-state transitions by `abs(gradient) * group_scale`.
Dense top-512 vocab rows are excluded because they override the ternary tied
vocab rows. Each accepted edit preserves the current group output scale, so
unchanged hard weights in that group remain unchanged. Frozen200 and validation
rows are reporting-only; repair fitness uses training rows.

Objective:

```text
J = loss + lambda_residual * residual + lambda_iters * iterations - lambda_halt * halt_rate
```

The hard-export model decides every coordinate or ES acceptance.
`--atlas-eval-batches` controls independent training batches used for each
decision. An edit must lower `J` on every batch, not only on their mean.

## Decision Rule

Promote if strict frozen200 rises by at least 4 percentage points, invalid stays
0%, valid hard-export exact does not regress, and at least one exact ternary
state transition is accepted.

Kill if any Promote condition fails, including any invalid output, valid
hard-export regression, or accepted latent edits that produce no hard ternary
state change.

## Seed 1 Results

Exp123 hard-export baseline: strict frozen200 `59.5%`, invalid `0%`, valid hard
exact `58.594%`, valid hard loss `0.054196`.

| repair | edits | strict frozen200 | valid hard exact | valid hard loss | verdict |
|---|---:|---:|---:|---:|---|
| head coordinate 100 | 836 | 36.5% | 27.344% | 0.107011 | Kill |
| head ES 100 | 392 | skipped after valid gate | 54.688% | 0.056338 | Kill |
| MLP coordinate 20 | 112 | skipped after valid gate | 58.594% | 0.055742 | Kill |
| MLP coordinate + ES 20 | 153 | skipped after loss gate | 59.375% | 0.054431 | Kill |
| MLP ES 20 | 64 | 60.0% | 59.375% | 0.052635 | Kill |
| MLP ES 100 | 308 | 60.5% | 59.375% | 0.053089 | Kill |

Best branch: MLP ES 100. Strict gain `+1.0pp` is below the pre-registered
`+4pp` gate and below the repo noise floor. Exp124 does not Promote.

Head repair is unsafe for this model: `tied_vocab` is both output head and input
embedding. The top atlas row was token `" *"`; changing it alters both sides.

## Checks

```powershell
rtk python -m pytest tests/test_exp124_atlas_guided_repair.py -q
rtk python -m experiments.discipline preflight-readme "experiments/Experiment 124 - Atlas Guided FPRM Repair/README.md"
```

## Run order

Use the completed Exp123 v1 -> v2 checkpoint as `BASE`.

### 1. Baseline only

```powershell
rtk python -u "experiments/Experiment 124 - Atlas Guided FPRM Repair/atlas_guided_repair.py" `
  --base-checkpoint $BASE `
  --repair-output-dir "artifacts/phase0_fprm_exp124/baseline_seed1" `
  --baseline-only --device cuda
```

### 2. Atlas only, head

```powershell
rtk python -u "experiments/Experiment 124 - Atlas Guided FPRM Repair/atlas_guided_repair.py" `
  --base-checkpoint $BASE `
  --repair-output-dir "artifacts/phase0_fprm_exp124/atlas_head_seed1" `
  --atlas-only --atlas-target head --atlas-topk 1024 --device cuda --amp
```

### 3. Coordinate repair, head

```powershell
rtk python -u "experiments/Experiment 124 - Atlas Guided FPRM Repair/atlas_guided_repair.py" `
  --base-checkpoint $BASE `
  --repair-output-dir "artifacts/phase0_fprm_exp124/coordinate_head_seed1" `
  --atlas-target head --atlas-mode coordinate `
  --atlas-repair-steps 100 --atlas-remap-interval 5 --atlas-topk 1024 `
  --coordinate-max-tests 64 --sft-batch-size 2 --device cuda --amp
```

### 4. ES repair, head

Use the same `BASE` for a clean comparison.

```powershell
rtk python -u "experiments/Experiment 124 - Atlas Guided FPRM Repair/atlas_guided_repair.py" `
  --base-checkpoint $BASE `
  --repair-output-dir "artifacts/phase0_fprm_exp124/es_head_seed1" `
  --atlas-target head --atlas-mode es `
  --atlas-repair-steps 100 --atlas-remap-interval 5 --atlas-topk 1024 `
  --atlas-population 8 --atlas-mutations-per-child 4 `
  --sft-batch-size 2 --device cuda --amp
```

MLP and combined modes run only after the head results are recorded.
