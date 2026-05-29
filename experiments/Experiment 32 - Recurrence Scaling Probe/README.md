# Experiment 32 - Recurrence Scaling Probe

## Question

Does increasing HRM recurrence depth at inference improve frozen arithmetic
accuracy without any retraining?

The Exp31 checkpoint (26% frozen accuracy at H=2, L=3) is evaluated across
a range of H_cycles values. If accuracy improves with more iterations, the
core thesis holds: a fixed-size model reasons better by thinking longer.

## Run Target

| Setting | Value |
|---|---:|
| base hidden size | 256 |
| checkpoint | Exp31 (h256_exp30_plus_v2_steps2000_seed1) |
| H_cycles sweep | 1, 2, 4, 6, 10, 20 |
| L_cycles | 3 |
| frozen eval rows | 200 |
| retraining | none (eval only) |

## Decision Rule

Promote if accuracy increases monotonically with H_cycles: core thesis
validated — more thinking time = better reasoning at fixed parameter count.

Investigate if accuracy peaks then degrades: recurrence helps but the model
wasn't trained for deep iteration; may diverge at high H_cycles.

Kill if accuracy is flat or decreases with more iterations: recurrence is
not being utilized for reasoning at this training stage.

## Command

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File `
  "experiments/Experiment 32 - Recurrence Scaling Probe/run_exp32_recurrence_sweep.ps1"
```

Or directly:

```powershell
rtk python -u "experiments/Experiment 32 - Recurrence Scaling Probe/recurrence_scaling_probe.py" --device cuda
```

## Artifacts

The runner writes:

```text
artifacts/phase0_recurrence_scaling/h256_sweep/metrics.json
artifacts/phase0_recurrence_scaling/h256_sweep/generation_examples.jsonl
experiments/Experiment 32 - Recurrence Scaling Probe/results_h256_recurrence_sweep.md
```

## Results

(pending)
