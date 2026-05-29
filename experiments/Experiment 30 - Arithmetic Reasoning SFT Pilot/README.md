# Experiment 30 - Arithmetic Reasoning SFT Pilot

## Question

Can the calibrated Phase 0 checkpoint learn to emit arithmetic reasoning chains
from the synthetic programmatic dataset?

This is not a new base pretrain. It is a short supervised continuation run from:

```text
artifacts/phase0_first_pretrain/h256_steps50000_seed1_exportcalib3000/checkpoint_fp32.pt
```

## Run Target

| Setting | Value |
|---|---:|
| base hidden size | 256 |
| SFT steps | 2,000 |
| batch size | 4 |
| sequence length | 128 |
| token exposures | 1.024M |
| learning rate | 1e-4 |
| train mode | hard-export |
| train rows | 100,000 |
| valid rows | 2,000 |

## Decision Rule

Promote if synthetic valid loss drops clearly, hard-export valid loss stays close
to train-mode valid loss, artifacts save correctly, and frozen chain generation
shows non-zero exact answers.

Kill if the run OOMs, synthetic valid loss does not move, hard-export drift comes
back materially, or frozen chain generation remains at zero after the pilot.

## Command

Run in the background:

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File `
  "experiments/Experiment 30 - Arithmetic Reasoning SFT Pilot/start_exp30_h256_sft_pilot.ps1"
```

Run in the current terminal:

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File `
  "experiments/Experiment 30 - Arithmetic Reasoning SFT Pilot/run_exp30_h256_sft_pilot.ps1"
```

## Artifacts

The runner writes:

```text
artifacts/phase0_arithmetic_sft_pilot/h256_steps2000_seed1/checkpoint_fp32.pt
artifacts/phase0_arithmetic_sft_pilot/h256_steps2000_seed1/checkpoint_packed.pt
artifacts/phase0_arithmetic_sft_pilot/h256_steps2000_seed1/metrics.json
artifacts/phase0_arithmetic_sft_pilot/h256_steps2000_seed1/generation_examples.jsonl
experiments/Experiment 30 - Arithmetic Reasoning SFT Pilot/results_h256_steps2000_seed1.md
```

## Results

The CUDA pilot completed and saved both FP32 and packed checkpoints.

Synthetic validation moved hard:

```text
valid_loss:      2.4300 -> 0.2136
valid_token_acc: 46.85% -> 92.04%
```

Hard-export stayed stable:

```text
train-mode valid loss: 0.2136
hard-export valid loss: 0.2108
hard-export gap:       -0.0028
```

The model learned the chain format, but exact reasoning is still weak:

```text
synthetic valid exact_acc:       0.0%
frozen chain generation acc:     4.0% on 50 examples
frozen chain generation invalid: 0.0%
```

Peak VRAM was 739.4 MB. Packed size stayed 13.82 MB. Wall time was 7.5
minutes. This is not a Phase 0 promote yet, but it proves the SFT path is
working and moved output behavior from zero to non-zero.
