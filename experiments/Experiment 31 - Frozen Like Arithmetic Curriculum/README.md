# Experiment 31 - Frozen Like Arithmetic Curriculum

## Question

Does a tighter two-digit arithmetic curriculum improve exact frozen answers
after the first SFT pilot learned the chain format?

This continues from the Experiment 30 checkpoint:

```text
artifacts/phase0_arithmetic_sft_pilot/h256_steps2000_seed1/checkpoint_fp32.pt
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
| train rows | 16,000 |
| valid rows | 2,000 |
| frozen eval rows | 200 |

## Decision Rule

Promote if frozen chain generation accuracy improves clearly over the 4.0%
pilot result, synthetic valid loss stays low, hard-export valid loss stays close
to train-mode valid loss, and artifacts save correctly.

Kill if frozen chain generation does not improve, synthetic valid loss regresses
badly, hard-export drift comes back materially, or the run OOMs.

## Command

Run in the background:

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File `
  "experiments/Experiment 31 - Frozen Like Arithmetic Curriculum/start_exp31_h256_sft_v2.ps1"
```

Run in the current terminal:

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File `
  "experiments/Experiment 31 - Frozen Like Arithmetic Curriculum/run_exp31_h256_sft_v2.ps1"
```

## Artifacts

The runner writes:

```text
artifacts/phase0_arithmetic_sft_v2/h256_exp30_plus_v2_steps2000_seed1/checkpoint_fp32.pt
artifacts/phase0_arithmetic_sft_v2/h256_exp30_plus_v2_steps2000_seed1/checkpoint_packed.pt
artifacts/phase0_arithmetic_sft_v2/h256_exp30_plus_v2_steps2000_seed1/metrics.json
artifacts/phase0_arithmetic_sft_v2/h256_exp30_plus_v2_steps2000_seed1/generation_examples.jsonl
experiments/Experiment 31 - Frozen Like Arithmetic Curriculum/results_h256_exp30_plus_v2_steps2000_seed1.md
```

## Results

The CUDA continuation completed and saved both FP32 and packed checkpoints.

Raw first run, using the old last-number extraction:

```text
valid_loss:              0.2489 -> 0.1019
valid_token_acc:         96.16%
valid_exact_acc:         26.56%
hard-export gap:         -0.0030
frozen generation acc:   8.0% on 200 examples
peak VRAM:               739.4 MB
wall time:               15.4 minutes
```

The generation examples showed an evaluator bug: the model often emitted the
first answer, then continued with extra text, and the scorer used the final
number in the whole output. After changing extraction to score the first number
after the first `Answer:` label, a 50-example check gave:

```text
frozen generation acc:   26.0% on 50 examples
valid_loss:              0.0949
hard-export gap:         -0.0045
```

The first answer-line stop rule was too aggressive and stopped after the first
digit. After changing it to wait for a non-digit delimiter after the answer
value, the full 200-example corrected eval gave:

```text
frozen generation acc:   26.0% on 200 examples
frozen invalid:          0.0%
valid_loss:              0.1019
valid_exact_acc:         26.56%
hard-export gap:         -0.0030
```

So the next gate is not more base training. The model now has real task signal,
but arithmetic is still the bottleneck, especially multiplication and carry /
borrow style digit math.
