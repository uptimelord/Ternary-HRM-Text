# Experiment 36 - Language Rehearsal EqR SFT

> **Status: completed.** Exp36 branches from the Exp35 mixed-pretrain
> checkpoint and changes only the final SFT diet: arithmetic SFT with 25%
> Dolmino continuation rehearsal mixed into each batch. Arithmetic remains
> useful and recurrence helps, but language is still not solved.

## Question

Exp35 improved arithmetic, but pure arithmetic SFT pulled normal language
prompts toward number/step fragments. Exp36 asks whether a small language
rehearsal stream can keep the arithmetic gain without as much language overwrite.

## Recipe

```text
Exp35 mixed-pretrain checkpoint
  -> EqR SFT on v2 frozen-like arithmetic
  -> 25% Dolmino continuation rehearsal in SFT batches
  -> eval200 H=2/4/6 before and after SFT + language probes
```

Default batch shape is 4 examples:

```text
3 arithmetic SFT examples
1 Dolmino continuation rehearsal example
```

## Run

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File `
  "experiments/Experiment 36 - Language Rehearsal EqR SFT/start_exp36_h256_rehearsal_sft.ps1"
```

Direct runner:

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File `
  "experiments/Experiment 36 - Language Rehearsal EqR SFT/run_exp36_h256_rehearsal_sft.ps1"
```

## Decision Rule

Promote if arithmetic stays near the Exp35 band, invalid rate stays at 0%,
language probes are less number/step-dominated than Exp35 final SFT, and H=4
remains at least roughly as good as H=2.

Kill if arithmetic falls back near the weak pre-SFT baseline, language probes are
still pure arithmetic fragments, or rehearsal makes recurrence unstable across
H=2/4/6.

## Results

Exp36 is a useful arithmetic/recurrence result, but not a language solve.

| Run | H=2 | H=4 | H=6 | invalid |
|---|---:|---:|---:|---:|
| Exp35 mixed final | 65.0% | 67.0% | 64.0% | 0.0% |
| Exp36 before SFT | 2.0% | 1.5% | 1.0% | 0.0% |
| Exp36 rehearsal SFT | 58.5% | 62.0% | 63.5% | 0.0% |

Read:

- Pretrain checkpoint alone is not enough for arithmetic generation.
- Rehearsal keeps arithmetic in a strong band, with H=6 best.
- Language no longer collapses mainly into arithmetic steps, but it still
  repeats weak Dolmino phrases like `"The New York Times"` / `"The"`.

Full result:

```text
experiments/Experiment 36 - Language Rehearsal EqR SFT/results_h256_exp35pretrain_rehearsal25_eqr10000_seed1.md
```
