# Experiment 18 - Weight Decay Phaseout

## Question

Does disabling weight decay in the last 20% of training help ternary vocab quality?

Relevant for production `pretrain.py` (`weight_decay=0.1` constant today). Laptop
smokes use `weight_decay=0.0` by default, so this experiment uses nonzero WD to
make the effect measurable.

## Model

`mixed_top512` tied vocab (Exp 14 production-relevant lane).

## Variants

| Variant | Weight decay schedule |
|---|---|
| `constant_wd` | `weight_decay=0.01` throughout |
| `wd_phaseout` | `weight_decay=0.01` until 80% of steps, then `0.0` |

## Command

```powershell
rtk python -u "experiments/Experiment 18 - Weight Decay Phaseout/weight_decay_phaseout.py" `
  --steps 500 `
  --weight-decay 0.01 `
  --phaseout-ratio 0.8 `
  --seeds 1 `
  --device cuda `
  --append-md "experiments/Experiment 18 - Weight Decay Phaseout/results_500.md"
```

## Decision rule

- Phaseout helps → keep for production pretrain recipe
- Flat → low priority for laptop smokes; retest with `weight_decay=0.1` later

## Results (2026-05-24, CUDA, seed 1, 500 steps, weight_decay=0.01)

| variant | final_eval | gap_vs_constant | phaseout_step | tok/s |
|---|---:|---:|---:|---:|
| constant_wd | 5.9428 | +0.0000 | - | 1907 |
| wd_phaseout | 5.9429 | +0.0000 | 400 | 1832 |

Full log: [`results_500.md`](results_500.md).

## Verdict: flat on laptop smoke

No measurable difference at `weight_decay=0.01` over 500 steps. This is expected
for short smokes with low WD. Not important for laptop experiments; still worth
testing later with production `weight_decay=0.1` in `pretrain.py`.

## Decision

- Do **not** prioritize WD phaseout for laptop smoke configs
- Keep on backlog for production pretrain recipe test at `weight_decay=0.1`
