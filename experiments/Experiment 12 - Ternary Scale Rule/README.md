# Experiment 12 - Ternary Scale Rule

## Goal

Test whether the ternary vocab loss comes from the scale rule.

Current default:

```text
scale = mean(abs(weights in group))
```

Alternatives:

```text
selected_mean_abs = mean(abs(weights that survive ternary threshold))
rms               = sqrt(mean(weight^2))
```

## Run

```bash
python "experiments/Experiment 12 - Ternary Scale Rule/scale_rule_sweep.py" \
  --steps 500 \
  --scale-modes mean_abs,selected_mean_abs,rms \
  --device cuda
```

## Read Template

Good outcome:

```text
selected_mean_abs or rms closes vocab gap without changing packed size much
```

## Results

Run:

```bash
python "experiments/Experiment 12 - Ternary Scale Rule/scale_rule_sweep.py" \
  --steps 500 \
  --scale-modes mean_abs,selected_mean_abs,rms \
  --device cuda
```

Machine:

- NVIDIA GeForce RTX 3050 Ti Laptop GPU
- train tokens: 3,999,951
- eval tokens: 999,987
- seed: 1
- ternary preset: `threshold=0.25`, `group_size=32`

| Variant | Scale mode | Eval | Gap | Packed size | Compression |
| --- | --- | ---: | ---: | ---: | ---: |
| `dense_tied_vocab` | `mean_abs` | 5.9669 | 0.0000 | 34.76 MB | 1.00x |
| `ternary_tied_vocab` | `mean_abs` | 6.0170 | +0.0502 | 4.86 MB | 7.16x |
| `ternary_tied_vocab` | `selected_mean_abs` | 6.0016 | +0.0347 | 4.86 MB | 7.16x |
| `ternary_tied_vocab` | `rms` | 6.0082 | +0.0413 | 4.86 MB | 7.16x |

## Read

Scale rule matters. `selected_mean_abs` is the best of this sweep and improves
the 500-step gap from `+0.0502` to `+0.0347` without changing packed size.

`rms` also helps, but less.

## Decision

Promote `selected_mean_abs` as the best scale rule for tuned ternary vocab.

The obvious combined preset was also checked:

```text
mixed_top512
threshold=0.25
group_size=32
scale_mode=selected_mean_abs
```

Result:

```text
dense_tied_vocab: 5.9669
mixed_top512 + selected_mean_abs: 5.9606
gap: -0.0063
packed: 5.11 MB
compression: 6.85x
```

That is good, but the original `mixed_top512` with `mean_abs` was still better
at `5.9428`. So:

```text
plain ternary vocab -> selected_mean_abs
mixed_top512 vocab  -> mean_abs
```
