# Experiment 10 - Wider Ternary Vocab Scale Test

## Goal

Check whether the ternary vocab gap shrinks when the model gets wider.

Claude's suggestion was simple: if hidden size grows, the model may have enough
capacity that ternary damage matters less.

## Run

```bash
python "experiments/Experiment 10 - Wider Ternary Vocab Scale Test/wider_scale_test.py" \
  --steps 300 \
  --hidden-sizes 128,192,256 \
  --device cuda
```

## Read Template

Good sign:

```text
gap at hidden=256 < gap at hidden=128
```

Bad sign:

```text
gap stays flat or grows as width grows
```

## Results

Run:

```bash
python "experiments/Experiment 10 - Wider Ternary Vocab Scale Test/wider_scale_test.py" \
  --steps 300 \
  --hidden-sizes 128,192,256 \
  --device cuda
```

Machine:

- NVIDIA GeForce RTX 3050 Ti Laptop GPU
- train tokens: 3,999,951
- eval tokens: 999,987
- seed: 1
- ternary preset: `threshold=0.25`, `group_size=32`

| Hidden size | Dense tied eval | Ternary tied eval | Gap | Packed size | Compression |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 6.2077 | 6.3109 | +0.1032 | 4.86 MB | 7.16x |
| 192 | 6.0682 | 6.1284 | +0.0602 | 8.22 MB | 6.46x |
| 256 | 5.8779 | 5.9355 | +0.0576 | 15.21 MB | 4.93x |

## Read

Width helps. The ternary gap drops from `+0.1032` at hidden 128 to `+0.0576`
at hidden 256.

This supports Claude's suggestion: ternary damage is less painful when the
model has more width.

## Decision

For future serious runs, do not judge ternary vocab only at hidden 128. The
better test is at hidden 192 or 256.

Hidden 256 gives the best loss but costs more packed size. Hidden 192 may be
the cleaner laptop-scale compromise.
