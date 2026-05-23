# Experiment 8 - Ternary Vocab Row Gain

## Goal

Exp 7 found the best simple ternary vocab preset so far:

```text
threshold=0.25
group_size=32
packed size=4.86 MB
2000-step gap=+0.1163
```

Exp 8 tests a small extra knob: one learned gain per vocab row.

## Idea

The ternary vocab still stores the row pattern as `{-1, 0, +1}` plus group
scales. A row gain adds one scalar per token:

```text
W_token = gain_token * ternary_weight_token
```

For vocab size 65,536, that is only 65,536 extra floats. Even FP32 is about
256 KB, tiny compared with the dense vocab matrix.

## Variants

| Variant | Vocab |
| --- | --- |
| `dense_tied_vocab` | tied dense baseline |
| `ternary_tuned` | Exp 7 quality preset, no row gain |
| `rowgain_0.5` | Exp 7 quality preset + gain in `[0.5, 1.5]` |
| `rowgain_1.0` | Exp 7 quality preset + gain in `[0.0, 2.0]` |

The row gain is tied: the same gained matrix is used for input embedding and
the LM head.

## Run

```bash
python "experiments/Experiment 8 - Ternary Vocab Row Gain/row_gain_sweep.py" \
  --steps 500 \
  --variants dense_tied_vocab,ternary_tuned,rowgain_0.5,rowgain_1.0 \
  --device cuda
```

If one row-gain variant wins clearly, confirm it at 2000 steps.

## Results

Run:

```bash
python "experiments/Experiment 8 - Ternary Vocab Row Gain/row_gain_sweep.py" \
  --steps 500 \
  --variants dense_tied_vocab,ternary_tuned,rowgain_0.5,rowgain_1.0 \
  --device cuda
```

Machine:

- NVIDIA GeForce RTX 3050 Ti Laptop GPU
- train tokens: 3,999,951
- eval tokens: 999,987
- seed: 1
- ternary preset: `threshold=0.25`, `group_size=32`

| Variant | Eval | Gap | Params | Packed size | Compression |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dense_tied_vocab` | 5.9669 | 0.0000 | 9,109,504 | 34.76 MB | 1.00x |
| `ternary_tuned` | 6.0170 | +0.0502 | 9,109,504 | 4.86 MB | 7.16x |
| `rowgain_0.5` | 6.0235 | +0.0566 | 9,175,040 | 5.11 MB | 6.85x |
| `rowgain_1.0` | 6.0274 | +0.0605 | 9,175,040 | 5.11 MB | 6.85x |

## Read

Row gain did **not** help. It added a small number of parameters and a small
packed-size cost, but both row-gain variants were worse than plain
`ternary_tuned`.

The useful result is negative:

```text
the simple missing piece is not just one better magnitude per token row
```

## Decision

Keep Exp 7's plain tuned ternary vocab as the default:

```text
threshold=0.25
group_size=32
packed size=4.86 MB
compression=7.16x
```

Do not add row gain to the main line.

The next quality knob should be more selective than row gain:

- keep rare or sensitive vocab rows dense,
- use a learned dense residual for only a small row subset,
- or change the ternary scale rule itself.
