# Experiment 9 - Mixed Precision Vocab Rows

## Goal

Keep the big ternary vocab compression win, but give dense capacity back to the
most common token rows.

## Idea

```text
most vocab rows       -> tuned ternary vocab
top-frequency rows    -> dense override rows
body                  -> dense
```

This is stronger than Exp 8 row gain. Row gain only changed one scalar per
token. Mixed rows restore a full dense hidden vector for selected tokens.

## Run

```bash
python "experiments/Experiment 9 - Mixed Precision Vocab Rows/mixed_vocab_rows.py" \
  --steps 500 \
  --variants dense_tied_vocab,ternary_tuned,mixed_top512,mixed_top1024,mixed_top2048,mixed_top4096 \
  --device cuda
```

## Read Template

The question is whether dense overrides can shrink the gap while packed size
stays far below dense tied vocab.

Good outcome:

```text
mixed_topK improves over ternary_tuned
packed size stays under roughly 8 MB
```

## Results

Run:

```bash
python "experiments/Experiment 9 - Mixed Precision Vocab Rows/mixed_vocab_rows.py" \
  --steps 500 \
  --variants dense_tied_vocab,ternary_tuned,mixed_top512,mixed_top1024,mixed_top2048,mixed_top4096 \
  --device cuda
```

Machine:

- NVIDIA GeForce RTX 3050 Ti Laptop GPU
- train tokens: 3,999,951
- eval tokens: 999,987
- seed: 1
- ternary preset: `threshold=0.25`, `group_size=32`, `scale_mode=mean_abs`

| Variant | Eval | Gap vs dense tied | Packed size | Compression |
| --- | ---: | ---: | ---: | ---: |
| `dense_tied_vocab` | 5.9669 | 0.0000 | 34.76 MB | 1.00x |
| `ternary_tuned` | 6.0170 | +0.0502 | 4.86 MB | 7.16x |
| `mixed_top512` | 5.9428 | -0.0240 | 5.11 MB | 6.85x |
| `mixed_top1024` | 5.9555 | -0.0114 | 5.36 MB | 6.57x |
| `mixed_top2048` | 5.9614 | -0.0055 | 5.87 MB | 6.09x |
| `mixed_top4096` | 5.9768 | +0.0099 | 6.89 MB | 5.34x |

## Read

This is the strongest compression result so far.

`mixed_top512` beat the dense tied baseline at 500 steps while still compressing
the model by **6.85x**.

The bigger mixed sets did not help more. At this scale, restoring too many
dense rows seems to reduce the useful regularization effect.

## Decision

Promote `mixed_top512` to the best current vocab preset:

```text
base ternary vocab: threshold=0.25, group_size=32
dense override rows: top 512 token-frequency rows
packed size: 5.11 MB
compression: 6.85x
```

This deserves a longer confirmation run.

## Combined Check - Selected Scale Mode

After Exp 12 showed `selected_mean_abs` helped plain ternary vocab, we checked
the obvious combination:

```bash
python "experiments/Experiment 9 - Mixed Precision Vocab Rows/mixed_vocab_rows.py" \
  --steps 500 \
  --variants dense_tied_vocab,ternary_tuned,mixed_top512 \
  --scale-mode selected_mean_abs \
  --device cuda
```

| Variant | Eval | Gap vs dense tied | Packed size | Compression |
| --- | ---: | ---: | ---: | ---: |
| `dense_tied_vocab` | 5.9669 | 0.0000 | 34.76 MB | 1.00x |
| `ternary_tuned` | 6.0016 | +0.0347 | 4.86 MB | 7.16x |
| `mixed_top512` | 5.9606 | -0.0063 | 5.11 MB | 6.85x |

`selected_mean_abs` helps plain ternary vocab, but the best mixed-row score is
still the original `mixed_top512` with `mean_abs`.
