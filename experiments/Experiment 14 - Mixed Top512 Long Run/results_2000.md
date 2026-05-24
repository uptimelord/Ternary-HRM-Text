# Experiment 14 - 2000 Step Results

Command:

```bash
rtk python "experiments/Experiment 9 - Mixed Precision Vocab Rows/mixed_vocab_rows.py" --steps 2000 --variants dense_tied_vocab,mixed_top512 --device cuda
```

Output summary:

| variant | eval | gap | params | packed_MB | compression | tok/s |
|---|---:|---:|---:|---:|---:|---:|
| dense_tied_vocab | 5.3831 | +0.0000 | 9,109,504 | 34.76 | 1.00x | 4892 |
| mixed_top512 | 5.3879 | +0.0048 | 9,175,040 | 5.11 | 6.85x | 4501 |

Best compressed variant: `mixed_top512`, with `+0.0048` eval gap and `6.85x`
packed compression.
