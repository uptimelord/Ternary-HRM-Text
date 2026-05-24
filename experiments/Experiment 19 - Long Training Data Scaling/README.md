# Experiment 19 - Long Training / Data Scaling

## Question

Does the best short-run compressed recipe hold or improve with longer training?

## Recipe selection (from Exp 16/17)

- **Best compressed vocab:** `mixed_top512_tequila` (Exp 16 gap `-0.0676` at 500 steps)
- **Exp 14 baseline:** `mixed_top512` standard (gap `+0.0048` at 2000 steps vs dense tied)
- Body transition failed in Exp 17 — no body ternary in this run

## Variants

| Variant | Description |
|---|---|
| `dense_tied_vocab` | Tied dense baseline (Exp 14 apples-to-apples) |
| `mixed_top512` | Standard STE mixed vocab |
| `mixed_top512_tequila` | Tequila STE mixed vocab (Exp 16 best) |

## Command (2000 steps default)

```powershell
rtk python -u "experiments/Experiment 19 - Long Training Data Scaling/long_training_data_scaling.py" `
  --steps 2000 `
  --seeds 1 `
  --device cuda `
  --append-md "experiments/Experiment 19 - Long Training Data Scaling/results_2000.md"
```

Longer runs (manual only):

```powershell
# 5000 steps
rtk python -u "experiments/Experiment 19 - Long Training Data Scaling/long_training_data_scaling.py" --steps 5000 ...

# 10000 steps
rtk python -u "experiments/Experiment 19 - Long Training Data Scaling/long_training_data_scaling.py" --steps 10000 ...
```

## Decision rule

- Gap shrinks with longer training → data scaling supports ternary path
- Gap grows → short-run recipe is not stable enough

## Results (2026-05-24, CUDA, seed 1, 2000 steps)

| variant | final_eval | gap_vs_dense_tied | packed_MB | compr | tok/s |
|---|---:|---:|---:|---:|---:|
| dense_tied_vocab | 5.3831 | +0.0000 | 34.76 | 1.00x | 2226 |
| mixed_top512 | 5.3879 | +0.0048 | 5.11 | 6.85x | 1782 |
| mixed_top512_tequila | 5.3753 | -0.0078 | 5.11 | 6.85x | 1584 |

Full log: [`results_2000.md`](results_2000.md).

## Exp 19b — 5000 steps (2026-05-24, CUDA, seed 1)

| variant | final_eval | gap_vs_dense_tied | packed_MB |
|---|---:|---:|---:|
| dense_tied_vocab | 5.1674 | +0.0000 | 34.76 |
| mixed_top512 | 5.1975 | +0.0301 | 5.11 |
| mixed_top512_tequila | 5.1922 | +0.0248 | 5.11 |

Full log: [`results_5000.md`](results_5000.md).

## Verdict (2000 + 5000)

- **2000 steps:** Tequila beats dense tied (`-0.0078`); standard near-tied (`+0.0048`).
- **5000 steps:** Both compressed lanes are worse than dense (`+0.0248` / `+0.0301`).
  Gap vs dense **grew** with longer training — single-seed, so treat as noisy but
  not a clean scaling win at 5000.
- Within the same 5000 run, Tequila still beats standard compressed (`+0.0248` vs
  `+0.0301`).

## Decision

- **Deploy baseline:** `mixed_top512` (standard STE) — safest export path (Exp 20).
- **Training candidate:** `mixed_top512_tequila` — use for training smokes; not
  proven to beat dense at 5000 steps on one seed.
- See [`../VOCAB_RECIPE.md`](../VOCAB_RECIPE.md) for locked labels.
