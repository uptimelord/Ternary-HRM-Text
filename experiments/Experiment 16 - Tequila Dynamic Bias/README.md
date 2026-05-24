# Experiment 16 - Tequila Dynamic Bias

## Question

Does Tequila-style dynamic bias fix the body ternary quality problem and the
body+vocab interaction penalty from Experiment 13?

## Why This Comes First

This is a layer-level change. If it works, every downstream ternary experiment
should use the improved ternary layer. If it fails, deadzone trapping is probably
not the main blocker, and the next best path is dense-to-ternary transition
training.

## Variants

| Variant | Vocab | Body | STE mode |
|---|---|---|---|
| `dense` | untied dense | dense | none |
| `mixed_top512_standard` | mixed top-512 dense rows | dense | standard vocab |
| `mixed_top512_tequila` | mixed top-512 dense rows | dense | Tequila vocab |
| `mlp_gate_up_standard` | untied dense | `mlp_gate_up` | standard body |
| `mlp_gate_up_tequila` | untied dense | `mlp_gate_up` | Tequila body |
| `stacked_standard` | mixed top-512 dense rows | `mlp_gate_up` | standard vocab/body |
| `stacked_tequila` | mixed top-512 dense rows | `mlp_gate_up` | Tequila vocab/body |

## Command (CUDA, seed 1, 500 steps)

```powershell
rtk python -u "experiments/Experiment 16 - Tequila Dynamic Bias/tequila_dynamic_bias.py" `
  --steps 500 `
  --warmup-steps 2 `
  --seeds 1 `
  --device cuda `
  --append-md "experiments/Experiment 16 - Tequila Dynamic Bias/results_500.md"
```

## Results (2026-05-24, CUDA, seed 1)

| variant | final_eval | gap_vs_dense | packed_MB | compr | tok/s | peak_VRAM_MB |
|---|---:|---:|---:|---:|---:|---:|
| dense | 6.0064 | +0.0000 | 66.76 | 1.00x | 3275 | 660.3 |
| mixed_top512_standard | 5.9428 | -0.0635 | 5.11 | 6.85x | 3264 | 602.7 |
| mixed_top512_tequila | 5.9388 | -0.0676 | 5.11 | 6.85x | 3055 | 672.2 |
| mlp_gate_up_standard | 5.9894 | -0.0169 | 65.81 | 1.01x | 2886 | 679.5 |
| mlp_gate_up_tequila | 5.9825 | -0.0239 | 65.81 | 1.01x | 2689 | 690.1 |
| stacked_standard | 6.0415 | +0.0351 | 4.17 | 8.40x | 2681 | 627.4 |
| stacked_tequila | 6.0341 | +0.0277 | 4.17 | 8.40x | 1906 | 702.0 |

Full live log: [`results_500.md`](results_500.md).

## Tequila read

| Lane | Standard gap | Tequila gap | Tequila better? |
|---|---:|---:|:---:|
| mixed_top512 vocab | -0.0635 | -0.0676 | yes (small) |
| mlp_gate_up body | -0.0169 | -0.0239 | yes (small) |
| stacked vocab+body | +0.0351 | +0.0277 | yes (small) |

## Verdict: ambiguous but directionally helpful

Tequila improves every lane, but gains are small (0.004–0.007 eval). Single-seed
500-step noise is likely in the same range, so treat this as a weak signal, not
proof.

- Body ternary: Tequila wins (`-0.0239` vs `-0.0169`). Use Tequila as default
  body STE in Exp 17 transition runs.
- Stacked interaction: Tequila closes the penalty (`+0.0277` vs `+0.0351`) but
  stacked is still worse than dense. Stacking remains risky.
- Vocab-only: Tequila is slightly better; not enough to change the mixed_top512
  default on its own.

## Decision

- Default body STE → **tequila** for downstream experiments.
- Include `transition_mlp_gate_up_tequila` in Exp 17.
- Do **not** rerun Exp 13 stacked yet; gap is still positive with Tequila.
- Exp 17 transition training is now the more important fix for body ternary.
