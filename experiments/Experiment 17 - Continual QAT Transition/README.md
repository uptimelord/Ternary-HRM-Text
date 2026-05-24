# Experiment 17 - Continual QAT Transition

## Question

Does dense-to-ternary transition training beat ternary-from-scratch for
`mlp_gate_up` body layers?

## Setup

- Same token slice and eval as Exp 13/16
- Transition at 20% of steps (step 100 of 500)
- Dense weights copied into ternary latent weights at transition
- Optimizer rebuilt at transition (AdamW momentum reset — smoke limitation)

## Variants

| Variant | Description |
|---|---|
| `dense` | Untied dense baseline |
| `ternary_from_scratch_mlp_gate_up` | Ternary from step 0, standard STE |
| `transition_mlp_gate_up_standard` | Dense → ternary at 20%, standard STE |
| `transition_mlp_gate_up_tequila` | Dense → ternary at 20%, Tequila STE (Exp 16 showed body help) |

## Command

```powershell
rtk python -u "experiments/Experiment 17 - Continual QAT Transition/continual_qat_transition.py" `
  --steps 500 `
  --transition-ratio 0.2 `
  --warmup-steps 2 `
  --seeds 1 `
  --device cuda `
  --append-md "experiments/Experiment 17 - Continual QAT Transition/results_500.md"
```

## Decision rule

- If transition beats scratch → use transition as default for body ternary tests
- If transition + Tequila beats both → stack for Exp 19
- If transition fails → body ternary needs mixed precision / layer selection

## Results (2026-05-24, CUDA, seed 1, transition at step 100)

| variant | final_eval | gap_vs_dense | packed_MB | tok/s |
|---|---:|---:|---:|---:|
| dense | 6.0064 | +0.0000 | 66.76 | 2769 |
| ternary_from_scratch_mlp_gate_up | 5.9894 | -0.0169 | 65.81 | 2110 |
| transition_mlp_gate_up_standard | 6.0396 | +0.0332 | 65.81 | 1540 |
| transition_mlp_gate_up_tequila | 6.0289 | +0.0226 | 65.81 | 1333 |

Full log: [`results_500.md`](results_500.md).

## Verdict: transition fails

Scratch ternary still wins (`-0.0169` vs dense). Both transition lanes are worse
than dense (`+0.0226` to `+0.0332`). Tequila transition is slightly better than
standard transition but does not close the gap to scratch.

Transition training is **not** the default for body ternary on this smoke setup.
Exp 19 should use **mixed_top512** vocab (Exp 16 best: `-0.0676` with Tequila)
as the compressed recipe, not body transition or stacked.
