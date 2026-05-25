# Experiment 22 - Vocab + Body Combo Confirmation

## Question

Does the best vocab-compression lane combine cleanly with the best body-lane
signal?

Prior evidence:

- Exp 19: `mixed_top512_tequila` is the best compressed vocab training lane at
  5000 steps, but still has a small average gap vs `dense_tied_vocab`.
- Exp 21: `L_mlp_gate_up` is the best body-only target at 5000 steps, but it is
  a quality/regularizer signal rather than a size win.

This experiment checks whether `L_mlp_gate_up` offsets the remaining compressed
vocab gap.

## Variants

| Variant | Vocab | Body |
|---|---|---|
| `dense_tied_vocab` | dense tied | dense |
| `mixed_top512_tequila` | top 512 dense rows, rest ternary, Tequila STE | dense |
| `dense_tied_vocab_L_mlp_gate_up` | dense tied | L-level `mlp_gate_up`, Tequila STE |
| `mixed_top512_tequila_L_mlp_gate_up` | top 512 dense rows, rest ternary, Tequila STE | L-level `mlp_gate_up`, Tequila STE |

## Command

```powershell
rtk python -u "experiments/Experiment 22 - Vocab Body Combo Confirmation/vocab_body_combo.py" `
  --steps 5000 `
  --warmup-steps 2 `
  --seeds 1,2,3 `
  --device cuda `
  --append-md "experiments/Experiment 22 - Vocab Body Combo Confirmation/results_5000_seeds123.md"
```

## Decision Rule

Promote if `mixed_top512_tequila_L_mlp_gate_up` matches or beats `mixed_top512_tequila` on quality per packed MB, stays meaningfully compressed, and does not lose catastrophic throughput.

Kill if the combo loses to `mixed_top512_tequila` on quality per packed MB, gives back the compression win, or is too slow to keep in the training loop.

## Results - 5000 steps, seeds 1/2/3

Command:

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File `
  "experiments/Experiment 22 - Vocab Body Combo Confirmation/start_exp22_5000_seeds123.ps1"
```

Full live log: [`results_5000_seeds123.md`](results_5000_seeds123.md).

Noise floor: `+/- 0.0203` eval loss, computed from repeated dense-tied
5000-step baselines in Exp 19 and Exp 22.

| Variant | Mean eval | Gap +/- noise floor | Quality/MB | Packed size | Compression | Mean tok/s |
|---|---:|---|---:|---:|---:|---:|
| `dense_tied_vocab` | 5.1633 | +0.0000 +/- 0.0203 (at noise floor) | 0.00557 | 34.76 MB | 1.00x | 7024 |
| `mixed_top512_tequila` | 5.1720 | +0.0087 +/- 0.0203 (at noise floor) | 0.03784 | 5.11 MB | 6.85x | 5598 |
| `dense_tied_vocab_L_mlp_gate_up` | 5.1555 | -0.0078 +/- 0.0203 (at noise floor) | 0.00566 | 34.28 MB | 1.01x | 5455 |
| `mixed_top512_tequila_L_mlp_gate_up` | 5.1570 | -0.0063 +/- 0.0203 (at noise floor) | 0.04179 | 4.64 MB | 7.55x | 4071 |

## Decision

Promote `mixed_top512_tequila_L_mlp_gate_up` to the current training candidate.

Do not oversell the raw loss gap: `-0.0063` is inside the `+/- 0.0203` noise
floor. The promotion is because the combo improves the actual project metric
(`0.04179` quality/MB vs `0.03784` for vocab-only Tequila) while improving
packed size (`4.64 MB`, `7.55x`).

Tradeoff: throughput drops versus vocab-only Tequila (`4071` vs `5598` tok/s).
That is acceptable for the next quality confirmation, but this lane still needs
an export-parity check before it replaces the deploy baseline.
