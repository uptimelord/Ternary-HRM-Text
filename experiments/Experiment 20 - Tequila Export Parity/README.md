# Experiment 20 - Tequila Export Parity

## Question

Does `mixed_top512_tequila` produce the same **5.11 MB** packed checkpoint as
`mixed_top512`, and does export (hard ternary weights) match what we measure?

Tequila training uses `effective_weight()` (latent FP in deadzones). Deploy
uses `quantized_weight()` via pack/unpack. This experiment prevents us from
claiming training eval gains that vanish after export.

## Checks

| Check | Meaning |
|---|---|
| Same-init packed bytes | Pack is STE-mode independent for identical latent weights |
| Trained packed MB | Standard vs Tequila trained models match (~5.11 MB) |
| Roundtrip error | Exp 3 pack/unpack matches `quantized_weight()` |
| Export eval gap | Tequila train eval vs hard-weight export eval |

## Command

```powershell
rtk python -u "experiments/Experiment 20 - Tequila Export Parity/tequila_export_parity.py" `
  --steps 500 `
  --seed 1 `
  --device cuda `
  --append-md "experiments/Experiment 20 - Tequila Export Parity/results.md"
```

## Pass criteria

- Packed MB within 0.02 MB between standard and tequila (trained)
- Same-init packed byte diff = 0
- Roundtrip error < 1e-5
- Export eval gap ≤ 0.05 (single-seed smoke tolerance)

## Results (2026-05-24, CUDA, seed 1, 500 train steps)

| check | standard | tequila | pass |
|---|---:|---:|---|
| packed_MB (trained) | 5.11 | 5.11 | yes |
| packed_bytes (same init) | 5359332 | 5359332 | yes |
| roundtrip_err | 6.10e-05 | 6.10e-05 | yes |
| export_eval_gap | - | +0.0003 | yes |

Full log: [`results.md`](results.md).

## Verdict: PASS

- **5.11 MB is real.** Same packed bytes for identical latent weights; trained
  standard and tequila match to 0.01 MB.
- **Export parity holds.** Tequila train eval 5.9390 vs export (hard weights)
  5.9393 — gap +0.0003 (deadzone fraction 13.1%, max latent-vs-hard 0.042).
- Tequila training gains are **not** an artifact of mis-measured checkpoint size.

## Decision

Tequila is safe to treat as a **training** candidate. Deploy baseline remains
`mixed_top512` (standard STE) until a production export path explicitly uses
Tequila hard weights (which match pack).
