# Experiment 3 — AdamW Gate

## Question

Can pure Metropolis CE search beat AdamW on the same tiny model and data slice?
(AdamW is **baseline only** — not part of the no-backprop method.)

## Method

Per seed: shared init → Metropolis arm (Exp 2) vs AdamW CE arm on identical batches.
Report eval CE, VRAM, and optimizer-state tensor count for AdamW.

## Decision Rule

- Promote if Metropolis beats AdamW eval CE on >= 2/3 seeds by >= +0.05 CE,
  VRAM <= 1.25 GB for Metropolis arm, no backprop in Metropolis arm.
- Kill if AdamW wins decisively — lane stays "fair comparison / VRAM savings only".

## Results

See `report.json`, `results_adamw_gate.md`.

| Summary | Value |
|---------|-------|
| Seeds | 3 |
| Metro beats AdamW | **0/3** |
| Mean edge vs AdamW | **-2.44 CE** (AdamW wins) |
| VRAM savings (metro) | ~666 MB vs AdamW |
| Verdict | **Kill beat-AdamW claim** — Metropolis is a valid no-backprop lane, not CE-competitive with AdamW from scratch |

```powershell
python "experiments/Sandbox - Blume-Capel Ternary Search/Experiment 3 - AdamW Gate/run_exp3.py" --device cuda --seeds 1,2,3
```
