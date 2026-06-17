# Experiment 4.7 — Cross-Entropy Method (CEM)

## Question

Can block-level Cross-Entropy Method beat **AdamW** on eval CE under a matched
wall-clock budget — without backprop in the CEM arm?

## Method

- Maintain categorical probs `[p_-1, p_0, p_1]` per weight on a focal ternary module.
- Each cycle: sample `pop_size` full-module configs, evaluate train CE, refit probs
  from top `elite_frac`, keep best member.
- After CEM finishes, reload init weights and run **AdamW baseline** for the same
  elapsed seconds (AdamW uses backward; baseline only).
- Helpers: `../sandbox_cem.py`, `../sandbox_dfo_common.py`.

## Decision Rule

- Promote if CEM beats AdamW on >= 2/3 seeds, mean edge >= +0.05 CE,
  VRAM <= 1.25 GB, no backprop in CEM arm.
- Kill if AdamW wins or edge within noise floor (+/- 0.0203).

## Results

**CUDA 3-seed (kill):** AdamW wins all seeds. Mean edge **+8.55 CE** (CEM worse). CEM ~10.4–10.6 vs AdamW ~1.8–2.0 eval CE. VRAM: CEM ~1.07 GB vs AdamW ~1.74 GB.

| seed | adamw CE | cem CE | edge |
|------|----------|--------|------|
| 1 | 2.04 | 10.58 | +8.53 |
| 2 | 1.88 | 10.38 | +8.50 |
| 3 | 1.83 | 10.44 | +8.62 |

Verdict: **kill** — global CEM does not beat AdamW; local-credit lane (4.9+) is next.

Pending: none. See `report.json`.

```powershell
python "experiments/Sandbox - Blume-Capel Ternary Search/Experiment 4.7 - Cross Entropy Method/run_exp4_7.py" --device cpu --seeds 1 --K 2 --cycles 1 --pop-size 3 --generations 2

python "experiments/Sandbox - Blume-Capel Ternary Search/Experiment 4.7 - Cross Entropy Method/run_exp4_7.py" --device cuda --seeds 1,2,3 --K 20 --cycles 20 --pop-size 8
```
