# Experiment 4.8 — Evolution Strategies (ES)

## Question

Can OpenAI-style Evolution Strategies on ternary latent weights beat **AdamW** on
eval CE under matched wall-clock — without backward in the ES arm?

## Method

- Each cycle: pick a random ternary module, hold latent vector `theta`.
- Sample `pop_size` Gaussian perturbations `theta + sigma * eps`, evaluate train CE
  (ternary quantize happens in forward).
- Rank-transform rewards and update `theta` via population gradient estimate.
- AdamW baseline runs for the same elapsed seconds after ES (backward allowed in baseline only).
- Helpers: `../sandbox_es.py`, `../sandbox_dfo_common.py`.

## Decision Rule

- Promote if ES beats AdamW on >= 2/3 seeds, mean edge >= +0.05 CE,
  VRAM <= 1.25 GB, no backward in ES arm.
- Kill if AdamW wins or edge within noise floor (+/- 0.0203).

## Results

**CUDA 3-seed (kill):** AdamW wins all seeds. Mean edge **+8.94 CE** (ES worse). ES ~10.7–11.1 vs AdamW ~1.8–2.1 eval CE.

| seed | adamw CE | es CE | edge |
|------|----------|-------|------|
| 1 | 2.10 | 10.71 | +8.61 |
| 2 | 1.90 | 11.12 | +9.22 |
| 3 | 1.85 | 10.85 | +9.00 |

Verdict: **kill** — global ES does not beat AdamW; local-credit lane (4.9+) continues.

See `report.json`.

```powershell
python "experiments/Sandbox - Blume-Capel Ternary Search/Experiment 4.8 - Evolution Strategies/run_exp4_8.py" --device cpu --seeds 1 --K 2 --cycles 1 --pop-size 3 --generations 2

python "experiments/Sandbox - Blume-Capel Ternary Search/Experiment 4.8 - Evolution Strategies/run_exp4_8.py" --device cuda --seeds 1,2,3 --K 20 --cycles 20 --pop-size 8
```
