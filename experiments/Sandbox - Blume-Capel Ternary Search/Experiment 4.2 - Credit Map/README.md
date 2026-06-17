# Experiment 4.2 — Credit Map

## Question

Can a bandit credit map over ternary blocks and shift transitions beat random-focus
Metropolis without extra forward budget?

## Method

Arms: blind, random-focus, credit-map (UCB-style block + transition scores from
per-flip CE reward). Same Metropolis accept/reject shell.

## Decision Rule

- Promote if credit map beats random-focus on >= 2/3 seeds, mean edge >= +0.0203 CE,
  VRAM <= 1.25 GB, no backprop.
- Kill if mean edge within noise despite occasional per-seed wins.

## Results

See `report.json`, `results_credit_map.md`.

| Summary | Value |
|---------|-------|
| Seeds | 3 |
| Credit beats random-focus (per-seed CE) | 2/3 |
| Mean edge vs random-focus | **-0.0003 CE** (within noise) |
| Verdict | **Kill** — bandit complexity not worth it |

```powershell
python "experiments/Sandbox - Blume-Capel Ternary Search/Experiment 4.2 - Credit Map/run_exp4_2.py" --device cuda --seeds 1,2,3 --K 20 --cycles 20
```
