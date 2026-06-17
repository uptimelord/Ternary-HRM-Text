# Experiment 1 — PoC Measurements (CPU)

## Question

Can Metropolis-style ternary flips improve eval CE on a tiny TRM body **without**
`loss.backward()` or optimizer state?

## Method

- Global blind Metropolis: random module, random coordinate block, random ternary shift.
- Energy = train CE (+ optional sparsity / flip penalty; defaults off).
- CPU smoke: h=32, 200 steps, 5 seeds.
- Baseline: same step budget with random flips only (no accept/reject).

## Decision Rule

- Promote if Metropolis final eval CE beats random-flip baseline on >= 4/5 seeds
  and mean edge >= +0.0203 CE, with actual ternary flips > 0 and no backprop.
- Kill if no edge beyond noise, hidden gradients, or held-out ID leak.

## Results

See root `../report.json` and runner `run_exp1.py`.

| Summary | Value |
|---------|-------|
| Seeds | 5 |
| Beats random | **5/5** |
| Mean edge vs random | ~+0.43 eval CE |
| Verdict | **Promote PoC** — proceed to CUDA scale (Exp 2) |

```powershell
python "experiments/Sandbox - Blume-Capel Ternary Search/Experiment 1 - PoC Measurements/run_exp1.py" --device cpu --steps 200 --seeds 1,2,3,4,5
```
