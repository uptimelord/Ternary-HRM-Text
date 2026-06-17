# Experiment 4.6 — Forward-Forward Lite

> **Status: Kill (CPU probe).** CUDA job removed from queue.

## Question

Can Hinton-style Forward-Forward goodness (positive vs corrupted negative batch)
drive Metropolis search better than random-focus CE alone — without backward?

## Method

- **Proposal:** random-focus (Exp 4.2 anchor).
- **Energy:** `-goodness(pos) + goodness(neg)` on TRM hidden states (mean `relu(h)^2`
  on supervised positions). Optional `ce_weight` tie-break.
- **Negative batch:** shuffle response tokens within each sequence.
- Shared helpers: `../sandbox_ff_energy.py`.

## Decision Rule

- Promote if FF beats random-focus eval CE on >= 2/3 seeds, mean edge >= +0.0203 CE,
  VRAM <= 1.25 GB on CUDA, ternary flips > 0, no backprop.
- Kill if no edge beyond noise or hidden gradients.

## Results

CPU 3-seed probe (K=3, cycles=2): **kill** — 0/3 beats random-focus, mean edge **-0.006 CE** (within noise). CUDA job queued but low priority after 4.4/4.5.

Full CUDA: pending queue slot. See `report.json` when run completes.

```powershell
python "experiments/Sandbox - Blume-Capel Ternary Search/Experiment 4.6 - Forward Forward Lite/run_exp4_6.py" --device cpu --seeds 1 --K 1 --cycles 1

python "experiments/Sandbox - Blume-Capel Ternary Search/Experiment 4.6 - Forward Forward Lite/run_exp4_6.py" --device cuda --seeds 1,2,3 --K 20 --cycles 20
```
