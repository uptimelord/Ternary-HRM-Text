# Experiment 4.10 — Equilibrium Propagation (Blume-Capel / Ternary)

## Question

Can **Equilibrium Propagation** update ternary weights via free/nudged phase
correlation differences — without `loss.backward()` — and beat AdamW?

## Method (from literature)

- **EP** (Scellier & Bengio, Frontiers 2017): minimize energy `E`; free phase
  settles to `s_*`; nudged phase adds `-β ∂ℓ/∂s` toward labels to get `s_*^β`.
- Weight update (contrastive Hebbian):

  `ΔW_ij = (1/β) (ρ(s_i^*) ρ(s_j^*) - ρ(s_i) ρ(s_j))`

- **Binary/ternary EP** (Laborieux et al., arXiv:2103.08953): gradient estimate
  is **ternary** `{−2/β, 0, +2/β}`; **BOP** flips discrete weights from momentum
  of EP estimate (no latent AdamW in method arm).

**Sandbox adaptation:**

1. Free forward: capture module pre/post activations.
2. Nudged forward: nudge output logits toward labels with strength `β`.
3. Compute batch-averaged contrastive `ΔW`; apply top-|ΔW| **ternary quant shifts**
   via BOP threshold (max flips per step capped).
4. Compare global eval CE vs AdamW under matched wall-clock.

Helpers: `../sandbox_ep.py`, `../sandbox_dfo_common.py`.

## Decision Rule

- Promote if EP beats AdamW on >= 2/3 seeds, mean edge >= +0.05 CE, VRAM <= 1.25 GB,
  no backward in EP arm, total ternary flips > 0.
- Kill if AdamW wins or edge within noise floor (+/- 0.0203).

## Results

**CUDA 3-seed (kill):** AdamW wins all seeds. Mean edge **+9.40 CE** vs AdamW. **Zero ternary flips** across all cycles — BOP threshold never fired; EP arm made no weight updates (implementation bug or miscalibrated β/threshold).

Verdict: **kill**. See `report.json`.

```powershell
python "experiments/Sandbox - Blume-Capel Ternary Search/Experiment 4.10 - Equilibrium Propagation/run_exp4_10.py" --device cpu --seeds 1 --K 2 --cycles 1

python "experiments/Sandbox - Blume-Capel Ternary Search/Experiment 4.10 - Equilibrium Propagation/run_exp4_10.py" --device cuda --seeds 1,2,3 --K 20 --cycles 20 --beta 0.5
```

## References

- Scellier & Bengio, *Equilibrium Propagation*, Frontiers in Computational Neuroscience 2017
- Laborieux et al., *Training Dynamical Binary Neural Networks with Equilibrium Propagation*, arXiv:2103.08953
