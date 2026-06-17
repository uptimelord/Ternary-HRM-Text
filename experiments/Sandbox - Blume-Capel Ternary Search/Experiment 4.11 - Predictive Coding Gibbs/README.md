# Experiment 4.11 — Predictive Coding via Local Gibbs/MH

## Question

Can **local predictive-coding energy** drive ternary weight search (Gibbs/MH per
weight) without global backprop — and beat AdamW?

## Method (from literature)

- **Predictive coding** (Rao & Ballard; Whittington & Bogacz 2017): layers
  minimize prediction error; total energy `E = 0.5 Σ ||e[l]||²`.
- **PC ≈ backprop** (Millidge et al., arXiv PdauS7wZBfC): PC weight updates use
  only local pre/post activity: `ΔW[l] ∝ x[l+1]ᵀ (e[l] * f'(pre))`.
- **MCPC** (PLOS Comp Bio 2024): add noise / sampling; when closed form fails,
  use Metropolis-Hastings on local energy.

**Sandbox adaptation (no backward in method arm):**

1. Forward once; capture module `(pre, post)`.
2. Local PC target = detached post (self-consistency baseline; upgrade to
   top-down target in follow-up).
3. Propose random ternary flip on one weight; accept if local `ΔE` passes
   Metropolis test — **O(out_features)** per proposal, not full-network forward.
4. Compare global eval CE vs AdamW under matched wall-clock.

Helpers: `../sandbox_pc.py`, `../sandbox_dfo_common.py`.

## Decision Rule

- Promote if PC-MH beats AdamW on >= 2/3 seeds, mean edge >= +0.05 CE, VRAM <= 1.25 GB,
  no backward in PC arm, accept rate > 0.
- Kill if AdamW wins or edge within noise floor (+/- 0.0203).

## Results

**CUDA 3-seed (kill):** AdamW wins all seeds. Mean edge **+9.71 CE** vs AdamW. MH accepted ~250 flips/seed but global eval CE stayed ~11.5 (local self-consistency target does not align with LM objective).

Verdict: **kill**. See `report.json`.

```powershell
python "experiments/Sandbox - Blume-Capel Ternary Search/Experiment 4.11 - Predictive Coding Gibbs/run_exp4_11.py" --device cpu --seeds 1 --K 2 --cycles 1

python "experiments/Sandbox - Blume-Capel Ternary Search/Experiment 4.11 - Predictive Coding Gibbs/run_exp4_11.py" --device cuda --seeds 1,2,3 --K 20 --cycles 20
```

## References

- Whittington & Bogacz, *An Approximation of the Error Backpropagation Algorithm*, Neural Computation 2017
- Millidge et al., *Predictive Coding Approximates Backprop Along Arbitrary Computation Graphs*, ICLR 2022
- Haesegawa & Domínguez, *Learning probability distributions with Monte Carlo predictive coding*, PLOS Comp Bio 2024
