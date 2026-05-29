# Equilibrium Reasoners: Learning Attractors Enables Scalable Reasoning

- **Authors:** Benhao Huang, Zhengyang Geng, Zico Kolter (CMU Locus Lab)
- **Venue:** ICML 2026
- **arXiv:** https://arxiv.org/abs/2605.21488
- **Code:** https://github.com/locuslab/eqr

## Core Idea

Weight-tied iterative models define a learned dynamical system. Generalizable
reasoning emerges when stable fixed points (attractors) correspond to correct
solutions. Scaling test-time compute works only when the attractor landscape
is well-shaped during training.

## Method

Iteration rule with damping and noise:

```
z_{k+1} = z_k + (1-λ)(f_θ(z_k; x) - z_k) + β·ε_k
```

Two-axis test-time scaling:
- **Depth:** more iterations per trajectory
- **Breadth:** multiple stochastic restarts, pick lowest residual ‖f(z) - z‖

Training interventions to shape attractor landscape:
1. **Randomized initialization (RI)** — diverse starting points broaden basins
2. **Noise injection (NI)** — perturbations during iteration avoid spurious attractors
3. **Segmented online training (SOT)** — truncated gradient segments, each supervised independently
4. **Adaptive computation time (ACT)** — learned halting, more iterations for harder instances

## Key Results

| Task | Feedforward | Baseline Iterative | EqR |
|------|-------------|-------------------|-----|
| Sudoku-Extreme | 2.6% | — | 99.8% |
| Maze-Unique | — | 44.9% | 93.0% |

- Trained with 16 iterations, tested at D=64 — equivalent to 40,000+ effective layers
- ACT reduces average forward evaluations by up to 17.4× with minimal accuracy loss
- Breadth scaling (multiple restarts) provides complementary gains to depth

## Relevance to BitNet-HRM

This paper directly explains our Exp32 result:
- H=2 works (trained attractor), H=4+ collapses (overshoots into spurious basin)
- The fix: add damping, noise injection, and randomized init to HRM training
- Their SOT = our truncated BPTT with bp_steps, but they add the landscape-shaping tricks
- Breadth scaling at inference = run multiple forward passes with noise, pick best convergence
- Proves the thesis: fixed-size models CAN reason better with more iterations, IF trained correctly

## Implementation Priority

1. Add damping factor λ to HRM iteration loop
2. Add noise injection β during training iterations
3. Train with randomized hidden state initialization
4. Monitor convergence residual ‖f(z) - z‖ as a diagnostic
5. Test depth scaling again after attractor-shaped training
