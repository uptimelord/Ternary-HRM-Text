# Experiment 4.12 — Random Projection Feedback (RPF) + Ternary Greedy

## Question

Can **scalable DFA-style RPF** (fixed Q/P projections, pseudo-gradient flip bias,
greedy CE accept) beat AdamW on eval CE — without backward?

## Method

Adapted to `build_trm_lmhead` (half_layers, H×L recurrence, ternary body):

1. One forward while recording **input activations `a`** to each ternary linear
   (all recurrence invocations via hooks).
2. Output error `e = softmax(logits) - one_hot(target)`; project `z = e @ Q`
   with fixed `Q ∈ R^{V×d}` (default d=64).
3. Per layer ℓ: `gW_ℓ = (z @ P_ℓ)^T @ a` accumulated over steps; `P_ℓ ∈ R^{d×out}`.
4. Propose ternary flips where `|gW| > threshold`; **greedy accept** if train CE drops.
5. Optional Blume-Capel: use `gW` sign to bias Glauber/Metropolis proposals.

**Memory at scale:** Q + all P matrices ≈ **12 MB** overhead at 300M (fixed d, independent of seq).

Helpers: `../sandbox_rpf.py`, `../sandbox_dfo_common.py`.

## Decision Rule

- Promote if RPF beats AdamW on >= 2/3 seeds, mean edge >= +0.05 CE, VRAM <= 1.25 GB,
  no backward, total flips > 0.
- Kill if AdamW wins or edge within noise floor (+/- 0.0203).

## Results

**CUDA 3-seed (kill):** AdamW wins all seeds. Mean edge **+9.10 CE** vs AdamW. RPF made ~205–247 flips/seed and dropped eval CE by ~0.16–0.38 (init ~11.5 → ~11.0 best) but never approached AdamW ~2.0.

Verdict: **kill**. See `report.json`.

```powershell
python "experiments/Sandbox - Blume-Capel Ternary Search/Experiment 4.12 - RPF Ternary Greedy/run_exp4_12.py" --device cpu --seeds 1 --K 2 --cycles 1

python "experiments/Sandbox - Blume-Capel Ternary Search/Experiment 4.12 - RPF Ternary Greedy/run_exp4_12.py" --device cuda --seeds 1,2,3 --K 20 --cycles 20 --d-bneck 64
```

## References

- Nøkland, *Direct Feedback Alignment*, 2016
- Align-Ada / random-projection feedback for scalable training without backprop
