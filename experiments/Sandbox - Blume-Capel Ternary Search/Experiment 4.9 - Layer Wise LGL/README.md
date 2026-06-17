# Experiment 4.9 — Layer-Wise Greedy Local Learning (InfoPro / LGL)

## Question

Does **greedy layer-wise discrete search** beat AdamW when each ternary module is
optimized against a **local** objective instead of global CE?

## Method (from literature)

- **InfoPro** (Wang et al., arXiv:2101.10832 / ICLR 2021): split network into
  gradient-isolated modules; avoid information collapse with an InfoPro loss that
  preserves input information while discarding task-irrelevant nuisance. Tractable
  surrogate: reconstruction + local CE on a small auxiliary head.
- **Greedy layer-wise** (Bengio et al., NeurIPS 2007): train one layer/module at a
  time, freeze earlier modules, move up the stack.

**Sandbox adaptation (no backprop in method arm):**

1. Fixed random projection head on each module's hidden activations (InfoPro-style
   local surrogate; we use MSE to random label projection as lightweight recon+CE proxy).
2. **CEM** on one ternary module at a time, minimizing **local** energy only.
3. Freeze module quants, proceed to next module (bottom → top).
4. Final headline metric: **global eval CE** vs AdamW under matched wall-clock.

Helpers: `../sandbox_lgl.py`, `../sandbox_cem.py`, `../sandbox_dfo_common.py`.

## Decision Rule

- Promote if LGL beats AdamW on >= 2/3 seeds, mean edge >= +0.05 CE, VRAM <= 1.25 GB,
  no backward in LGL arm, actual ternary updates > 0.
- Kill if AdamW wins or edge within noise floor (+/- 0.0203).

## Results

**CUDA 3-seed (kill):** AdamW wins all seeds. Mean edge **+9.57 CE** vs AdamW.

Verdict: **kill** — InfoPro random-surrogate LGL does not beat AdamW. Exp 4.13 tests real low-rank local LM CE instead.

See `report.json`.

```powershell
python "experiments/Sandbox - Blume-Capel Ternary Search/Experiment 4.9 - Layer Wise LGL/run_exp4_9.py" --device cpu --seeds 1 --K 2 --cycles 1 --pop-size 3 --generations 2

python "experiments/Sandbox - Blume-Capel Ternary Search/Experiment 4.9 - Layer Wise LGL/run_exp4_9.py" --device cuda --seeds 1,2,3 --K 20 --cycles 10 --pop-size 8
```

## References

- Wang et al., *InfoPro: Locally Supervised Deep Learning by Maximizing Information Propagation*, arXiv:2101.10832
- Bengio et al., *Greedy Layer-Wise Training of Deep Networks*, NeurIPS 2007
