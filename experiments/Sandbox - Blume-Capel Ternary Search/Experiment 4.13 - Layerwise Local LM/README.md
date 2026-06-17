# Experiment 4.13 — Greedy Layer-Wise Local LM + Ternary Greedy

## Question

Can **block-by-block training** with a low-rank **local LM head** (real local CE,
not global signal) beat AdamW — without backward?

## Method

Distinct from Exp 4.9 (InfoPro random surrogate): uses **H → rank → V** local head
(default rank=256) and **local cross-entropy** on SFT labels.

Per physical block ℓ (`n_phys = n_layers // 2` under half_layers):

1. Data flows through **frozen** blocks `< ℓ`; full H×L recurrence on prefix.
2. Capture block ℓ outputs at each recurrence step; local head → **local CE**.
3. Update block ℓ ternary weights via **RPF-biased greedy flips** on local CE
   (same-batch accept/revert).
4. **Freeze** block ℓ; advance.

**Memory at scale:** one block alive ≈ **250 MB** peak (independent of total depth).

Helpers: `../sandbox_layerwise_lm.py`, `../sandbox_rpf.py`, `../sandbox_dfo_common.py`.

## Decision Rule

- Promote if layer-wise LM beats AdamW on >= 2/3 seeds, mean edge >= +0.05 CE,
  VRAM <= 1.25 GB, no backward, total flips > 0.
- Kill if AdamW wins or edge within noise floor (+/- 0.0203).

## Results

**CUDA 3-seed (kill):** AdamW wins all seeds. Mean edge **+9.72 CE** vs AdamW. Only block 0 trained (h=64, L=2 → 1 physical block). ~75–107 flips/seed; global eval CE barely moved (~0.12 drop best case); seed 3 **worsened**.

Verdict: **kill**. See `report.json`.

```powershell
python "experiments/Sandbox - Blume-Capel Ternary Search/Experiment 4.13 - Layerwise Local LM/run_exp4_13.py" --device cpu --seeds 1 --steps-per-block 2 --train-batches 2

python "experiments/Sandbox - Blume-Capel Ternary Search/Experiment 4.13 - Layerwise Local LM/run_exp4_13.py" --device cuda --seeds 1,2,3 --steps-per-block 20 --local-rank 256
```

## References

- Greedy layer-wise / Cramming-BERT local-loss transformers
- InfoPro arXiv:2101.10832 (4.9 uses random surrogate; this uses low-rank local LM CE)
