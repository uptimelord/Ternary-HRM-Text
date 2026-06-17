# Experiment 2 — CUDA Scale

## Question

Does blind Metropolis still beat a random-flip baseline at CUDA scale within ~1 GB VRAM?

## Method

Same as Exp 1 PoC, scaled to h=64, 200 Metropolis steps, 3 seeds on CUDA.
Train/eval batches from Exp70 logic JSONL (train-visible only).

## Decision Rule

- Promote if Metropolis beats random on >= 2/3 seeds, mean edge >= +0.0203 CE,
  peak VRAM <= 1.25 GB, ternary flips > 0, no backprop.
- Kill if CUDA regression vs CPU PoC direction, VRAM blowup, or no edge beyond noise.

## Results

See `report.json`, `results_cuda_seed123.md`.

| Summary | Value |
|---------|-------|
| Seeds | 3 |
| Beats random | **3/3** |
| Mean edge vs random | **+0.431 CE** |
| Mean metro delta | +0.406 CE |
| Peak VRAM | ~1066 MB |
| Verdict | **Promote scale** — fair no-backprop baseline established |

```powershell
python "experiments/Sandbox - Blume-Capel Ternary Search/Experiment 2 - CUDA Scale/run_exp2.py" --device cuda --steps 200 --seeds 1,2,3
```
