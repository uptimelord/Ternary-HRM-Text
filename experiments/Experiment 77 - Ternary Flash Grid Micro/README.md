# Experiment 77 — Ternary Flash Grid Micro

> **Status: smoke + oracle + distil completed; do not promote overlay yet.** Distil
> lowers valid loss versus baseline, but random-oracle search found **0 / 64**
> trials beating baseline — rank-8 injection point may be wrong for this checkpoint.

## Question

Can a **frozen** `mixed_top512_tequila_L_mlp_gate_up` backbone plus a tiny
**HyperBuilder (ROM)** and rank-8 **ternary overlay (RAM)** injected into L-level
MLP outputs improve arithmetic SFT loss without retraining the body?

This is the arithmetic DistIL pattern (`models/fast_weight_overlay.py`) at micro
scale before Exp74 verified-logic DistIL.

## Method

Frozen checkpoint + trainable Builder only. Ternary snap uses repo
`TernaryLinear158Init` STE convention (`{-1,0,+1}`). Default checkpoint order:

1. `artifacts/exp76_smoke/plain_sft/checkpoint_fp32.pt`
2. `artifacts/phase0_first_pretrain/h256_steps50000_seed1_exportcalib3000/checkpoint_fp32.pt`

| Mode | Purpose |
|------|---------|
| `smoke` | Wiring, backward on Builder only, peak VRAM |
| `baseline` | Frozen checkpoint without overlay |
| `oracle` | Random ternary rank-8 search ceiling |
| `distil` | Train Builder on arithmetic SFT (backbone frozen) |

Design references (no extra pip deps):

- [shyamsn97/hyper-nn](https://github.com/shyamsn97/hyper-nn) — hypernet plumbing
- [labml_nn/fast_weights](https://github.com/labmlai/annotated_deep_learning_paper_implementations/tree/master/labml_nn/transformers/fast_weights) — forward injection

Runner: `ternary_flash_grid.py`. Rank 8, inject scale 0.25, ~1.63M Builder params.

## Decision Rule

Promote if `distil` mode yields `valid_overlay.loss` at least **0.05** below
`valid_baseline.loss` **and** oracle search finds at least one trial beating
baseline by ≥ 0.05 (overlay rank/inject point is real, not lucky distil).

Kill if oracle finds **0 / N** trials beating baseline, or distil loss wins only
while frozen exact accuracy stays at 0% with no heldout parity path.

## Results

### Smoke

```powershell
rtk python -u "experiments/Experiment 77 - Ternary Flash Grid Micro/ternary_flash_grid.py" `
  --mode smoke --steps 10 --device cuda
```

Pass: finite loss, peak VRAM < 1200 MB, no crash. Covered by
`tests/test_exp77_ternary_flash_grid.py`.

### Oracle (64 trials, seed 1)

`results_smoke_seed1.md` (oracle block):

| Metric | Value |
|---|---:|
| `baseline_loss` | 2.010 |
| `best_oracle_loss` | 2.010 |
| `oracle_improved_trials` | **0 / 64** |
| `delta_loss` | 0.0 |

Random rank-8 ternary overlays do not beat the frozen baseline on the valid slice.

### Distil (500 steps, seed 1)

`results_distil_seed1.md`, checkpoint
`artifacts/exp76_smoke/plain_sft/checkpoint_fp32.pt`:

| Split | loss | token_acc | exact_acc |
|---|---:|---:|---:|
| valid baseline | 2.053 | 46.9% | 0.0% |
| valid overlay | **1.757** | 51.0% | 0.0% |
| train last (overlay) | 1.823 | 49.1% | 0.0% |

Peak VRAM (distil): **590.8 MB**. Valid loss delta: **−0.296** (overlay wins on
loss), but exact accuracy remains 0% on this short run.

## Read

Distil shows the Builder can nudge token loss, but the oracle ceiling says the
chosen rank-8 inject geometry is not obviously right — random ternary weights
never beat baseline. Do not promote to default training. Next step: try a
different inject site, higher rank, or Exp70 logic checkpoint before Exp74
verified-logic DistIL.

## Commands

```powershell
# smoke
rtk python -u "experiments/Experiment 77 - Ternary Flash Grid Micro/ternary_flash_grid.py" `
  --mode smoke --steps 10 --device cuda

# oracle ceiling
rtk python -u "experiments/Experiment 77 - Ternary Flash Grid Micro/ternary_flash_grid.py" `
  --mode oracle --oracle-trials 128 --device cuda

# distil train
rtk python -u "experiments/Experiment 77 - Ternary Flash Grid Micro/ternary_flash_grid.py" `
  --mode distil --steps 500 --batch-size 4 --device cuda `
  --output-dir artifacts/exp77_flash_grid/distil_seed1 `
  --append-md "experiments/Experiment 77 - Ternary Flash Grid Micro/results_distil_seed1.md"

pytest tests/test_exp77_ternary_flash_grid.py -q
```
