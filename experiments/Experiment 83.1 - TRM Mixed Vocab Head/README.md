# Experiment 83.1 - TRM Mixed Vocab Head (train-time mixed_top512_tequila)

> Follow-up to Exp83's promote caveat. Exp83 promoted the Tied Recursive Block
> on `q/mb` but used `--ternary-body` only — the **vocab head was left fp32**, so
> 64 of its 64.08 MB packed size was an uncompressed head. This experiment applies
> the promoted deploy vocab recipe (`mixed_top512_tequila`) to the head **during
> training** (not post-hoc), so the head learns ternary-robust weights and packs
> ~3.5 MB instead of 64 MB.

## Motivation (the receipt)

From Exp83 `report.json`, both arms (seed 1):

| arm | head_params | head_packed_mb | body_packed_mb | total packed | q/mb | q/body_mb |
|---|---:|---:|---:|---:|---:|---:|
| HRM | 16.77M | **64.00** | 2.28 | 66.28 | 0.0065 | 0.17 |
| TRM | 16.77M | **64.00** | 0.08 | 64.08 | 0.0422 | **33.8** |

99.9% of packed size is an identical, uncompressed vocab head. The TRM body is
already the smallest in the repo (`q/body_mb` = 33.8 vs HRM 0.17, ~200×), but the
headline `q/mb` buries it under the fat head. Compressing the head with the deploy
recipe should drop total packed to ~3.5–4 MB and lift `q/mb` ~18× (~0.042 → ~0.7).

## What changed (train-time, not post-hoc)

`--head-recipe mixed_top512` (default stays `dense` = the Exp83 run, unchanged):
- vocab head becomes a tied **ternary base** (1.58-bit, group 32, threshold 0.25,
  **tequila STE**) + **dense top-512 override rows** (most-frequent tokens kept fp32).
- Reuses `models.layers.TernaryLinear158Init` + Exp9 `MixedPrecisionTiedVocabHead`
  via the shared helper `training.arch_backbone.apply_mixed_top512_head`.
- Applied to **both arms** before training, so `q/mb` is apples-to-apples (both
  heads ~3.5 MB; the gap then reflects body + pretrain loss, as it should).
- The Exp13 packer already recognizes `TernaryLinear158Init`, so `packed_exact`
  stays `true`.

Post-hoc compression of a trained fp32 head is explicitly **not** used — the head
trains tequila-aware so capability holds through quantization (deploy-preset method).

## Decision Rule

Promote the mixed-head pack as the TRM default if, vs the Exp83 dense-head TRM
baseline (strict logic hard **0.205**, 2 seeds):

- **Promote if** `packed_mb ≤ 5 MB` AND strict logic hard within noise (±2.03 pp)
  of 0.205 on both seeds AND invalid 0% AND `packed_exact = true`.
- **Kill if** strict logic hard drops > 5 pp under head compression (the head
  carries capability the body can't absorb — keep the fp32 head, accept the size),
  OR `packed_mb` does not fall below 10 MB (compression failed).

`q/mb` is expected to rise sharply, but the **gate is strict logic + packed size**,
not q/mb (q/mb numerator is pretrain loss — a proxy; Exp71's lesson).

## Run

```powershell
# smoke (CPU crash check)
python "experiments/Experiment 83 - Tied Recursive Block/tied_recursive_block.py" --mode smoke --device cpu --head-recipe mixed_top512

# decision-grade (CUDA, seeds 1,2, 5000 pretrain / 8000 logic SFT, frozen-200)
python "experiments/Experiment 83 - Tied Recursive Block/tied_recursive_block.py" --mode full --device cuda --head-recipe mixed_top512
```

Shares the Exp83 runner (`tied_recursive_block.py`) — no forked stack. The report
records `head_recipe`; results land in this folder as `results_full_seed1.md`.

## Status

Infra-ready / awaiting decision-grade CUDA run. Wiring validated on CPU:
`tests/test_exp83_1_mixed_head.py` (mixed head packs < 50% of dense, trains one
step through tequila STE) + the existing Exp83 regression suite stays green.
GPU run waits for a gap (serialize-only; Codex owns Exp81/84).

## Results

_Pending decision-grade run._
