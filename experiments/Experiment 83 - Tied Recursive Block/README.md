# Experiment 83 - Tied Recursive Block

> Architecture brief **C5**: TRM tied recursive block vs HRM control — denominator / quality_per_mb play.

## Decision Rule

Promote if quality_per_mb beats HRM control by > noise floor AND strict logic within 2 pp of control.

Kill if pretrain unstable or strict logic drops > 5 pp.

Report both:

- `fp32_mb`: normal state dict size.
- `packed_mb`: packed ternary-aware state dict size.
- `q/mb`: full model quality over `packed_mb`, including vocab head.
- `q/body_mb`: backbone quality over packed body MB. This is the direct TRM denominator check, because the vocab head can hide body shrinkage.

TRM uses `--ternary-body` by default, so both arms are ternary. **Recipe
asymmetry caveat:** the recipes are not identical — the TRM arm ternarizes the
whole body (`target: "body"`, all attention + MLP), while the HRM control uses
the Phase-0 `L_mlp_gate_up` subset only. q/mb comparisons therefore mix
architecture and ternary-coverage effects; read `q/body_mb` with that in mind.
Use `--no-ternary-body` only for an explicit dense-body ablation — note it
de-ternarizes the TRM arm only (HRM stays tequila-ternary), so the arms diverge
further in recipe under that flag.

## Run

```powershell
python "experiments/Experiment 83 - Tied Recursive Block/tied_recursive_block.py" --mode smoke --device cpu
python "experiments/Experiment 83 - Tied Recursive Block/tied_recursive_block.py" --mode full --device cuda
```

The full command above is already decision-grade: defaults are seeds `1,2`,
pretrain 5000 steps, logic SFT 8000 steps, logic-hard eval n=200, ternary body
on, true packed-MB denominators (`packed_exact` must be `true` in the report).

## Results

Canonical result: `results_full_seed1.md` (+ `report.json`), decision-grade
CUDA run 2026-06-13, seeds 1,2, `packed_exact: true`, peak VRAM 462 MB.

| arm | packed_mb | q/mb (mean) | strict logic hard (mean, n=200) | pretrain loss |
|---|---:|---:|---:|---:|
| HRM control | 66.28 | 0.0065 | 0.1925 | 2.58 / 2.08 |
| TRM tied | 64.08 | 0.0422 | 0.2050 | 0.369 / 0.371 |

Superseded run logs: `results_*_codex.md` files in this folder are historical
run logs from before true packed-MB accounting (their `packed_mb` is fp32
bytes); do not use them for promote/kill decisions.

## Verdict

**Promote.** TRM q/mb 0.0422 vs HRM 0.0065 (Δ 0.0356, far above noise) and
strict logic Δ +1.25 pp mean (within the 2 pp guard; per-seed −1.5 / +4.0 pp).

Caveat: the q/mb gap is dominated by the pretrain-loss numerator (TRM 0.37 vs
HRM ~2.3), not the packed-MB denominator (64 vs 66 MB). Exp71's lesson stands —
pretrain loss is a proxy; the downstream strict-logic guard is what held here.
Follow-up before relying on TRM downstream: verify the loss gap is not an
objective/recipe artifact of the tied block.
