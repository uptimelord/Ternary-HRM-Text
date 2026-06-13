# Experiment 80 - Abacus Digit Embedding Probe

> Architecture brief **C1**: per-digit positional embeddings (KB-scale) vs matched control.

## Decision Rule

Promote if strict frozen add/sub pass@1 improves **at least 5 pp** over control on both seeds, invalid 0%.

Kill if gap is at or below noise on both seeds or tokenizer needs multi-digit BPE surgery (`abacus_ready=false`).

## Run

```powershell
python "experiments/Experiment 80 - Abacus Digit Embedding Probe/abacus_digit_probe.py" --mode smoke --device cpu
python "experiments/Experiment 80 - Abacus Digit Embedding Probe/abacus_digit_probe.py" --mode full --device cuda --resume
```

Decision-grade run (full frozen-200 + word heldout, seeds 1,2 trained
internally — the default 40-row limits are too small for the 5 pp promote bar):

```powershell
python "experiments/Experiment 80 - Abacus Digit Embedding Probe/abacus_digit_probe.py" --mode full --device cuda --resume --frozen-limit 200 --word-heldout-limit 200
```

Full mode defaults to 2000 steps from the Exp34.1 EQR-pretrain checkpoint
(`...eqrpretrain_plain2000_then_eqr..._steps10000/checkpoint_fp32.pt` — not an
SFT checkpoint). It trains
four arms: control/abacus for seeds 1 and 2. Use `--resume` for long runs; each
completed arm is written to `report_full.json`, and resume will refuse to skip
old arms if the run settings changed.

`--digit-order msd` is the default. `--digit-order lsd` runs the literature
variant where the units digit gets position 1.

## Results

Decision-grade result: `results_full_seed1.md` (seeds 1,2 from
`report_full.json`). Full CUDA run (4 arms, 2000 steps, frozen-200 +
word-heldout-200) on 2026-06-13, `abacus_ready=true` (real signal, not floor):

| seed | control frozen | abacus frozen | frozen Δ pp | add/sub Δ pp | invalid |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `0.025` | `0.020` | `-0.5` | `0.0` | `0.000` |
| 2 | `0.015` | `0.015` | `0.0` | `0.0` | `0.000` |

Smoke (floor) result: `results_smoke_seed1.md` — the `inconclusive_floor` case,
not used for the decision.

Superseded run logs: `results_*_codex.md` files in this folder are historical
run logs from before the per-row metric / floor-guard fixes; do not use them
for promote/kill decisions. `results_full_codex.md` in particular is the
all-zeros floor run the remediation plan flagged as inconclusive.

## Verdict

Kill / do not promote. The decision-grade run shows the abacus arm at or below
control on both seeds (frozen Δ −0.5 pp / 0.0 pp, add/sub Δ 0.0 pp), far under
the +5 pp promote bar and within the ±2.03 pp noise floor. `abacus_ready=true`
and invalid 0% rule out the tokenizer-surgery and `inconclusive_floor` escape
clauses — this is a genuine no-effect result. C1 per-digit-positional lane
closes at this scale.
