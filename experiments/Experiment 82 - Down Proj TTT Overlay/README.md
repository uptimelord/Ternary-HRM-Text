# Experiment 82 - Down_Proj TTT Overlay

> Architecture brief **C3**: fast-weight overlay at MLP `down_proj` + masked next-token TTT (one-retry-then-kill).

## Decision Rule

Promote if strict logic hard improves **at least +5 pp** vs frozen baseline on 2 seeds, overlay wiped between tasks.

Kill if improvement is at or below noise on both seeds; lane closes.

## Run

```powershell
python "experiments/Experiment 82 - Down Proj TTT Overlay/down_proj_ttt_overlay.py" --mode smoke --device cpu
python "experiments/Experiment 82 - Down Proj TTT Overlay/down_proj_ttt_overlay.py" --mode full --device cuda --eval-domain logic --frozen-eval-limit 200 --seeds 1,2
```

Default checkpoint is the promoted Exp70 logic SFT checkpoint:

```text
artifacts/exp70_comparative_logic_sft/h256_30k_steps8000_seed1_term/checkpoint_fp32.pt
```

## Results

Decision-grade result: `results_full_seed1.md`.

Full CUDA run (`--eval-domain logic --frozen-eval-limit 200 --seeds 1,2`) on
2026-06-13:

| seed | baseline pass@1 | adapted pass@1 | delta pp |
| --- | ---: | ---: | ---: |
| 1 | `0.775` | `0.775` | `0.0` |
| 2 | `0.775` | `0.775` | `0.0` |

Smoke result: `results_smoke_seed1.md`.

Superseded run logs: `results_*_codex.md` files in this folder are historical
run logs from before checkpoint/eval-slice provenance was recorded (incl. the
unexplained frozen 0.700 baseline); do not use them for promote/kill decisions.

The markdown records the full checkpoint path and eval slice
`offset=<n> limit=<n>`. Use `--eval-offset` for chunked reruns.

## Verdict

Kill / do not promote. The decision-grade full run shows `0.0 pp` improvement on
both seeds, so the down-proj TTT overlay lane closes under the decision rule.
