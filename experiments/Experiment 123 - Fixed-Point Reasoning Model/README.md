# Experiment 123 - Fixed-Point Reasoning Model

Text pretraining check for the FPRM method in [arXiv:2606.18206v1](https://arxiv.org/abs/2606.18206v1).

The vocab uses the Exp34.1 `mixed_top512_tequila` recipe: one tied input/output
weight, top 512 rows dense, remaining rows ternary. The FPRM body uses Tequila
ternary weights with group 128 and threshold 0.5. Attention stays causal for
text. Pretraining and SFT use the same adaptive fixed-point loop. There is no
HRM H-depth schedule or H sweep. `max_iters` is only the safety cap; residual
halting picks the actual iteration count.

Deep-supervision windows keep the exact full-vocabulary CE objective. Their
projection/loss is activation-checkpointed one window at a time, so peak VRAM
does not scale with a stacked `[windows, tokens, vocab]` logits tensor.

Full pretraining writes `pretrain_progress.pt` atomically every 500 steps and
auto-resumes from it. `pretrain_progress.json` is the live step/loss summary.
The completed pretrain checkpoint is saved before SFT; the combined checkpoint
is saved after SFT.

## Decision Rule

Promote if two seeds beat the matched Exp34.1 text-loss baseline by more than
`0.0203` without worse strict frozen arithmetic, while packed size and peak VRAM
stay inside the Phase 0 envelope.

Kill if either seed leaks future tokens, fails fixed-point halting, exceeds the
VRAM envelope, or misses the loss gate.

## Check

```powershell
rtk python -m pytest tests/test_exp123_fprm.py -q
rtk python -m experiments.discipline preflight-readme "experiments/Experiment 123 - Fixed-Point Reasoning Model/README.md"
```

## Smoke run

```powershell
& "experiments/Experiment 123 - Fixed-Point Reasoning Model/run_fprm_full_pretrain_then_sft.ps1"
```

## Fair Exp34.1 curriculum check

Uses the saved FPRM pretrain checkpoint, then the same winning Exp34.1 SFT
curriculum: broad v1 for 2,000 steps, frozen-like v2 for 10,000 steps, then
strict frozen200 generation.

```powershell
rtk python -u "experiments/Experiment 123 - Fixed-Point Reasoning Model/fprm_curriculum_sft.py" --device cuda
```

Each stage writes `progress.json` plus an atomic resumable checkpoint every 500
steps. The v1 and final v2 model checkpoints are saved separately.

## Full Adam8bit + AMP rerun

Matched h256 Exp123 rerun: 50,000-step text pretrain, direct v2 SFT, then the
fair v1 2,000 -> v2 10,000 curriculum. Architecture, data, and step budgets stay
fixed. Only the optimizer changes to bitsandbytes Adam8bit and training forwards
use CUDA bf16 autocast. Pretrain and both curriculum stages remain resumable.

```powershell
& "experiments/Experiment 123 - Fixed-Point Reasoning Model/run_fprm_adam8_amp_full.ps1"
```
