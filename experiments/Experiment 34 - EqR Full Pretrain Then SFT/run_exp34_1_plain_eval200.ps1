$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot

[System.Diagnostics.Process]::GetCurrentProcess().PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal

& rtk python -u "experiments/Experiment 30 - Arithmetic Reasoning SFT Pilot/arithmetic_sft_pilot.py" `
    --base-checkpoint "artifacts/phase0_eqr_full/h256_exp34_1_eqrpretrain_plain2000_then_eqr_d015_zl010_h246_bp4_steps10000_seed1/eqr_sft/checkpoint_fp32.pt" `
    --train-jsonl "data/synthetic_arithmetic_reasoning/v2_frozen_like/train.jsonl" `
    --valid-jsonl "data/synthetic_arithmetic_reasoning/v2_frozen_like/valid.jsonl" `
    --steps 0 `
    --seed 1 `
    --batch-size 4 `
    --total-len 128 `
    --lr 1e-4 `
    --bp-steps 4 `
    --eval-batches 32 `
    --device cuda `
    --generation-eval-limit 200 `
    --generation-max-new-tokens 64 `
    --output-dir "artifacts/phase0_eqr_full/h256_exp34_1_plain_eval200" `
    --append-md "experiments/Experiment 34 - EqR Full Pretrain Then SFT/results_h256_exp34_1_plain_eval200.md"
