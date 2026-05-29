$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot

[System.Diagnostics.Process]::GetCurrentProcess().PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal

& rtk python -u "experiments/Experiment 33 - EqR Lite Recurrence Stability/eqr_lite_recurrence_sft.py" `
    --device cuda `
    --base-checkpoint "artifacts/phase0_arithmetic_sft_pilot/h256_steps2000_seed1/checkpoint_fp32.pt" `
    --train-jsonl "data/synthetic_arithmetic_reasoning/v2_frozen_like/train.jsonl" `
    --valid-jsonl "data/synthetic_arithmetic_reasoning/v2_frozen_like/valid.jsonl" `
    --steps 6000 `
    --seed 1 `
    --batch-size 4 `
    --total-len 128 `
    --lr 1e-4 `
    --bp-steps 2 `
    --eval-batches 32 `
    --train-h-values "2,4,6" `
    --eval-h-values "1,2,4,6" `
    --damping-lambda 0.15 `
    --noise-beta 0.01 `
    --ri-z-h-std 0.0 `
    --ri-z-l-std 0.10 `
    --generation-eval-limit 0 `
    --generation-max-new-tokens 64 `
    --output-dir "artifacts/phase0_eqr_lite_recurrence/h256_exp33_1_from_exp30pilot_d015_zl010_h246_steps6000_seed1" `
    --append-md "experiments/Experiment 33 - EqR Lite Recurrence Stability/results_h256_exp33_1_from_exp30pilot_d015_zl010_h246_steps6000_seed1.md"
