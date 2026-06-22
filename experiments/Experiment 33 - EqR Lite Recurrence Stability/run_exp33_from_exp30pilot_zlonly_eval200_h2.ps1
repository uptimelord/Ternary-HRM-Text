$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot

[System.Diagnostics.Process]::GetCurrentProcess().PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal

& rtk python -u "experiments/Experiment 33 - EqR Lite Recurrence Stability/eqr_lite_recurrence_sft.py" `
    --device cuda `
    --base-checkpoint "artifacts/phase0_eqr_lite_recurrence/h256_exp33_from_exp30pilot_zlonly_steps2000_seed1/checkpoint_fp32.pt" `
    --train-jsonl "datasets/synthetic_arithmetic_reasoning/v2_frozen_like/train.jsonl" `
    --valid-jsonl "datasets/synthetic_arithmetic_reasoning/v2_frozen_like/valid.jsonl" `
    --steps 0 `
    --seed 1 `
    --batch-size 4 `
    --total-len 128 `
    --lr 1e-4 `
    --bp-steps 2 `
    --eval-batches 32 `
    --train-h-values "1,2,4,6" `
    --eval-h-values "2" `
    --damping-lambda 0.05 `
    --noise-beta 0.01 `
    --ri-z-h-std 0.0 `
    --ri-z-l-std 1.0 `
    --generation-eval-limit 200 `
    --generation-max-new-tokens 64 `
    --output-dir "artifacts/phase0_eqr_lite_recurrence/h256_exp33_from_exp30pilot_zlonly_eval200_h2" `
    --append-md "experiments/Experiment 33 - EqR Lite Recurrence Stability/results_h256_exp33_from_exp30pilot_zlonly_eval200_h2.md"
