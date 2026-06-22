$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot

[System.Diagnostics.Process]::GetCurrentProcess().PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal

& rtk python -m experiments.discipline preflight-readme `
    "experiments/Experiment 34 - EqR Full Pretrain Then SFT/README.md"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& rtk python -u "experiments/Experiment 30 - Arithmetic Reasoning SFT Pilot/arithmetic_sft_pilot.py" `
    --base-checkpoint "artifacts/phase0_eqr_full/h256_exp34_eqr_d015_zl010_h246_bp4_steps50000_seed2/pretrain/checkpoint_fp32.pt" `
    --train-jsonl "datasets/synthetic_arithmetic_reasoning/v1/train.jsonl" `
    --valid-jsonl "datasets/synthetic_arithmetic_reasoning/v1/valid.jsonl" `
    --steps 2000 `
    --seed 1 `
    --batch-size 4 `
    --total-len 128 `
    --lr 1e-4 `
    --bp-steps 2 `
    --eval-batches 32 `
    --device cuda `
    --generation-eval-limit 50 `
    --generation-max-new-tokens 64 `
    --output-dir "artifacts/phase0_eqr_full/h256_exp34_1_eqrpretrain_seed2_plain2000_sftseed1_then_eqr_d015_zl010_h246_bp4_steps10000_sftseed1/plain_sft" `
    --append-md "experiments/Experiment 34 - EqR Full Pretrain Then SFT/results_h256_exp34_1_seed2pretrain_sftseed1_plain_bridge_steps2000_seed1.md"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& rtk python -u "experiments/Experiment 33 - EqR Lite Recurrence Stability/eqr_lite_recurrence_sft.py" `
    --device cuda `
    --base-checkpoint "artifacts/phase0_eqr_full/h256_exp34_1_eqrpretrain_seed2_plain2000_sftseed1_then_eqr_d015_zl010_h246_bp4_steps10000_sftseed1/plain_sft/checkpoint_fp32.pt" `
    --train-jsonl "datasets/synthetic_arithmetic_reasoning/v2_frozen_like/train.jsonl" `
    --valid-jsonl "datasets/synthetic_arithmetic_reasoning/v2_frozen_like/valid.jsonl" `
    --steps 10000 `
    --seed 1 `
    --batch-size 4 `
    --total-len 128 `
    --lr 1e-4 `
    --bp-steps 4 `
    --eval-batches 32 `
    --train-h-values "2,4,6" `
    --eval-h-values "2,4,6" `
    --damping-lambda 0.15 `
    --noise-beta 0.01 `
    --ri-z-h-std 0.0 `
    --ri-z-l-std 0.10 `
    --generation-eval-limit 0 `
    --generation-max-new-tokens 64 `
    --output-dir "artifacts/phase0_eqr_full/h256_exp34_1_eqrpretrain_seed2_plain2000_sftseed1_then_eqr_d015_zl010_h246_bp4_steps10000_sftseed1/eqr_sft" `
    --append-md "experiments/Experiment 34 - EqR Full Pretrain Then SFT/results_h256_exp34_1_eqrpretrain_seed2_plain2000_sftseed1_then_eqr_d015_zl010_h246_bp4_steps10000_sftseed1.md"
