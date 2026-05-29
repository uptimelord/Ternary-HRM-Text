$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot

[System.Diagnostics.Process]::GetCurrentProcess().PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal

& rtk python -m experiments.discipline preflight-readme `
    "experiments/Experiment 34 - EqR Full Pretrain Then SFT/README.md"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& rtk python -u "experiments/Experiment 34 - EqR Full Pretrain Then SFT/eqr_full_pretrain_then_sft.py" `
    --device cuda `
    --seed 1 `
    --pretrain-steps 50000 `
    --export-calibration-steps 3000 `
    --sft-steps 10000 `
    --warmup-steps 2 `
    --hidden-size 256 `
    --n-layers 4 `
    --num-heads 4 `
    --numseqs 4 `
    --prefix-len 64 `
    --causal-len 64 `
    --eval-batches 8 `
    --sft-eval-batches 32 `
    --pretrain-lr 3e-4 `
    --export-calibration-lr 1e-4 `
    --sft-lr 1e-4 `
    --bp-min-steps 4 `
    --bp-max-steps 4 `
    --sft-bp-steps 4 `
    --train-h-values "2,4,6" `
    --eval-h-values "2,4,6" `
    --damping-lambda 0.15 `
    --noise-beta 0.01 `
    --ri-z-h-std 0.0 `
    --ri-z-l-std 0.10 `
    --generation-eval-limit 50 `
    --generation-max-new-tokens 64 `
    --output-dir "artifacts/phase0_eqr_full/h256_exp34_eqr_d015_zl010_h246_bp4_steps50000_sft10000_seed1" `
    --append-md "experiments/Experiment 34 - EqR Full Pretrain Then SFT/results_h256_exp34_eqr_d015_zl010_h246_bp4_steps50000_sft10000_seed1.md"
