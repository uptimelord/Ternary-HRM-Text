$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot

[System.Diagnostics.Process]::GetCurrentProcess().PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal

# Note: Adjust --pretrain-steps and --sft-steps for full runs (e.g. 50000 and 10000)
& rtk python -u "experiments/Experiment 123 - Fixed-Point Reasoning Model/fprm_full_pretrain_then_sft.py" `
    --device cuda `
    --seed 1 `
    --pretrain-steps 100 `
    --export-calibration-steps 0 `
    --sft-steps 100 `
    --sft-eval-batches 1 `
    --generation-eval-limit 0 `
    --pretrain-checkpoint-interval 0 `
    --no-resume-pretrain `
    --hidden-size 256 `
    --n-layers 4 `
    --max-iters 20 `
    --tau 0.1

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
