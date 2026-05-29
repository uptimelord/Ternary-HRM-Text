$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot

[System.Diagnostics.Process]::GetCurrentProcess().PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal

& rtk python -m experiments.discipline preflight-readme `
    "experiments/Experiment 32 - Recurrence Scaling Probe/README.md"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& rtk python -u "experiments/Experiment 32 - Recurrence Scaling Probe/recurrence_scaling_probe.py" `
    --checkpoint "artifacts/phase0_arithmetic_sft_v2/h256_exp30_plus_v2_steps2000_seed1/checkpoint_fp32.pt" `
    --device cuda `
    --h-values "1,2,4,6,10,20" `
    --l-cycles 3 `
    --generation-eval-limit 200 `
    --generation-max-new-tokens 64 `
    --bp-steps 2
