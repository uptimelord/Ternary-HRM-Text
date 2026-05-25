$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot

[System.Diagnostics.Process]::GetCurrentProcess().PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal

& rtk python -m experiments.discipline preflight-readme `
    "experiments/Experiment 23 - Combo Export Parity/README.md"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& rtk python -u "experiments/Experiment 23 - Combo Export Parity/combo_export_parity.py" `
    --steps 500 `
    --warmup-steps 2 `
    --seed 1 `
    --device cuda `
    --append-md "experiments/Experiment 23 - Combo Export Parity/results_500_seed1.md"
