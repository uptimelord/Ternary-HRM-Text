$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot

[System.Diagnostics.Process]::GetCurrentProcess().PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal

& rtk python -m experiments.discipline preflight-readme `
    "experiments/Experiment 22 - Vocab Body Combo Confirmation/README.md"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& rtk python -u "experiments/Experiment 22 - Vocab Body Combo Confirmation/vocab_body_combo.py" `
    --steps 5000 `
    --warmup-steps 2 `
    --seeds 1,2,3 `
    --device cuda `
    --append-md "experiments/Experiment 22 - Vocab Body Combo Confirmation/results_5000_seeds123.md"
