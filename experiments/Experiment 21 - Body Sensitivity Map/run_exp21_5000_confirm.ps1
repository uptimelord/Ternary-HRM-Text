$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot

[System.Diagnostics.Process]::GetCurrentProcess().PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal

& rtk python -u "experiments/Experiment 21 - Body Sensitivity Map/body_sensitivity_map.py" `
    --steps 5000 `
    --warmup-steps 2 `
    --seeds 2,3 `
    --variants dense,L_mlp_gate_up,both_mlp_gate_up `
    --device cuda `
    --append-md "experiments/Experiment 21 - Body Sensitivity Map/results_5000_confirm_seeds23.md"
