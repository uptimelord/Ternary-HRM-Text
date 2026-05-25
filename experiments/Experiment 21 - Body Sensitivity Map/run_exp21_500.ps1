$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot

& rtk python -u "experiments/Experiment 21 - Body Sensitivity Map/body_sensitivity_map.py" `
    --steps 500 `
    --warmup-steps 2 `
    --seeds 1 `
    --device cuda `
    --append-md "experiments/Experiment 21 - Body Sensitivity Map/results_500.md"
