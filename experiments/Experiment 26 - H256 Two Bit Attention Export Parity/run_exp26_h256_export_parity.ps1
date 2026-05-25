$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot

[System.Diagnostics.Process]::GetCurrentProcess().PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal

& rtk python -m experiments.discipline preflight-readme `
    "experiments/Experiment 26 - H256 Two Bit Attention Export Parity/README.md"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& rtk python -u "experiments/Experiment 26 - H256 Two Bit Attention Export Parity/h256_twobit_export_parity.py" `
    --steps 500 `
    --warmup-steps 2 `
    --seed 1 `
    --hidden-size 256 `
    --device cuda `
    --append-md "experiments/Experiment 26 - H256 Two Bit Attention Export Parity/results_500_seed1_h256.md"
