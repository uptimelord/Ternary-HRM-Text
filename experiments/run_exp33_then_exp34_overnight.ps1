$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

& rtk powershell -NoProfile -ExecutionPolicy Bypass -File `
    "experiments/Experiment 33 - EqR Lite Recurrence Stability/run_exp33_h256_eqr_lite.ps1"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& rtk powershell -NoProfile -ExecutionPolicy Bypass -File `
    "experiments/Experiment 34 - EqR Full Pretrain Then SFT/run_exp34_h256_eqr_full.ps1"
