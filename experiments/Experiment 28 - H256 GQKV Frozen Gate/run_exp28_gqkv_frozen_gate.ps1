$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot

[System.Diagnostics.Process]::GetCurrentProcess().PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal

& rtk python -m experiments.discipline preflight-readme `
    "experiments/Experiment 28 - H256 GQKV Frozen Gate/README.md"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& rtk python -u "experiments/Experiment 28 - H256 GQKV Frozen Gate/h256_gqkv_frozen_gate.py" `
    --steps 2000 `
    --warmup-steps 2 `
    --seeds 1,2 `
    --hidden-size 256 `
    --variants combo_baseline,combo_2bit_attention_gqkv `
    --device cuda `
    --append-md "experiments/Experiment 28 - H256 GQKV Frozen Gate/results_2000_seeds12_h256.md"
