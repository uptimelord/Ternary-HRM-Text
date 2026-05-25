$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot

[System.Diagnostics.Process]::GetCurrentProcess().PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal

& rtk python -m experiments.discipline preflight-readme `
    "experiments/Experiment 25 - Stacked Two Bit Compression/README.md"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& rtk python -u "experiments/Experiment 25 - Stacked Two Bit Compression/stacked_twobit_compression.py" `
    --steps 500 `
    --warmup-steps 2 `
    --seeds 1 `
    --hidden-size 128 `
    --variants combo_baseline,combo_2bit_attention_gqkv,combo_2bit_attention_o,combo_2bit_attention `
    --device cuda `
    --append-md "experiments/Experiment 25 - Stacked Two Bit Compression/results_500_seed1_h128.md"
