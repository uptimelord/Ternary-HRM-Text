$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot

[System.Diagnostics.Process]::GetCurrentProcess().PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal

& rtk python -m experiments.discipline preflight-readme `
    "experiments/Experiment 24 - Two Bit Body Sensitivity/README.md"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& rtk python -u "experiments/Experiment 24 - Two Bit Body Sensitivity/twobit_body_sensitivity.py" `
    --steps 500 `
    --warmup-steps 2 `
    --seeds 1 `
    --hidden-size 128 `
    --variants dense,both_mlp_gate_up,H_mlp_gate_up,L_mlp_gate_up,both_mlp_down,both_attention_o,both_attention_gqkv `
    --device cuda `
    --append-md "experiments/Experiment 24 - Two Bit Body Sensitivity/results_500_seed1_h128.md"
