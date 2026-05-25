$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot

[System.Diagnostics.Process]::GetCurrentProcess().PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal

& rtk python -m experiments.discipline preflight-readme `
    "experiments/Experiment 24 - Two Bit Body Sensitivity/README.md"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$Variants = "dense,both_attention_gqkv,both_attention_o,both_mlp_gate_up"
$HiddenSizes = @(128, 256)
$StepCounts = @(500, 2000)

foreach ($Hidden in $HiddenSizes) {
    foreach ($Steps in $StepCounts) {
        $Result = "experiments/Experiment 24 - Two Bit Body Sensitivity/results_top_targets_h${Hidden}_steps${Steps}_seed1.md"
        & rtk python -u "experiments/Experiment 24 - Two Bit Body Sensitivity/twobit_body_sensitivity.py" `
            --steps $Steps `
            --warmup-steps 2 `
            --seeds 1 `
            --hidden-size $Hidden `
            --variants $Variants `
            --device cuda `
            --append-md $Result
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }
}
