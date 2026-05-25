$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot

[System.Diagnostics.Process]::GetCurrentProcess().PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal

& rtk python -m experiments.discipline preflight-readme `
    "experiments/Experiment 25 - Stacked Two Bit Compression/README.md"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$Variants = "combo_baseline,combo_2bit_attention_gqkv,combo_2bit_attention"
$HiddenSizes = @(128, 256)
$StepCounts = @(500, 2000)

foreach ($Hidden in $HiddenSizes) {
    foreach ($Steps in $StepCounts) {
        $Result = "experiments/Experiment 25 - Stacked Two Bit Compression/results_confirm_h${Hidden}_steps${Steps}_seed1.md"
        & rtk python -u "experiments/Experiment 25 - Stacked Two Bit Compression/stacked_twobit_compression.py" `
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
