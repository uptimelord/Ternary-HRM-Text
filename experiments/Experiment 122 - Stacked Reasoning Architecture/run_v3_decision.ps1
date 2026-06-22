$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Runner = Join-Path $PSScriptRoot "runner.py"
Set-Location $RepoRoot

$CommonArgs = @(
    $Runner,
    "--mode", "decision",
    "--device", "cuda",
    "--train-limit", "2000",
    "--eval-limit", "200",
    "--steps", "2000",
    "--batch-size", "4",
    "--lr", "0.0003",
    "--max-length", "128",
    "--max-prompt-tokens", "32",
    "--max-new-tokens", "64",
    "--d-model", "128",
    "--factor-dim", "8",
    "--layers", "2",
    "--d-state", "16",
    "--kan-basis", "6",
    "--sdm-address-bits", "64",
    "--sdm-locations", "512",
    "--sdm-k-active", "32",
    "--sdm-seed-rows", "2000",
    "--log-interval", "100"
)

foreach ($Seed in 1, 2) {
    $OutputDir = Join-Path $RepoRoot "artifacts/exp122_stacked_reasoning_v3_seed$Seed"
    Write-Output "START seed=$Seed output=$OutputDir"
    & rtk python @CommonArgs --seed $Seed --output-dir $OutputDir
    if ($LASTEXITCODE -ne 0) {
        throw "Exp122 v3 seed $Seed failed with exit code $LASTEXITCODE"
    }
    Write-Output "DONE seed=$Seed"
}
