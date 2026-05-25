$ErrorActionPreference = "Stop"

$Runner = Join-Path $PSScriptRoot "run_5000_confirmations.ps1"
$LogDir = Join-Path (Split-Path -Parent $PSScriptRoot) "experiments\_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Process = Start-Process powershell `
    -WindowStyle Hidden `
    -PassThru `
    -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $Runner
    )

Start-Sleep -Seconds 1
try {
    $Process.PriorityClass = "BelowNormal"
} catch {
    Write-Host "priority=unchanged"
}

Write-Host "pid=$($Process.Id)"
Write-Host "runner=$Runner"
Write-Host "logs=$LogDir"
Write-Host "results_vocab=experiments/Experiment 19 - Long Training Data Scaling/results_5000_seeds23.md"
Write-Host "results_body=experiments/Experiment 16 - Tequila Dynamic Bias/results_5000_body_seeds23.md"
