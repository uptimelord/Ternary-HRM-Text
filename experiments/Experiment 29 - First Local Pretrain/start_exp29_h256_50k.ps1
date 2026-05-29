$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot

$LogDir = Join-Path $RepoRoot "experiments\_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Log = Join-Path $LogDir "exp29_h256_50k_$Stamp.log"
$ErrLog = Join-Path $LogDir "exp29_h256_50k_$Stamp.err.log"
$Result = "experiments/Experiment 29 - First Local Pretrain/results_h256_steps50000_seed1.md"
$Artifacts = "artifacts/phase0_first_pretrain/h256_steps50000_seed1"
$Runner = Join-Path $PSScriptRoot "run_exp29_h256_50k.ps1"
$QuotedRunner = '"' + $Runner + '"'

$Process = Start-Process "powershell" `
    -WindowStyle Hidden `
    -PassThru `
    -WorkingDirectory $RepoRoot `
    -RedirectStandardOutput $Log `
    -RedirectStandardError $ErrLog `
    -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File $QuotedRunner"

Start-Sleep -Seconds 1
try {
    $Process.PriorityClass = "BelowNormal"
} catch {
    Write-Host "priority=unchanged"
}

Write-Host "pid=$($Process.Id)"
Write-Host "log=$Log"
Write-Host "err=$ErrLog"
Write-Host "result=$Result"
Write-Host "artifacts=$Artifacts"
