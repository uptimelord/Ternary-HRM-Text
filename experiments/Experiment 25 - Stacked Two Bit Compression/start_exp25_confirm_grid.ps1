$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot

$LogDir = Join-Path $RepoRoot "experiments\_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Log = Join-Path $LogDir "exp25_confirm_grid_$Stamp.log"
$ErrLog = Join-Path $LogDir "exp25_confirm_grid_$Stamp.err.log"
$ResultDir = "experiments/Experiment 25 - Stacked Two Bit Compression"
$Runner = Join-Path $PSScriptRoot "run_exp25_confirm_grid.ps1"
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
Write-Host "result_dir=$ResultDir"
