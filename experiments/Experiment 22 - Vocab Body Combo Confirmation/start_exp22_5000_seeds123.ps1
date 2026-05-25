$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot

$LogDir = Join-Path $RepoRoot "experiments\_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Log = Join-Path $LogDir "exp22_vocab_body_combo_5000_$Stamp.log"
$ErrLog = Join-Path $LogDir "exp22_vocab_body_combo_5000_$Stamp.err.log"
$Result = "experiments/Experiment 22 - Vocab Body Combo Confirmation/results_5000_seeds123.md"
$Runner = Join-Path $PSScriptRoot "run_exp22_5000_seeds123.ps1"
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
