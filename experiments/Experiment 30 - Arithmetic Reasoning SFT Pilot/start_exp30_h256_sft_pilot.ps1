$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$LogDir = Join-Path $RepoRoot "experiments/_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutLog = Join-Path $LogDir "exp30_h256_sft_pilot_$Stamp.log"
$ErrLog = Join-Path $LogDir "exp30_h256_sft_pilot_$Stamp.err.log"
$Runner = Join-Path $PSScriptRoot "run_exp30_h256_sft_pilot.ps1"

$Process = Start-Process `
    -FilePath "powershell" `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $Runner) `
    -WorkingDirectory $RepoRoot `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog `
    -WindowStyle Hidden `
    -PassThru

Write-Host "started_pid=$($Process.Id)"
Write-Host "stdout=$OutLog"
Write-Host "stderr=$ErrLog"
