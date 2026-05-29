$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$LogDir = Join-Path $RepoRoot "experiments/_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutLog = Join-Path $LogDir "exp34_1_eval200_h246_$Stamp.log"
$ErrLog = Join-Path $LogDir "exp34_1_eval200_h246_$Stamp.err.log"
$Runner = Join-Path $PSScriptRoot "run_exp34_1_eval200_h246.ps1"

$Process = Start-Process `
    -FilePath "rtk" `
    -ArgumentList @("powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ('"{0}"' -f $Runner)) `
    -WorkingDirectory $RepoRoot `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog `
    -WindowStyle Hidden `
    -PassThru

Write-Host "started_pid=$($Process.Id)"
Write-Host "stdout=$OutLog"
Write-Host "stderr=$ErrLog"
