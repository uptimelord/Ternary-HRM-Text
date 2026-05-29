$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$LogDir = Join-Path $RepoRoot "experiments/_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutLog = Join-Path $LogDir "exp33_5_from_exp30pilot_d015_zl010_h246_bp4_steps10000_nogen_$Stamp.log"
$ErrLog = Join-Path $LogDir "exp33_5_from_exp30pilot_d015_zl010_h246_bp4_steps10000_nogen_$Stamp.err.log"
$Runner = Join-Path $PSScriptRoot "run_exp33_5_from_exp30pilot_d015_zl010_h246_bp4_steps10000_nogen.ps1"

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
