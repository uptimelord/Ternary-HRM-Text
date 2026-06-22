$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$RunScript = Join-Path $PSScriptRoot "run_fprm_adam8_amp_full.ps1"
$LogDir = Join-Path $RepoRoot "artifacts/phase0_fprm_exp123/h256_fprm_adam8_amp_max20_bp4_steps50000_sft10000_seed1"
$Stdout = Join-Path $LogDir "full_run.log"
$Stderr = Join-Path $LogDir "full_run.err.log"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Process = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$RunScript`"") `
    -WorkingDirectory $RepoRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $Stdout `
    -RedirectStandardError $Stderr `
    -PassThru

Write-Output "pid=$($Process.Id)"
Write-Output "stdout=$Stdout"
Write-Output "stderr=$Stderr"
