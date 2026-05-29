$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$LogDir = Join-Path $RepoRoot "experiments\_logs"
$Runner = Join-Path $PSScriptRoot "run_exp32_recurrence_sweep.ps1"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$StdoutLog = Join-Path $LogDir "exp32_recurrence_sweep_$Stamp.log"
$StderrLog = Join-Path $LogDir "exp32_recurrence_sweep_$Stamp.err.log"
$StatusLog = Join-Path $LogDir "exp32_recurrence_sweep_$Stamp.status.txt"

Set-Location $RepoRoot

"started_at=$(Get-Date -Format o)" | Set-Content -Path $StatusLog -Encoding ascii
"runner=$Runner" | Add-Content -Path $StatusLog -Encoding ascii
"stdout_log=$StdoutLog" | Add-Content -Path $StatusLog -Encoding ascii
"stderr_log=$StderrLog" | Add-Content -Path $StatusLog -Encoding ascii

& powershell -NoProfile -ExecutionPolicy Bypass -File $Runner 1>> $StdoutLog 2>> $StderrLog
$ExitCode = $LASTEXITCODE

"completed_at=$(Get-Date -Format o)" | Add-Content -Path $StatusLog -Encoding ascii
"exit_code=$ExitCode" | Add-Content -Path $StatusLog -Encoding ascii

shutdown.exe /s /t 60 /c "Exp32 recurrence sweep finished with exit code $ExitCode"
exit $ExitCode
