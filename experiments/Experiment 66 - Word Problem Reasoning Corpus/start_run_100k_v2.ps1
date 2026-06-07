$ErrorActionPreference = "Stop"

$script = Join-Path $PSScriptRoot "run_100k_v2.ps1"
$stdout = Join-Path $PSScriptRoot "run_100k_v2.out.log"
$stderr = Join-Path $PSScriptRoot "run_100k_v2.err.log"
$arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$script`""

$p = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList $arguments `
    -WorkingDirectory $PSScriptRoot `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -WindowStyle Hidden `
    -PassThru

"pid=$($p.Id)"
"stdout=$stdout"
"stderr=$stderr"
