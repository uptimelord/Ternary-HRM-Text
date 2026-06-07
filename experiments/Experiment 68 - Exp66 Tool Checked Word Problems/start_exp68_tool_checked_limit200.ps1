$ErrorActionPreference = "Stop"

$script = Join-Path $PSScriptRoot "run_exp68_tool_checked_limit200.ps1"
$stdout = Join-Path $PSScriptRoot "exp68_tool_checked_limit200.out.log"
$stderr = Join-Path $PSScriptRoot "exp68_tool_checked_limit200.err.log"
$arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$script`""

Remove-Item -LiteralPath $stdout, $stderr -ErrorAction SilentlyContinue

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
