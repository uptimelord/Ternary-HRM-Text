$ErrorActionPreference = "Stop"

$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Resolve-Path (Join-Path $dir "..\..")
$out = Join-Path $dir "run_2000.out.log"
$err = Join-Path $dir "run_2000.err.log"
$run = Join-Path $dir "run_2000.ps1"

if (Test-Path $out) {
    Remove-Item $out
}
if (Test-Path $err) {
    Remove-Item $err
}

$args = "-NoProfile -ExecutionPolicy Bypass -File `"$run`""
$process = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList $args `
    -WorkingDirectory $repo `
    -RedirectStandardOutput $out `
    -RedirectStandardError $err `
    -PassThru `
    -WindowStyle Hidden

Write-Output $process.Id
