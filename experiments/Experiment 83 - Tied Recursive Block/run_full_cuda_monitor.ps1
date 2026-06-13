param(
    [string]$Exp = "experiments\Experiment 83 - Tied Recursive Block"
)

$ErrorActionPreference = "Stop"

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runner = Join-Path $Exp "tied_recursive_block.py"
$log = Join-Path $Exp ("run_full_cuda_" + $stamp + ".log")
$err = Join-Path $Exp ("run_full_cuda_" + $stamp + ".err")
$vram = Join-Path $Exp ("vram_full_cuda_" + $stamp + ".csv")
$results = Join-Path $Exp ("results_full_cuda_" + $stamp + ".md")
$exitFile = Join-Path $Exp ("run_full_cuda_" + $stamp + ".exitcode")
$worker = Join-Path $Exp ("run_full_cuda_" + $stamp + "_worker.ps1")
$workerOut = Join-Path $Exp ("run_full_cuda_" + $stamp + "_worker.out")
$workerErr = Join-Path $Exp ("run_full_cuda_" + $stamp + "_worker.err")
$pidFile = Join-Path $Exp "run_full_cuda.pid"
$metaFile = Join-Path $Exp "run_full_cuda_meta.json"

$repo = (Get-Location).Path
$workerScript = @"
`$ErrorActionPreference = "Stop"
Set-Location -LiteralPath '$repo'
try {
    & python '$runner' --mode full --device cuda --output-dir '$Exp' --results-md '$results' 1> '$log' 2> '$err'
    `$exitCode = `$LASTEXITCODE
} catch {
    (`$_ | Out-String) | Set-Content -LiteralPath '$err' -Encoding UTF8
    `$exitCode = 1
}
Set-Content -LiteralPath '$exitFile' -Value `$exitCode -Encoding ASCII
exit `$exitCode
"@
Set-Content -LiteralPath $worker -Value $workerScript -Encoding UTF8
$workerAbs = (Resolve-Path -LiteralPath $worker).Path
$workerArg = '-NoProfile -ExecutionPolicy Bypass -File "' + $workerAbs + '"'

$process = Start-Process `
    -FilePath "powershell" `
    -ArgumentList $workerArg `
    -WorkingDirectory $repo `
    -RedirectStandardOutput $workerOut `
    -RedirectStandardError $workerErr `
    -PassThru `
    -WindowStyle Hidden

Set-Content -LiteralPath $pidFile -Value $process.Id -Encoding ASCII

$monitorCommand = @"
`$pidToWatch = $($process.Id)
`$vramPath = '$vram'
'timestamp,memory_used_mb' | Set-Content -LiteralPath `$vramPath -Encoding ASCII
while (Get-Process -Id `$pidToWatch -ErrorAction SilentlyContinue) {
    `$mem = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>`$null | Select-Object -First 1).Trim()
    if (`$mem) {
        ((Get-Date).ToString('o') + ',' + `$mem) | Add-Content -LiteralPath `$vramPath -Encoding ASCII
    }
    Start-Sleep -Seconds 1
}
`$mem = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>`$null | Select-Object -First 1).Trim()
if (`$mem) {
    ((Get-Date).ToString('o') + ',' + `$mem) | Add-Content -LiteralPath `$vramPath -Encoding ASCII
}
"@

$monitor = Start-Process `
    -FilePath "powershell" `
    -ArgumentList @("-NoProfile", "-Command", $monitorCommand) `
    -PassThru `
    -WindowStyle Hidden

$meta = [ordered]@{
    stamp = $stamp
    pid = $process.Id
    monitor_pid = $monitor.Id
    log = $log
    err = $err
    vram = $vram
    exitcode = $exitFile
    worker = $worker
    worker_out = $workerOut
    worker_err = $workerErr
    results_md = $results
    report_full = (Join-Path $Exp "report_full.json")
    report_json = (Join-Path $Exp "report.json")
    command = 'python "experiments/Experiment 83 - Tied Recursive Block/tied_recursive_block.py" --mode full --device cuda --output-dir "experiments/Experiment 83 - Tied Recursive Block" --results-md "' + $results + '"'
}

$meta | ConvertTo-Json | Set-Content -LiteralPath $metaFile -Encoding UTF8
$meta | ConvertTo-Json
