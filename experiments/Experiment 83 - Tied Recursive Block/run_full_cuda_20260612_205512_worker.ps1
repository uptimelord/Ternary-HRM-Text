$ErrorActionPreference = "Stop"
Set-Location -LiteralPath 'C:\Users\Dos\Documents\GRAM\BitNet-HRM'
try {
    & python 'experiments\Experiment 83 - Tied Recursive Block\tied_recursive_block.py' --mode full --device cuda --output-dir 'experiments\Experiment 83 - Tied Recursive Block' --results-md 'experiments\Experiment 83 - Tied Recursive Block\results_full_cuda_20260612_205512.md' 1> 'experiments\Experiment 83 - Tied Recursive Block\run_full_cuda_20260612_205512.log' 2> 'experiments\Experiment 83 - Tied Recursive Block\run_full_cuda_20260612_205512.err'
    $exitCode = $LASTEXITCODE
} catch {
    ($_ | Out-String) | Set-Content -LiteralPath 'experiments\Experiment 83 - Tied Recursive Block\run_full_cuda_20260612_205512.err' -Encoding UTF8
    $exitCode = 1
}
Set-Content -LiteralPath 'experiments\Experiment 83 - Tied Recursive Block\run_full_cuda_20260612_205512.exitcode' -Value $exitCode -Encoding ASCII
exit $exitCode
