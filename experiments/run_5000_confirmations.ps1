param(
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$LogDir = Join-Path $RepoRoot "experiments\_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$SummaryLog = Join-Path $LogDir "run_5000_confirmations_$Stamp.log"

function Run-Step {
    param(
        [string]$Name,
        [string[]]$PythonArgs
    )

    $StepLog = Join-Path $LogDir "$Name`_$Stamp.log"
    "[$(Get-Date -Format o)] START $Name" | Tee-Object -FilePath $SummaryLog -Append
    "log=$StepLog" | Tee-Object -FilePath $SummaryLog -Append

    & rtk python -u @PythonArgs 2>&1 | Tee-Object -FilePath $StepLog
    if ($LASTEXITCODE -ne 0) {
        "[$(Get-Date -Format o)] FAIL $Name exit=$LASTEXITCODE" | Tee-Object -FilePath $SummaryLog -Append
        throw "$Name failed with exit code $LASTEXITCODE"
    }

    "[$(Get-Date -Format o)] END $Name" | Tee-Object -FilePath $SummaryLog -Append
}

Run-Step "exp19c_vocab_5000_seeds23" @(
    "experiments/Experiment 19 - Long Training Data Scaling/long_training_data_scaling.py",
    "--steps", "5000",
    "--warmup-steps", "2",
    "--seeds", "2,3",
    "--device", $Device,
    "--append-md", "experiments/Experiment 19 - Long Training Data Scaling/results_5000_seeds23.md"
)

Run-Step "exp16_body_5000_seeds23" @(
    "experiments/Experiment 16 - Tequila Dynamic Bias/tequila_dynamic_bias.py",
    "--steps", "5000",
    "--warmup-steps", "2",
    "--seeds", "2,3",
    "--variants", "dense,mlp_gate_up_tequila",
    "--device", $Device,
    "--append-md", "experiments/Experiment 16 - Tequila Dynamic Bias/results_5000_body_seeds23.md"
)

"[$(Get-Date -Format o)] COMPLETE all steps" | Tee-Object -FilePath $SummaryLog -Append
