$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot

[System.Diagnostics.Process]::GetCurrentProcess().PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal

& rtk python -m experiments.discipline preflight-readme `
    "experiments/Experiment 33.6 - EqR Convergence Probe/README.md"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& rtk python -u "experiments/Experiment 33.6 - EqR Convergence Probe/eqr_convergence_probe.py" `
    --checkpoint "artifacts/phase0_eqr_lite_recurrence/h256_exp33_5_from_exp30pilot_d015_zl010_h246_bp4_steps10000_seed1/checkpoint_fp32.pt" `
    --device cuda `
    --h-values "1,2,3,4,5,6,8,10,12,16" `
    --bp-steps 4 `
    --damping-lambda 0.15 `
    --generation-eval-limit 200 `
    --generation-max-new-tokens 64 `
    --residual-batches 32
