$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot

[System.Diagnostics.Process]::GetCurrentProcess().PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal

& rtk python -u "experiments/Experiment 33.6 - EqR Convergence Probe/eqr_convergence_probe.py" `
    --checkpoint "artifacts/phase0_eqr_full/h256_exp34_1_eqrpretrain_seed2_plain2000_then_eqr_d015_zl010_h246_bp4_steps10000_seed2/eqr_sft/checkpoint_fp32.pt" `
    --device cuda `
    --h-values "2,4,6" `
    --bp-steps 4 `
    --damping-lambda 0.15 `
    --noise-beta 0.01 `
    --ri-z-h-std 0.0 `
    --ri-z-l-std 0.10 `
    --generation-eval-limit 0 `
    --generation-max-new-tokens 64 `
    --residual-batches 64 `
    --output-dir "artifacts/phase0_eqr_convergence_probe/h256_exp34_1_seed2_bp4_residual_only_h246" `
    --append-md "experiments/Experiment 33.6 - EqR Convergence Probe/results_h256_exp34_1_seed2_bp4_residual_only_h246.md"
