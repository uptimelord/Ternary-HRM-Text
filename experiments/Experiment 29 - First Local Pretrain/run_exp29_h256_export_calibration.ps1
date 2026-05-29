$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot

[System.Diagnostics.Process]::GetCurrentProcess().PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal

& rtk python -m experiments.discipline preflight-readme `
    "experiments/Experiment 29 - First Local Pretrain/README.md"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& rtk python -u "experiments/Experiment 29 - First Local Pretrain/first_local_pretrain.py" `
    --resume-fp32-checkpoint "artifacts/phase0_first_pretrain/h256_steps50000_seed1/checkpoint_fp32.pt" `
    --steps 0 `
    --export-calibration-steps 3000 `
    --export-calibration-lr 1e-4 `
    --warmup-steps 0 `
    --seed 1 `
    --hidden-size 256 `
    --n-layers 4 `
    --num-heads 4 `
    --numseqs 4 `
    --prefix-len 64 `
    --causal-len 64 `
    --eval-batches 8 `
    --device cuda `
    --frozen-eval-limit 200 `
    --generation-eval-limit 200 `
    --output-dir "artifacts/phase0_first_pretrain/h256_steps50000_seed1_exportcalib3000" `
    --append-md "experiments/Experiment 29 - First Local Pretrain/results_h256_steps50000_seed1_exportcalib3000.md"
