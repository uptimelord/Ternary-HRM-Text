$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot

[System.Diagnostics.Process]::GetCurrentProcess().PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal

& rtk python -m experiments.discipline preflight-readme `
    "experiments/Experiment 31 - Frozen Like Arithmetic Curriculum/README.md"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& rtk python -u "experiments/Experiment 30 - Arithmetic Reasoning SFT Pilot/arithmetic_sft_pilot.py" `
    --base-checkpoint "artifacts/phase0_arithmetic_sft_pilot/h256_steps2000_seed1/checkpoint_fp32.pt" `
    --train-jsonl "data/synthetic_arithmetic_reasoning/v2_frozen_like/train.jsonl" `
    --valid-jsonl "data/synthetic_arithmetic_reasoning/v2_frozen_like/valid.jsonl" `
    --steps 2000 `
    --seed 1 `
    --batch-size 4 `
    --total-len 128 `
    --lr 1e-4 `
    --bp-steps 2 `
    --eval-batches 32 `
    --device cuda `
    --generation-eval-limit 200 `
    --generation-max-new-tokens 64 `
    --output-dir "artifacts/phase0_arithmetic_sft_v2/h256_exp30_plus_v2_steps2000_seed1" `
    --append-md "experiments/Experiment 31 - Frozen Like Arithmetic Curriculum/results_h256_exp30_plus_v2_steps2000_seed1.md"
