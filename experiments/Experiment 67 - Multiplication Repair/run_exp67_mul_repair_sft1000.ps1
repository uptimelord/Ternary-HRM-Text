$ErrorActionPreference = "Stop"

Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))

rtk python -u "experiments/Experiment 30 - Arithmetic Reasoning SFT Pilot/arithmetic_sft_pilot.py" `
    --base-checkpoint "artifacts/phase0_exp66_word_reasoning/h256_exp34_1_word100k_sft2000_seed1/checkpoint_fp32.pt" `
    --train-jsonl "data/exp67_mul_repair_sft/v1/train.jsonl" `
    --valid-jsonl "data/exp67_mul_repair_sft/v1/valid.jsonl" `
    --output-dir "artifacts/phase0_exp67_mul_repair/h256_exp66_word_sft2000_mulrepair_sft1000_seed1" `
    --append-md "experiments/Experiment 67 - Multiplication Repair/results_h256_exp66_mulrepair_sft1000_seed1.md" `
    --steps 1000 `
    --batch-size 4 `
    --total-len 128 `
    --lr 1e-4 `
    --bp-steps 4 `
    --eval-batches 32 `
    --log-interval 100 `
    --max-prompt-tokens 64 `
    --max-response-tokens 96 `
    --generation-eval-limit 100 `
    --generation-max-new-tokens 64 `
    --device auto

if ($LASTEXITCODE -ne 0) {
    throw "Exp67 multiplication repair SFT failed with exit code $LASTEXITCODE"
}
