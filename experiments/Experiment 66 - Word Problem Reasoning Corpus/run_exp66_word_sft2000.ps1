$ErrorActionPreference = "Stop"

Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))

rtk python -u "experiments/Experiment 30 - Arithmetic Reasoning SFT Pilot/arithmetic_sft_pilot.py" `
    --base-checkpoint "artifacts/phase0_eqr_full/h256_exp34_1_eqrpretrain_plain2000_then_eqr_d015_zl010_h246_bp4_steps10000_eval200_h246/checkpoint_fp32.pt" `
    --train-jsonl "datasets/exp66_word_reasoning_sft/v1/train.jsonl" `
    --valid-jsonl "datasets/exp66_word_reasoning_sft/v1/valid.jsonl" `
    --output-dir "artifacts/phase0_exp66_word_reasoning/h256_exp34_1_word100k_sft2000_seed1" `
    --append-md "experiments/Experiment 66 - Word Problem Reasoning Corpus/results_h256_exp34_1_word100k_sft2000_seed1.md" `
    --steps 2000 `
    --batch-size 4 `
    --total-len 128 `
    --lr 1e-4 `
    --bp-steps 4 `
    --eval-batches 32 `
    --log-interval 100 `
    --max-prompt-tokens 64 `
    --max-response-tokens 96 `
    --generation-eval-limit 200 `
    --generation-max-new-tokens 64 `
    --device auto

if ($LASTEXITCODE -ne 0) {
    throw "Exp66 word SFT failed with exit code $LASTEXITCODE"
}
