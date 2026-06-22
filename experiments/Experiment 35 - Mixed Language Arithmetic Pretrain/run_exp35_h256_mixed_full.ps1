$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot

$TokenDir = "datasets/exp35_mixed_language_arithmetic"
$TokenPath = "$TokenDir/tokens_flat.npy"

if (-not (Test-Path $TokenPath)) {
    rtk python -u "experiments/Experiment 35 - Mixed Language Arithmetic Pretrain/prepare_exp35_mixed_tokens.py" `
        --target-tokens 8000000 `
        --output-dir $TokenDir
}

rtk python -u "experiments/Experiment 35 - Mixed Language Arithmetic Pretrain/exp35_mixed_pretrain_then_sft.py" `
    --device cuda `
    --seed 1 `
    --tokens-path $TokenPath `
    --token-manifest "$TokenDir/manifest.json" `
    --pretrain-steps 97656 `
    --export-calibration-steps 3000 `
    --plain-bridge-steps 2000 `
    --eqr-sft-steps 10000 `
    --hidden-size 256 `
    --n-layers 4 `
    --num-heads 4 `
    --expansion 2.0 `
    --numseqs 4 `
    --prefix-len 64 `
    --causal-len 64 `
    --pretrain-lr 3e-4 `
    --export-calibration-lr 1e-4 `
    --plain-bridge-lr 1e-4 `
    --eqr-sft-lr 1e-4 `
    --bp-min-steps 4 `
    --bp-max-steps 4 `
    --plain-bridge-bp-steps 2 `
    --eqr-sft-bp-steps 4 `
    --train-h-values "2,4,6" `
    --eval-h-values "2,4,6" `
    --damping-lambda 0.15 `
    --noise-beta 0.01 `
    --ri-z-h-std 0.0 `
    --ri-z-l-std 0.10 `
    --generation-eval-limit 200 `
    --generation-max-new-tokens 64 `
    --language-max-new-tokens 40 `
    --log-interval 2000 `
    --sft-log-interval 200 `
    --output-dir "artifacts/phase0_exp35_mixed/h256_exp35_mixed50m_plain2000_eqr10000_seed1" `
    --append-md "experiments/Experiment 35 - Mixed Language Arithmetic Pretrain/results_h256_exp35_mixed50m_plain2000_eqr10000_seed1.md"
