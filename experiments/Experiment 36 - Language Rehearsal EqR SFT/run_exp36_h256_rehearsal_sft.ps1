$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot

$TokenDir = "datasets/exp35_mixed_language_arithmetic"
$TokenPath = "$TokenDir/tokens_flat.npy"
$ManifestPath = "$TokenDir/manifest.json"

if (-not (Test-Path $TokenPath) -or -not (Test-Path $ManifestPath)) {
    rtk python -u "experiments/Experiment 35 - Mixed Language Arithmetic Pretrain/prepare_exp35_mixed_tokens.py" `
        --target-tokens 8000000 `
        --output-dir $TokenDir
}

rtk python -u "experiments/Experiment 36 - Language Rehearsal EqR SFT/exp36_language_rehearsal_sft.py" `
    --device cuda `
    --seed 1 `
    --base-checkpoint "artifacts/phase0_exp35_mixed/h256_exp35_mixed50m_plain2000_eqr10000_seed1/pretrain/checkpoint_fp32.pt" `
    --tokens-path $TokenPath `
    --token-manifest $ManifestPath `
    --eqr-sft-steps 10000 `
    --eqr-sft-lr 1e-4 `
    --eqr-sft-bp-steps 4 `
    --sft-batch-size 4 `
    --sft-total-len 128 `
    --language-prompt-tokens 64 `
    --language-rehearsal-ratio 0.25 `
    --language-max-sequences 50000 `
    --language-eval-sequences 256 `
    --train-h-values "2,4,6" `
    --eval-h-values "2,4,6" `
    --damping-lambda 0.15 `
    --noise-beta 0.01 `
    --ri-z-h-std 0.0 `
    --ri-z-l-std 0.10 `
    --generation-eval-limit 200 `
    --generation-max-new-tokens 64 `
    --language-max-new-tokens 40 `
    --sft-log-interval 200 `
    --output-dir "artifacts/phase0_exp36_rehearsal/h256_exp35pretrain_rehearsal25_eqr10000_seed1" `
    --append-md "experiments/Experiment 36 - Language Rehearsal EqR SFT/results_h256_exp35pretrain_rehearsal25_eqr10000_seed1.md"
