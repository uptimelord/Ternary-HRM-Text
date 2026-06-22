$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot

rtk python -u "scripts/generate_deepseek_custom_dataset.py" `
    --output-dir "datasets/deepseek_custom_rehearsal/v1" `
    --train-count 100000 `
    --valid-count 4000 `
    --model "deepseek-v4-flash" `
    --deepseek-batch-size 40 `
    --max-tokens 12000 `
    --temperature 0.9 `
    --sleep-seconds 0.15
