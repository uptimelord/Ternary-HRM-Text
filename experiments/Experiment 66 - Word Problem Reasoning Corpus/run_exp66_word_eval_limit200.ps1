$ErrorActionPreference = "Stop"

Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))

rtk python -u "experiments/Experiment 66 - Word Problem Reasoning Corpus/eval_exp66_word_reasoning.py" `
    --limit 200 `
    --h-values 4 `
    --device auto `
    --out "experiments/Experiment 66 - Word Problem Reasoning Corpus/results_exp66_word_sft2000_strict_eval_limit200.json"

if ($LASTEXITCODE -ne 0) {
    throw "Exp66 strict eval limit200 failed with exit code $LASTEXITCODE"
}
