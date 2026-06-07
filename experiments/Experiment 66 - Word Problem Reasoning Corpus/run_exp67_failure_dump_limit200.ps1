$ErrorActionPreference = "Stop"

Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))

rtk python -u "experiments/Experiment 66 - Word Problem Reasoning Corpus/eval_exp66_word_reasoning.py" `
    --limit 200 `
    --h-values 4 `
    --device auto `
    --out "experiments/Experiment 66 - Word Problem Reasoning Corpus/results_exp67_failure_dump_limit200.json" `
    --save-generations "experiments/Experiment 66 - Word Problem Reasoning Corpus/generations_exp67_failure_dump_limit200.jsonl"

if ($LASTEXITCODE -ne 0) {
    throw "Exp67 failure dump limit200 failed with exit code $LASTEXITCODE"
}
