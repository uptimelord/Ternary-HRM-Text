$ErrorActionPreference = "Stop"

Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))

rtk python -u "experiments/Experiment 68 - Exp66 Tool Checked Word Problems/exp66_tool_checked_word_problems.py" `
    --limit 200 `
    --h 4 `
    --device auto `
    --out "experiments/Experiment 68 - Exp66 Tool Checked Word Problems/results_exp68_tool_checked_limit200.json" `
    --records-out "experiments/Experiment 68 - Exp66 Tool Checked Word Problems/records_exp68_tool_checked_limit200.jsonl"

if ($LASTEXITCODE -ne 0) {
    throw "Exp68 tool-checked limit200 failed with exit code $LASTEXITCODE"
}
