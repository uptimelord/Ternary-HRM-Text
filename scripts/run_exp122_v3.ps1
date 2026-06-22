$ErrorActionPreference = "Stop"

& (Join-Path $PSScriptRoot "..\experiments\Experiment 122 - Stacked Reasoning Architecture\run_v3_decision.ps1")
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
