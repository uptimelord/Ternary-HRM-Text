$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot

$MainOut = "artifacts/phase0_fprm_exp123/h256_fprm_adam8_amp_max20_bp4_steps50000_sft10000_seed1"
$CurriculumOut = "artifacts/phase0_fprm_exp123/h256_fprm_adam8_amp_v1_2000_v2_10000_seed1"
$MainResults = "experiments/Experiment 123 - Fixed-Point Reasoning Model/results_fprm_adam8_amp_h256_steps50000_sft10000_seed1.md"
$CurriculumResults = "experiments/Experiment 123 - Fixed-Point Reasoning Model/results_fprm_adam8_amp_v1_2000_v2_10000_seed1.md"

& rtk python -u "experiments/Experiment 123 - Fixed-Point Reasoning Model/fprm_full_pretrain_then_sft.py" `
    --device cuda `
    --seed 1 `
    --pretrain-steps 50000 `
    --export-calibration-steps 0 `
    --sft-steps 10000 `
    --sft-eval-batches 32 `
    --generation-eval-limit 200 `
    --pretrain-checkpoint-interval 500 `
    --hidden-size 256 `
    --n-layers 4 `
    --num-heads 4 `
    --max-iters 20 `
    --tau 0.1 `
    --optimizer adam8bit `
    --amp `
    --output-dir $MainOut `
    --append-md $MainResults

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& rtk python -u "experiments/Experiment 123 - Fixed-Point Reasoning Model/fprm_curriculum_sft.py" `
    --device cuda `
    --base-checkpoint "$MainOut/pretrain/checkpoint_fp32.pt" `
    --optimizer adam8bit `
    --amp `
    --output-dir $CurriculumOut `
    --append-md $CurriculumResults

exit $LASTEXITCODE
