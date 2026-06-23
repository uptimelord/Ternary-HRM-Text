$ErrorActionPreference = "Stop"

# Exp125 head-hard flip-vs-LR sweep, 100 steps, seed 1.
# Purpose: find a head LR where mean ternary flip rate enters the Promote band
# [0.0001, 0.01] while eval loss still improves and no NaN. The loss gate and
# NaN guard prevent gaming the flip gate. Fresh dirs + --resume False so each
# arm is an independent deterministic init (no cross-contamination).

$lrs = @(60.0, 100.0, 300.0, 1000.0)
$tags = @("lr6e1", "lr1e2", "lr3e2", "lr1e3")
for ($i = 0; $i -lt $lrs.Length; $i++) {
    $lr = $lrs[$i]
    $tag = $tags[$i]
    rtk python -u "experiments/Experiment 125 - No-BP Hard Ternary FPRM/exp125_nobp_hard.py" `
      --train-rule nobp-head-hard `
      --device cuda `
      --numseqs 1 `
      --hidden-size 256 `
      --n-layers 4 `
      --num-heads 4 `
      --max-iters 20 `
      --prefix-len 64 `
      --causal-len 64 `
      --vocab-size 65536 `
      --vocab-chunk-size 2048 `
      --pretrain-steps 100 `
      --warmup-steps 0 `
      --export-calibration-steps 0 `
      --sft-steps 0 `
      --nobp-head-lr $lr `
      --nobp-beta 0.03 `
      --nobp-residual-lambda 0.003 `
      --checkpoint-interval 100 `
      --log-interval 20 `
      --no-resume `
      --output-dir "artifacts/phase0_fprm_exp125/head_hard_h256_steps100_${tag}_seed1" `
      --append-md "experiments/Experiment 125 - No-BP Hard Ternary FPRM/results_head_hard_h256_steps100_${tag}_seed1.md"
}
