$ErrorActionPreference = "Stop"

# Exp125 spsa-hard control, 1000 steps, seed 1.
# Purpose (preregistered in README): rule out that the no-BP loss drop is a
# perturbation artifact of the toy 1-sequence setup. spsa-hard estimates the
# tied-vocab gradient from a SINGLE random +/-1 direction per step (2 forwards,
# no exact chunked-CE gradient, no body DFA feedback) and updates along it.
# If spsa ALSO improves eval loss > 0.0203, the no-BP win is devalued (any head
# update works on this toy setup); if spsa fails, the exact-CE head gradient is
# shown to do the work and the DFA Promote is strengthened.
# Fair step size: head_lr 0.3 matches the DFA run so spsa gets a comparable
# update magnitude (update_clip 1.0 caps the crude scalar estimate). epsilon
# 1e-3 is the README default. 1 seed is enough for a control kill; a second
# seed only if it surprisingly passes.

rtk python -u "experiments/Experiment 125 - No-BP Hard Ternary FPRM/exp125_nobp_hard.py" `
  --train-rule spsa-hard `
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
  --pretrain-steps 1000 `
  --warmup-steps 0 `
  --export-calibration-steps 0 `
  --sft-steps 0 `
  --nobp-head-lr 0.3 `
  --spsa-epsilon 1e-3 `
  --nobp-beta 0.03 `
  --nobp-residual-lambda 0.003 `
  --checkpoint-interval 100 `
  --log-interval 20 `
  --no-resume `
  --output-dir "artifacts/phase0_fprm_exp125/spsa_h256_steps1000_h03_seed1" `
  --append-md "experiments/Experiment 125 - No-BP Hard Ternary FPRM/results_spsa_h256_steps1000_h03_seed1.md"
