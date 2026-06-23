$ErrorActionPreference = "Stop"

# Exp125 head-hard, 1000 steps, head LR 60, seed 1.
# Promote attempt: the 100-step sweep found lr=60 is the ONLY head LR where mean
# ternary flip rate enters the Promote band [0.0001, 0.01] (1.009e-4) while eval
# loss still improves (gap -1.45) and no NaN. lr>=100 diverges loss; lr<=30
# misses the flip floor. Flip rate saturates ~1.8e-4, so no comfortable mid-band
# LR exists -- this is a knife-edge candidate. Quantizer (group 32, threshold
# 0.25, mean-abs) is untouched.
# Measure-twice: seed 1 first. Run seed 2 only if seed 1 clears all gates.

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
  --pretrain-steps 1000 `
  --warmup-steps 0 `
  --export-calibration-steps 0 `
  --sft-steps 0 `
  --nobp-head-lr 60.0 `
  --nobp-beta 0.03 `
  --nobp-residual-lambda 0.003 `
  --checkpoint-interval 100 `
  --log-interval 20 `
  --no-resume `
  --output-dir "artifacts/phase0_fprm_exp125/head_hard_h256_steps1000_lr6e1_seed1" `
  --append-md "experiments/Experiment 125 - No-BP Hard Ternary FPRM/results_head_hard_h256_steps1000_lr6e1_seed1.md"
