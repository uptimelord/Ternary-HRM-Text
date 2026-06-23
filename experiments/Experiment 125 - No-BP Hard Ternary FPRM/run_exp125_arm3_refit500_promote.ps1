$ErrorActionPreference = "Stop"

# Exp125 ARM 3 PROMOTE: online-refined DFA (periodic bounded-BP refit), 5000 steps,
# both seeds. Headline arm-3 result. Builds on arm-2 (bp-warmup-seeded DFA): every
# K=500 no-BP steps, re-anchor the 19 fitted matrices to the current weights via a
# bounded 20-step BP mini-batch (no optimizer step), then resume no-BP. Removes
# the arm-2 staleness; narrows the BP gap 2.58 -> 1.47 nats while keeping every
# no-BP invariant (no autograd at training time, no optimizer state, steady-state
# 478 MiB). See README "Arm 3".
# Quantizer untouched (group 32, threshold 0.25, mean-abs).

# --- seed 1 ---
rtk python -u "experiments/Experiment 125 - No-BP Hard Ternary FPRM/exp125_nobp_hard.py" `
  --train-rule nobp-dfa-full-hard `
  --device cuda --seed 1 --numseqs 1 `
  --hidden-size 256 --n-layers 4 --num-heads 4 --max-iters 20 `
  --prefix-len 64 --causal-len 64 --vocab-size 65536 --vocab-chunk-size 2048 `
  --pretrain-steps 5000 --warmup-steps 0 --export-calibration-steps 0 --sft-steps 0 `
  --nobp-head-lr 0.3 --nobp-core-lr 0.3 --nobp-beta 0.03 --nobp-residual-lambda 0.003 `
  --nobp-feedback-mode bp-warmup --nobp-warmup-steps 50 --nobp-warmup-ridge 1e-3 `
  --nobp-refit-interval 500 --nobp-refit-steps 20 --nobp-refit-ridge 1e-3 `
  --checkpoint-interval 500 --log-interval 500 --no-resume `
  --output-dir "artifacts/phase0_fprm_exp125/dfafull_h256_steps5000_h03_cr03_bpwarm50_refit500_seed1" `
  --append-md "experiments/Experiment 125 - No-BP Hard Ternary FPRM/results_dfafull_h256_steps5000_h03_cr03_bpwarm50_refit500_seed1.md"

# --- seed 2 ---
rtk python -u "experiments/Experiment 125 - No-BP Hard Ternary FPRM/exp125_nobp_hard.py" `
  --train-rule nobp-dfa-full-hard `
  --device cuda --seed 2 --numseqs 1 `
  --hidden-size 256 --n-layers 4 --num-heads 4 --max-iters 20 `
  --prefix-len 64 --causal-len 64 --vocab-size 65536 --vocab-chunk-size 2048 `
  --pretrain-steps 5000 --warmup-steps 0 --export-calibration-steps 0 --sft-steps 0 `
  --nobp-head-lr 0.3 --nobp-core-lr 0.3 --nobp-beta 0.03 --nobp-residual-lambda 0.003 `
  --nobp-feedback-mode bp-warmup --nobp-warmup-steps 50 --nobp-warmup-ridge 1e-3 `
  --nobp-refit-interval 500 --nobp-refit-steps 20 --nobp-refit-ridge 1e-3 `
  --checkpoint-interval 500 --log-interval 500 --no-resume `
  --output-dir "artifacts/phase0_fprm_exp125/dfafull_h256_steps5000_h03_cr03_bpwarm50_refit500_seed2" `
  --append-md "experiments/Experiment 125 - No-BP Hard Ternary FPRM/results_dfafull_h256_steps5000_h03_cr03_bpwarm50_refit500_seed2.md"
