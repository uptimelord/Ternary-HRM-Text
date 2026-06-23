$ErrorActionPreference = "Stop"

# Exp125 PLATEAU TEST: 5000 steps, numseqs=1, h256 x 4 (one lever = steps, 5x the
# Promote run). Question: does no-BP flatten while BP keeps dropping?
# Pre-registered decision rule:
#   Plateau confirmed (-> learned-feedback research justified):
#     both no-BP seeds 5000-step eval >= 8.9 (barely moved from 1000-step mean
#     9.14) AND BP 5000-step eval < 5.5 (keeps dropping below 1000-step 6.21).
#   Still improving (-> current DFA viable with patience):
#     no-BP 5000-step mean <= 8.5 (improves > 0.6 over 9.14).
# Thresholds 15-30x the 0.0203 noise floor. Quantizer untouched.
# Arms: no-BP s1, no-BP s2, BP s1 (BP seed2 redundant: gap is 3+ nats, 150x
# noise -- a second seed cannot close it, would only burn GPU).

# --- no-BP seed 1, 5000 steps ---
rtk python -u "experiments/Experiment 125 - No-BP Hard Ternary FPRM/exp125_nobp_hard.py" `
  --train-rule nobp-dfa-full-hard `
  --device cuda --seed 1 --numseqs 1 `
  --hidden-size 256 --n-layers 4 --num-heads 4 --max-iters 20 `
  --prefix-len 64 --causal-len 64 --vocab-size 65536 --vocab-chunk-size 2048 `
  --pretrain-steps 5000 --warmup-steps 0 --export-calibration-steps 0 --sft-steps 0 `
  --nobp-head-lr 0.3 --nobp-core-lr 0.1 --nobp-beta 0.03 --nobp-residual-lambda 0.003 `
  --checkpoint-interval 500 --log-interval 500 --no-resume `
  --output-dir "artifacts/phase0_fprm_exp125/dfafull_h256_steps5000_h03_cr01_nseq1_seed1" `
  --append-md "experiments/Experiment 125 - No-BP Hard Ternary FPRM/results_dfafull_h256_steps5000_h03_cr01_nseq1_seed1.md"

# --- no-BP seed 2, 5000 steps ---
rtk python -u "experiments/Experiment 125 - No-BP Hard Ternary FPRM/exp125_nobp_hard.py" `
  --train-rule nobp-dfa-full-hard `
  --device cuda --seed 2 --numseqs 1 `
  --hidden-size 256 --n-layers 4 --num-heads 4 --max-iters 20 `
  --prefix-len 64 --causal-len 64 --vocab-size 65536 --vocab-chunk-size 2048 `
  --pretrain-steps 5000 --warmup-steps 0 --export-calibration-steps 0 --sft-steps 0 `
  --nobp-head-lr 0.3 --nobp-core-lr 0.1 --nobp-beta 0.03 --nobp-residual-lambda 0.003 `
  --checkpoint-interval 500 --log-interval 500 --no-resume `
  --output-dir "artifacts/phase0_fprm_exp125/dfafull_h256_steps5000_h03_cr01_nseq1_seed2" `
  --append-md "experiments/Experiment 125 - No-BP Hard Ternary FPRM/results_dfafull_h256_steps5000_h03_cr01_nseq1_seed2.md"

# --- BP control seed 1, 5000 steps (matched) ---
rtk python -u "experiments/Experiment 125 - No-BP Hard Ternary FPRM/exp125_nobp_hard.py" `
  --train-rule bp `
  --device cuda --seed 1 --numseqs 1 `
  --hidden-size 256 --n-layers 4 --num-heads 4 --max-iters 20 `
  --prefix-len 64 --causal-len 64 --vocab-size 65536 --vocab-chunk-size 2048 `
  --pretrain-steps 5000 --warmup-steps 0 --export-calibration-steps 0 --sft-steps 0 `
  --pretrain-lr 3e-4 --optimizer adamw --bp-steps 4 --dense-top-k 0 `
  --checkpoint-interval 500 --log-interval 500 --no-resume `
  --output-dir "artifacts/phase0_fprm_exp125/bp_h256_steps5000_lr3e4_nseq1_seed1" `
  --append-md "experiments/Experiment 125 - No-BP Hard Ternary FPRM/results_bp_h256_steps5000_lr3e4_nseq1_seed1.md"
