$ErrorActionPreference = "Stop"

# Exp125 SCALED pretrain: numseqs 1 -> 4 (4x gradient signal/step), 1000 steps,
# h256 x 4, vocab 65536. One lever changed vs the Promote run; everything else
# identical. Serialized on the single 4 GB GPU: no-BP seed1, no-BP seed2, BP control.
#
# Pre-registered decision rule:
#   Promote-if: both no-BP seeds at numseqs=4 pass all original gates
#     (flip in [1e-4,1e-2], eval gap < -0.0203, finite CE, no autograd, no opt
#     state, hard step 0, peak VRAM <= 3800, no NaN).
#   Kill-if: any gate fails or NaN.
#   Report (not gated): no-BP vs BP gap at scale vs numseqs=1 baselines
#     (no-BP 9.34/8.95, BP 6.21) -- does scaling narrow or widen the gap.
# Quantizer untouched (group 32, threshold 0.25, mean-abs).

# --- no-BP seed 1 ---
rtk python -u "experiments/Experiment 125 - No-BP Hard Ternary FPRM/exp125_nobp_hard.py" `
  --train-rule nobp-dfa-full-hard `
  --device cuda --seed 1 --numseqs 4 `
  --hidden-size 256 --n-layers 4 --num-heads 4 --max-iters 20 `
  --prefix-len 64 --causal-len 64 --vocab-size 65536 --vocab-chunk-size 2048 `
  --pretrain-steps 1000 --warmup-steps 0 --export-calibration-steps 0 --sft-steps 0 `
  --nobp-head-lr 0.3 --nobp-core-lr 0.1 --nobp-beta 0.03 --nobp-residual-lambda 0.003 `
  --checkpoint-interval 100 --log-interval 100 --no-resume `
  --output-dir "artifacts/phase0_fprm_exp125/dfafull_h256_steps1000_h03_cr01_nseq4_seed1" `
  --append-md "experiments/Experiment 125 - No-BP Hard Ternary FPRM/results_dfafull_h256_steps1000_h03_cr01_nseq4_seed1.md"

# --- no-BP seed 2 ---
rtk python -u "experiments/Experiment 125 - No-BP Hard Ternary FPRM/exp125_nobp_hard.py" `
  --train-rule nobp-dfa-full-hard `
  --device cuda --seed 2 --numseqs 4 `
  --hidden-size 256 --n-layers 4 --num-heads 4 --max-iters 20 `
  --prefix-len 64 --causal-len 64 --vocab-size 65536 --vocab-chunk-size 2048 `
  --pretrain-steps 1000 --warmup-steps 0 --export-calibration-steps 0 --sft-steps 0 `
  --nobp-head-lr 0.3 --nobp-core-lr 0.1 --nobp-beta 0.03 --nobp-residual-lambda 0.003 `
  --checkpoint-interval 100 --log-interval 100 --no-resume `
  --output-dir "artifacts/phase0_fprm_exp125/dfafull_h256_steps1000_h03_cr01_nseq4_seed2" `
  --append-md "experiments/Experiment 125 - No-BP Hard Ternary FPRM/results_dfafull_h256_steps1000_h03_cr01_nseq4_seed2.md"

# --- BP control, seed 1 (matched scale) ---
rtk python -u "experiments/Experiment 125 - No-BP Hard Ternary FPRM/exp125_nobp_hard.py" `
  --train-rule bp `
  --device cuda --seed 1 --numseqs 4 `
  --hidden-size 256 --n-layers 4 --num-heads 4 --max-iters 20 `
  --prefix-len 64 --causal-len 64 --vocab-size 65536 --vocab-chunk-size 2048 `
  --pretrain-steps 1000 --warmup-steps 0 --export-calibration-steps 0 --sft-steps 0 `
  --pretrain-lr 3e-4 --optimizer adamw --bp-steps 4 --dense-top-k 0 `
  --checkpoint-interval 100 --log-interval 100 --no-resume `
  --output-dir "artifacts/phase0_fprm_exp125/bp_h256_steps1000_lr3e4_nseq4_seed1" `
  --append-md "experiments/Experiment 125 - No-BP Hard Ternary FPRM/results_bp_h256_steps1000_lr3e4_nseq4_seed1.md"
