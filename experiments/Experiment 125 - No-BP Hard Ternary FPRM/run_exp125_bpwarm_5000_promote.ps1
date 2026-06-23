$ErrorActionPreference = "Stop"

# Exp125 LEARNED-FEEDBACK PROMOTE: bp-warmup-seeded DFA, 5000 steps, both seeds.
# Headline result. Fits the 19 DFA feedback matrices by least squares from a
# bounded offline BP warmup (50 steps, no optimizer step, weights at init), then
# runs the no-BP trainer with the fitted matrices seeded. Removes the 5000-step
# drift that killed fixed-random; halves the BP gap (5.12 -> 2.58 nats) while
# keeping every no-BP invariant (no autograd at training time, no optimizer
# state, batch-invariant 467 MiB). See README "Learned-feedback variant".
# core_lr 0.3 is the tuned scale knob (direction = learned; loss gate binding).
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
  --checkpoint-interval 500 --log-interval 500 --no-resume `
  --output-dir "artifacts/phase0_fprm_exp125/dfafull_h256_steps5000_h03_cr03_bpwarm50_nseq1_seed1" `
  --append-md "experiments/Experiment 125 - No-BP Hard Ternary FPRM/results_dfafull_h256_steps5000_h03_cr03_bpwarm50_nseq1_seed1.md"

# --- seed 2 ---
rtk python -u "experiments/Experiment 125 - No-BP Hard Ternary FPRM/exp125_nobp_hard.py" `
  --train-rule nobp-dfa-full-hard `
  --device cuda --seed 2 --numseqs 1 `
  --hidden-size 256 --n-layers 4 --num-heads 4 --max-iters 20 `
  --prefix-len 64 --causal-len 64 --vocab-size 65536 --vocab-chunk-size 2048 `
  --pretrain-steps 5000 --warmup-steps 0 --export-calibration-steps 0 --sft-steps 0 `
  --nobp-head-lr 0.3 --nobp-core-lr 0.3 --nobp-beta 0.03 --nobp-residual-lambda 0.003 `
  --nobp-feedback-mode bp-warmup --nobp-warmup-steps 50 --nobp-warmup-ridge 1e-3 `
  --checkpoint-interval 500 --log-interval 500 --no-resume `
  --output-dir "artifacts/phase0_fprm_exp125/dfafull_h256_steps5000_h03_cr03_bpwarm50_nseq1_seed2" `
  --append-md "experiments/Experiment 125 - No-BP Hard Ternary FPRM/results_dfafull_h256_steps5000_h03_cr03_bpwarm50_nseq1_seed2.md"
