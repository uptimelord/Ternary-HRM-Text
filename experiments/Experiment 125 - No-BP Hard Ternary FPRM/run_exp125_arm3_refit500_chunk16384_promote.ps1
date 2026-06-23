$ErrorActionPreference = "Stop"

# Exp125 ARM 3 PROMOTE (optimized config): online-refined DFA, 5000 steps, both
# seeds, vocab_chunk_size 16384. Identical to run_exp125_arm3_refit500_promote.ps1
# except chunk 2048 -> 16384, a pure speed lever (exact logsumexp math unchanged;
# only the reduction iteration count drops 32 -> 4 chunks). Validated: both seeds
# pass every Promote gate at 16384 with a 2-seed mean eval identical to chunk 2048
# (6.693 vs 6.692) and 1.42x faster (20.0 -> 14.1 min/seed, incl. thermal). VRAM
# unchanged (477.9 MiB steady-state). See README "Arm 3 / Optimization".
# This is the canonical arm3 config of record.

# --- seed 1 ---
rtk python -u "experiments/Experiment 125 - No-BP Hard Ternary FPRM/exp125_nobp_hard.py" `
  --train-rule nobp-dfa-full-hard `
  --device cuda --seed 1 --numseqs 1 `
  --hidden-size 256 --n-layers 4 --num-heads 4 --max-iters 20 `
  --prefix-len 64 --causal-len 64 --vocab-size 65536 --vocab-chunk-size 16384 `
  --pretrain-steps 5000 --warmup-steps 0 --export-calibration-steps 0 --sft-steps 0 `
  --nobp-head-lr 0.3 --nobp-core-lr 0.3 --nobp-beta 0.03 --nobp-residual-lambda 0.003 `
  --nobp-feedback-mode bp-warmup --nobp-warmup-steps 50 --nobp-warmup-ridge 1e-3 `
  --nobp-refit-interval 500 --nobp-refit-steps 20 --nobp-refit-ridge 1e-3 `
  --checkpoint-interval 500 --log-interval 500 --no-resume `
  --output-dir "artifacts/phase0_fprm_exp125/dfafull_h256_steps5000_h03_cr03_bpwarm50_refit500_chunk16384_seed1" `
  --append-md "experiments/Experiment 125 - No-BP Hard Ternary FPRM/results_dfafull_h256_steps5000_h03_cr03_bpwarm50_refit500_chunk16384_seed1.md"

# --- seed 2 ---
rtk python -u "experiments/Experiment 125 - No-BP Hard Ternary FPRM/exp125_nobp_hard.py" `
  --train-rule nobp-dfa-full-hard `
  --device cuda --seed 2 --numseqs 1 `
  --hidden-size 256 --n-layers 4 --num-heads 4 --max-iters 20 `
  --prefix-len 64 --causal-len 64 --vocab-size 65536 --vocab-chunk-size 16384 `
  --pretrain-steps 5000 --warmup-steps 0 --export-calibration-steps 0 --sft-steps 0 `
  --nobp-head-lr 0.3 --nobp-core-lr 0.3 --nobp-beta 0.03 --nobp-residual-lambda 0.003 `
  --nobp-feedback-mode bp-warmup --nobp-warmup-steps 50 --nobp-warmup-ridge 1e-3 `
  --nobp-refit-interval 500 --nobp-refit-steps 20 --nobp-refit-ridge 1e-3 `
  --checkpoint-interval 500 --log-interval 500 --no-resume `
  --output-dir "artifacts/phase0_fprm_exp125/dfafull_h256_steps5000_h03_cr03_bpwarm50_refit500_chunk16384_seed2" `
  --append-md "experiments/Experiment 125 - No-BP Hard Ternary FPRM/results_dfafull_h256_steps5000_h03_cr03_bpwarm50_refit500_seed2.md"
