$ErrorActionPreference = "Stop"

# Exp125 head-hard, 1000 steps, head LR 0.3, single seed (seed 1).
# Run-order step 3: full 1000-step head run after the 100-step mechanics passed.
# Expected verdict: flip-gate miss (mean ternary flip rate below 0.0001 floor);
# one-seed development evidence, not a Promote attempt (Promote needs both seeds).

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
  --nobp-head-lr 0.3 `
  --nobp-beta 0.03 `
  --nobp-residual-lambda 0.003 `
  --checkpoint-interval 100 `
  --log-interval 20 `
  --output-dir "artifacts/phase0_fprm_exp125/head_hard_h256_steps1000_lr3_seed1" `
  --append-md "experiments/Experiment 125 - No-BP Hard Ternary FPRM/results_head_hard_h256_steps1000_lr3_seed1.md"
