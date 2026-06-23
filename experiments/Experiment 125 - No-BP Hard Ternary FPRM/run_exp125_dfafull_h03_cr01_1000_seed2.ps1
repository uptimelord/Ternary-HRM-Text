$ErrorActionPreference = "Stop"

# Exp125 nobp-dfa-full-hard, 1000 steps, head LR 0.3 + core LR 0.1, seed 2.
# Second seed for the Promote verdict. Seed 1 cleared all gates:
#   mean flip 4.314e-4 (in band, stable), eval gap -3.161 (12.50 -> 9.34),
#   no autograd, no optimizer state, hard from step 0, peak VRAM 506.6 MiB.
# Identical config to seed 1; only --seed and output paths differ.

rtk python -u "experiments/Experiment 125 - No-BP Hard Ternary FPRM/exp125_nobp_hard.py" `
  --train-rule nobp-dfa-full-hard `
  --device cuda `
  --seed 2 `
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
  --nobp-core-lr 0.1 `
  --nobp-beta 0.03 `
  --nobp-residual-lambda 0.003 `
  --checkpoint-interval 100 `
  --log-interval 20 `
  --no-resume `
  --output-dir "artifacts/phase0_fprm_exp125/dfafull_h256_steps1000_h03_cr01_seed2" `
  --append-md "experiments/Experiment 125 - No-BP Hard Ternary FPRM/results_dfafull_h256_steps1000_h03_cr01_seed2.md"
