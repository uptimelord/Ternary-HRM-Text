$ErrorActionPreference = "Stop"

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
  --nobp-head-lr 3e-4 `
  --nobp-beta 0.03 `
  --nobp-residual-lambda 0.003 `
  --checkpoint-interval 100 `
  --log-interval 20 `
  --output-dir "artifacts/phase0_fprm_exp125/head_hard_h256_steps1000_seed1" `
  --append-md "experiments/Experiment 125 - No-BP Hard Ternary FPRM/results_head_hard_h256_steps1000_seed1.md"
