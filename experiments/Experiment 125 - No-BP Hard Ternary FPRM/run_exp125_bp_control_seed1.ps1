$ErrorActionPreference = "Stop"

# Exp125 BP control, 1000 steps, seed 1.
# Purpose: the no-BP Promote (nobp-dfa-full-hard) proves no-autograd hard-ternary
# pretrain improves its own loss with flips in band. It does NOT prove that is
# competitive with BP. This control runs the autograd + AdamW + tequila-STE path
# on the SAME model, data, and scale (h256 x 4, vocab 65536, 1 seq, 1000 steps,
# dense_top_k 0) so the only difference is BP vs no-BP.
# Canonical LR: exp123's validated run uses the default pretrain-LR (3e-4); no
# tuning either way. BP legitimately uses autograd + optimizer state -- that is
# the point of contrast, not a gate it must pass. Peak VRAM expected ~1 GB
# (exp123 measured 1010.9 MiB), well under the 3800 MiB cap and the 4 GB card.

rtk python -u "experiments/Experiment 125 - No-BP Hard Ternary FPRM/exp125_nobp_hard.py" `
  --train-rule bp `
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
  --pretrain-lr 3e-4 `
  --optimizer adamw `
  --bp-steps 4 `
  --dense-top-k 0 `
  --checkpoint-interval 100 `
  --log-interval 20 `
  --no-resume `
  --output-dir "artifacts/phase0_fprm_exp125/bp_h256_steps1000_lr3e4_seed1" `
  --append-md "experiments/Experiment 125 - No-BP Hard Ternary FPRM/results_bp_h256_steps1000_lr3e4_seed1.md"
