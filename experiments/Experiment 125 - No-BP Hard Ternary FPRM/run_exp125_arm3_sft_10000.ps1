$ErrorActionPreference = "Stop"

# Exp125 ARM 3 no-BP SFT stage.
# See diagnosis: SFT collapse is head outrunning body (direct head_lr=0.3 vs
# effective body ~ core_lr*beta=0.015) + SFT response-only tokens cause under-flip
# at pretrain-tuned core_lr=0.5 (observed flip 8.36e-5 < 1e-4 floor).
# Body barely adapts; head memorizes format template → 0% exact/frozen.
#
# Use --train-rule nobp-head-hard for curriculum (learn format first), then
# a second pass with dfa-full + tuned rates.
# Recommended starting probe for arithmetic (per diagnosis A/B):
#   raise core_lr (0.7-1.0), raise beta (0.1), or lower head_lr.
# Always 100-step probe first to watch for divergence.

# Baseline (reproduces the collapse):
# rtk python ... --nobp-core-lr 0.5 ...

# Try this (higher core to restore flip band under SFT token density):
rtk python -u "experiments/Experiment 125 - No-BP Hard Ternary FPRM/arm3_nobp_sft.py" `
  --pretrain-checkpoint "artifacts/phase0_fprm_exp125/arm3_full_nseq4_steps50000_seed1/pretrain/checkpoint_fp32.pt" `
  --device cuda --seed 1 `
  --sft-steps 10000 --sft-batch-size 4 --sft-total-len 128 --sft-eval-batches 32 `
  --bp-steps 4 --vocab-chunk-size 16384 --vocab-size 65536 `
  --train-rule nobp-dfa-full-hard `
  --nobp-head-lr 0.3 --nobp-core-lr 1.0 --nobp-beta 0.1 --nobp-residual-lambda 0.003 --nobp-update-clip 1.0 `
  --nobp-warmup-steps 50 --nobp-warmup-ridge 1e-3 `
  --nobp-refit-interval 250 --nobp-refit-steps 20 --nobp-refit-ridge 1e-3 `
  --generation-max-new-tokens 64 --generation-eval-limit 200 `
  --log-interval 500 --checkpoint-interval 1000 `
  --output-dir "artifacts/phase0_fprm_exp125/arm3_sft_10000_core1p0_beta0p1_seed1"
