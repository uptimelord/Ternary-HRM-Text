$ErrorActionPreference = "Stop"

# Exp125 nobp-dfa-full-hard, 1000 steps, head LR 0.3 + core LR 0.1, seed 1.
# Promote attempt via the core stage.
#
# Why the core stage, not the head: the head stage is structurally flip-unpromotable.
# A full sweep showed flip rate saturates ~1.8e-4 and only reaches the [1e-4,1e-2]
# band at head LR ~= 60, which passes flips at 1000 steps (1.08e-4) but regresses
# eval loss 12.50 -> 29.80 (gap +17.30, Kill on strict-eval-regression). No head
# LR satisfies both the flip floor and loss stability.
#
# The core stage separates the two concerns: head LR 0.3 stays loss-safe (head
# flips ~6e-7, negligible in the weighted mean) while a separate core LR flips the
# body. nobp-dfa-full-hard updates all body projections (~4.2M params), the best
# flip-gate geometry (body needs only ~5e-4 flip to pull the head+body mean to
# the 1e-4 floor). A 100-step core_lr sweep found core_lr=0.1 gives mean flip
# 4.20e-4 (in band, 4.2x margin) with eval gap -2.292 (loss improves, 113x over
# the 0.0203 noise floor) -- margin on BOTH gates, unlike the head knife-edge.
#
# This is a run-order deviation (README gates core stages behind head promotion;
# the head is structurally blocked, documented below). All Promote gates remain
# binding: flip in [1e-4,1e-2], loss improves > 0.0203, no autograd, no optimizer
# state, hard from step 0, peak VRAM <= 3800 MiB, BOTH seeds. Quantizer untouched
# (group 32, threshold 0.25, mean-abs). Measure-twice: seed 1 first; seed 2 only
# if seed 1 clears all gates.

rtk python -u "experiments/Experiment 125 - No-BP Hard Ternary FPRM/exp125_nobp_hard.py" `
  --train-rule nobp-dfa-full-hard `
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
  --nobp-core-lr 0.1 `
  --nobp-beta 0.03 `
  --nobp-residual-lambda 0.003 `
  --checkpoint-interval 100 `
  --log-interval 20 `
  --no-resume `
  --output-dir "artifacts/phase0_fprm_exp125/dfafull_h256_steps1000_h03_cr01_seed1" `
  --append-md "experiments/Experiment 125 - No-BP Hard Ternary FPRM/results_dfafull_h256_steps1000_h03_cr01_seed1.md"
