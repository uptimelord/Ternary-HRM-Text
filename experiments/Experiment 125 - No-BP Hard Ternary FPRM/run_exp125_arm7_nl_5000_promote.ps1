$ErrorActionPreference = "Stop"

# Exp125 ARM 7 PROMOTE: hybrid feedback -- frozen MLP predictors for the 4
# deep-qkv layers (where arm 6 proved the ~0.85 linear-fit ceiling is a
# nonlinearity limit, and arm 7 Phase 1 confirmed a regularized MLP beats linear
# there on both seeds) + linear ridge matrices for the other 15 DFA feedback
# layers (where linear already wins, per Phase 1). Otherwise identical to the
# arm-3 promote config: bp-warmup 50 steps, core_lr 0.3, refit every 500 (20-step
# bounded BP mini-batch, weights unchanged), chunk 16384, 5000 steps, both seeds.
# Phase-1 caveats carried in: the deep-qkv win is ~4% relative and the MLP loses
# on the other 15 layers, so this is a HYBRID (not full-MLP); the Promote-if
# gate eval < 6.3 (beats arm-3 6.69 by > 0.39) is not guaranteed by Phase 1.

$nlLayers = "model.resonance_core.layers.0.attn.qkv,model.resonance_core.layers.1.attn.qkv,model.resonance_core.layers.2.attn.qkv,model.resonance_core.layers.3.attn.qkv"

# --- seed 1 ---
rtk python -u "experiments/Experiment 125 - No-BP Hard Ternary FPRM/exp125_nobp_hard.py" `
  --train-rule nobp-dfa-full-hard `
  --device cuda --seed 1 --numseqs 1 `
  --hidden-size 256 --n-layers 4 --num-heads 4 --max-iters 20 `
  --prefix-len 64 --causal-len 64 --vocab-size 65536 --vocab-chunk-size 16384 `
  --pretrain-steps 5000 --warmup-steps 0 --export-calibration-steps 0 --sft-steps 0 `
  --nobp-head-lr 0.3 --nobp-core-lr 0.3 --nobp-beta 0.03 --nobp-residual-lambda 0.003 `
  --nobp-feedback-mode bp-warmup-nl --nobp-warmup-steps 50 --nobp-warmup-ridge 1e-3 `
  --nobp-nl-layers $nlLayers --nobp-nl-hidden 256 --nobp-nl-epochs 300 --nobp-nl-lr 1e-3 --nobp-nl-wd 1e-2 `
  --nobp-refit-interval 500 --nobp-refit-steps 20 --nobp-refit-ridge 1e-3 `
  --checkpoint-interval 500 --log-interval 500 --no-resume `
  --output-dir "artifacts/phase0_fprm_exp125/arm7_nl_h256_steps5000_qkv4_seed1" `
  --append-md "experiments/Experiment 125 - No-BP Hard Ternary FPRM/results_arm7_nl_h256_steps5000_qkv4_seed1.md"

# --- seed 2 ---
rtk python -u "experiments/Experiment 125 - No-BP Hard Ternary FPRM/exp125_nobp_hard.py" `
  --train-rule nobp-dfa-full-hard `
  --device cuda --seed 2 --numseqs 1 `
  --hidden-size 256 --n-layers 4 --num-heads 4 --max-iters 20 `
  --prefix-len 64 --causal-len 64 --vocab-size 65536 --vocab-chunk-size 16384 `
  --pretrain-steps 5000 --warmup-steps 0 --export-calibration-steps 0 --sft-steps 0 `
  --nobp-head-lr 0.3 --nobp-core-lr 0.3 --nobp-beta 0.03 --nobp-residual-lambda 0.003 `
  --nobp-feedback-mode bp-warmup-nl --nobp-warmup-steps 50 --nobp-warmup-ridge 1e-3 `
  --nobp-nl-layers $nlLayers --nobp-nl-hidden 256 --nobp-nl-epochs 300 --nobp-nl-lr 1e-3 --nobp-nl-wd 1e-2 `
  --nobp-refit-interval 500 --nobp-refit-steps 20 --nobp-refit-ridge 1e-3 `
  --checkpoint-interval 500 --log-interval 500 --no-resume `
  --output-dir "artifacts/phase0_fprm_exp125/arm7_nl_h256_steps5000_qkv4_seed2" `
  --append-md "experiments/Experiment 125 - No-BP Hard Ternary FPRM/results_arm7_nl_h256_steps5000_qkv4_seed2.md"
