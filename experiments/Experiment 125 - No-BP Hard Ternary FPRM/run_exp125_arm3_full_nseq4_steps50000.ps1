$ErrorActionPreference = "Stop"

# Exp125 ARM 3 vs Exp123 -- FULL apples-to-apples pretrain (same data, same steps).
# Trains arm3 no-BP at exp123's FULL pretrain config: numseqs=4, 50000 steps,
# h256x4, prefix/causal 64/64, vocab 65536, eval-batches 8 -- identical data
# pipeline and scale to exp123's run_fprm_adam8_amp_full.ps1 (the
# h256_fprm_adam8_amp_max20_bp4_steps50000_sft10000_seed1 run). exp123 is NOT
# retrained; its 50000-step pretrain evals are read from the existing
# pretrain/metrics.json (final soft 4.129, hard_export 4.366).
#
# Arm3 hyperparams = the promoted config (head_lr 0.3, bp-warmup 50, refit
# K=500/20, ridge 1e-3, chunk 16384) with ONE legitimate scale-knob change:
# core_lr 0.3 -> 0.5. At numseqs=4 the 4x-averaged gradient is cleaner, so the
# promoted core_lr 0.3 (tuned at numseqs=1) under-flips (7.1e-5, below the 1e-4
# floor); 0.5 restores the flip band (1.25e-4 mid-band, loss stable), 1.0
# diverges. core_lr is the README's sanctioned scale knob (direction = learned,
# scale = hyperparameter); the flip band and noise floor are NOT widened. Probes:
# results_arm3_full_nseq4_probe_cr{0.3,0.5,1.0}_seed1.md. Quantizer untouched.
#
# SCOPE: pretrain-stage only. exp123's full pipeline adds 10000 SFT; arm3 v1 is
# pretrain-only (no no-BP SFT exists). So this compares 50k pretrain (hard eval)
# vs 50k pretrain hard_export_eval. The SFT stage is a separate, unbuilt piece.

# --- seed 1 (matches exp123's seed 1 full run) ---
rtk python -u "experiments/Experiment 125 - No-BP Hard Ternary FPRM/exp125_nobp_hard.py" `
  --train-rule nobp-dfa-full-hard `
  --device cuda --seed 1 --numseqs 4 `
  --hidden-size 256 --n-layers 4 --num-heads 4 --max-iters 20 `
  --prefix-len 64 --causal-len 64 --vocab-size 65536 --vocab-chunk-size 16384 `
  --pretrain-steps 50000 --warmup-steps 0 --export-calibration-steps 0 --sft-steps 0 `
  --nobp-head-lr 0.3 --nobp-core-lr 0.5 --nobp-beta 0.03 --nobp-residual-lambda 0.003 `
  --nobp-feedback-mode bp-warmup --nobp-warmup-steps 50 --nobp-warmup-ridge 1e-3 `
  --nobp-refit-interval 500 --nobp-refit-steps 20 --nobp-refit-ridge 1e-3 `
  --eval-batches 8 `
  --checkpoint-interval 1000 --log-interval 1000 --no-resume `
  --output-dir "artifacts/phase0_fprm_exp125/arm3_full_nseq4_steps50000_seed1" `
  --append-md "experiments/Experiment 125 - No-BP Hard Ternary FPRM/results_arm3_full_nseq4_steps50000_seed1.md"
