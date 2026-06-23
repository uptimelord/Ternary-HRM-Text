$ErrorActionPreference = "Stop"

# Exp125 core-stage flip-vs-core_lr probe, 100 steps, seed 1.
# Context: the head stage is structurally flip-unpromotable (flip floor 1e-4 and
# loss stability are incompatible at every head LR; flip rate saturates ~1.8e-4
# only at LRs that diverge loss over 1000 steps -- lr=60 passed flips at 1000
# steps but regressed loss 12.50 -> 29.80, Kill). The legitimate next lever is a
# core stage: a SEPARATE core_lr flips the body while head_lr=0.3 stays loss-safe
# (head flips ~6e-7, negligible in the mean). nobp-dfa-full-hard updates all body
# projections (~4.2M params), so the body needs to flip at only ~5e-4 to pull the
# weighted mean (head 16.8M + body 4.2M) to the 1e-4 floor.
# This is a 100-step mechanics probe (analogous to the README's step-2 head
# check), NOT a Promote claim. Quantizer untouched (group 32, threshold 0.25).

$headLr = 0.3
$coreLrs = @(0.03, 0.1, 0.3)
$tags = @("cr003", "cr01", "cr03")
for ($i = 0; $i -lt $coreLrs.Length; $i++) {
    $coreLr = $coreLrs[$i]
    $tag = $tags[$i]
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
      --pretrain-steps 100 `
      --warmup-steps 0 `
      --export-calibration-steps 0 `
      --sft-steps 0 `
      --nobp-head-lr $headLr `
      --nobp-core-lr $coreLr `
      --nobp-beta 0.03 `
      --nobp-residual-lambda 0.003 `
      --checkpoint-interval 100 `
      --log-interval 20 `
      --no-resume `
      --output-dir "artifacts/phase0_fprm_exp125/dfafull_h256_steps100_h03_${tag}_seed1" `
      --append-md "experiments/Experiment 125 - No-BP Hard Ternary FPRM/results_dfafull_h256_steps100_h03_${tag}_seed1.md"
}
