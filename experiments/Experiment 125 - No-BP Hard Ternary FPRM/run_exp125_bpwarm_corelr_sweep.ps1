$ErrorActionPreference = "Stop"

# Exp125 bp-warmup core_lr sweep, 1000 steps, seed 1, h256, numseqs=1.
# At core_lr=0.1 the fitted matrices give flip 4.0e-5 (below 1e-4 floor) but eval
# 7.75 (beats fixed-random 9.34 by 1.59 nats). The fitted matrices are ~10x
# gentler than random, so body update magnitude is too small to flip symbols into
# band. core_lr is the legitimate scale knob (direction = learned; scale = hyper).
# Find core_lr where flip enters [1e-4, 1e-2] with loss still improving.
# Quantizer untouched. Fixed-random baseline at core_lr=0.1: flip 4.3e-4, eval 9.34.

$coreLrs = @(0.2, 0.3, 0.5)
$tags = @("cr02", "cr03", "cr05")
for ($i = 0; $i -lt $coreLrs.Length; $i++) {
    $coreLr = $coreLrs[$i]
    $tag = $tags[$i]
    rtk python -u "experiments/Experiment 125 - No-BP Hard Ternary FPRM/exp125_nobp_hard.py" `
      --train-rule nobp-dfa-full-hard `
      --device cuda --seed 1 --numseqs 1 `
      --hidden-size 256 --n-layers 4 --num-heads 4 --max-iters 20 `
      --prefix-len 64 --causal-len 64 --vocab-size 65536 --vocab-chunk-size 2048 `
      --pretrain-steps 1000 --warmup-steps 0 --export-calibration-steps 0 --sft-steps 0 `
      --nobp-head-lr 0.3 --nobp-core-lr $coreLr --nobp-beta 0.03 --nobp-residual-lambda 0.003 `
      --nobp-feedback-mode bp-warmup --nobp-warmup-steps 50 --nobp-warmup-ridge 1e-3 `
      --checkpoint-interval 0 --log-interval 200 --no-resume `
      --output-dir "artifacts/phase0_fprm_exp125/dfafull_h256_steps1000_h03_${tag}_bpwarm50_nseq1_seed1" `
      --append-md "experiments/Experiment 125 - No-BP Hard Ternary FPRM/results_dfafull_h256_steps1000_h03_${tag}_bpwarm50_nseq1_seed1.md"
}
