$ErrorActionPreference = "Stop"

# Exp125 S2: hybrid BP SFT -- no-BP pretrain (arm 3, 50k, nseq4) + BP+AdamW SFT.
# Tests whether the 1.22-nat weaker no-BP pretrain degrades the SFT result.
# exp123 (BP pretrain + BP SFT) reaches frozen 51%; estimate for this hybrid
# 35-45%. HYBRID pipeline: no-BP at pretrain (4.1x VRAM win), BP at SFT (small,
# fits easily). Not no-BP end-to-end. Reuses exp123's exact SFT config (10000
# steps, batch 4, lr 1e-4, AdamW, tequila STE).

rtk python -u "experiments/Experiment 125 - No-BP Hard Ternary FPRM/arm3_bp_sft.py" `
  --pretrain-checkpoint "artifacts/phase0_fprm_exp125/arm3_full_nseq4_steps50000_seed1/pretrain/checkpoint_fp32.pt" `
  --device cuda --seed 1 `
  --sft-steps 10000 --sft-batch-size 4 --sft-total-len 128 --sft-eval-batches 32 `
  --bp-steps 4 --vocab-size 65536 `
  --sft-lr 1e-4 --optimizer adamw --no-amp `
  --generation-max-new-tokens 64 --generation-eval-limit 200 `
  --log-interval 500 `
  --output-dir "artifacts/phase0_fprm_exp125/arm3_bp_sft_10000_seed1"
