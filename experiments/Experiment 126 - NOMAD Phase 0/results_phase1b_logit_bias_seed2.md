# Experiment 126 - Phase 1B-logit-bias: memory as direct logit bias

checkpoint: `C:\Users\Dos\Documents\GRAM\BitNet-HRM\artifacts\phase0_nomad_exp126\seed1\pretrain\checkpoint_fp32.pt` (frozen, Delta-theta-core=0, Delta-theta-head=0)
bias: z' = W_o h + beta * b_M ; b_M = retrieved chunks' token-freq dist (decay-weighted)
trained: beta + trust (per-token) (500 steps, eta_beta=1.0, eta_trust=4000.0), beta_init=0.0 -> 1.2023
retrieval: per-position, exact only, top_k=4, stride=32

## Results (averaged over eval batches)

| test | loss | top1 | top5 | top10 | mean_rank | ece |
|------|------|------|------|-------|-----------|-----|
| baseline (off) | 7.9450 | 0.1533 | 0.2949 | 0.3738 | 7025.1 | 0.1136 |
| on relevant (beta trained) | 7.8667 | 0.1541 | 0.3101 | 0.3894 | 7012.8 | 0.1138 |
| on distractor (beta trained) | 7.9436 | 0.1533 | 0.2947 | 0.3745 | 7023.6 | 0.1137 |

## New-doc insertion (answer chunk in memory, Delta-theta=0 except beta)

- seq0: retrieval_hit=True ans_tok=50 loss 8.413->8.325 top5 0.189->0.213 rank 5191->5105
- seq1: retrieval_hit=True ans_tok=468 loss 8.837->8.573 top5 0.110->0.157 rank 8636->8226
- seq2: retrieval_hit=True ans_tok=1777 loss 7.144->7.107 top5 0.409->0.409 rank 6008->5954
- seq3: retrieval_hit=True ans_tok=236 loss 7.526->7.490 top5 0.433->0.425 rank 6964->6844
