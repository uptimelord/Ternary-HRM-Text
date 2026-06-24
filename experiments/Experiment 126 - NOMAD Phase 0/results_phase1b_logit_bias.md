# Experiment 126 - Phase 1B-logit-bias: memory as direct logit bias

checkpoint: `C:\Users\Dos\Documents\GRAM\BitNet-HRM\artifacts\phase0_nomad_exp126\seed1\pretrain\checkpoint_fp32.pt` (frozen, Delta-theta-core=0, Delta-theta-head=0)
bias: z' = W_o h + beta * b_M ; b_M = retrieved chunks' token-freq dist (decay-weighted)
trained: beta only (500 steps, eta_beta=1.0), beta_init=0.0 -> 1.2811
retrieval: per-position, exact only, top_k=4, stride=32

## Results (averaged over eval batches)

| test | loss | top1 | top5 | top10 | mean_rank | ece |
|------|------|------|------|-------|-----------|-----|
| baseline (off) | 7.9450 | 0.1533 | 0.2949 | 0.3738 | 7025.1 | 0.1136 |
| on relevant (beta trained) | 7.9314 | 0.1538 | 0.2944 | 0.3743 | 7020.3 | 0.1132 |
| on distractor (beta trained) | 7.9448 | 0.1533 | 0.2949 | 0.3738 | 7024.4 | 0.1136 |

## New-doc insertion (answer chunk in memory, Delta-theta=0 except beta)

- seq0: retrieval_hit=True ans_tok=50 loss 8.413->8.397 top5 0.189->0.189 rank 5191->5121
- seq1: retrieval_hit=True ans_tok=468 loss 8.837->8.810 top5 0.110->0.110 rank 8636->8287
- seq2: retrieval_hit=True ans_tok=1777 loss 7.144->7.125 top5 0.409->0.409 rank 6008->5979
- seq3: retrieval_hit=True ans_tok=236 loss 7.526->7.493 top5 0.433->0.433 rank 6964->6855
