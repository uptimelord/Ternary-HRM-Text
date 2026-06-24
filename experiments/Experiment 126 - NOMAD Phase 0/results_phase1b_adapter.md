# Experiment 126 - Phase 1B: train memory adapter on frozen 0C core

checkpoint: `C:\Users\Dos\Documents\GRAM\BitNet-HRM\artifacts\phase0_nomad_exp126\seed1\pretrain\checkpoint_fp32.pt` (frozen, Delta-theta-core=0, Delta-theta-head=0)
adapter: A=[256,256] + gate gamma, trained 300 steps (eta=0.01, alpha=1.0, gamma_init=0.1)
retrieval: exact only (compression DISABLED -- it hurt in 1A), top_k=4

## Results (averaged over eval batches)

| test | loss | top1 | top5 | top10 | mean_rank | ece |
|------|------|------|------|-------|-----------|-----|
| baseline (off, adapter init) | 7.9450 | 0.1533 | 0.2949 | 0.3738 | 7025.1 | 0.1136 |
| off (adapter trained) | 7.9450 | 0.1533 | 0.2949 | 0.3738 | 7025.1 | 0.1136 |
| on relevant memory | 7.9400 | 0.1533 | 0.2949 | 0.3735 | 7023.1 | 0.1132 |
| on distractor | 7.9451 | 0.1533 | 0.2954 | 0.3738 | 7025.8 | 0.1136 |

## New-doc insertion (Delta-theta=0 except adapter)

- seq0: retrieval_hit=True loss 8.413->8.407 top5 0.189->0.189 rank 5191->5190
- seq1: retrieval_hit=True loss 8.837->8.833 top5 0.110->0.110 rank 8636->8633
- seq2: retrieval_hit=True loss 7.144->7.138 top5 0.409->0.409 rank 6008->6001
- seq3: retrieval_hit=True loss 7.526->7.515 top5 0.433->0.433 rank 6964->6937
