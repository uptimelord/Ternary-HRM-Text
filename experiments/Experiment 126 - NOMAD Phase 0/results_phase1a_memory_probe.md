# Experiment 126 - Phase 1A: retrieval-only memory probe (frozen 0C/10k)

checkpoint: `C:\Users\Dos\Documents\GRAM\BitNet-HRM\artifacts\phase0_nomad_exp126\seed1\pretrain\checkpoint_fp32.pt` (step 10000, frozen, Delta-theta=0)
eval: 4 batches x 16 seqs x 128 tokens, prefix_len=64, top_k=4
relevant memory: 2790 chunks (train corpus), distractor: 2844 chunks (random ids)

## Inference tests (averaged over eval batches)

| test | loss | top1 | top5 | top10 | mean_rank | median_rank | ece |
|------|------|------|------|-------|-----------|-------------|-----|
| baseline (off) | 7.9450 | 0.1533 | 0.2949 | 0.3738 | 7025.1 | 35.5 | 0.1136 |
| exact | 7.9360 | 0.1523 | 0.2913 | 0.3748 | 6958.1 | 35.5 | 0.1130 |
| exact+compression | 7.9915 | 0.1487 | 0.2888 | 0.3708 | 7057.5 | 34.8 | 0.1119 |
| distractor | 7.9618 | 0.1504 | 0.2917 | 0.3730 | 6992.2 | 36.0 | 0.1105 |
