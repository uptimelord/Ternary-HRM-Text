# DiffusionBlocks: Block-wise Neural Network Training via Diffusion Interpretation

- **Authors:** Makoto Shing, Masanori Koyama, Takuya Akiba (Sakana AI / University of Tokyo)
- **Venue:** ICLR 2026
- **arXiv:** https://arxiv.org/abs/2506.14202
- **Code:** https://github.com/SakanaAI/DiffusionBlocks

## Core Idea

Residual connections in transformers are discretized steps of a reverse diffusion
process. This lets you partition a network into independent blocks, each trained
with a local denoising objective (score matching) instead of end-to-end BPTT.
Memory scales with block size, not total depth.

## Method

1. **Block partitioning** — divide L layers into B blocks
2. **Noise range assignment** — each block handles a slice of the noise schedule
   using equi-probability partitioning (equal probability mass under log-normal)
3. **Noise conditioning** — augment each block with AdaLN so it learns to denoise
   its assigned noise range independently

Training requires gradients for only one block at a time → B× memory reduction.

## Key Results

| Architecture | DiffusionBlocks | End-to-End | Memory Savings |
|---|---|---|---|
| ViT (CIFAR-100) | 59.30% | 60.25% | significant |
| DiT (ImageNet) | FID 10.63 | FID 12.09 | 3× |
| Huginn (recurrent-depth LM) | MAUVE 0.70 | MAUVE 0.49 | eliminates 32 training iterations |

- Outperforms Forward-Forward (7.85% vs 59.30% on CIFAR-100)
- For recurrent-depth models: replaces iterative BPTT with single-pass training
- Up to K-fold reduction in training computation for K-iteration recurrent models

## Relevance to BitNet-HRM

The main bottleneck for training HRM with deep bp_steps is VRAM — BPTT through
many iterations multiplies memory linearly. DiffusionBlocks offers a way out:

1. Reframe each HRM iteration as a denoising step at a specific noise level
2. Train each "iteration block" independently — no unrolling needed
3. This could let us train with effective bp_steps=10-20 on 4GB VRAM
4. Combined with EqR's attractor shaping, enables deep recurrence training
   without the memory cost

The Huginn result is directly analogous — same recurrent-depth architecture
pattern as HRM, and DiffusionBlocks improved quality while eliminating BPTT.

## Implementation Considerations

- Requires adding noise-level conditioning (AdaLN) to HRM layers
- Equi-probability partitioning needs tuning for HRM's H×L iteration structure
- May interact with ternary quantization — noise injection during training
  could conflict with STE gradients (needs investigation)
- The memory savings are most impactful at scale (130M+ params)
