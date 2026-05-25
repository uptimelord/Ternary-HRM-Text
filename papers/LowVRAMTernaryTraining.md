# Low-VRAM Ternary Pretraining: Literature Survey

> Papers and techniques to enable **true low-VRAM pretraining** of 1-bit/1.58-bit/2-bit models — eliminating the master weight and optimizer state bottleneck.

---

## The Problem: Why 1.58-bit Models Still Need 20× VRAM to Train

Even though your ternary model weights are ~1.58 bits/param, **training** requires:

| Component | Precision | Per-Param Cost | 1B Model |
|---|---|---|---|
| **Master/shadow weights** | FP32 | 4 bytes | 4.0 GB |
| **Gradients** | BF16 | 2 bytes | 2.0 GB |
| **Adam momentum (m)** | FP32 | 4 bytes | 4.0 GB |
| **Adam variance (v)** | FP32 | 4 bytes | 4.0 GB |
| Activations (varies) | BF16 | ~2-8 bytes | 2-8 GB |
| **Total training** | — | **16-22 bytes/param** | **16-22 GB** |
| Inference (ternary) | 1.58-bit | 0.2 bytes | **0.2 GB** |

> [!CAUTION]
> **Your ternary model is 100× smaller for inference but needs the same VRAM as dense for training.** The master weights + optimizer states are the bottleneck, not the model weights.

---

## 🏆 Tier 1 — Directly Eliminates Master Weights

### 1. ECO: Error-Compensating Optimizer (Eliminate Master Weights)
- **Paper:** arXiv:2601.22101 (January 2026)
- **URL:** https://arxiv.org/abs/2601.22101
- **What it does:** Removes the FP32 master weight buffer entirely. Updates are applied **directly to quantized parameters**. Quantization error from each step is injected into the optimizer's momentum buffer, creating an error-feedback loop that preserves convergence.
- **Memory savings:**

| Component | Standard QAT | ECO |
|---|---|---|
| Master weights | 4 bytes/param | **0 bytes** |
| Quantized weights | — | 0.2 bytes/param (ternary) |
| Momentum (m) | 4 bytes | 4 bytes (carries error) |
| Variance (v) | 4 bytes | 4 bytes |
| **Total** | **16 bytes** | **~8.2 bytes** |

- **Key result:** Near-lossless accuracy for FP8/INT4 training of models up to 2.1B params. Proven on Gemma-3 1B.
- **Convergence:** Formally proven under standard assumptions with decaying LR.

> [!IMPORTANT]
> **This is the single highest-impact paper for your goal.** ECO directly addresses the #1 memory consumer (master weights). For a ternary model, the quantized weights are just 0.2 bytes/param instead of 4 bytes — a **20× reduction** in the weight storage component.

**Relevance to your project:**
- Your [TernaryLinear158Init](file:///c:/Users/Dos/Documents/GRAM/BitNet-HRM/models/layers.py#L88-L173) stores FP32 `self.weight` (master weight) and computes quantized weight on-the-fly. ECO would eliminate `self.weight` and store only the ternary representation + error in momentum.
- Would need to modify `quantized_weight()` to update ternary weights in-place and feed error to Adam.

---

### 2. Stochastic Ternary Momentum
- **Paper:** arXiv:2410.09734 (October 2024)
- **URL:** https://arxiv.org/abs/2410.09734
- **What it does:** Goes further than ECO — quantizes the **optimizer states themselves** to ternary. Parameters, gradients, AND momentum are all {-1, 0, +1} with stochastic Bernoulli sampling for updates.
- **Memory savings:**

| Component | Standard | Ternary Momentum |
|---|---|---|
| Weights | 4 bytes | 0.2 bytes (ternary) |
| Gradients | 2 bytes | 0.2 bytes (ternary) |
| Momentum | 4 bytes | **0.2 bytes (ternary)** |
| Variance | 4 bytes | **eliminated** |
| **Total** | **14 bytes** | **~0.8 bytes** |

- **Key result:** Up to **95% reduction** in optimizer memory. Uses integer-valued updates via stochastic sampling.
- **Tradeoff:** More noisy optimization — may need more training steps to converge. Works best for simpler architectures.

> [!WARNING]
> This is more experimental than ECO. The fully-ternary optimizer hasn't been validated at LLM pretraining scale yet. But the memory reduction is extraordinary — potentially enabling 1B models on **<2 GB VRAM** for weights + optimizer.

---

## 🥈 Tier 2 — Compress Optimizer States (Keep Master Weights)

### 3. GaLore: Gradient Low-Rank Projection
- **Paper:** arXiv:2403.03507 (March 2024)
- **URL:** https://arxiv.org/abs/2403.03507
- **What it does:** Projects gradients onto a low-rank subspace, reducing optimizer state memory by **>80%**. Full-parameter learning (not LoRA — all parameters update), but optimizer stores only the low-rank projection.
- **Key result:** Pretrained LLaMA-7B on a single RTX 4090 (24GB) — normally requires 4×A100s.
- **Memory savings:** ~80% reduction in optimizer states. Combined with 8-bit optimizer, can approach ~4 bytes/param total.

**Relevance:**
- Could combine with ECO: use ECO to eliminate master weights + GaLore to compress whatever optimizer state remains
- Proven at LLM pretraining scale (up to 7B)
- Available as a drop-in PyTorch optimizer wrapper

---

### 4. LOMO: Low-Memory Optimization
- **URL:** https://arxiv.org/abs/2306.09782
- **What it does:** Fuses gradient computation with weight updates in a single backward pass — never materializes the full gradient tensor. Layer-wise updates mean only one layer's gradient exists in memory at a time.
- **Memory savings:** Eliminates gradient storage entirely (~2 bytes/param saved). Combined with low-precision optimizer, approaches SGD-level memory usage.
- **Tradeoff:** Slower convergence than Adam (no momentum/variance). Best for fine-tuning; pretraining quality not fully validated.

---

### 5. 8-bit Adam (bitsandbytes)
- **Available:** [github.com/TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- **What it does:** Quantizes Adam momentum and variance to INT8 with dynamic quantization. Drop-in replacement for `torch.optim.AdamW`.
- **Memory savings:**

| Component | FP32 Adam | 8-bit Adam |
|---|---|---|
| Momentum | 4 bytes | 1 byte |
| Variance | 4 bytes | 1 byte |
| **Optimizer total** | 8 bytes | 2 bytes |

- **Maturity:** Production-ready, widely used. Works with any model.

---

## 🥉 Tier 3 — Better Quantized Training (Complementary)

### 6. QuEST: Quantized, Efficient, and Stable Training
- **Paper:** arXiv:2407.xxxxx (2024, ICML 2025)
- **URL:** Check [IST-DASLab/QuEST](https://github.com/IST-DASLab/QuEST)
- **What it does:** Replaces STE with a "trust gradient estimator" that minimizes error between the quantized and true gradient. Plus Hadamard normalization for better quantization fitting. Enables stable pretraining down to **1-bit** precision.
- **Relevance:** Not a memory technique per se, but enables **lower-bit weight formats** that are more memory-efficient. Could replace your STE approach for even tighter quantization.

---

### 7. StoSignSGD: State-Free Optimizer for Low-Precision Training
- **Paper:** 2026
- **What it does:** A state-free optimizer (no momentum, no variance) that works well in low-precision regimes where AdamW fails. Injects structural stochasticity into sign-based updates.
- **Memory savings:** Eliminates ALL optimizer states. Only weights + gradients in memory.
- **Tradeoff:** State-free optimizers converge slower. May need 2-5× more training steps.

---

## VRAM Budget Calculator: Stacking Techniques

### Scenario: 1B-parameter ternary HRM-Text pretraining

| Technique Stack | Weights | Optimizer | Gradients | Activations¹ | **Total** |
|---|---|---|---|---|---|
| **Baseline (your current)** | 4.0 GB | 8.0 GB | 2.0 GB | ~4 GB | **~18 GB** |
| + 8-bit Adam | 4.0 GB | 2.0 GB | 2.0 GB | ~4 GB | ~12 GB |
| + ECO (no master weights) | 0.2 GB | 2.0 GB | 2.0 GB | ~4 GB | **~8.2 GB** |
| + ECO + GaLore (80% opt reduction) | 0.2 GB | 0.4 GB | 2.0 GB | ~4 GB | **~6.6 GB** |
| + ECO + 8-bit Adam + grad checkpoint | 0.2 GB | 0.4 GB | 0.2 GB² | ~1 GB³ | **~1.8 GB** |
| + Ternary momentum (everything ternary) | 0.2 GB | 0.2 GB | 0.2 GB | ~1 GB³ | **~1.6 GB** |

¹ Activation memory depends on batch size and sequence length
² Layer-wise gradient computation (LOMO-style)
³ With aggressive gradient checkpointing (recompute every layer)

> [!TIP]
> The **ECO + 8-bit Adam + gradient checkpointing** stack could theoretically fit a **600M–1B ternary model on 4 GB VRAM** with batch_size=1, seq_len=512, and gradient accumulation. This would be *slow* but *possible*.

---

## Recommended Combination for Your Project

For your Ternary-HRM-Text project targeting 4 GB VRAM pretraining:

```
┌─────────────────────────────────────────────┐
│           4GB VRAM Training Stack           │
├─────────────────────────────────────────────┤
│ 1. ECO optimizer (no master weights)        │ ← Biggest win: -4GB
│ 2. 8-bit Adam states (bitsandbytes)         │ ← Easy: -6GB optimizer
│ 3. Gradient checkpointing (every layer)     │ ← Trades compute for RAM
│ 4. Layer-wise gradient (LOMO-style)         │ ← No full gradient tensor
│ 5. BF16 gradients + micro-batching          │ ← Standard practice
│ 6. Mixed_top512 ternary vocab               │ ← Your existing best recipe
└─────────────────────────────────────────────┘
```

### Implementation Priority

| # | Technique | Paper | Effort | VRAM Saved | Cumulative |
|---|---|---|---|---|---|
| 1 | 8-bit Adam | bitsandbytes | ⚡ 1 line | 6 GB | 12 GB |
| 2 | Gradient checkpointing | PyTorch native | ⚡ 5 lines | 3-6 GB | 6-9 GB |
| 3 | ECO (no master weights) | arXiv:2601.22101 | 🔧 ~100 lines | 3.8 GB | ~4 GB |
| 4 | Layer-wise grad updates | LOMO-inspired | 🔧 ~50 lines | 1.8 GB | ~2 GB |
| 5 | Ternary momentum | arXiv:2410.09734 | 🏗️ ~200 lines | 1.6 GB | ~1.6 GB |

---

## Key Insight: The "Ternary Everything" Vision

The endgame for truly low-VRAM ternary training is:

| Component | Current | Goal |
|---|---|---|
| **Model weights** | FP32 master + ternary forward | **Ternary only** (ECO) |
| **Optimizer momentum** | FP32 | **Ternary** (Stochastic Ternary Momentum) |
| **Optimizer variance** | FP32 | **Eliminated** (sign-based optimizer) |
| **Gradients** | BF16 | **Ternary or 8-bit** |
| **Activations** | BF16 | **8-bit** (Spectra 1.1 style) + checkpointing |

When ALL components are ternary/low-bit, the model trains with ~1 byte/param instead of ~18 bytes/param — an **18× VRAM reduction**. A 1B model would need ~1 GB for weights+optimizer instead of ~18 GB.

> [!CAUTION]
> **No single paper has demonstrated the full stack yet.** ECO eliminates master weights. Ternary momentum eliminates optimizer states. But combining them for ternary LLM pretraining at scale (>1B params, >100B tokens) is unexplored territory — and a potential contribution for your project.

---

## Papers Referenced

| Paper | arXiv | Focus |
|---|---|---|
| **ECO** | [2601.22101](https://arxiv.org/abs/2601.22101) | Eliminate master weights |
| **Stochastic Ternary Momentum** | [2410.09734](https://arxiv.org/abs/2410.09734) | Ternary optimizer states |
| **GaLore** | [2403.03507](https://arxiv.org/abs/2403.03507) | Low-rank optimizer states |
| **LOMO** | [2306.09782](https://arxiv.org/abs/2306.09782) | Layer-wise gradient updates |
| **QuEST** | IST-DASLab | Trust gradient estimator |
| **StoSignSGD** | 2026 | State-free optimizer |
| **8-bit Adam** | bitsandbytes | INT8 optimizer states |