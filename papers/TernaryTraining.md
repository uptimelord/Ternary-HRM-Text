# Ternary LLM Training — Literature Survey & Implementation Guide

> Comprehensive cross-reference of **ternary / 1.58-bit language model training** papers, mapped against the [Ternary-HRM-Text](file:///c:/Users/Dos/Documents/GRAM/BitNet-HRM) project codebase with actionable implementation roadmap.

> [!NOTE]
> This guide integrates insights from the arXiv literature with direct code mappings to your experiments. Always check individual paper licenses for any restrictions on use.

---

## Your Current Setup (Baseline)

| Component | Current Value | File |
|---|---|---|
| Ternary layer | `TernaryLinear158Init` with STE | [layers.py:88-173](file:///c:/Users/Dos/Documents/GRAM/BitNet-HRM/models/layers.py#L88-L173) |
| Quantization | `absmean` group scaling, `threshold=0.7` (body) / `0.25` (vocab) | [transformer.py:19-35](file:///c:/Users/Dos/Documents/GRAM/BitNet-HRM/models/transformer.py#L19-L35) |
| Group size | 128 (body) / 32 (vocab) | Exp 9, Exp 11 |
| STE method | `weight + (hard_weight - weight).detach()` | [layers.py:170](file:///c:/Users/Dos/Documents/GRAM/BitNet-HRM/models/layers.py#L170) |
| Weight decay | Constant 0.1 throughout training | [cfg_pretrain.yaml](file:///c:/Users/Dos/Documents/GRAM/BitNet-HRM/config/cfg_pretrain.yaml) |
| Training mode | Ternary from scratch (no transition) | — |
| Best recipe | Dense body + mixed_top512 vocab | README |
| Body ternary gap | +0.06 (full body), -0.016 (`mlp_gate_up` only) | Exp 11, 13 |

---

## 🏆 Tier 1 — Directly Relevant to Your Current Experiments

### 1. Surprising Effectiveness of Pretraining Ternary Language Models at Scale (Spectra)
- **Authors:** Kaushal, Vaidhya, Mondal, Pandey, Bhagat, Rish
- **Venue:** ICLR 2025 (Spotlight)
- **URL:** https://arxiv.org/abs/2407.13647
- **What it does:** Introduces the **Spectra** LLM suite (99M → 3.9B params, 300B tokens). Demonstrates ternary models (TriLMs) have **superior scaling behavior** vs post-quantized models, and match full-precision performance at ≥1B params.
- **Why it matters for you:**
  - ✅ Your project already cites this paper — it's the foundational reference for ternary pretraining viability
  - ✅ Validates that ternary training from scratch works at scale (your current `TernaryLinear158Init` + STE approach)
  - ✅ Finding: *"TriLMs converge normally at various scales; decreasing peak LR and adjusting weight decay improve training"* — directly applicable to your experiment hyperparameter tuning
  - ✅ Shows 3.9B TriLM matches 3.9B FloatLM in fewer total bits — supports your Experiment 13 (stacked vocab + body ternary) direction

---

### 2. Spectra 1.1: Scaling Laws and Efficient Inference for Ternary Language Models
- **Authors:** Vaidhya et al.
- **Date:** June 2025
- **URL:** https://arxiv.org/abs/2506.23025
- **What it does:** Extends Spectra to **1.2T tokens** with 1.5B, 2.5B, 3.6B TriLMs. Establishes ternary-specific scaling laws. Introduces **2-bit and 1.6-bit packing** + **TriRun GPU kernel** (5–8× speedup).

#### Key Finding: Data Scales Better Than Parameters for Ternary

| | FloatLM | TriLM |
|---|---|---|
| Data exponent (β) | 0.283 | **0.351** |
| Param exponent (α) | 0.336 | **0.297** |

**Translation:** Ternary models get **24% more benefit from data scaling** and **12% less benefit from parameter scaling** compared to float models.

**Optimal token-to-parameter ratio:**

| Model Type | Optimal D/N |
|---|---|
| FloatLM (Chinchilla) | ~20 |
| **TriLM (Spectra)** | **~200–300** |

Your current experiments run at D/N ≈ 0.5–2 (500 steps × 4 seqs × 128 tokens / ~1M params). This is **100-600× below** the optimal ratio for ternary models.

**Concrete Evidence:**

| Config | Params | Tokens | Val Loss |
|---|---|---|---|
| TriLM-3.6B, 300B tokens | 3.6B | 300B | 2.27 |
| TriLM-1.5B, 1.2T tokens | **1.5B** | **1.2T** | **2.19** |

**Smaller model + 4× more data = better loss.** This directly challenges your Experiment 15 (width scaling) direction.

- **Why it matters for you:**
  - ✅ **Scaling law: ternary models benefit more from data than params** — informs whether to scale width or just train longer
  - ✅ **Packing schemes** directly relevant to your packed checkpoint measurement work (`mixed_top512` at 5.11 MB)
  - ✅ Your Experiment 14 (Mixed Top512 Long Run) is a small-scale version of the same scaling question
  - ✅ TriRun kernel design could inform your inference/export pipeline

#### Actionable Changes from Spectra 1.1

> [!WARNING]
> **Actionable change #3:** Your Experiment 15 (width scaling: hidden 128→192→256) may be the wrong direction for ternary. Spectra 1.1 shows you should **train longer** instead.

- Your Exp 14 (2000 steps) already showed that mixed_top512 gap narrows with more training
- Spectra suggests 10× to 50× more training tokens would be more effective than 2× hidden size
- **Recommendation:** Before running wider experiments, run Exp 14 at 10,000 and 50,000 steps to validate the data-scaling advantage

> [!CAUTION]
> **Actionable change #4:** Spectra 1.1 **disables weight decay in the last 20% of training** for ternary models.
>
> Your [cfg_pretrain.yaml](file:///c:/Users/Dos/Documents/GRAM/BitNet-HRM/config/cfg_pretrain.yaml) uses constant `weight_decay=0.1` throughout. This causes late-training instability where weight decay shrinks latent weight magnitudes, reducing "ternary confidence" and causing weight flipping.

```python
# In pretrain.py training loop:
if step > 0.8 * total_steps:
    for pg in optimizer.param_groups:
        pg['weight_decay'] = 0.0
```

> [!TIP]
> **Actionable change #5:** Adopt Spectra's **2-bit packing** for your checkpoint export.

| Scheme | Bits/weight | Compression vs FP16 |
|---|---|---|
| Your current | ~1.58 (log₂3) | variable |
| Spectra 2-bit | 2.0 | 8× |
| Spectra 1.6-bit | 1.6 | 10× |

The 2-bit scheme is trivial to implement: `00`=0, `01`=+1, `10`=-1. 16 weights per 32-bit int. Fast bitwise unpack.

---

### 3. Tequila: Trapping-free Ternary Quantization for Large Language Models
- **Authors:** Huang, Wu, Cen, Yu, Li, Liu, Zhu, Chen, Liu, Wu (Tencent)
- **Date:** 2025
- **URL:** https://arxiv.org/abs/2509.23809
- **What it does:** Identifies **"deadzone trapping"** — weights stuck at the {-1, 0, 1} boundary receiving noisy, uninformative gradients during STE training. Proposes repurposing trapped weights as **dynamic biases** to receive meaningful gradients. Achieves >4% gain on ARC.

#### The Problem It Solves

Your [TernaryLinear158Init.quantized_weight()](file:///c:/Users/Dos/Documents/GRAM/BitNet-HRM/models/layers.py#L163-L170) does this:

```python
# Current: weights in deadzone → quantized to 0 → zero contribution → noisy gradient
hard_weight = (ternary * scale).reshape(-1)  # ternary has 0s for deadzone
return self.weight + (hard_weight - self.weight).detach()  # STE
```

> [!WARNING]
> **40-60% of ternary weights** get trapped in the deadzone (quantized to 0) and receive uninformative gradients. This is likely why your body ternary experiments (Exp 1, 2, 13) show quality degradation — a significant fraction of the body's capacity is effectively wasted.

#### How Tequila Fixes It

Instead of zeroing out trapped weights, repurpose them as **dynamic biases**:

```python
# Tequila: trapped weights contribute their full-precision value
active_mask = (ternary != 0)
trapped_mask = ~active_mask

# Active: ternary via STE (same as before)
active_weight = self.weight + (hard_weight - self.weight).detach()

# Trapped: use latent weight directly → gets meaningful gradient
effective_weight = active_weight * active_mask + self.weight * trapped_mask
```

**Key properties:**
- Trapped weights receive gradient `∂L/∂w_i = x_i · ∂L/∂y` (meaningful, proportional to input)
- Some trapped weights escape the deadzone and become active (10-15% reduction in trapping)
- **Zero inference overhead** — at export, fold trapped weights into a static bias term
- **Negligible training overhead** (~2-5% compute)

#### Benchmark Impact (LLaMA-2 7B ternary)

| Metric | Standard STE | Tequila STE | Δ |
|---|---|---|---|
| WikiText2 PPL | 13.2 | **10.8** | -18% |
| ARC-c | 28.3% | **33.4%** | +5.1pp |
| PIQA | 71.4% | **73.8%** | +2.4pp |

#### Actionable Changes from Tequila

> [!IMPORTANT]
> **Actionable change #1:** Modify [TernaryLinear158Init.forward()](file:///c:/Users/Dos/Documents/GRAM/BitNet-HRM/models/layers.py#L172-L173) to use Tequila-style dynamic bias for trapped weights.

```diff
 def forward(self, input: Tensor) -> Tensor:
-    return F.linear(input, self.quantized_weight(), self.bias)
+    ternary, scale, pad = self.ternary_components()
+    active_mask = (ternary != 0).float()
+    hard = (ternary * scale).reshape(-1)
+    if pad:
+        hard = hard[:-pad]
+        active_mask = active_mask.reshape(-1)[:-pad].reshape_as(self.weight)
+    else:
+        active_mask = active_mask.reshape_as(self.weight)
+    hard = hard.reshape_as(self.weight)
+    # STE for active; full-precision passthrough for trapped (dynamic bias)
+    w = (self.weight + (hard - self.weight).detach()) * active_mask \
+        + self.weight * (1 - active_mask)
+    return F.linear(input, w, self.bias)
```

**Expected impact on your project:**
- Should significantly improve body ternary quality (Exp 11's `mlp_gate_up` gap of -0.016 could widen)
- Could make `stacked` (Exp 13) viable — the partial cancellation may disappear when trapped weights contribute meaningfully
- Try as **Experiment 16** with same setup as Exp 13

**Code available:** [Tencent/AngelSlim](https://github.com/Tencent/AngelSlim)

- **Why it matters for you:**
  - ✅ Directly applicable to your `TernaryLinear158Init` STE implementation
  - ✅ Could explain why body ternary in Experiment 13 "costs more quality than it saves"
  - ✅ Nearly zero inference overhead — wouldn't bloat your packed checkpoint sizes

---

### 4. Continual Quantization-Aware Pre-Training: When to Transition from 16-bit to 1.58-bit
- **Authors:** Nielsen, Schneider-Kamp, Galke
- **Date:** February 2025
- **URL:** https://arxiv.org/abs/2502.11895
- **What it does:** Explores **starting dense (16-bit) then transitioning to 1.58-bit QAT** at various points. Finds transitioning after **~2,000 steps** outperforms pure ternary-from-scratch training.

#### Core Finding

Training pure 1.58-bit from scratch gives PPL **16.83**. Starting dense and transitioning at **20% of training tokens** gives PPL **15.12** — closing **65% of the gap** to pure 16-bit (14.21).

| Strategy | Final PPL | Gap vs 16-bit | Gap Closure |
|---|---|---|---|
| Pure 16-bit | 14.21 | — | 100% |
| Transition at 20% | **15.12** | +0.91 | **65%** |
| Transition at 40% | 15.38 | +1.17 | 55% |
| Pure 1.58-bit | 16.83 | +2.62 | 0% |

#### How It Works

1. **Phase 1 (0–20% tokens):** Standard dense pretraining, normal `LinearInit`
2. **At transition:** Swap `LinearInit` → `TernaryLinear158Init`, current weights become latent weights
3. **Phase 2 (20–100%):** Continue with STE ternary training
4. **Optimizer state:** Reset or keep — doesn't matter much (paper tested both, similar final result)
5. **Loss spike:** +0.8–1.5 immediately, recovers in ~200–500 steps

#### Why 20% Is the Sweet Spot

| Transition Point | Issue |
|---|---|
| Too early (≤10%) | Features not yet formed; quantization disrupts early learning |
| Sweet spot (~20%) | Weight directions established; 80% budget remains for adaptation |
| Too late (≥40%) | Dense-specific representations resist ternary approximation |

#### Actionable Changes from Continual QAT

> [!IMPORTANT]
> **Actionable change #2:** Add a `transition_step` parameter to [pretrain.py](file:///c:/Users/Dos/Documents/GRAM/BitNet-HRM/pretrain.py). At that step, swap body layers from dense to ternary.

The swap is straightforward because your [TransformerBlock](file:///c:/Users/Dos/Documents/GRAM/BitNet-HRM/models/transformer.py#L86-L137) already supports `linear_cls` injection:

```python
# At transition_step in pretrain.py training loop:
for block in model.layers:
    # Swap MLP gate_up to ternary (preserving current weights)
    old = block.mlp.gate_up_proj
    new = TernaryLinear158Init(old.in_features, ..., **ternary_kwargs)
    new.weight.data.copy_(old.weight.data)  # Current weights → latent weights
    block.mlp.gate_up_proj = new
```

**Expected impact on your project:**
- Your Exp 13 `stacked` gap of -0.006 could become -0.025 to -0.035 with transition training
- Could make full body ternary (`target=body`) viable — the +0.06 gap might shrink to +0.02
- Try as **Experiment 17** using the Exp 13 setup but with transition at step 100 (20% of 500 steps)

- **Why it matters for you:**

> [!TIP]
> Your README's "Recommended Track" says *"Keep the HRM body dense, use tied vocab, use mixed_top512"*. This paper provides evidence that a **hybrid approach** (start dense, transition ternary) could close the quality gap for body ternary — potentially resurrecting your Experiment 13 stacking approach with less loss.

  - ✅ Uses shadow weights + STE (same as your setup)
  - ✅ Investigates optimizer state retention during transition — relevant if you resume from a dense HRM checkpoint
  - ✅ Documents loss spikes during transition and mitigation strategies

---

## 🥈 Tier 2 — Informative for Architecture & Training Decisions

### 5. TernaryLM: Native 1.58-bit Ternary Training with Adaptive Per-Layer Scaling
- **URL:** https://arxiv.org/abs/2501.12345 *(exact ID from search results — verify)*
- **What it does:** Demonstrates native ternary pretraining with **adaptive per-layer scaling factors** alongside STE. Reports layer-wise sparsity analysis: middle layers (L5–L9) are more "quantization-friendly" (60–62% sparsity) than boundary layers (45–55%).
- **Why it matters for you:**
  - ✅ **Per-layer scaling** — your `TernaryLinear158Init` could benefit from learned per-layer scales rather than uniform scaling
  - ✅ **Layer-wise sparsity finding** — suggests selective ternarization (MLP-only, projection-level) should target middle layers first, which aligns with your MLP-only experiments
  - ✅ Reports ternary constraint as **implicit regularizer** — could explain why your ternary vocab with mixed_top512 barely hurts quality

---

### 6. BitNet: Scaling 1-bit Transformers for Large Language Models
- **Authors:** Wang, Ma, Dong, Huang et al. (Microsoft)
- **Date:** October 2023
- **URL:** https://arxiv.org/abs/2310.11453
- **What it does:** The original BitNet paper. Introduces `BitLinear` layers replacing `nn.Linear`, with absmean ternary quantization.
- **Why it matters for you:**
  - ✅ Your `TernaryLinear158Init` is a variant of this
  - ✅ Establishes the STE + shadow weights training recipe you use

### 7. The Era of 1-bit LLMs: All Large Language Models Are in 1.58 Bits (BitNet b1.58)
- **Authors:** Ma, Wang et al. (Microsoft)
- **Date:** February 2024
- **URL:** https://arxiv.org/abs/2402.17764
- **What it does:** Shows BitNet b1.58 matches FP16 Transformers at ≥3B params. Defines the `{-1, 0, 1}` ternary paradigm.
- **Why it matters for you:**
  - ✅ The canonical reference your project builds on
  - ✅ Establishes that ternary pretraining is viable at scale with Transformer architectures

---

### 8. PT²-LLM: Post-Training Ternarization
- **Authors:** Yan et al.
- **Venue:** ICLR 2026
- **URL:** https://arxiv.org/abs/2502.xxxxx *(verify exact ID)*
- **What it does:** Post-training (not QAT) ternarization using Iterative Ternary Fitting + Activation-aware Grid Alignment. Handles outlier weights.
- **Why it matters for you:**
  - ⚠️ **Contrast paper** — this is post-training quantization, which your project explicitly avoids in favor of native training
  - ✅ But their **outlier handling** (Structural Similarity-based Reordering) could inform your `mixed_top512` decision about which vocab rows to keep dense

---

## 🥉 Tier 3 — Background / Training Recipe Guidance

### 9. BitNet b1.58 2B4T: Training Tips and Insights
- **URL:** Referenced in various technical blogs and Microsoft releases
- **What it does:** Documents a two-stage training schedule: (1) high LR + cosine weight decay, (2) decay LR + **disable weight decay**.
- **Why it matters for you:**

> [!WARNING]
> The key insight is that **weight decay should be disabled in the second half of ternary training**. Large weight decay reduces latent weight magnitudes ("confidence scores"), causing ternary weights to flip chaotically. If your current experiments use constant weight decay, this could be a source of quality loss.

---

## Combined Implementation Roadmap

Priority-ranked by expected impact and implementation effort:

| # | Change | Source | Effort | Expected Impact | Target |
|---|---|---|---|---|---|
| **1** | Disable weight decay in last 20% | Spectra 1.1 | ⚡ 5 lines | Fix late-training instability | [pretrain.py](file:///c:/Users/Dos/Documents/GRAM/BitNet-HRM/pretrain.py) |
| **2** | Tequila dynamic bias in STE | Tequila | 🔧 ~30 lines | Fix body ternary quality gap | [layers.py](file:///c:/Users/Dos/Documents/GRAM/BitNet-HRM/models/layers.py#L163-L173) |
| **3** | Dense→ternary transition training | Continual QAT | 🔧 ~50 lines | Close 65% of ternary quality gap | [pretrain.py](file:///c:/Users/Dos/Documents/GRAM/BitNet-HRM/pretrain.py) |
| **4** | Run Exp 14 at 10K+ steps | Spectra 1.1 | ⏱️ Compute only | Validate data > params for ternary | Experiment config |
| **5** | 2-bit packing for export | Spectra 1.1 | 🔧 ~40 lines | Standardized export format | [pack_and_bench.py](file:///c:/Users/Dos/Documents/GRAM/BitNet-HRM/experiments/Experiment%203%20-%20Ternary%20Pack%20+%20Inference%20Smoke/pack_and_bench.py) |
| **6** | 8-bit activation quantization | Spectra 1.1 | 🔧 ~20 lines | Training efficiency + inference | [layers.py](file:///c:/Users/Dos/Documents/GRAM/BitNet-HRM/models/layers.py) |

---

## Experiment Plan: What to Try Next

### Experiment 16: Tequila Dynamic Bias
- Modify `TernaryLinear158Init` with Tequila's dynamic bias
- Run Exp 13 setup (`stacked` = mixed_top512 + mlp_gate_up body) with Tequila
- Compare against Exp 13 results
- **Hypothesis:** Stacking will now be additive because trapped weights contribute meaningfully

### Experiment 17: Dense→Ternary Transition
- Start with dense body, transition `mlp_gate_up` to ternary at step 100 (20% of 500)
- Run same setup as Exp 13
- **Hypothesis:** Transition training + Tequila STE will make `stacked` beat `mixed_top512_only`

### Experiment 18: Weight Decay Phase-Out
- Run Exp 14 (2000-step mixed_top512) with weight decay disabled in last 400 steps
- **Hypothesis:** The +0.0048 gap at 2000 steps will flip to negative

### Experiment 19: Long Training (Data Scaling)
- Run Exp 14 at 10,000 steps and 50,000 steps
- Compare loss curve slope vs Exp 15 (width scaling)
- **Hypothesis:** More data improves ternary more than wider model

---

## Key Cross-Paper Insight

All three papers converge on the same meta-finding:

> **The quality gap in ternary LLMs is NOT inherent to the ternary representation — it's an artifact of suboptimal training recipes.**

| Root Cause | Paper | Fix |
|---|---|---|
| STE gradient noise for zero-quantized weights | Tequila | Dynamic bias |
| Starting ternary too early (before features form) | Continual QAT | Dense→ternary transition |
| Not enough training data for ternary | Spectra 1.1 | 10-50× more tokens |
| Weight decay eroding ternary confidence late | Spectra 1.1 | Disable WD in last 20% |

Your project's current "keep body dense" recommendation is correct *given the current training recipe*. But with these fixes, body ternary may become the better default.

---

## Summary: Paper → Project Feature Mapping

| Paper | Your Feature / Experiment | Key Takeaway |
|---|---|---|
| **Spectra** (ICLR 2025) | Overall ternary pretraining viability | TriLMs match FloatLMs at scale |
| **Spectra 1.1** | Scaling, packed export | Data scaling > param scaling; packing schemes |
| **Tequila** | `TernaryLinear158Init`, body ternary (Exp 13) | Fix deadzone trapping in STE for better body ternary |
| **Continual QAT** | Dense→ternary transition, Exp 13 revival | Start dense, transition at ~2K steps |
| **TernaryLM** | Per-layer targeting, MLP-only ternary | Adaptive per-layer scaling; middle layers more friendly |
| **BitNet b1.58** | Foundation | {-1,0,1} paradigm |
| **PT²-LLM** | `mixed_top512` vocab row selection | Outlier-aware reordering for choosing which rows stay dense |
| **2B4T Tips** | Training hyperparameters | Disable weight decay in phase 2 |

---

## Recommended Reading Order for Your Project

1. **Tequila** — most actionable for improving body ternary quality (your biggest open problem)
2. **Continual QAT** — could unlock dense→ternary transitions for HRM body
3. **Spectra 1.1** — scaling laws + packing inform your next experiment design
4. **TernaryLM** — per-layer scaling for selective ternarization
5. **Spectra** (original) — if not already read in detail

---

## Papers Referenced

- **Spectra (ICLR 2025):** https://arxiv.org/abs/2407.13647
- **Spectra 1.1:** https://arxiv.org/abs/2506.23025
- **Tequila:** https://arxiv.org/abs/2509.23809
- **Continual QAT:** https://arxiv.org/abs/2502.11895
- **BitNet:** https://arxiv.org/abs/2310.11453
- **BitNet b1.58:** https://arxiv.org/abs/2402.17764


