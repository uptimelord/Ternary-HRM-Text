# Body Compression Next Steps — Research Synthesis

> Comprehensive analysis combining current experiment results, existing papers, and new arXiv literature search to chart the path forward for compressing the HRM body using 1-bit/1.58-bit/2-bit architectures.

> [!NOTE]
> arXiv literature search was used to find papers cited in this document. All URLs are listed in the [Papers Referenced](#papers-referenced) section. Always check individual paper licenses for restrictions on use.

---

## Current State of the Project

### 2026-05-25 Update

This note is now partially stale:

- Exp 22 promoted `mixed_top512_tequila_L_mlp_gate_up`.
- Exp 23 passed hard-export parity for that combo, so it is now the deploy
  baseline.
- Exp 24 started the 2-bit body path. The h=128, 500-step pilot found the
  strongest signal in `both_attention_gqkv` (`-0.0660 +/- 0.0203` vs dense),
  followed by `both_attention_o` and `both_mlp_gate_up`.
- Exp 25 stacked 2-bit attention on the compressed combo baseline. The h=128,
  500-step pilot found `combo_2bit_attention` at `3.47 MB` (`10.08x`) with
  eval gap `+0.0011 +/- 0.0203` versus the combo baseline.
- Exp 25 confirmation showed the result is size-sensitive: the same stacked
  recipe failed at `h128 / 2000`, but `combo_2bit_attention` passed both h=256
  checkpoints and saved `4.67 MB`.
- Exp 26 passed h=256 export parity for `combo_2bit_attention`: export gap
  `+0.0011 +/- 0.0203`, packed size `9.15 MB`, ternary roundtrip `5.96e-05`,
  and 2-bit roundtrip `3.05e-05`.
- Exp 27 failed the wider-deploy gate. Frozen answer-loss did not fail, but the
  second seed's normal eval gap was `+0.0410 +/- 0.0203`, so the two-seed mean
  eval gap rose above the noise floor.
- Exp 28 tested the less aggressive h=256 `combo_2bit_attention_gqkv` path.
  Normal eval looked fine on seed 1 (`+0.0059 +/- 0.0203`), but frozen
  answer-loss failed hard (`+0.5914 +/- 0.0203`), so the run stopped early by
  the pre-registered kill rule.

The next useful body-compression confirmation is no longer "try 2-bit attention"
in the abstract. The h=256 2-bit attention path is export-clean but not robust
enough for wider deploy, and the narrower gqkv path failed the frozen benchmark.
Do not spend the next run on another attention-only packing check.

```text
keep mixed_top512_tequila_L_mlp_gate_up as deploy baseline
move any next 2-bit body test away from attention, and keep the frozen gate
```

### What Works

| Recipe | Gap vs dense | Packed size | Compression |
|---|---|---|---|
| `mixed_top512` (vocab) | +0.0048 @ 2k steps | 5.11 MB | 6.85× |
| `mixed_top512` @ h=256 | +0.0092 @ 2k steps | — | 4.81× |

**Vocab compression is solved** — `mixed_top512` is the default. The open problem is **body compression**.

### What Failed

| Approach | Experiment | Result | Why |
|---|---|---|---|
| **Stacking** (vocab + body ternary) | Exp 13 | +0.078 gap @ 2k | Interaction penalty; body + vocab ternary hurt each other |
| **Dense→ternary transition** | Exp 17 | +0.023–0.033 gap | Scratch ternary beats transition at this scale |
| **Weight decay phaseout** | Exp 18 | No effect | Too low WD (0.01) and too short (500 steps) |

### What Shows Promise

| Approach | Experiment | Result | Status |
|---|---|---|---|
| **L_mlp_gate_up** (fast-level MLP only) | Exp 21 | **−0.035** (beats dense!) | 🔬 Needs 5k-step confirmation |
| **Tequila STE** | Exp 16, 19, 20 | Consistent small gains; export parity verified | ✅ Training candidate |
| **both_mlp_gate_up** | Exp 21 | **−0.024** | Promising, but H-level adds less than L-level |

> [!IMPORTANT]
> **The breakthrough finding from Experiment 21** is that body ternary quality is not uniform across HRM levels. The **fast L-level MLP gate/up projections** are the best target — they actually **beat** dense at 500 steps. This is the first clear positive signal for body compression.

---

## Sensitivity Map from Experiment 21

```
Best ◄─────────────────────────────────────────────────────► Worst

L_mlp_gate_up   both_mlp_gate_up   both_mlp_down   both_attn_o   H_mlp_gate_up   both_attn_gqkv
   -0.035           -0.024            -0.006         +0.004        +0.009          +0.026
```

**Key insight:** The L-level (fast recurrence) is far more ternary-friendly than the H-level (slow recurrence). Attention layers resist ternary.

This aligns with **TernaryLM**'s finding that middle layers are more "quantization-friendly" (60-62% sparsity vs 45-55% at boundaries), and with the broader literature on **anchor layers** — keeping sensitive layers at higher precision while aggressively quantizing robust ones.

---

## New Papers from arXiv Search — Directly Relevant to Body Compression

### 1. Sparse-BitNet (March 2026) — arXiv:2603.05168

**What:** Combines 1.58-bit ternary quantization with semi-structured N:M sparsity (6:8 pattern) for additional compression.

**Why it matters for body compression:**
- Ternary models already have ~42% intrinsic zero weights → natural compatibility with sparsity
- 6:8 sparsity on top of ternary gives additional 1.3× speedup with minimal quality loss
- Could be applied **selectively** to the L_mlp_gate_up target (which already tolerates ternary)

> [!TIP]
> **Proposed Experiment:** After confirming L_mlp_gate_up at 5k steps, add 6:8 sparsity to those layers. This would compound: ternary (1.58 bits) × sparsity (6/8 utilization) ≈ **1.19 effective bits/param** for those layers, with hardware-acceleratable sparse tensor core operations.

### 2. BitNet v2 (2025) — H-BitLinear with Hadamard Transformation

**What:** Applies online Hadamard transformation to activations before quantization, enabling native 4-bit activation quantization alongside 1.58-bit ternary weights.

**Why it matters for body compression:**
- Your body ternary experiments quantize only **weights** — activations remain BF16
- Activation outliers are a known problem for ternary layers (especially in attention)
- H-BitLinear smooths spiky activation distributions, potentially making **attention layers ternary-viable**
- Could unlock `both_attn_gqkv` (currently +0.026 gap) by solving the activation outlier problem

> [!IMPORTANT]
> **Proposed Experiment:** Apply Hadamard normalization to activations in `both_attn_gqkv` layers and re-run Exp 21's sensitivity map. If the gap drops from +0.026 to near zero, this unlocks full-body ternary.

### 3. SURGE: Surrogate Gradient Adaptation (ICML 2026) — arXiv:2605.10989

**What:** Replaces STE with a dual-path gradient compensator (DPGC) for binary/ternary networks. A parallel full-precision auxiliary branch provides less biased gradient estimates.

**Why it matters for body compression:**
- Your STE implementation (even with Tequila) is known to introduce gradient bias
- SURGE provides **norm-based adaptive gradient scaling** between binary and full-precision branches
- Could improve convergence speed and final quality for body ternary layers
- Specifically designed for from-scratch training (matches your approach)

> [!TIP]
> SURGE is a **gradient estimator upgrade** — it sits at the same level as Tequila but is more principled. If Tequila gives 0.004–0.007 improvement (Exp 16), SURGE could give more. Test as a drop-in replacement for your STE in `L_mlp_gate_up`.

### 4. QuEST: Quantized Efficient Stable Training (ICML 2025)

**What:** Trust gradient estimator + Hadamard normalization for stable QAT down to W1A1. Identifies 4-bit as Pareto-optimal for accuracy vs inference cost.

**Why it matters for body compression:**
- The trust gradient estimator **explicitly minimizes error** between quantized and true gradients (vs STE which ignores the error)
- Could replace your STE for body layers specifically
- Hadamard normalization synergizes with BitNet v2's approach for activation handling
- GPU kernel support available at [IST-DASLab/QuEST](https://github.com/IST-DASLab/QuEST)

### 5. ZeroQAT (August 2025) — Backpropagation-Free QAT

**What:** Uses zeroth-order optimization (forward-pass only) for QAT. Eliminates activation gradient storage.

**Why it matters for body compression:**
- Dramatic VRAM reduction during body ternary training
- Could enable ternarizing **all** body layers simultaneously (currently limited by activation memory)
- Particularly interesting for your "4 GB VRAM training" goal from LowVRAMTernaryTraining.md
- Works well in W4A4 — needs testing for W1.58A-anything

### 6. Sub-1-Bit: NanoQuant & BTC-LLM (2026)

**NanoQuant** (arXiv, Feb 2026): Low-rank binary factorization for PTQ down to sub-1-bit. Compresses 70B to 5.35 GB (fits 8 GB GPU).

**BTC-LLM** (arXiv, Apr 2026): Binary pattern clustering with learnable transformations. LLaMA-2-13B at **0.8 bits** with only 3.1% accuracy drop, 1.6× speedup over FP16.

**Why they matter:**
- These are post-training methods, but the **binary codebook** idea from BTC-LLM could inspire a native training approach
- If body weights can be represented in <1 bit (via redundancy exploitation), the compression ratio goes beyond ternary's theoretical limit
- Your `mixed_top512` approach already uses a form of mixed precision — sub-1-bit for the tail + dense for the top rows

> [!WARNING]
> Sub-1-bit native training is **unexplored territory**. No paper has demonstrated from-scratch training at <1 bit for LLMs. This is high-risk/high-reward and not recommended for the immediate next experiments.

---

## Synthesis: Three Paths to Body Compression

### Path A: Selective Ternary (Low Risk, Near-Term) ✅

Build on Exp 21's success with `L_mlp_gate_up`:

```
Step 1: Confirm L_mlp_gate_up at 5k steps (next experiment)
Step 2: Add Tequila STE → L_mlp_gate_up_tequila
Step 3: Add 6:8 sparsity (Sparse-BitNet) for additional compression
Step 4: Gradually expand: L_mlp_down → L_mlp_all → both_mlp_gate_up
```

**Expected outcome:** 20-40% of body params at 1.58 bits + 6:8 sparsity ≈ 1.19 bits for those params. Combined with `mixed_top512` vocab, total model compression could reach 3-5× vs dense.

### Path B: Better Gradient Estimators (Medium Risk, Medium-Term) 🔬

Replace STE with more principled approaches:

```
Step 1: Implement SURGE dual-path gradient compensator for body ternary layers
Step 2: Compare SURGE vs Tequila vs QuEST on L_mlp_gate_up
Step 3: Apply winner to expand ternary targets (attention, H-level)
Step 4: Combine with BitNet v2's Hadamard activation normalization
```

**Expected outcome:** Better gradient estimation could unlock previously-failed targets (stacking, attention layers). Could make full-body ternary viable.

### Path C: 2-Bit Body (Low Risk, Orthogonal) 🆕

Instead of ternary {-1, 0, +1}, use 2-bit weights {-1, -⅓, +⅓, +1}:

```
Step 1: Implement 2-bit linear layer with STE
Step 2: Run Exp 21's sensitivity map with 2-bit instead of 1.58-bit
Step 3: If gaps are smaller, 2-bit body + ternary vocab = good tradeoff
```

**Why 2-bit:**
- Only 0.42 bits more than ternary (2.0 vs 1.58)
- 4 levels instead of 3 → much less quantization noise
- Simple packing: 16 weights per uint32 (same as Spectra's 2-bit scheme)
- Could close the stacking penalty entirely — the body gets just enough extra precision
- BitNet a4.8 and Spectra 1.1 both show 2-bit packing works well

> [!IMPORTANT]
> **Path C (2-bit body) is the most underexplored and potentially highest-value next step.** Your stacking failure (Exp 13) and the H-level sensitivity (Exp 21) both suggest that 1.58 bits is slightly too few for robust body compression — but 2 bits might be just right. The literature supports this: the Scaling Laws for Precision paper (arXiv:2411.04330) shows effective parameter count drops sharply below ~2 bits.

---

## Prioritized Experiment Roadmap

| Priority | Experiment | What | Based On | Risk |
|---|---|---|---|---|
| **1** | **Exp 22: L_mlp_gate_up 5k confirm** | Run `dense` vs `L_mlp_gate_up` vs `both_mlp_gate_up` at 5000 steps with Tequila STE | Exp 21 | Low |
| **2** | **Exp 23: 2-Bit Body Sweep** | Implement `TwobitLinear` with 4-level quantization. Run Exp 21 sensitivity map with 2-bit instead of 1.58-bit | Scaling Laws for Precision, Spectra 1.1 | Low |
| **3** | **Exp 24: 2-Bit Body + Ternary Vocab Stack** | If 2-bit body works, stack with `mixed_top512` vocab | Exp 13 (retry with 2-bit body) | Medium |
| **4** | **Exp 25: SURGE Gradient Estimator** | Replace STE/Tequila with SURGE for `L_mlp_gate_up`, compare | SURGE (ICML 2026) | Medium |
| **5** | **Exp 26: Sparse-BitNet L_mlp** | Add 6:8 semi-structured sparsity to confirmed L_mlp_gate_up target | Sparse-BitNet | Medium |
| **6** | **Exp 27: Hadamard Activation + Attention Ternary** | Apply BitNet v2's H-BitLinear to attention layers, re-run sensitivity | BitNet v2 | Medium-High |
| **7** | **Exp 28: QuEST Trust Gradient** | Replace STE with QuEST's trust gradient estimator for body ternary | QuEST (ICML 2025) | Medium |

---

## Key Cross-Paper Insights for Body Compression

### Why Body Ternary Has Been Hard (Root Causes)

| Root Cause | Evidence | Fix |
|---|---|---|
| **STE gradient bias** | Tequila shows 40-60% weights trapped in deadzone | SURGE or QuEST gradient estimators |
| **Activation outliers** in attention | Exp 21: `both_attn_gqkv` gap +0.026 | BitNet v2 Hadamard normalization |
| **H-level is more sensitive** than L-level | Exp 21: H_mlp_gate_up +0.009 vs L_mlp_gate_up −0.035 | Selective targeting (L-level first) |
| **1.58 bits may be too few** for body layers | Stacking penalty in Exp 13 | 2-bit body weights |
| **Intrinsic sparsity** is untapped | Ternary models have ~42% zero weights naturally | Sparse-BitNet 6:8 pattern |

### The "Just Right" Compression Hypothesis

The emerging picture from experiments + literature:

```
┌─────────────────────────────────────────┐
│        Compression Sweet Spot           │
├─────────────────────────────────────────┤
│ Vocab embedding: 1.58-bit ternary      │ ← Solved (mixed_top512)
│ L-level MLP:     1.58-bit ternary      │ ← Confirmed (Exp 21)
│ H-level MLP:     2-bit (proposed)      │ ← Test in Exp 23
│ Attention:       Dense (for now)       │ ← Needs Hadamard activation
│ Embedding head:  Dense top-512 rows    │ ← Solved (mixed_top512)
└─────────────────────────────────────────┘
```

This mixed-precision body strategy assigns precision by sensitivity, not uniformly — exactly what TernaryLM's adaptive per-layer scaling and the anchor-layer literature recommend.

---

## Papers Referenced

### Already in `/papers/` (Project Literature)
- **Spectra** (ICLR 2025): https://arxiv.org/abs/2407.13647
- **Spectra 1.1** (June 2025): https://arxiv.org/abs/2506.23025
- **Tequila** (Tencent, 2025): https://arxiv.org/abs/2509.23809
- **Continual QAT** (Feb 2025): https://arxiv.org/abs/2502.11895
- **BitNet** (Oct 2023): https://arxiv.org/abs/2310.11453
- **BitNet b1.58** (Feb 2024): https://arxiv.org/abs/2402.17764
- **Scaling Laws for Precision** (Nov 2024): https://arxiv.org/abs/2411.04330
- **ECO** (Jan 2026): https://arxiv.org/abs/2601.22101
- **FairyFuse** (Apr 2026): https://arxiv.org/abs/2604.20913
- **BWTA** (Apr 2026): https://arxiv.org/abs/2604.03957

### New from arXiv Search
- **Sparse-BitNet** (Mar 2026): https://arxiv.org/abs/2603.05168
- **SURGE** (ICML 2026): https://arxiv.org/abs/2605.10989
- **QuEST** (ICML 2025): https://github.com/IST-DASLab/QuEST
- **BTC-LLM** (Apr 2026): Sub-1-bit binary pattern clustering
- **NanoQuant** (Feb 2026): Low-rank binary factorization PTQ
- **ZeroQAT** (Aug 2025): Backpropagation-free QAT
- **BitNet v2** (2025): H-BitLinear with Hadamard activation normalization
- **BitNet a4.8** (2024/2025): 4-bit activation + 1-bit weight hybrid
- **TernaryLM** (2026): Adaptive layer-wise scaling for ternary LLMs
- **PST — Polynomial Surrogate Training** (Mar 2026): https://arxiv.org/abs/2603.00302
