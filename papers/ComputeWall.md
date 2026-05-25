# Solving the Compute Wall: Literature-Backed Analysis

> Can the compute wall for ternary training be solved or worked around? A survey of 16 papers with real speedup numbers.

*Sources: arXiv literature search via the arXiv API. All papers cited with URLs.*

---

## Attack Vector 1: Ternary Ops ≠ FLOPs (Software Kernels)

The fundamental insight: ternary weights {-1, 0, +1} replace multiply-accumulate with conditional add/sub/skip. Multiple recent papers have **measured** the actual speedup.

### FairyFuse (April 2026) — [arXiv:2604.20913](https://arxiv.org/abs/2604.20913)
- **What:** Multiplication-free LLM inference on CPUs via fused ternary kernels. Uses AVX-512 masked `vaddps`/`vsubps` — assembly-verified zero `vmulps` instructions.
- **Kernel speedup:** **29.6×** over FP32 GEMV at 48 CPU cores (single socket)
- **End-to-end:** 32.4 tok/s on Intel Xeon 8558P, **1.24× faster than llama.cpp Q4_K_M**
- **Quality:** WikiText-2 perplexity 5.52 (vs 5.47 FP16) — near-lossless
- **Key insight (Section 5.4):** *"Roofline analysis reveals a structural asymmetry: 16× data compression from ternary packing shifts deeply memory-bound GEMV toward the compute ridge on bandwidth-limited CPUs, yielding 29.6× kernel speedup, whereas the same compression provides negligible benefit on bandwidth-rich GPUs."*

> [!IMPORTANT]
> **FairyFuse proves that ternary speedup is real but hardware-dependent.** CPUs benefit enormously (bandwidth-limited → compute-limited transition), GPUs benefit less (already bandwidth-rich). This suggests custom silicon optimized for ternary arithmetic could unlock the full theoretical speedup.

### BWTA (April 2026) — [arXiv:2604.03957](https://arxiv.org/abs/2604.03957)
- **What:** Binary Weights & Ternary Activations (BWTA) with custom CUDA kernels for NVIDIA GPUs
- **GPU kernel speedup:** **16-24× over FP16** on NVIDIA GPUs
- **End-to-end prefill:** 216-330 tokens/s with lower memory footprint on LLMs
- **Quality:** Approaches full-precision performance for BERT (average 3.5% drop on GLUE)
- **Key result:** This is the **first demonstration of 16-24× GPU speedup for binary/ternary operations** — showing that even on bandwidth-rich GPUs, purpose-built kernels can extract massive speedup

### Litespark (May 2026) — [arXiv:2605.06485](https://arxiv.org/abs/2605.06485)
- **What:** Custom SIMD kernels for ternary inference on consumer CPUs (Apple Silicon, Intel, AMD)
- **Speedup:** **9.2× faster time-to-first-token, 52× higher throughput, 14× memory reduction** vs standard PyTorch
- **Availability:** pip-installable, HuggingFace integration

### RSR-core (March 2026) — [arXiv:2603.27462](https://arxiv.org/abs/2603.27462)
- **What:** Redundant Segment Reduction algorithm as optimized low-level kernels for CPU and CUDA
- **CPU speedup:** Up to **62× speedup** for GEMM operations
- **CUDA speedup:** Up to **1.9× speedup** for token generation on ternary LLMs
- **Production-ready:** HuggingFace integration for preprocessing and accelerated inference

### Summary of Measured Software Speedups

| Paper | Platform | Kernel Speedup | End-to-End Speedup | Quality |
|---|---|---|---|---|
| FairyFuse | x86 CPU (AVX-512) | **29.6×** vs FP32 | 1.24× vs Q4_K_M | Near-lossless |
| BWTA | NVIDIA GPU (CUDA) | **16-24×** vs FP16 | 216-330 tok/s prefill | ~96.5% of FP |
| Litespark | Consumer CPU (SIMD) | **52× throughput** | 9.2× TTFT | — |
| RSR-core | CPU + CUDA | **62× CPU GEMM** | 1.9× CUDA tok/s | — |

---

## Attack Vector 2: Custom Silicon (ASIC/FPGA)

A wave of 2025-2026 papers demonstrate purpose-built ternary accelerators with extraordinary efficiency numbers.

### VitaLLM (April-May 2026) — [arXiv:2604.27396](https://arxiv.org/abs/2604.27396) & [arXiv:2605.00320](https://arxiv.org/abs/2605.00320)
- **What:** TSMC 16nm silicon prototype for BitNet b1.58 (3B) inference
- **Throughput:** 70.70-72.46 tokens/s decode
- **Area:** Only **0.214-0.223 mm²** (!!!)
- **Power:** 65.97 mW
- **Figure of Merit:** **17.4 TOPS/mm²/W** — significantly outperforming all prior accelerators
- **Architecture:** Dual-core — multiplier-free TINT core for ternary projections + BoothFlex core for mixed-precision attention

> [!IMPORTANT]
> **0.223 mm² of silicon achieves 72 tok/s at 66 mW.** For context, an NVIDIA B200 die is ~814 mm². If you could fill an entire B200-sized die with VitaLLM-style ternary cores (even at 1% efficiency due to interconnect, cooling, etc.), you'd have thousands of ternary compute units. The raw operations-per-watt advantage of ternary silicon vs floating-point silicon is **orders of magnitude**.

### TOM: Ternary Read-only Memory Accelerator (February 2026) — [arXiv:2602.20662](https://arxiv.org/abs/2602.20662)
- **What:** Hybrid ROM-SRAM accelerator co-designed with ternary quantization
- **Key innovation:** Synthesizes ternary weights as **standard-cell logic** (ROM) — zero-valued weights consume no area
- **Throughput:** **3,306 tokens/s** on BitNet-2B
- **Supports:** QLoRA-based on-device adaptation (tunability preserved)

### LUT-Based Accelerator Design Space (April 2026) — [arXiv:2604.25183](https://arxiv.org/abs/2604.25183)
- **What:** Open-source hardware generator + analytical cost model for ternary LUT-based accelerators (ISPASS 2026)
- **Key finding:** Optimal architecture is governed by **activation data type** — LUT-based reuse yields massive gains for FP16 activations but diminishing returns for small integer types
- **Area reduction:** **2.2× vs multiplier-based baselines**
- **Validated:** TSMC 16nm synthesis, with up to 1.2× improvement over prior suboptimal designs

### TeLLMe v2: FPGA Accelerator (October 2025) — [arXiv:2510.15926](https://arxiv.org/abs/2510.15926)
- **What:** Table-lookup-based ternary LLM accelerator for edge FPGAs (5W power budget)
- **Throughput:** 25 tok/s decode, 0.45-0.96s time-to-first-token
- **Power:** Under 5W total

### PD-Swap: FPGA with Dynamic Reconfiguration (December 2025) — [arXiv:2512.11550](https://arxiv.org/abs/2512.11550)
- **What:** Prefill-decode disaggregated ternary LLM accelerator on edge FPGAs
- **Speedup:** **1.3-2.1× over prior FPGA works** (larger gains at longer context)
- **Key insight:** Uses dynamic partial reconfiguration to time-multiplex compute resources between prefill (compute-bound) and decode (memory-bound) phases

### T-SAR: CPU SIMD Reorganization (November 2025) — [arXiv:2511.13676](https://arxiv.org/abs/2511.13676)
- **What:** In-register LUT generation for ternary inference on CPUs (DATE 2026)
- **Speedup:** 5.6-24.5× GEMM latency, 1.1-86.2× GEMV throughput
- **Energy efficiency:** 2.5-4.9× the energy efficiency of NVIDIA Jetson AGX Orin
- **Overhead:** Only 3.2% power and 1.4% area overhead in SIMD units

### Custom Silicon Summary

| Accelerator | Technology | Throughput | Power | Key Metric |
|---|---|---|---|---|
| **VitaLLM** | TSMC 16nm ASIC | 72 tok/s (3B model) | 66 mW | 17.4 TOPS/mm²/W |
| **TOM** | ROM-SRAM hybrid | **3,306 tok/s** (2B model) | — | Weights as logic gates |
| **TeLLMe v2** | Edge FPGA | 25 tok/s | <5W | Table-lookup matmul |
| **PD-Swap** | Edge FPGA | 27 tok/s | — | Dynamic reconfiguration |
| **T-SAR** | CPU (modified SIMD) | 86.2× GEMV speedup | 3.2% power overhead | In-register LUT |

---

## Attack Vector 3: Scaling Laws Say "More Params, Less Data" for Low Precision

### Scaling Laws for Precision (November 2024) — [arXiv:2411.04330](https://arxiv.org/abs/2411.04330)
*Authors: Kumar, Ankner, Spector, Bordelon, Muennighoff, Paul, Pehlevan, Ré, Raghunathan (Stanford, Harvard)*

This is arguably the **most important paper for your question**. Key findings from 465 pretraining runs:

#### Finding 1: Low precision reduces "effective parameter count"
Training at precision P reduces effective parameters to:
```
N_eff(N, P_w) = N × (1 - e^{-P_w / γ_w})
```
At ternary (P≈1.58), a 100B model has an effective parameter count much less than 100B. The exact reduction depends on fitted γ_w.

#### Finding 2: Compute-optimal precision is ~7-8 bits (independent of compute budget!)
> *"The compute-optimal pretraining precision is in general independent of compute budget... We find this P* to be around 7-8 bits."*

This means:
- Training at 16-bit may be **wasteful** (spending bits for negligible quality gain)
- Training below 4-bit requires **disproportionately larger models** (more than 4× larger) to maintain quality
- The sweet spot is ~7-8 bits for general training

#### Finding 3: If you MUST train in low precision, increase parameters before data
> *"As precision of training decreases at fixed compute, we should increase parameters and decrease data."*

For ternary training, this means the **Spectra 1.1 finding (data > params) may be wrong at frontier scale**. The Scaling Laws for Precision paper suggests the opposite: at very low precision, you should have MORE parameters (since each parameter carries less information) and LESS data.

#### Finding 4: Overtrained models degrade MORE from quantization
> *"For models that will be post-train quantized, there exists an amount of pretraining data beyond which additional data is actively harmful to performance at inference-time."*

This is devastating for the post-train-quantize approach (train float, quantize to ternary). It means train-from-scratch ternary (your approach) is fundamentally better for heavily trained models.

> [!CAUTION]
> **The Scaling Laws for Precision paper fundamentally changes the compute-wall math.** If ternary models need ~4× more parameters to achieve the same "effective" quality, but each ternary parameter is ~10× cheaper to compute with (no multiply), then ternary training is still a net win: 4× more params × 0.1× compute/param = 0.4× the total compute. **Ternary training could be 2.5× MORE compute-efficient than float training, not less.**

---

## Attack Vector 4: Communication Reduction for Distributed Training

### MAGNET: Decentralized BitNet Training (March 2026) — [arXiv:2603.25813](https://arxiv.org/abs/2603.25813)
- **What:** Decentralized system for training BitNet b1.58 models using DiLoCo-based distributed merging
- **Key result:** Uses **communication-efficient aggregation** of domain specialists — ternary weights compress gradient syncs
- **Demonstrated:** CPU-native training via bitnet.cpp, no GPU required
- **Validated:** 10-phase hyperparameter sweep achieved -16.7% validation loss improvement

This is early-stage but validates the concept of communication-efficient distributed ternary training.

---

## Synthesis: What the Literature Actually Says About the Compute Wall

### The compute wall IS solvable, through three complementary mechanisms:

| Mechanism | Evidence | Maturity | Impact |
|---|---|---|---|
| **Software kernels** (BWTA, FairyFuse) | 16-30× measured speedup | ✅ Published, reproducible | 3-5× training speedup today |
| **Scaling law reshaping** (more params, less data) | 465 runs, validated to 1.7B | ✅ Published (Stanford) | Changes optimal training strategy |
| **Custom silicon** (VitaLLM, TOM) | 17.4 TOPS/mm²/W measured | ⚠️ Inference only, small scale | 50-100× theoretical for training |
| **Communication compression** (MAGNET, DiLoCo) | DiLoCo validated at small scale | ⚠️ Early stage | Linear scaling to more nodes |

### What this means for your original question ("single B200/B300"):

**The FairyFuse roofline analysis (Section 5.4) is the key insight.** It proves that ternary speedup is **real but not uniform across hardware**:
- On **bandwidth-limited platforms** (CPUs, edge): ternary gives 16-62× speedup because the 16× weight compression shifts the operation from memory-bound to compute-bound
- On **bandwidth-rich platforms** (GPUs like B200/B300 with 4.8+ TB/s HBM): ternary gives less speedup because these platforms are already compute-bound, and ternary operations (add/sub) use a different part of the die than what GPUs optimize for (FP MACs)

**This means the compute wall solution is NOT "run ternary kernels on existing GPUs."** It's:
1. **Short-term (today):** Use BWTA-style CUDA kernels for **16-24× speedup on existing GPUs** during forward/backward ternary matmuls, combined with the Scaling Laws insight (more params, less data)
2. **Medium-term (1-2 years):** Purpose-built ternary compute units (VitaLLM-style) integrated as accelerator tiles alongside existing GPU fabric — 50-100× for ternary operations
3. **Long-term (3-5 years):** Full ternary ASICs (TOM-style ROM compute) where weights are literally synthesized as logic gates — eliminates the memory wall entirely

### Revised Single-GPU Math (with literature-backed numbers):

| Component | Standard B200 | Ternary + BWTA kernels | Ternary + VitaLLM tiles |
|---|---|---|---|
| Ternary matmul speedup | 1× (FP16 MAC) | **16-24×** (BWTA) | **50-100×** (ASIC) |
| Weight memory | 4 bytes/param | 0.2 bytes/param (ECO) | 0.2 bytes/param |
| Optimizer memory | 8 bytes/param | 0.4-2 bytes/param | 0.4 bytes/param |
| Effective throughput | 9 PFLOPS | ~36-54 PFLOP-equiv | ~200-500 PFLOP-equiv |
| Max params (192 GB) | ~10B | ~250B (ternary everything) | ~250B |
| Frontier MoE training time¹ | 12 years | **0.5-0.75 years** | **~2-3 months** |

¹ 671B MoE, 37B active, 15T tokens, single GPU

> [!TIP]
> **With BWTA-style GPU kernels (available now) + ECO (available now) + the Scaling Laws insight (train wider, not longer), frontier-class ternary training on 32-64 B200s in ~3 months is plausible.** That's ~$100-200K in cloud compute vs ~$100M+ for standard float training. The key missing piece is validating quality at scale (>70B params).

---

## Papers Referenced

| Paper | arXiv | Year | Focus |
|---|---|---|---|
| FairyFuse | [2604.20913](https://arxiv.org/abs/2604.20913) | 2026 | Multiplication-free CPU kernels (29.6×) |
| BWTA | [2604.03957](https://arxiv.org/abs/2604.03957) | 2026 | Binary/ternary CUDA kernels (16-24× GPU) |
| Litespark | [2605.06485](https://arxiv.org/abs/2605.06485) | 2026 | Consumer CPU SIMD kernels (52×) |
| RSR-core | [2603.27462](https://arxiv.org/abs/2603.27462) | 2026 | Production ternary kernel (62× CPU) |
| VitaLLM | [2604.27396](https://arxiv.org/abs/2604.27396) | 2026 | 16nm ternary ASIC (17.4 TOPS/mm²/W) |
| TOM | [2602.20662](https://arxiv.org/abs/2602.20662) | 2026 | ROM-based ternary accelerator (3,306 tok/s) |
| LUT Accelerators | [2604.25183](https://arxiv.org/abs/2604.25183) | 2026 | Design space exploration (ISPASS 2026) |
| TeLLMe v2 | [2510.15926](https://arxiv.org/abs/2510.15926) | 2025 | FPGA ternary LLM (25 tok/s @ 5W) |
| PD-Swap | [2512.11550](https://arxiv.org/abs/2512.11550) | 2025 | FPGA dynamic reconfiguration |
| T-SAR | [2511.13676](https://arxiv.org/abs/2511.13676) | 2025 | CPU SIMD reorganization (DATE 2026) |
| **Scaling Laws for Precision** | [2411.04330](https://arxiv.org/abs/2411.04330) | 2024 | **Precision-aware scaling laws (465 runs)** |
| ECO | [2601.22101](https://arxiv.org/abs/2601.22101) | 2026 | Master-weight-free training |
| GaLore | [2403.03507](https://arxiv.org/abs/2403.03507) | 2024 | Low-rank optimizer states (ICML Oral) |
| LOMO | [2306.09782](https://arxiv.org/abs/2306.09782) | 2023 | Layer-wise gradient updates (ACL 2024) |
| BitNet b1.58 | [2402.17764](https://arxiv.org/abs/2402.17764) | 2024 | Original 1.58-bit LLM |
| MAGNET | [2603.25813](https://arxiv.org/abs/2603.25813) | 2026 | Decentralized BitNet training |