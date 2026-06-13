# Training Efficiency Brief — GPU-hours & Max-Scale Envelope on 4 GB

**Date:** 2026-06-10
**Scope:** Objective 1 — verified capability per GPU-hour at fixed config. Objective 2 — largest trainable config on RTX 3050 Ti for the endgoal pretrain. Research only; no code changed.
**⚠ Run order ≠ ID order.** Canonical schedule: [`EXECUTION_ORDER.md`](EXECUTION_ORDER.md). Note this brief is *split* there: enablers **Exp85 (AMP) + Exp86 (chunked CE)** run early (Tier 1, critical path); the scale ladder **Exp87–89** stays parked (Tier 5, gated).
**Companion:** `Architecture Research Brief - 2026-06-10.md` (intelligence per packed MB). This brief supplies the training envelope that gates its C4/C5 and the contention rule for C2/C3.
**Inputs:** measured peaks from Exp29/30/38/66/67/69/70/72/77/78 results files, `config/arch/*`, `papers/LowVRAMTernaryTraining.md`, literature sweep 2024–2026.

---

## 1. Executive summary

- **Best efficiency move (fixed config): turn on what already exists, then kill the logits tensor.** `--amp` is implemented in Exp30/69 runners, measured at **1.70× faster**, and *not used in any baseline run*. 8-bit Adam (Exp72) is measured at **−112.7 MB (−14.7%)** with the parity gate pending. Neither is research; both are switch-flips.
- **The hidden activation hog is the vocabulary head, not the body.** vocab=65,536 at FP32 means one logits materialization at batch 16 is ~537 MB; CE backward multiplies it. This is why Exp69 (batch 16) peaked at 2,221 MB while batch-4 runs sit near 767 MB. **Chunked/fused cross-entropy is the single highest-leverage new technique** — it unlocks batch 32–64 at h256, which is the cheapest path to tokens/hour.
- **Best scale move:** AMP + 8-bit Adam + chunked CE + gradient checkpointing puts **h512×6 (~59M params) comfortably under 3.5 GB**; even h1024×12 (size B, ~268M) is borderline-feasible at batch 1–4 with Tier 3. VRAM is *not* the binding constraint.
- **The real cliff is wall-clock, not VRAM.** Measured throughput: 6,387 tok/s (fp32, batch 16, h256×2). The GPU runs at <25% VRAM in every standard lane today. A 150M+ model trains at maybe 2–4k tok/s — ternary-optimal data budgets (D/N ≈ 200–300×, Spectra 1.1) are months away at that rate. Params are cheap to *fit* and ruinous to *feed*.
- **Recommended philosophy: C now, B as north star, A capped as one gated probe.** Current pretrains are massively undertrained (25.6M tokens at h256 vs any reasonable optimum) — the marginal token beats the marginal parameter. Speed up the current lanes 3–5× first (C), keep deploy-scale training as default (B), and run mode A only as a pre-registered ladder probe (Exp88) capped at h512×6 — promoted *only* if it beats h256 on the strict bridge at matched wall-clock.
- **Direct answer — largest laptop pretrain to plan for:** primary spec **h512×6, ~59M params, seq 128, batch 16×accum 2, AMP bf16 + 8-bit Adam + chunked CE + checkpointed cycles, predicted peak ~2.6–3.2 GB, ~300M tokens in a 15–20 h weekend run**. Conservative fallback: **h384×6 (~39M), ~1.8–2.2 GB, overnight class**. Both export through the existing Phase 0 recipe; deploy preset is unchanged.
- ECO (master-weight elimination) is real ([arXiv 2601.22101](https://arxiv.org/abs/2601.22101), Google/ISTA, proven to 2.1B) but only matters above ~100M params on our budget — file as a Tier-3 spike, gated on mode A actually being chosen.
- Exp79's generate+verify overhead is VRAM-trivial (inference at h256 + ms-class verifiers); its cost is wall-clock per step, linear in K — Exp81's breadth curve (Architecture brief C2) is the right tool to pick K, not VRAM analysis.
- Roadmap: **P0 finish Exp72 gate → Exp85 AMP parity gate → Exp86 chunked CE → Exp87 checkpointed cycles → Exp88 scale ladder → (gated) Exp89 ECO spike.** Each has a numeric promote/kill bar (§6).
- Architecture-brief coordination: this envelope **unblocks C5** (TRM pretrain is tiny; use the Tier-1.5 stack), **covers C4** without DiffusionBlocks up to bp_steps ≈ 16 via checkpointing, declares **C3's HyperBuilder negligible** (~26 MB train-side), and sets the **C2 contention rule**: serialize inference sweeps with training runs; don't co-schedule.

---

## 2. Constraint envelope

| Field | Value | Source |
|---|---|---|
| GPU | RTX 3050 Ti laptop, 4,096 MB | repo-wide |
| Hard cap (5% margin) | peak_vram_mb ≤ **3,500** working target | this brief |
| Measured peaks today | Exp29 pretrain 953.8 · Exp30 SFT 739.4 · Exp70 766.7 · Exp69 (b16) **2,221.3** · Exp72 8-bit 654.0 | results files per experiment |
| Throughput today | 6,387 tok/s fp32 b16; `--amp` = 1.70× (measured, unused in baselines) | Exp69/Exp30 READMEs |
| Wall-clock anchors | 50k steps h256 b4 = 128.8 min (Exp29); 18k steps b16 = 96.2 min (Exp69); 8k steps b4 = 12.9 min (Exp70) | results files |
| Train dtype today | FP32 everywhere; **no gradient checkpointing in any runner** (grep-verified) | agent sweep |
| Quality gates | strict pass@1 parity / eval loss within ±0.0203 @ 5k; invalid 0% | DISCIPLINE.md |
| Deploy preset | `mixed_top512_tequila_L_mlp_gate_up` ~4.64 MB — **unchanged, lane closed** | PHASE0_DEPLOY_PRESET.md |
| Local model family | h128–h256 × 2–4 layers, vocab 65,536 tied, seq 128, HRM H=2/L=3, bp_steps 2–5 | Exp29/34/69 runners, `config/arch/net/hrm.yaml` |

Note: upstream `config/arch/size/{B,L,XL,XXL}` (12–72 layers, h1024–2560) are datacenter presets; the local lane has its own h128/h256 configs in experiment runners. The envelope below maps both.

---

## 3. VRAM bottleneck diagnosis (evidence-linked)

**Calibration.** h256×4, vocab 65,536 tied ⇒ 19.79M params (Exp29). FP32 AdamW states = weights 79 MB + grads 79 MB + m/v 158 MB ≈ **317 MB**. Measured SFT peak 767 MB ⇒ activations+logits ≈ **450 MB** at batch 4. At batch 16 (Exp69, h256×2, states ≈ 290 MB): peak 2,221 MB ⇒ activations ≈ **1,930 MB**. Activation cost scales ~linearly with batch at ~120 MB per batch-unit — and the dominant term is the **vocab head**: logits (b×128×65,536×4 B) = 134 MB at b4, 537 MB at b16, with CE forward+backward holding 2–3× that. Body activations (h256, 4 layers, seq 128, bp_steps 2) are a minor term.

**Ranked bottlenecks (today's lanes, h256-class):**

1. **Logits/CE activations** — dominant at any batch ≥ 8. Mechanism: FP32 × 65,536 vocab. Fix: chunked CE (§6 Exp86) + AMP. This is what blocks batch scaling, which is what blocks tokens/hour.
2. **Optimizer states (m, v)** — 158 MB at 20M params; grows linearly and becomes dominant above ~100M params (1.6 GB at 100M). Fix: 8-bit Adam (measured, Exp72) → ECO/APOLLO at A-scale.
3. **FP32 master weights + grads** — 158 MB at 20M; 1.2 GB at 150M. Fix: ECO at A-scale only; irrelevant below ~60M.
4. **BPTT/recurrence activations** — linear in effective unrolled depth (bp_steps × cycle layers). At today's bp_steps=2–5, h256: small. At C4-style deep recurrence or h512+, it matters — checkpointing covers it (§6 Exp87); DiffusionBlocks not needed at this scale.
5. **Vocab/embed weight tensor** — 16.8M of 19.8M params *is* the vocab. At train time it's FP32 (67 MB) regardless of ternary forward. Exp5b's Triton packed path saves ~32 MB but at 0.4× speed — wrong trade for training.
6. **Exp79 generate+verify** — inference-mode forward (no_grad) + ms-class verifier: est. < 200 MB transient on top of training state. Wall-clock cost linear in rollouts K; not a VRAM problem.

**Headline:** every standard lane uses < 25% of the card. The project has been paying a 1.7× speed tax (no AMP) and a 4× batch ceiling (full-vocab CE) for free.

---

## 4. Technique stack tiers

Cumulative VRAM predicted at **h256×4, seq 128, batch 16** (calibrated to measured 2,221 MB Exp69-class baseline; body slightly larger here so figures rounded up):

| Tier | Stack | Peak VRAM est. | Steps/hour effect | Effort | Strict-eval risk |
|---|---|---|---|---|---|
| 0 | FP32 AdamW (today) | ~2,350 MB | 1.0× | — | — |
| 1 | + 8-bit Adam (Exp72) | ~2,270 MB | ~0.8–1.0× (20% slower observed at b4) | **done, gate pending** | Low — LR-tunable parity already shown at 3k |
| 1.5 | + AMP bf16 autocast | ~1,300 MB | **~1.7×** (measured flag) | S (flag exists) | Low–med — STE/ternary quant under autocast must pass parity; Exp72 memory says no STE-latent blowup with optimizer changes, AMP needs its own gate |
| 2 | + chunked/fused CE (vocab 65,536) | ~700–800 MB | ~1.0× per step, but **unlocks batch 32–64 ⇒ 2–4× tokens/hour** | M (~30 lines manual chunking; or Liger-style fused kernel) | Low — numerically equivalent up to reduction order |
| 2.5 | + gradient checkpointing on H/L cycle blocks | ~550–650 MB (b16) — or same budget at **h512×6 b8–16** | 0.7–0.8× per step (recompute) | M (~20 lines, `torch.utils.checkpoint` around cycle bodies) | Low — exact gradients if RNG handled |
| 3 | + ECO (no FP32 master weights) | states −4 B/param; matters ≥100M params | ~1.0× | L (~100 lines into `TernaryLinear158Init`, per papers/ survey) | **Med** — error-feedback changes optimization; needs its own 2-seed parity gate |
| 4 | APOLLO-Mini / GaLore / LOMO | states → near-SGD | varies | L | Med–high; only if A-scale chosen and Tier 3 insufficient |

Killed in one line: **ternary momentum** (unvalidated at LM scale; we're not state-bound below 100M) · **LOMO** (drops Adam moments; HRM small-model training is noise-sensitive) · **activation offloading** (laptop PCIe; checkpointing dominates per 2024 lit) · **Triton packed matmul in training** (0.4× speed for 32 MB) · **Muon/Stable-SPAM** (perplexity plays, not memory plays) · **DiffusionBlocks** (solves deep-BPTT VRAM we don't have; revisit only if C4 needs bp_steps ≫ 16).

---

## 5. Max-scale envelope

**Parametric model** (vocab 65,536 tied, expansion 4): params ≈ 65,536·h + n_layers·16·h². States: FP32 AdamW 16 B/param → T1 10 B → T1.5+T1 ~10 B → T3 ~5–6 B. Activations: body ≈ c·b·seq·h·depth_eff (checkpointable to ~⅓); logits ≈ 2.5×(b·seq·65,536·bytes) unless chunked.

**OOM cliff chart** (fits = predicted peak ≤ 3,500 MB; **calibrated** at h256 row, extrapolated elsewhere — flag: ±20% uncertainty, must be re-measured by Exp88):

| Config (params) | T0 b4 s128 | T1.5 b16 s128 | T2 (chunked CE) b16 s128 | T2.5 b4 s512 | T3 b4 s128 |
|---|---|---|---|---|---|
| h256×4 (19.8M) | ✅ 0.77 GB meas. | ✅ ~1.3 GB | ✅ ~0.8 GB | ✅ ~1.2 GB | ✅ |
| h384×6 (38.9M) | ✅ ~1.3 GB | ✅ ~1.9 GB | ✅ ~1.2 GB | ✅ ~1.8 GB | ✅ |
| h512×6 (58.8M) | ✅ ~1.9 GB | ⚠️ ~2.9 GB | ✅ ~1.7 GB | ✅ ~2.4 GB | ✅ |
| h768×8 (125.8M) | ⚠️ ~3.3 GB | ❌ | ⚠️ ~3.0 GB | ⚠️ ~3.3 GB | ✅ ~2.2 GB |
| h1024×12 / size B (268M) | ❌ ~6 GB | ❌ | ❌ ~4.6 GB | ❌ | ⚠️ ~3.2 GB b1–2 |
| L/XL/XXL (0.7B+) | ❌ | ❌ | ❌ | ❌ | ❌ — out of laptop class at any tier |

Grad-accum column omitted deliberately: accumulation is VRAM-free (grads in place) and trades wall-clock only — use accum 2–32 freely at any cell above.

**Wall-clock (the honest constraint).** Throughput anchors: 6.4k tok/s fp32 b16 h256×2 (measured); AMP ⇒ ~11k; chunked CE batch 32 ⇒ est. 15–20k tok/s at h256. Larger bodies scale per-token cost roughly with body FLOPs (vocab head amortizes):

| Run | Tokens | Est. wall-clock |
|---|---|---|
| 5k-step probe, h256 b4 | 2.6M | **~8 min** (measured class) |
| 50k pretrain, h256 b4 fp32 (today) | 25.6M | **2.1 h** (measured) |
| Endgoal-B: h256, T2 stack, b32 | 500M | **~7–10 h (overnight)** |
| Endgoal-A primary: h512×6, T2+2.5, b16×accum2 | 300M | **~15–20 h (weekend)** ±50% |
| h768×8 at ternary-optimal D/N (≥10B tok) | — | **months — not plannable** |

**Recommended laptop-max spec (primary):**

```text
config:        local_h512x6 (hidden 512, 6 layers, heads 4×128, vocab 65536 tied, HRM H=2 L=3, bp_steps ≤5)
params:        ~58.8M
seq/batch:     128 × 16, grad_accum 2 (effective 4096 tok/step)
stack:         AMP bf16 + 8-bit Adam + chunked CE + checkpointed H/L cycles (Tier 2.5)
peak_vram:     predicted 2.6–3.2 GB (MUST be measured at 500 steps before committing; kill >3.5 GB)
wall-clock:    ~300M tokens ≈ 15–20 h (weekend run, checkpoint_interval small, resumable)
deploy path:   Phase 0 recipe applies unchanged (mixed_top512 vocab + L_mlp gate_up);
               packed est. ~15–25 MB at h512 — this is a CAPABILITY-DISCOVERY artifact,
               not a deploy candidate; deploy north star stays h128/h256 (~4.64 MB preset)
```

**Conservative fallback:** h384×6, ~39M, Tier 1.5+2 only (no checkpointing), predicted ~1.8–2.2 GB, overnight class.

**Philosophy verdict (A/B/C).** The default hypothesis ("A for endgoal") survives only in capped form. Evidence: (i) VRAM headroom is real but wall-clock makes ≥100M-param ternary-optimal training non-plannable on this card; (ii) current checkpoints are undertrained in tokens, not starved of params (Exp69's 83×-token win is the repo's single biggest capability jump; HRM's design point is sample efficiency); (iii) Architecture brief C5 pushes params *down*, not up. **Recommendation: C immediately (Tier 1–2 on existing lanes, 3–5× tokens/hour), B as the default for all capability work, A as exactly one pre-registered ladder probe (Exp88) capped at h512×6 — and if its strict bridge doesn't beat h256 at matched wall-clock, A is dead on this hardware and we say so loudly.**

---

## 6. Candidate experiments

**P0 — Finish Exp72 8,000-step heldout gate** *(implementation priority 0)*
Question: does 8-bit Adam hold strict parity at the full Exp70 gate? Mechanism: m/v INT8 (−112.7 MB measured). Runner: Exp72's existing harness, 8k steps. Decision Rule (already pre-registered in Exp72 README): Promote if ≥10% VRAM saving AND heldout logic parity at 8k. Literature agrees this is safe (bitsandbytes production-grade). Blocks: T1 inclusion in every later stack.

**Exp85 — AMP bf16 parity gate**
Question: does `--amp` (already implemented, 1.70× measured) hold strict eval parity with the ternary STE path? VRAM mechanism: bf16 activations ≈ −45% activation memory + speed. Predicted peak: 767 → ~550 MB at b4. Runner: Exp70 8k logic SFT, `--amp` on/off, 2 seeds. **Promote if** logic easy/hard pass@1 within 2 pp of fp32 on both seeds AND ≥1.5× steps/hour AND invalid 0%. **Kill if** strict drop > 2 pp on any seed (then AMP stays eval-only). Unblocks: every wall-clock number in §5.

**Exp86 — Chunked cross-entropy**
Question: does chunking the 65,536-vocab CE (e.g., 8 × 8,192-token chunks, logits never fully materialized) cut peak VRAM enough to double batch? Mechanism: kills the dominant activation term (537 MB logits at b16). Predicted: b16 peak 2,221 → ~1,500 MB (fp32) / ~900 MB (with AMP); enables b32–64. Runner: Exp69 word SFT, 2k-step slice, loss-curve overlay vs unchunked. **Promote if** peak_vram −≥30% at b16 AND loss curve identical within fp tolerance AND strict word heldout parity on 1 seed (numerics-only change). **Kill if** step-time overhead > 15%. Unblocks: Endgoal-B overnight spec; biggest tokens/hour lever.

**Exp87 — Checkpointed H/L cycles**
Question: does `torch.utils.checkpoint` around cycle bodies buy h512-class training under 3.5 GB at ≤1.35× step time? Mechanism: recompute cycle activations in backward; memory ∝ √depth-ish instead of linear in bp_steps × layers. Runner: Exp29 pretrain runner, h256 control + h512×6, 500 steps, measure peak + step time. **Promote if** h512×6 b8 fits < 3.0 GB with ≤1.35× step-time vs no-checkpoint AND loss bitwise-class identical. **Kill if** recompute overhead > 1.5×. Unblocks: Exp88 upper rungs; answers C4's deep-BPTT question (covers bp_steps ≈ 16 without DiffusionBlocks).

**Exp88 — Max-scale ladder probe (mode-A test)**
Question: does any config above h256 beat h256 on quality per wall-clock-hour under the Tier-2.5 stack? Ladder: h384×6 → h512×6 (each: 5k-step pretrain, 2 seeds, measure peak_vram_mb, tok/s, pretrain eval loss; then Exp64-style strict arithmetic bridge). **Promote (mode A lives) if** a rung beats h256 matched-wall-clock on the strict bridge by > noise AND peak < 3.5 GB. **Kill (mode A dies on this card) if** no rung beats h256 at matched hours — record loudly per the brief's mandate. Depends on: Exp85+86+87. Gates: the §5 laptop-max spec; also informs C5's pretrain budget.

**Exp89 — ECO feasibility spike** *(gated on Exp88 promoting and a >100M ambition)*
Paper-only first: map ECO's update rule onto `TernaryLinear158Init` master-weight path (~100 lines per papers/LowVRAMTernaryTraining.md), estimate interaction with Tequila STE and EMA buffer. **Go if** Exp88 shows scale pays and Tier 2.5 caps out; **No-go if** B/C verdict stands — below ~60M params ECO saves <250 MB, not worth optimizer-dynamics risk.

---

## 7. Ranked roadmap

```text
NOW      P0  Exp72 8k parity gate          — stack unchanged, decision rule pre-registered
NEXT     Exp85 AMP gate                    — cheapest speed 1.7×, flag already in runners
         Exp86 chunked CE                  — batch ceiling 4×; pairs with Exp85
THEN     Exp87 checkpointed cycles         — h512 enabler; measured peak on 500-step probe
GATED    Exp88 scale ladder                — the A/B/C verdict, pre-registered kill
GATED    Exp89 ECO spike                   — only if Exp88 promotes scale
BEFORE   Architecture C5 (TRM pretrain)    — use Tier 1.5+2 stack; envelope: trivial fit
                                             (TRM-class ≤10M params, est. < 1 GB peak)
PARALLEL C2/Exp81 breadth sweep            — inference-only ~0.6 GB; SERIALIZE with training
                                             runs (no co-scheduling on 4 GB), fill idle gaps
```

Coordination answers for the Architecture brief: **C5 unblocked** (any tied-block config in its plan fits; spec above). **C4** — deep BPTT fits via Exp87 checkpointing up to ~16 unrolled steps at h256–h512; DiffusionBlocks stays parked. **C3** — HyperBuilder (1.63M params) adds ~26 MB FP32 train states: in budget, ignore. **C2** — zero training VRAM, but serialize with training jobs; the card can't host both.

---

## 8. Explicit non-goals

| Non-goal | Why |
|---|---|
| New deploy compression | Lane closed (Exp71, PHASE0_DEPLOY_PRESET) |
| Multi-GPU / cloud pretrain | Violates constraint thesis; L/XL/XXL presets stay datacenter docs |
| Bigger params for their own sake | Exp88 is the only sanctioned scale test, with a kill bar |
| Kernel hacks without VRAM story | Exp5b precedent: 0.4× speed for 32 MB — measure or skip |
| Replacing sound verifiers | Architecture brief moat; unchanged here |
| Activation offloading to host | Laptop PCIe bandwidth; checkpointing strictly dominates at this scale |

---

## 9. Open questions for the owner

1. **Endgoal token budget:** is the endgoal pretrain a 500M-token overnight (B-class) or a 300M-token weekend at h512 (A-class probe)? Pick before Exp88 so the ladder has a fixed wall-clock denominator.
2. **Lock A/B/C:** this brief recommends C→B with A capped at h512×6 via Exp88. Approve, or override the cap?
3. **Max acceptable wall-clock** for a single endgoal run: overnight (~10 h), weekend (~20 h), or week-class (~100 h)? Determines whether h512 or h768 is even on the table.
4. **quality_per_mb denominator:** count train-side ephemera (HyperBuilder ~1.63M, optimizer choice) or deployed bytes only? (Same question as Architecture brief §7.3 — one ruling covers both.)
5. **GPU scheduling:** confirm serialize-only rule — training runs own the card; Exp81/C2 inference sweeps fill gaps between runs. OK?

---

## 10. Citation index

| Topic | Paper | Year | Link |
|---|---|---|---|
| Master-weight elimination | ECO: Quantized Training without Full-Precision Master Weights | 2026 | https://arxiv.org/abs/2601.22101 |
| SGD-level optimizer memory | APOLLO: SGD-like Memory, AdamW-level Performance (MLSys) | 2025 | https://arxiv.org/abs/2412.05270 |
| Low-rank optimizer states | GaLore | 2024 | https://arxiv.org/abs/2403.03507 |
| Optimizer taxonomy / minimalist design | A Minimalist Optimizer Design for LLM Pretraining | 2025 | https://arxiv.org/abs/2506.16659 |
| Layer-wise fused updates | LOMO | 2023 | https://arxiv.org/abs/2306.09782 |
| INT8 optimizer states | 8-bit Adam (bitsandbytes) | 2022+ | https://github.com/TimDettmers/bitsandbytes |
| Ternary optimizer states | Stochastic Ternary Momentum | 2024 | https://arxiv.org/abs/2410.09734 |
| BPTT checkpointing, recurrent | Optimal Gradient Checkpointing for Sparse & Recurrent Architectures | 2024 | https://arxiv.org/abs/2412.11810 |
| Blockwise recurrence training | DiffusionBlocks (papers/DiffusionBlocks.md) | 2025/26 | https://arxiv.org/abs/2506.14202 |
| Ternary D/N scaling (data-hungry) | Spectra 1.1 (via papers/TernaryTraining.md) | 2025 | https://arxiv.org/abs/2506.23025 |
| Low-VRAM stack survey | papers/LowVRAMTernaryTraining.md (repo) | 2026 | — |
