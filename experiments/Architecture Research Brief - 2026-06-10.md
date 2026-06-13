# Architecture Research Brief — Intelligence per Packed MB

**Date:** 2026-06-10
**Scope:** Architecture proposals to raise `verified pass@k × difficulty coverage ÷ packed MB` on RTX 3050 Ti class hardware. Research track only — no code changed. Companion to the Exp79 lane currently in flight.
**⚠ Run order ≠ ID order.** Experiment numbers are identifiers (authoring order), not a schedule. Canonical run order across all briefs: [`EXECUTION_ORDER.md`](EXECUTION_ORDER.md). *Status: Exp80 kill · Exp82 kill · Exp83 promote · Exp81/84 pending (2026-06-13).*
**Inputs:** VISION.md, Experimental Log Summary.md, DISCIPLINE.md, papers/ corpus, Exp77/78/79 READMEs, `models/fast_weight_overlay.py`, `models/smt_memory_training.py`, web literature sweep (2024–2026).

---

## 1. Executive summary

- **Single most important bet (next 90 days): verifier-selected breadth at inference (Candidate C2).** We already own the rarest asset — sound, strict, cheap verifiers. Sampling K diverse candidates and letting the verifier pick converts free inference compute into verified pass@1, at **zero packed MB**. PTRM showed +11 pts on Sudoku-Extreme (87.4→98.75%) with this exact move and no retraining.
- The repo's own evidence says the bottleneck is **not recurrence depth** (Exp33.5: depth hurts; Exp34.1: flat across H=2/4/6; externally confirmed by the HRM Misconceptions audit — H-module contributes ~nothing, 2–4 steps suffice).
- The bottleneck is also **not raw capacity**: Exp70 logic hits 91%/83% at the same size that scores 8.5% strict frozen arithmetic. Capability is domain- and representation-dependent, not MB-limited.
- Two levers confirmed by Exp69: **READ lives in weights** (62.5% raw word), **COMPUTE lives in tools** (~98% with calculator). Architecture work should make reading/planning denser, not chase in-weights multiplication (Exp67 kill confirms).
- Strict frozen arithmetic at 8.5% is partly an **input-representation artifact**: digit-position embeddings (Abacus) take small transformers from ~chance to 99% on long addition. A cheap probe (C1) tells us how much of the 8.5% is representation vs capability.
- Exp77's overlay failure (oracle 0/64) matches the 2026 In-Place TTT finding: the productive fast-weight site is the **final MLP projection with a next-token objective**, not gate_up outputs with oracle/distil search. One more shot at the overlay with corrected geometry is justified (C3).
- TRM ("Less is More") beat HRM with **4× fewer params** by collapsing H+L into one tiny recursed block with full backprop through recursion. That attacks our denominator (packed MB) directly (C5).
- Attractor-trained recurrence (CMM, 2026) revives depth **only under equilibrium-loss training and only on constraint-structured domains** — file under the logic/LDT rung, not word problems (C4, gated).
- Literature caution on RLVR: it mainly sharpens sampling efficiency, it does not create new capability (pass@k critique). Exp79's `tool_supervised` mode injects *external* information (solver-corrected chains) — that is the mode with genuine new signal; `verified_filter` and `rlvr` redistribute existing mass. Sequence accordingly.
- V-STaR's lesson for Phase 3: **keep the failed traces.** Our TraceBuffer schema already stores failures; make sure they are used (DPO-style or contrastive) instead of discarded.
- Retrieval stays parked (Phase 5): current failures are reading/planning failures, not knowledge-recall failures. No evidence in any of our lanes that missing facts cost us points.
- Nothing proposed here adds packed MB beyond noise; C5 *reduces* it. Nothing requires hardware beyond the 3050 Ti. Every candidate has a pre-registered kill bar.
- Ordering: C2 (inference-only, feeds Exp79's K choice) → C1 (tiny probe) → C3 (overlay retry) → C5 (TRM-ization, needs a pretrain) → C4 (gated on returning to constraint domains).
- Where literature contradicts our baselines, it is flagged inline (§4, §6) — notably TRM's data-augmentation confound and CMM's domain restriction.

---

## 2. Constraint envelope

| Constraint | Value | Source |
|---|---|---|
| GPU | RTX 3050 Ti laptop, ~4 GB VRAM | CLAUDE.md, repo-wide |
| Deploy size class | ~4.64 MB packed (`mixed_top512_tequila_L_mlp_gate_up`); h256 lane ~9.15 MB | Log summary baselines |
| Model scale | h128/h256, HRM H_cycles=2, L_cycles=3, half_layers | `config/arch/net/hrm.yaml`, `config/arch/size/*` |
| Realistic train budget | 5k–50k steps per run (Exp29 did 50k locally); full-epoch SFT feasible (Exp69) | Log summary |
| Noise floor | ±0.0203 eval loss @ 5000 steps; pp-level bars per experiment | DISCIPLINE.md |
| Verifier cost | `ArithmeticExactVerifier` is regex+Decimal, ~ms/task; thousands/hour trivial | `evaluation/arithmetic_verifier.py` |
| Compression lane | **CLOSED** at 5.46×/7.55× — proposals must not assume "quantize more" | PHASE0_DEPLOY_PRESET.md, Exp71 |
| Leakage rules | frozen eval200 + 40 held-out reporting-only; guard_rail mandatory | Phase 0.5, CLAUDE.md |

Implication: inference-time compute is the cheapest resource we have (small model, ms-class verifiers, idle GPU between training runs). Packed bytes are the most expensive. Proposals are ranked by how hard they exploit that asymmetry.

---

## 3. Bottleneck diagnosis (evidence-linked)

**Where capability is lost today, ranked by evidence strength:**

1. **Compute-in-weights is lost and should stay lost.** Strict frozen 8.5% (Exp64) vs loose 55.5% (Exp34.1): the model has CoT *shape* without digit *truth*. Exp67 (kill): repair SFT moves token loss, not strict accuracy. Exp57 + Exp65 + Exp69: external sound solvers carry compute to ~98%. Verdict: don't spend architecture on in-weights multiplication. The open question is only whether cheap representation fixes (digit embeddings) move addition/reading — see C1.
2. **Reading/planning is the weight-side bottleneck.** Exp69: full-epoch training (83× token exposure on the membrane) lifted word from 32% → 62.5%. HRM's design point is 1k-example sample efficiency — the earlier gap was training utilization, not data volume. Remaining 62.5 → 100 gap is the READ lane; that is where density per MB is won.
3. **Recurrence depth is currently dead weight.** Exp33.5: frozen 57.5/50.0/46.0% for H=2/4/6 (depth *hurts*). Exp34.1: flat. External audit (HRM Misconceptions, arXiv 2510.00355): H-module removable, ACT unhelpful, trained HRM collapses to 2–4 effective steps. We carry H+L machinery — and its parameters — for ~2 effective iterations. This is *negative* density. C5 attacks it.
4. **The verifier sits at eval, not yet in training or inference selection.** Exp73: GRPO with verifier reward stays honest (gamed_frac 0.0) but was mild. Exp79 harness built, capability verdict pending. Meanwhile pass@k under strict verification — the literal north-star numerator — has never been measured as a function of sampling breadth. We do not know our own pass@16. C2 fixes that first.
5. **Fast-weight/overlay capacity exists but is mis-aimed.** Exp77: rank-8 ternary overlay at L-MLP gate_up output, oracle 0/64 improvements — the *site and objective* failed, not necessarily the idea (distil variant did move loss −0.296 nats). Exp78 (kill): z_L is a depth scratchpad, not temporal memory; BPTT beat SMT 1.917 vs 4.117 CE. In-Place TTT (2026) points to down_proj-as-fast-weights with a next-token-aligned objective.

---

## 4. Candidate architectures

### C1 — Digit-structured input representation ("Abacus probe")

- **Mechanism (one sentence):** Add a per-digit positional embedding (position of digit within its number, Abacus-style), optionally with least-significant-digit-first ordering and input injection, so the model can *address* digits instead of pattern-matching token shapes.
- **Why per MB:** One extra embedding table (~30 positions × hidden ≈ 4–8 KB packed at h128/h256 — rounding error). Capability gain, if any, is nearly free in bytes. Literature: small 16-layer transformer goes to **99% on 100-digit addition** trained on ≤20-digit numbers; combined with looped architecture + input injection, 92.9% → 99.1% OOD ([Transformers Can Do Arithmetic with the Right Embeddings, 2024](https://arxiv.org/abs/2405.17399); [Position Coupling, 2024](https://arxiv.org/abs/2405.20671)).
- **Predicted lift:** Strict frozen arithmetic (8.5% baseline) on +/− tasks specifically; secondary lift on word heldout via better operand extraction for the tool path (model must read numbers correctly even when the calculator computes).
- **What it does NOT cost:** No packed-MB change beyond KB; no VRAM change; no new leakage surface; SFT-scale training only (reuse Exp30 stack).
- **Risk / failure mode:** Maps to *Overbuilt architecture* if the 8.5% is planning failure rather than digit-addressing failure. Hard dependency: **BPE tokenizer must emit single-digit tokens** for digit-position embeddings to be well-defined; if the tokenizer merges digit spans, this becomes a tokenizer change (bigger lift) — open question §7.
- **Smallest experiment:** `Experiment 80 - Abacus Digit Embedding Probe`. Runner: Exp30 SFT stack + one embedding-table addition behind a flag; train two matched SFT runs (with/without) on the Exp69 corpus slice; eval strict frozen + heldout word, 2 seeds.
  **Decision Rule draft — Promote if:** strict frozen exact pass@1 (add/sub subset) improves ≥ 5 pp over matched no-abacus control on both seeds, invalid rate 0%. **Kill if:** gap ≤ noise on both seeds, or tokenizer requires multi-digit surgery to even start.

### C2 — Verifier-selected breadth (pass@k harvesting + diverse trajectories)

- **Mechanism:** Sample K candidates per task (temperature and/or PTRM-style Gaussian noise on initial z), run the strict verifier on each, report verified pass@k and verifier-picked pass@1; later feed survivors to Exp79's trace buffer.
- **Why per MB:** Zero new weights. This is the north-star numerator measured and then *harvested* directly. PTRM: Sudoku-Extreme 87.4 → 98.75% with noise-injected parallel trajectories and selection, **no retraining** ([Probabilistic TRM, 2026](https://arxiv.org/abs/2605.19943)). We are strictly better positioned than PTRM: they select with a learned Q-head; we select with a *sound* verifier — selection cannot be gamed (Exp73 failure mode designed out by construction).
- **Predicted lift:** Word heldout strict (62.5% pass@1 baseline) — even modest diversity typically yields pass@8 ≫ pass@1; logic hard (83%) likewise. Also produces the K-vs-yield curve Exp79 `verified_filter` needs to set its rollout budget rationally.
- **What it does NOT cost:** No packed MB, no training, no leakage (held-out tasks never enter the buffer — guard_rail path already exists). Costs only inference time; verifier is ms-class.
- **Risk / failure mode:** *Trivial true outputs* if K-sampling is harvested into training naively on easy tasks — mitigate with difficulty stratification already present in eval sets. Literature caution: in small discrete answer spaces, brute-force K eventually "solves" everything — report pass@k alongside consistency (G-Pass@k-style) so the curve is honest ([Don't Pass@k, 2025](https://arxiv.org/html/2510.04265v1); RLVR-sharpens-sampling critique in the same family).
- **Smallest experiment:** `Experiment 81 - Verified Breadth Sweep`. Runner: Exp69 + Exp70 checkpoints, K ∈ {1,2,4,8,16}, two diversity sources (temp 0.7 vs z-noise), strict verify all, report pass@k curve + verifier-picked pass@1, 2 seeds. Pure inference — can run while Exp79 CUDA lane is queued.
  **Decision Rule draft — Promote if:** verifier-picked pass@1 at K=8 beats single-sample pass@1 by ≥ 5 pp on heldout word or logic hard, invalid 0%. **Kill if:** pass@k curve is flat (diversity collapse — all samples agree and are wrong the same way), which itself is a publishable diagnosis of mode collapse.

### C3 — Fast-weight overlay v2: down_proj site + next-token objective (Exp77 corrected)

- **Mechanism:** Re-aim Exp77's ternary rank overlay at the MLP **final projection** (down_proj) and replace the oracle/distil objective with an objective aligned to next-token prediction, updated per-prompt (ephemeral RAM, wiped after each task), per the In-Place TTT recipe.
- **Why per MB:** Overlay weights are ephemeral — **zero packed MB**; the only persistent cost is the HyperBuilder (~1.63M params, and that's trainable-side, not deploy-side if kept as a tool). Capability added per deployed byte is the whole point. Literature: [In-Place TTT, 2026](https://arxiv.org/abs/2604.06169) (final-projection fast weights, drop-in, next-token-aligned objective); [TTT-RNN, 2024](https://arxiv.org/abs/2407.04620); [LaCT, 2025](https://arxiv.org/abs/2505.23884) (large-chunk updates for hardware efficiency); [Titans, 2025](https://arxiv.org/abs/2501.00663) (memory-as-context framing).
- **Predicted lift:** Strict word heldout and logic hard via within-task adaptation (long word problems where early context must persist); this is the honest successor to Exp78's killed temporal-memory ambition.
- **What it does NOT cost:** Packed MB (ephemeral); pretraining (frozen backbone, builder-only training as in Exp77); leakage risk unchanged (same SFT data paths + guard_rail).
- **Risk / failure mode:** *Overbuilt architecture* — two strikes already in this lane (Exp77 oracle 0/64, Exp78 kill). This is explicitly a **one-retry-then-kill** lane: the literature pinpoints what Exp77 got wrong (site + objective), so one corrected shot is fair; a second miss closes the lane until new evidence.
- **Smallest experiment:** `Experiment 82 - Down_Proj TTT Overlay`. Runner: extend `models/fast_weight_overlay.py` hook target to down_proj (read-only inspection says hooks currently target L-level SwiGLU output); objective = masked next-token CE on the prompt prefix; eval = Exp77's protocol on Exp70 logic checkpoint (per Exp77's own "next" note).
  **Decision Rule draft — Promote if:** strict logic hard ≥ +5 pp vs frozen baseline (83%) on 2 seeds with overlay wiped between tasks (no cross-task leakage). **Kill if:** ≤ noise on both seeds — and lane closes.

### C4 — Attractor-trained recurrence (CMM recipe) on the constraint/LDT rung — *gated*

- **Mechanism:** Train recurrence with an equilibrium loss + stability criterion (contractive dynamics) + noise so correct solutions become attractors, making depth *useful* instead of harmful — applied only to constraint-structured domains (logic, LDT lattice), never word arithmetic.
- **Why per MB:** Depth reuses the same weights — if depth converts to accuracy, capability per parameter rises with zero packed-MB growth. CMM numbers: 5M params 93.7% Sudoku-Extreme (vs TRM 5M 87.4%, HRM 27M 55%); 0.26M model holds 85.4% — extreme density ([CMM / HRM Dynamical Systems Theory, 2026](https://arxiv.org/abs/2603.22871); cross-supported by [Equilibrium Reasoners, ICML 2026](https://arxiv.org/abs/2605.21488), already in papers/).
- **Predicted lift:** Logic hard (83%) and any future LDT/constraint rung; explicitly **not** word problems (no attractor structure there — CMM's own domain restriction).
- **What it does NOT cost:** Packed MB; new data (reuses Exp70 corpus).
- **Risk / failure mode:** *Recurrence overthinking* + repeating the Exp33/34 EqR arc. Mitigation is the gate: per the standing analysis, plain-HRM depth is dead (our data + Misconceptions audit agree); CMM is the only recipe with external evidence of reviving it, and only on attractor domains. **Gated on: (a) Exp79 verdict landed, (b) a depth-vs-accuracy replication on our logic task showing CMM-trained depth N_L=6 beats N_L=2.**
- **Smallest experiment:** `Experiment 84 - Attractor Logic Recurrence` (number after C5). 2×2: {plain, CMM-loss} × {shallow, deep}, Exp70 data, strict logic eval.
  **Decision Rule draft — Promote if:** CMM-deep beats plain-shallow on logic hard by ≥ 5 pp, 2 seeds, AND depth monotonicity holds (deep ≥ shallow under CMM). **Kill if:** depth still flat/harmful under CMM training — recurrence-depth lane closes for good at this scale.

### C5 — Single recursive block, weight-tied (TRM-ization of the backbone)

- **Mechanism:** Collapse the H+L two-module HRM into one tiny (~2-layer) block recursed with full backprop through recursion and deep supervision, per TRM — removing the H-module that external audit says contributes nothing.
- **Why per MB:** This is the only candidate that shrinks the **denominator**. TRM: 7M params, 2 layers, beats 27M HRM on Sudoku/Maze/ARC (87.4% Sudoku-Extreme) — ~4× fewer params at higher accuracy ([Less is More: TRM, 2025](https://arxiv.org/abs/2510.04871); training-contract refinement in [Recursive Stem Model, 2026](https://arxiv.org/abs/2603.15641) — depth-agnostic transition operator, detached warm-up, loss at final step — cheaper to train). If it transfers, packed MB drops toward ~2–3 MB class at equal capability: density roughly doubles even at flat accuracy.
- **Predicted lift:** quality_per_mb (DISCIPLINE.md metric) on the pretrain gate; strict logic/word at minimum flat. Honest framing: this is a denominator play; numerator flat is a win.
- **What it does NOT cost:** New mechanisms (it *removes* machinery); inference latency class (fewer params, more iterations — net measured, gate at 1.5× per Phase 0 rule).
- **Risk / failure mode:** *Self-deception via domain transfer* — TRM's wins are puzzle grids with ~1000× data augmentation; text SFT is different. Honest contradiction flag: critics note TRM-vs-LLM comparisons confound data augmentation with architecture. Our test is TRM-block vs our-HRM on *our* tasks at matched tokens — that comparison is clean. Also requires a fresh pretrain (the expensive part, ~Exp29-class run).
- **Smallest experiment:** `Experiment 83 - Tied Recursive Block`. New `config/arch/net/trm_tied.yaml` (config exists for trm already — extend), h128, Exp29-style short pretrain (5k steps, 2×2 per DISCIPLINE), then Exp70-style logic SFT; compare quality_per_mb and strict logic vs matched HRM control.
  **Decision Rule draft — Promote if:** quality_per_mb beats HRM control by > noise floor AND strict logic hard within 2 pp of control (i.e., ≥ half the params at ≤ small capability cost), 2 seeds. **Kill if:** pretrain unstable under ternary recipe or strict logic drops > 5 pp.

### Killed at the brainstorm stage (one line each)

- **SMT/temporal z_L memory:** Exp78 already killed it fairly (BPTT 1.917 vs 4.117); C3 is its honest successor.
- **DiffusionBlocks-style blockwise recurrence training:** solves a VRAM problem we don't have at h128–h256; revisit only at larger size.
- **Retrieval now:** Phase 5 by design; no observed knowledge-recall failures — reading, not facts, is the gap.
- **MoE / wider models:** packed-MB denominator explodes; contradicts the north star outright.
- **Learned verifier / process reward model (V-STaR verifier, rStar-Math PPM):** we own *sound* programmatic verifiers — replacing them with a learnable one reintroduces the gaming surface Exp73 was designed to detect; keep only the keep-failed-traces idea.
- **Coconut-style continuous latent thoughts:** documented training instability; looped/recurrent-depth is the trainable instantiation of the same idea ([formal comparison, 2025/26](https://arxiv.org/abs/2509.25239)) — covered by C4/C5.
- **More aggressive quantization (activations, attention, Hadamard):** lane closed by Exp71/PHASE0_DEPLOY_PRESET; TWLA/HGF ideas stay parked in papers/ memory for a future reopen.
- **ACT/adaptive halting:** external audit — max-steps beats learned halting; nothing to build.

---

## 5. Ranked roadmap

```text
NOW (parallel with Exp79 CUDA run; inference-only, no contention for training lane)
  Exp81  C2 Verified Breadth Sweep        — zero-MB numerator measurement + harvest curve
                                            feeds K-budget into Exp79 verified_filter mode

NEXT (SFT-scale, after Exp79 verdict lands; order by cost)
  Exp80  C1 Abacus Digit Probe            — KB-scale rep fix; answers "is 8.5% representation?"
  Exp82  C3 Down_Proj TTT Overlay         — one-retry-then-kill; zero packed MB

AFTER (pretrain-scale; gated)
  Exp83  C5 Tied Recursive Block          — denominator attack; needs Exp29-class pretrain
  Exp84  C4 Attractor Logic Recurrence    — gated on Exp83/79 verdicts + depth replication

Dependency graph (Phase alignment):
  C2 ──► Exp79 mode B budget ──► Phase 6 loop          (Phase 2/3/6)
  C1 ──► word/arithmetic READ lane                      (Phase 2)
  C3 ──► per-task adaptation, successor to Exp77/78     (Phase 4-adjacent)
  C5 ──► new backbone candidate ──► re-run Phase 0 gates(Phase 0/4)
  C4 ──► constraint/LDT rung only, after C5 verdict     (Phase 4, gated)
```

Rationale for the order: C2 is free, fast, cannot leak, and de-risks Exp79's most expensive hyperparameter (rollout count). C1/C3 are SFT-scale with pre-registered kill bars. C5/C4 each require a pretrain and should only spend that budget after the cheap probes have either moved the numerator or been killed.

---

## 6. Explicit non-goals

| Non-goal | Why |
|---|---|
| In-weights multiplication / long-digit compute | Exp67 kill + Exp57/65 tool path at ~98%; tools are the COMPUTE organ |
| Deeper plain-HRM recurrence | Exp33.5/34.1 + Misconceptions audit: depth without attractor training is harmful |
| More compression | Phase 0 closed; Exp71 showed −6 pts hard reasoning per 0.95 MB saved |
| Learned verifiers | Soundness is our moat; learnable judges reopen the gaming surface |
| Retrieval integration | Phase 5 gate not reached; no knowledge-failure evidence in any lane |
| Bigger models / cloud training | Violates the constraint thesis; nothing here needs >4 GB VRAM |
| More SFT steps alone | Excluded by task definition; Exp69 already banked the utilization win |

---

## 7. Open questions for the owner

1. **Tokenizer digit granularity (blocks C1):** does the BPE at `data_io/trained_tokenizers/bpe/tokenizer.json` emit single-digit tokens, or are there multi-digit merges? Multi-digit merges turn the Abacus probe into a tokenizer change — different cost class.
2. **Is strict in-weights arithmetic still a target at all,** or formally delegated to tools with Exp64 kept as an honesty gauge only? Decides whether C1's promote bar is on strict frozen or on word-heldout operand-extraction.
3. **Packed-MB accounting for ephemeral machinery:** does the north-star denominator count the overlay HyperBuilder (~1.63M params) and any retrieval index, or only deployed backbone bytes? C3's density claim depends on the answer.
4. **Difficulty coverage term:** currently implicit (logic easy/hard, digit counts). Should we formalize it (e.g., strata-weighted pass@1) before C2 reports its first pass@k curves, so the north-star metric is computable from day one?
5. **GPU scheduling:** may Exp81 (inference-only breadth sweep) run on idle GPU time before the Exp79 CUDA run completes, or does the training lane own the GPU until verdict?

---

## Appendix: citation index

| Topic | Paper | Year | Link |
|---|---|---|---|
| Recursion > size | Less is More: Recursive Reasoning with Tiny Networks (TRM) | 2025 | https://arxiv.org/abs/2510.04871 |
| Test-time breadth for tiny recursive models | Probabilistic TRM | 2026 | https://arxiv.org/abs/2605.19943 |
| Cheaper recursive training contract | Recursive Stem Model | 2026 | https://arxiv.org/abs/2603.15641 |
| HRM component audit | HRM Perspectives & Misconceptions | 2025 | https://arxiv.org/abs/2510.00355 |
| Attractor-trained depth | CMM / HRM Dynamical Systems Theory | 2026 | https://arxiv.org/abs/2603.22871 |
| Attractor landscape shaping | Equilibrium Reasoners (papers/EquilibriumReasoners.md) | 2026 | https://arxiv.org/abs/2605.21488 |
| Digit positional embeddings | Transformers Can Do Arithmetic with the Right Embeddings | 2024 | https://arxiv.org/abs/2405.17399 |
| Same idea, concurrent | Position Coupling | 2024 | https://arxiv.org/abs/2405.20671 |
| Fast weights, corrected inject site | In-Place Test-Time Training | 2026 | https://arxiv.org/abs/2604.06169 |
| Fast-weight RNN foundation | Learning to (Learn at Test Time) | 2024 | https://arxiv.org/abs/2407.04620 |
| Large-chunk TTT efficiency | Test-Time Training Done Right (LaCT) | 2025 | https://arxiv.org/abs/2505.23884 |
| Neural long-term memory | Titans | 2025 | https://arxiv.org/abs/2501.00663 |
| Verifier-augmented self-training | V-STaR | 2024 | https://arxiv.org/abs/2402.06457 |
| Self-training root | STaR | 2022 | https://arxiv.org/abs/2203.14465 |
| SLM + process reward + search | rStar-Math (ICML 2025) | 2025 | https://icml.cc/virtual/2025/poster/46400 |
| pass@k measurement caveats | Don't Pass@k: Bayesian framework | 2025 | https://arxiv.org/html/2510.04265v1 |
| SLM self-correction needs strong verifiers | Small LMs Need Strong Verifiers | 2024 | https://arxiv.org/html/2404.17140v2 |
| Latent vs explicit CoT formal comparison | Formal Comparison CoT vs Latent Thought | 2025/26 | https://arxiv.org/abs/2509.25239 |
| Latent reasoning survey | A Survey on Latent Reasoning | 2025 | https://arxiv.org/abs/2507.06203 |
