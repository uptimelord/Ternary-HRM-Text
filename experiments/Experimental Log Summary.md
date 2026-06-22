# Experimental Log Summary

Living index of every experiment under `experiments/Experiment */`. Each row is
written to stand alone: **why we ran it**, **what we measured**, and **what it
means**. Full commands, decision rules, and generation dumps live in the folder
README.

**How to update:** add a row (or new README + row), refresh **Current Baselines**
if a lane is promoted/killed, bump the footer count.

**Run order ≠ ID order.** Experiment numbers are identifiers, not a schedule. For *what runs when* across
all research briefs, see [EXECUTION_ORDER.md](EXECUTION_ORDER.md) (Tier 0→5; machine milestones Exp112–116 last).

Governance: [DISCIPLINE.md](DISCIPLINE.md) · Locked vocab:
[VOCAB_RECIPE.md](VOCAB_RECIPE.md) · LDT harness detail (44–53):
[SURVEY_44_53_LDT_STATUS.md](SURVEY_44_53_LDT_STATUS.md) · North star:
[VISION.md](../VISION.md)

---

## Roadmap (why these experiments exist)

```text
Phase 0 — Ternary engine (Exp 1–28, 5b/5c)
  Can we ship a ~5 MB HRM that trains on a 3050 Ti without quality collapse?

Phase 0 — First real checkpoint (Exp 29–42)
  Does the locked recipe pretrain, export cleanly, and pass deploy/frozen gates?

Phase 0.5 — Arithmetic & recurrence (Exp 30–36, 32, 64–67)
  Can the small model do verified digit math and benefit from more H_cycles?

Phase 1 — Verifier harness (Exp 43–53, 64–65)
  Parse → rank rules → exact verify, without cheating on held-out splits.

Phase 1 — LDT / closure lattice (Exp 43–44, 54–63)
  Faithful carry-lattice solving: sound closure, no wrong singleton returns.

Phase 1.5 — Word & logic SFT (Exp 66–75, 73)
  READ problems in weights; COMPUTE with tools; VERIFY with strict scorers.

Architecture probes (Exp 77–84)
  Fast-weight overlay, memory, breadth, and architecture controls — only if they beat simpler baselines.
```

**Dependency chains worth remembering:**

- Compression: Exp 9 → 14 → 19 → 22 → **23 (deploy lock)** → Exp 38 (2×2 confirm)
- Arithmetic: Exp 29 → 30 → 31 → 34.1 → 64 (strict truth) → 66 → 68 → **69**
- Logic: Exp 29 → **70** → gates Exp 71/72/74/75
- LDT carry: Exp 55 (unsound) → 56 (0% coverage) → **57 (promote)** → 58/59
- Verifier shape: Exp 44 → 45 → 47 → 48 → 49 → 50 → **51 (full loop)** → 52/53 (safety)

---

## Glossary

| Term | Meaning |
|---|---|
| **frozen eval200** | Held-out arithmetic file — never train on it; reporting only |
| **loose vs strict** | Loose = answer string match; strict = `ArithmeticExactVerifier` checks every step |
| **pass@1 / coverage** | Fraction of eval rows where the model/solver returns one correct verified answer |
| **gap (nats)** | Eval loss minus dense baseline — lower is better; noise floor ±0.0203 @ 5000 steps |
| **mixed_top512_tequila** | Top-512 vocab rows stay dense; rest ternary; Tequila STE on body |
| **L_mlp_gate_up** | Ternary only the L-level MLP gate+up projection (best body target) |
| **H_cycles** | Recurrence depth at inference — “think longer” without more parameters |
| **EqR** | Equilibrium-recurrence training — random H during SFT for depth robustness |
| **tool-checked** | Model writes plan; exact calculator recomputes each step; verifier scores final |
| **promote / kill** | Survives to default path vs abandoned lane (see each README Decision Rule) |
| **LDT** | Lattice Deduction Transformer — narrow a candidate set monotonically toward one answer |
| **closure-led** | Sound carry equations run before neural elimination (Exp 57 fix) |

---

## Current Baselines

| Lane | Recipe / checkpoint | Why this is the baseline | Key numbers |
|---|---|---|---|
| **Deploy compression** | `mixed_top512_tequila_L_mlp_gate_up` | Only recipe that passes export parity + 2×2 deploy lock (Exp 22–23, 38) | ~4.64 MB packed at h128 (7.55×); size target unlocked 2026-06-20 — bounded by 4 GB GPU envelope (peak <= 3,800 MiB), not a byte ceiling |
| **Locked arithmetic (loose)** | **Exp34.1** | Best EqR-from-scratch arithmetic lock before word-reasoning fork | Frozen eval200: H=2/4/6 **55.5/56.5/55.5%**, invalid 0% |
| **Locked arithmetic (strict)** | Exp64 on Exp34.1 | Same checkpoint under honest verifier — real compute ability | Strict frozen **~8.5%** (not 55%); CoT shape without digit reliability |
| **Word reasoning (raw)** | **Exp69** | Full-epoch SFT fixes READING; best raw word checkpoint | heldout_word **62.5%**, frozen chain **72%** |
| **Word reasoning (tool)** | Exp69 + Exp68 calculator | COMPUTE lever: model reads, tool adds/muls correctly | Held-out overall **~98%** (word **94%**) |
| **Comparative logic** | **Exp70** (term path) | First strong logic generator; gates compression/optimizer probes | easy **91%**, hard **83%**, invalid 0% |
| **Closure solver** | **Exp57** | Sound carry lattice that actually returns answers | **100%** coverage, `returned_wrong == 0` |
| **h256 2-bit attention** | Exp26 `combo_2bit_attention` | Extra compression at h256 only — not global deploy | Packed **9.15 MB**; frozen gate passed once |

---

## Number Gaps & README Hygiene

| Item | Status |
|---|---|
| **Missing folder** | **76** only |
| **Indexed READMEs** | **93** folders, each with README |
| **Decision Rule** | **59** post-DISCIPLINE READMEs; **28** legacy (Exp 1–28, 5b, 5c) intentionally not retrofitted |
| **Pending runs** | **Exp32** (recurrence sweep), **Exp37** (dataset gen needs API key), **Exp74** (planned), **Exp81** (infra-ready / awaiting decision-grade run), **Exp84** (unblocked: Exp83 promote + Exp79 verdict in). *Exp80 (kill), Exp82 (kill) decided 2026-06-13.* |

> Do not use the deleted `_exp575859_run.log` summary. Exp57–59 READMEs + JSON are canonical.

---

## All Experiments

### Ternary / vocab / compression (1–28, 5b, 5c)

*Theme: shrink packed size toward ~5 MB without eval loss above noise floor.*

| Exp | Name | Why (question) | Key result | Verdict |
|---:|---|---|---|---|
| 1 | [Ternary HRM Body](Experiment%201%20-%20Ternary%20HRM%20Body/README.md) | Can 1.58-bit ternary layers drop into HRM-Text as a new track? | Wiring only: dense embed + ternary body + dense head | open |
| 2 | [Ternary HRM Smoke Train](Experiment%202%20-%20Ternary%20HRM%20Smoke%20Train/README.md) | At 100 steps, how bad is ternary_mlp vs ternary_body vs dense? | mlp +0.048 nats; body +0.351 nats vs dense | open |
| 2b | [Longer Ternary Train](Experiment%202b%20-%20Longer%20Ternary%20Train/README.md) | Does the body gap shrink if we train 500 steps instead of 100? | Gaps −34% / −68%; body ~+0.112 nats @ 500 | open |
| 2b-long | [2000-step Quality Check](Experiment%202b-long%20-%202000-step%20Quality%20Check/README.md) | Does the gap keep closing at 2000 steps? | Body plateaus ~+0.09 nats — not closing to zero | open |
| 2c | [Ternary Hyperparam Sweep](Experiment%202c%20-%20Ternary%20Hyperparam%20Sweep/README.md) | Which threshold × group_size minimizes body ternary damage? | Best thr=0.5, gs=128 (+0.085) — **Because** body gap was stuck at defaults | **promote** |
| 4 | [Ternary Tied Vocab](Experiment%204%20-%20Ternary%20Tied%20Vocab/README.md) | Vocab is 64/67 MB — must we ternarize tied embed+head to move headline size? | 7.76× packed (+0.06 nats) — **Because** body-only ternarization saved ~4% disk | **promote** |
| 6 | [Ternary Tied Vocab Long Run](Experiment%206%20-%20Ternary%20Tied%20Vocab%20Long%20Run/README.md) | Does the vocab win survive 2000 steps? | Gap grew +0.096→+0.125; compression holds — **Because** short-run wins can drift | **promote** (lane; tune quantizer) |
| 7 | [Ternary Vocab Quantizer Sweep](Experiment%207%20-%20Ternary%20Vocab%20Quantizer%20Sweep/README.md) | Can we tune quantizer to shrink the 2000-step vocab gap? | Best thr=0.25, gs=32 (+0.050 @ 500) — **Because** Exp6 gap stayed +0.125 | **promote** |
| 8 | [Ternary Vocab Row Gain](Experiment%208%20-%20Ternary%20Vocab%20Row%20Gain/README.md) | Does per-row learned gain fix residual vocab error? | Worse than plain tuned ternary — **Because** extra knob overfit noise | **kill** |
| 9 | [Mixed Precision Vocab Rows](Experiment%209%20-%20Mixed%20Precision%20Vocab%20Rows/README.md) | Can dense top-frequency rows recover quality cheaply? | mixed_top512 +0.0048 nats, 6.85× — **Because** rare tokens matter less | **promote** |
| 10 | [Wider Ternary Vocab Scale Test](Experiment%2010%20-%20Wider%20Ternary%20Vocab%20Scale%20Test/README.md) | Does wider hidden size absorb ternary damage? | Gap 128→256: +0.103→+0.058 | open |
| 11 | [Selective Body Ternary](Experiment%2011%20-%20Selective%20Body%20Ternary/README.md) | Which body projections tolerate ternary? | mlp_gate_up −0.016; attn_gqkv +0.025 — **Because** blanket body ternarization failed | **promote** mlp; **kill** attn default |
| 12 | [Ternary Scale Rule](Experiment%2012%20-%20Ternary%20Scale%20Rule/README.md) | Is the scale rule causing vocab loss? | selected_mean_abs best plain; mean_abs best mixed_top512 — **Because** wrong scale inflated gap | **promote** |
| 13 | [Stacked Vocab + Body Ternary](Experiment%2013%20-%20Stacked%20Vocab%20%2B%20Body%20Ternary/README.md) | Can we stack the two best 500-step wins? | Stack +0.078; vocab-only +0.012 — **Because** interaction penalty between stacks | **kill** stack; **promote** vocab-only |
| 14 | [Mixed Top512 Long Run](Experiment%2014%20-%20Mixed%20Top512%20Long%20Run/README.md) | Does mixed_top512 hold at 2000 vs tied dense baseline? | +0.0048 nats, 5.11 MB — **Because** Exp13 needed clean long-run confirm | **promote** |
| 15 | [Mixed Top512 Width Scaling](Experiment%2015%20-%20Mixed%20Top512%20Width%20Scaling/README.md) | Which width is the deploy sweet spot? | h128/192/256 gaps +0.0048/+0.0194/+0.0092 — **Because** h256 balances gap vs capacity | **promote** |
| 16 | [Tequila Dynamic Bias](Experiment%2016%20-%20Tequila%20Dynamic%20Bias/README.md) | Does Tequila STE fix body+vocab interaction penalty? | Small gains 0.004–0.007; stack still +0.028 | open |
| 17 | [Continual QAT Transition](Experiment%2017%20-%20Continual%20QAT%20Transition/README.md) | Is dense→ternary transition better than scratch? | Scratch wins −0.017 — **Because** transition preserves bad basins | **kill** |
| 18 | [Weight Decay Phaseout](Experiment%2018%20-%20Weight%20Decay%20Phaseout/README.md) | Does WD phaseout in last 20% help ternary vocab? | Flat at wd=0.01 over 500 steps | open |
| 19 | [Long Training Data Scaling](Experiment%2019%20-%20Long%20Training%20Data%20Scaling/README.md) | Do short-run winners hold at 2000/5000 steps? | 2000 tequila −0.008; 5000 compressed +0.025–0.030 — **Because** deploy needs long runs | **promote** |
| 20 | [Tequila Export Parity](Experiment%2020%20-%20Tequila%20Export%20Parity/README.md) | Does packed export match training eval? | Export gap +0.0003 — **Because** deploy uses packed weights | **promote** |
| 21 | [Body Sensitivity Map](Experiment%2021%20-%20Body%20Sensitivity%20Map/README.md) | Which H/L body targets tolerate ternary before long runs? | L_mlp_gate_up −0.035 best — **Because** avoid expensive blind body sweeps | open |
| 22 | [Vocab Body Combo Confirmation](Experiment%2022%20-%20Vocab%20Body%20Combo%20Confirmation/README.md) | Does best vocab + best body combine at 5000 steps? | Combo −0.0063 (noise); 0.04179 quality/MB — **Because** final deploy recipe candidate | **promote** |
| 23 | [Combo Export Parity](Experiment%2023%20-%20Combo%20Export%20Parity/README.md) | Does hard-export combo match training? | Gap +0.0033; 4.64 MB — **Because** ship path is packed inference | **promote** (deploy lock; MB target later unlocked 2026-06-20) |
| 24 | [Two Bit Body Sensitivity](Experiment%2024%20-%20Two%20Bit%20Body%20Sensitivity/README.md) | Does 2-bit beat 1.58-bit body penalty? | both_attention_gqkv −0.066 — opposite of 1.58-bit map | open |
| 25 | [Stacked Two Bit Compression](Experiment%2025%20-%20Stacked%20Two%20Bit%20Compression/README.md) | Does 2-bit attention stack on deploy combo? | Fails h128; passes h256 (−4.67 MB, +0.0134) | open (h256 only) |
| 26 | [H256 Two Bit Attention Export Parity](Experiment%2026%20-%20H256%20Two%20Bit%20Attention%20Export%20Parity/README.md) | Does h256 2-bit attention export cleanly? | Gap +0.0011; 9.15 MB — **Because** extra MB only if export-safe | **promote** (h256 candidate) |
| 27 | [H256 Frozen Generalization Gate](Experiment%2027%20-%20H256%20Frozen%20Generalization%20Gate/README.md) | Does h256 2-bit hold on seed2 + frozen arithmetic? | Seed2 +0.041; frozen favored compressed | open |
| 28 | [H256 GQKV Frozen Gate](Experiment%2028%20-%20H256%20GQKV%20Frozen%20Gate/README.md) | Does lighter 2-bit GQKV pass frozen gate? | Frozen answer-loss +0.5914 — **Because** aggressive attn quant fails generalization | **kill** |
| 5b | [Triton Packed Matmul](Experiment%205b%20-%20Triton%20Packed%20Matmul/README.md) | Can fused Triton cut VRAM while keeping latency? | −32 MB vocab VRAM; 0.40–0.66× latency vs cached dense | open |
| 5c | [Triton Packed Matmul Tuning](Experiment%205c%20-%20Triton%20Packed%20Matmul%20Tuning/README.md) | Can row-scale specialization beat generic Triton? | Row-scale slower — **Because** specialization did not help this shape | **kill** row-scale |

### Training infra / kernels (3, 5, 29, 40, 72)

*Theme: make packed weights fast to run and cheap to train on 4 GB VRAM.*

| Exp | Name | Why (question) | Key result | Verdict |
|---:|---|---|---|---|
| 3 | [Ternary Pack + Inference Smoke](Experiment%203%20-%20Ternary%20Pack%20%2B%20Inference%20Smoke/README.md) | What is the real disk and latency payoff of ternarization? | Per-layer ~10–13× disk; inference ~40% slower (embed dominates) | open |
| 5 | [Packed Matmul Kernel](Experiment%205%20-%20Packed%20Matmul%20Kernel/README.md) | Can we cache quantized weights to fix inference latency? | `cached` 1.56× vs STE — **Because** re-quantizing every forward was the bottleneck | **promote** cached eval |
| 29 | [First Local Pretrain](Experiment%2029%20-%20First%20Local%20Pretrain/README.md) | Can locked recipe finish 50k-step Phase 0 on 3050 Ti? | 50k done; frozen exact 1.5% — first real checkpoint artifact | open |
| 40 | [ECO Optimizer](Experiment%2040%20-%20ECO%20Optimizer/README.md) | Can master-weight-free ECO cut optimizer state on ternary layers? | Architecture mismatch — honest negative — **Because** STE latent already differs from ECO assumptions | **kill** |
| 72 | [8-bit Adam Optimizer Probe](Experiment%2072%20-%208-bit%20Adam%20Optimizer%20Probe/README.md) | Can 8-bit Adam save VRAM during logic SFT without hurting Exp70? | −15% VRAM @ 3000 steps; 8000-step parity gate pending | open |

### Recurrence / EqR / memory overlays (32–36, 33.6, 77, 78)

*Theme: does “think longer” (H_cycles / memory) buy verified reasoning on a fixed 20M model?*

| Exp | Name | Why (question) | Key result | Verdict |
|---:|---|---|---|---|
| 32 | [Recurrence Scaling Probe](Experiment%2032%20-%20Recurrence%20Scaling%20Probe/README.md) | Does more H_cycles at inference help Exp31 without retraining? | **(pending)** — no sweep artifacts on disk | open |
| 33 | [EqR Lite Recurrence Stability](Experiment%2033%20-%20EqR%20Lite%20Recurrence%20Stability/README.md) | Can EqR-lite SFT survive random H during training? | Exp33.5 frozen: H=2/4/6 **57.5/50.0/46.0%** — depth hurts | open |
| 33.6 | [EqR Convergence Probe](Experiment%2033.6%20-%20EqR%20Convergence%20Probe/README.md) | Is H>2 decline “overthinking” (good state then drift)? | Diagnostic harness for per-cycle acc + residual | open |
| 34 | [EqR Full Pretrain Then SFT](Experiment%2034%20-%20EqR%20Full%20Pretrain%20Then%20SFT/README.md) | What if EqR dynamics exist from pretrain start, not just SFT? | **Exp34.1:** 55.5/56.5/55.5% frozen; seed2 ~40% repro fail | open (34.1 locked) |
| 35 | [Mixed Language Arithmetic Pretrain](Experiment%2035%20-%20Mixed%20Language%20Arithmetic%20Pretrain/README.md) | Can mixed Dolmino+arithmetic pretrain fix language collapse? | Arithmetic 65/67/64% beats 34.1; language still junk | open |
| 36 | [Language Rehearsal EqR SFT](Experiment%2036%20-%20Language%20Rehearsal%20EqR%20SFT/README.md) | Can rehearsal during EqR SFT preserve language without losing math? | 58.5/62.0/63.5%; language still weak | open |
| 77 | [Ternary Flash Grid Micro](Experiment%2077%20-%20Ternary%20Flash%20Grid%20Micro/README.md) | Can rank-8 ternary overlay on frozen backbone improve arithmetic SFT loss? | Oracle 0/64 beat baseline; distil loss wins only — **Because** inject point/rank may be wrong | open (oracle **kill**) |
| 78 | [SMT z_L Probe](Experiment%2078%20-%20SMT%20z_L%20Probe/README.md) | Does paper SMT+DMT beat BPTT for temporal memory on pretrain text? | Fair: BPTT CE **1.92** vs SMT+DMT **4.12** — **Because** z_L is depth scratchpad today, not seq memory | **kill** (deploy) |
| 79 | [Verifier In Loop Training](Experiment%2079%20-%20Verifier%20In%20Loop%20Training/README.md) | Does verifier-gated SFT (tool_supervised) lift strict pass@1? | seed1: heldout +3.5 pp (bar ≥5), frozen **−3.0 pp** (bar ≥+3); tool pass dropped 0.02→0.005 — **Because** neither promote bar met and frozen regressed | open (no promote, seed1) |

### Architecture brief probes (80–84)

*Theme: repair validity first, then run decision-grade CUDA gates.*

| Exp | Name | Why (question) | Key result | Verdict |
|---:|---|---|---|---|
| 80 | [Abacus Digit Embedding Probe](Experiment%2080%20-%20Abacus%20Digit%20Embedding%20Probe/README.md) | Do per-digit positions lift add/sub beyond matched control? | Decision-grade CUDA, 4 arms × frozen-200: abacus−control frozen delta **−0.5 pp (seed1) / 0.0 pp (seed2)**, add/sub delta 0.0 pp both seeds, invalid 0%, `abacus_ready=true` (not floor) — **Because** per-digit positions add nothing over matched control at this scale; both seeds at/below the +5 pp bar | **kill** |
| 81 | [Verified Breadth Sweep](Experiment%2081%20-%20Verified%20Breadth%20Sweep/README.md) | Does K-sampling plus label-free selection lift pass@1? | Oracle metric renamed `oracle_any_pass@k`; solver/derived pickers wired; stale logs marked superseded | infra-ready / awaiting rerun |
| 82 | [Down Proj TTT Overlay](Experiment%2082%20-%20Down%20Proj%20TTT%20Overlay/README.md) | Can one-retry TTT at `down_proj` improve logic hard? | Decision-grade CUDA (Exp70 logic SFT ckpt, hard-1k offset0 limit200): adapted−baseline **0.0 pp both seeds** (0.775→0.775) — **Because** one-retry TTT at down_proj moves nothing on logic hard; C3 one-retry-then-kill bar fails | **kill** (C3 lane closes) |
| 83 | [Tied Recursive Block](Experiment%2083%20-%20Tied%20Recursive%20Block/README.md) | Does TRM improve quality per packed MB vs HRM? | Full CUDA 2 seeds: TRM q/mb **0.0422** vs HRM 0.0065 (packed_exact), strict logic Δ +1.25 pp (within 2 pp), peak VRAM 462 MB — **Because** q/mb gap is pretrain-loss-driven (0.37 vs 2.3); downstream strict guard held | **promote** |
| 83.1 | [TRM Mixed Vocab Head](Experiment%2083.1%20-%20TRM%20Mixed%20Vocab%20Head/README.md) | Does train-time `mixed_top512_tequila` on the vocab head shrink TRM pack (64→~3.5 MB) while holding logic? | Exp83 left the vocab head fp32 (64 of 64.08 MB); `--head-recipe mixed_top512` wires the deploy recipe into both arms train-time (reuses Exp9 head + Exp13 packer); CPU wiring validated | infra-ready / awaiting run |
| 90.1 | [Messy Comparative Robustness](Experiment%2090.1%20-%20Messy%20Comparative%20Robustness/README.md) | On **real** deepseek paraphrases, does a prose-reader or a relation-lattice survive? Same 227k-param/0.87 MB model. | Text-readers collapse (TRM **0.02**, transformer **0.00**); relation lattice (Exp92, reads structured edges) holds at **0.81/0.795 neural, 1.00 constrained** (2 seeds) — **Because** the lattice is phrasing-invariant by construction; ordering is robust given clean edges | **promote** lattice schema for comparative; parser (prose→edges) is the open step |
| 90.2 | [Prose To Lattice Parser](Experiment%2090.2%20-%20Prose%20To%20Lattice%20Parser/README.md) | Can a tiny reader parse messy prose → relation lattice → exact solver (the READ half Exp90.1 left open)? | Ladder: prose-reader 0.02 → rank head 0.245 → +token-tagging 0.305 → **atomic-edge pair head 0.59/0.55 (2 seeds, mean 0.57)** on real paraphrases — **Because** per-pair directional edge supervision + confidence-thresholded topo-sort cleanly splits READ (parse edges) from SOLVE (rank). Fixed 3 traps: gold-edge leak, BCE-on-closure collapse, noise-edge decode. | **promote** parse-then-solve shape; residual lever = reader data diversity |
| 90.3 | [Shared Reachability / Resonance](Experiment%2090.3%20-%20Shared%20Reachability%20Machine/README.md) | Can comparative + logic share one reachability machine without over-crediting the solver? | Solver-assisted closure hits **0.9925/0.995** (diagnostic only); deep resonance seed1 reaches **0.7225** combined (comparative 0.715, logic 0.730), peak VRAM 2827 MB — **Because** resonance reduces handholding but has not cleared the 0.80 two-seed bar | partial / continue |
| 119 | [Stepwise Reachability](Experiment%20119%20-%20Stepwise%20Reachability/README.md) | Does stepwise message-passing (one hop per round) extrapolate to longer chains than trained on? | seed1, train ≤4-hop: K 0.989 → K+1 0.185 → K+2 0.019 (comparative K+2 0.000). Baseline (one-shot comparative) was K+2 0.602 — the halt-fix made comparative actually take 3–5 rounds (was 1) and extrapolation got **worse** — **Because** residual message-passing converges to a wrong-but-stable attractor on unseen chain lengths; thesis falsified for comparative in this form | **kill** |
| 84 | [Attractor Logic Recurrence](Experiment%2084%20-%20Attractor%20Logic%20Recurrence/README.md) | Does CMM revive depth on logic hard? | `n_super` semantics fixed; base-recipe label, repo control, and `l6` depth axis wired | infra-ready / awaiting run |

### Phase 0 adapters & deploy locks (37–42)

*Theme: confirm deploy claim, probe next compression/optimizer ideas on locked preset.*

| Exp | Name | Why (question) | Key result | Verdict |
|---:|---|---|---|---|
| 37 | [DeepSeek Custom Rehearsal Dataset](Experiment%2037%20-%20DeepSeek%20Custom%20Rehearsal%20Dataset/README.md) | Can we build 100k verified-arithmetic + DeepSeek language rehearsal cheaply? | Generator ready; **not run** (API key needed) | planned |
| 38 | [Deploy Lock 2x2](Experiment%2038%20-%20Deploy%20Lock%202x2/README.md) | Is deploy claim true across h{128,256}×steps{500,2000}×3 seeds? | All cells within noise; quality/MB wins — **Because** single-cell lock was discipline debt | **promote** |
| 39 | [NM Sparsity Gate Up](Experiment%2039%20-%20NM%20Sparsity%20Gate%20Up/README.md) | Does 6:8 sparsity on gate_up add free speed structure without quality loss? | No win at matched VRAM — **Because** zeros live inside already-ternary groups | **kill** (park) |
| 41 | [QK Norm Attention](Experiment%2041%20-%20QK%20Norm%20Attention/README.md) | Does QK-Norm stabilize dense attention before any future attn quant retry? | No lift under protocol — **Because** outlier problem not solved this way | **kill** |
| 42 | [Token Param Probe](Experiment%2042%20-%20Token%20Param%20Probe/README.md) | On 3050 Ti budget, do tokens or params buy more loss drop (Spectra hypothesis)? | Token doubling competitive — informs budget, not deploy gate | **promote** (probe) |

### LDT lattice & closure harness (43, 54–63)

*Theme: faithful carry-lattice solving — coverage without ever returning a wrong singleton.*

| Exp | Name | Why (question) | Key result | Verdict |
|---:|---|---|---|---|
| 43 | [LDT Arithmetic Probe](Experiment%2043%20-%20LDT%20Arithmetic%20Probe/README.md) | Can recurrent lattice on pre-parsed ops beat HRM frozen baseline? | Frozen **5.5%** — **Because** independent digit slots wrong shape for carry | **kill** |
| 54 | [Faithful LDT Arithmetic Minimal](Experiment%2054%20-%20Faithful%20LDT%20Arithmetic%20Minimal/README.md) | Does powerset answer lattice (faithful LDT object) learn arithmetic? | 0% coverage; safe abstain — **Because** 19999-answer lattice too coarse without search | open (no promote) |
| 55 | [Carry Lattice LDT Addition](Experiment%2055%20-%20Carry%20Lattice%20LDT%20Addition/README.md) | Can carry cells (not answer digits) make LDT faithful to addition? | Labels learn; wrong singletons returned — **Because** solver loop not sound yet | open (no promote) |
| 56 | [Sound Carry Closure](Experiment%2056%20-%20Sound%20Carry%20Closure/README.md) | Can carry equations prevent impossible singleton returns? | Sound but **0%** coverage — **Because** neural elim killed true path first | open |
| 57 | [Closure Led Elimination](Experiment%2057%20-%20Closure%20Led%20Elimination/README.md) | If closure runs before neural elim, do we get coverage + soundness? | **100%** coverage, wrong=0 all splits — **Because** fixes Exp56 ordering bug | **promote** |
| 58 | [Top State Curriculum](Experiment%2058%20-%20Top%20State%20Curriculum/README.md) | Does top-state warmup before on-policy fix zero coverage? | ~**95%** mean frozen coverage, sound — **Because** random on-policy too harsh cold-start | **promote** |
| 59 | [Soft Elimination Threshold](Experiment%2059%20-%20Soft%20Elimination%20Threshold/README.md) | Can softer threshold fix on-mode elim without turning it off? | On-mode 0%; off-mode **100%** — **Because** threshold still primary eliminator | **promote** off-mode |
| 60 | [Closure Controller Policy](Experiment%2060%20-%20Closure%20Controller%20Policy/README.md) | Can neural side choose branches instead of deleting candidates? | Harness works; policies tie — domain too easy | open |
| 61 | [DeepSeek Two Unknown Addition Policy](Experiment%2061%20-%20DeepSeek%20Two%20Unknown%20Addition%20Policy/README.md) | Can model learn search-state policy on DeepSeek-verified puzzles? | Sound wrong=0; learned ≈ first-cell — no branch pressure | open |
| 62 | [Finite Domain Constraint Policy](Experiment%2062%20-%20Finite%20Domain%20Constraint%20Policy/README.md) | Can one harness cover many constraint types (not one math family)? | All policies 100% on easy tasks — need harder search | open |
| 63 | [Hard Constraint Scale](Experiment%2063%20-%20Hard%20Constraint%20Scale/README.md) | Can we scale to 10k verified constraint rows + train path? | 10k dataset + train path works; eval still too easy | open |

### Verifier harness probes (44–53)

*Theme: Phase 1 shape — noisy text → parse → rank rule → exact verify. Not faithful LDT.*
Detail: [SURVEY_44_53_LDT_STATUS.md](SURVEY_44_53_LDT_STATUS.md)

| Exp | Name | Why (question) | Key result | Verdict |
|---:|---|---|---|---|
| 44 | [Arithmetic Latent Structure](Experiment%2044%20-%20Arithmetic%20Latent%20Structure%20Probe/README.md) | After Exp43 fail, can state labels beat direct-answer heads? | Frozen compose **100%** add/sub/mul — **Because** carry-coupled states not digit slots | **promote** decomposition |
| 45 | [Logic Sparse Field Probe](Experiment%2045%20-%20Logic%20Sparse%20Field%20Probe/README.md) | Does Exp44 sparse-rule shape work on symbolic logic? | Sparse 100% when families present; field 52.4% OOD | **promote** mechanism |
| 46 | [Logic Rule Ranker Probe](Experiment%2046%20-%20Logic%20Rule%20Ranker%20Probe/README.md) | Can learned scorer pick right rule from candidates? | Template-OOD 100%; rule-family-OOD **50%** | split |
| 47 | [Semantic Rule Ranker Probe](Experiment%2047%20-%20Semantic%20Rule%20Ranker%20Probe/README.md) | Is Exp46 failure weak features or missing rules? | Semantic ranker **100%** rule-family-OOD — **Because** generic match features fix OOD | **promote** |
| 48 | [Reusable Semantic Rule Ranker](Experiment%2048%20-%20Reusable%20Semantic%20Rule%20Ranker/README.md) | Can ranking become reusable machinery under wording noise? | 100% noisy OOD; BoW 50% — **Because** component survives paraphrase | **promote** |
| 49 | [Phase0 Adapter Rule Ranker](Experiment%2049%20-%20Phase0%20Adapter%20Rule%20Ranker/README.md) | Can frozen Phase 0 features + tiny adapter rank rules? | **100%** rule-family-OOD — validates Phase0↔Phase1 socket | **validate** |
| 50 | [Logic Text Parser Robustness](Experiment%2050%20-%20Logic%20Text%20Parser%20Robustness/README.md) | Can robust parser recover fields from noisy text? | Robust 100% parse; strict 0% (correct fail-closed) — **Because** membrane before ranker | **promote** |
| 51 | [Raw Text Verified Logic Loop](Experiment%2051%20-%20Raw%20Text%20Verified%20Logic%20Loop/README.md) | Does full loop work: noisy text → parse → Phase0 → verify? | **100%** both OOD splits with robust path | **validate** loop |
| 52 | [Adversarial Parser Boundary](Experiment%2052%20-%20Adversarial%20Parser%20Boundary/README.md) | On adversarial wording, is parsed_wrong always 0? | 0% parsed_wrong; adversarial 100% fail_closed — **Because** safety before capability | **lock** safety |
| 53 | [Generated Parser Stress Suite](Experiment%2053%20-%20Generated%20Parser%20Stress%20Suite/README.md) | Does parser stay safe across generated rule families? | **0/152** parsed_wrong — stress confirms Exp52 boundary | **lock** safety |

### Arithmetic / word reasoning / tools (30–31, 64–75)

*Theme: teach the real Phase 0 checkpoint to reason — with honest verifiers and tool leverage.*

| Exp | Name | Why (question) | Key result | Verdict |
|---:|---|---|---|---|
| 30 | [Arithmetic Reasoning SFT Pilot](Experiment%2030%20-%20Arithmetic%20Reasoning%20SFT%20Pilot/README.md) | Can Phase 0 learn to emit arithmetic CoT chains? | Token acc 92%; frozen gen **4.0%** — format without compute | open |
| 31 | [Frozen Like Arithmetic Curriculum](Experiment%2031%20-%20Frozen%20Like%20Arithmetic%20Curriculum/README.md) | Does tighter curriculum improve exact frozen answers after Exp30? | Frozen gen **26.0%**; mul still bottleneck | open |
| 64 | [Phase0 Verifier Bridge](Experiment%2064%20-%20Phase0%20Verifier%20Bridge/README.md) | What is real Phase-0 pass@1 under strict verifier (not loose match)? | Strict frozen **8.5%** vs loose **55.5%** — **Because** model has CoT shape not digit math | **validate** (measurement) |
| 65 | [Tool Checked Arithmetic Steps](Experiment%2065%20-%20Tool%20Checked%20Arithmetic%20Steps/README.md) | Does student+calculator fix per-step errors on Exp64 checkpoint? | Tool ~62–63% but tool_wrong **37%** on frozen — plan still unreliable | **kill** |
| 66 | [Word Problem Reasoning Corpus](Experiment%2066%20-%20Word%20Problem%20Reasoning%20Corpus/README.md) | Can DeepSeek stories + Python truth build a leak-free 100k corpus? | heldout_word **32%** after 8% epoch — reading undertrained | **kill** pilot; corpus kept |
| 67 | [Multiplication Repair](Experiment%2067%20-%20Multiplication%20Repair/README.md) | Can focused mul SFT fix Exp66’s 0.8% `*` accuracy? | Token loss down; strict `*` **0%** — **Because** repair overfits shape not ops | **kill** |
| 68 | [Exp66 Tool Checked Word Problems](Experiment%2068%20-%20Exp66%20Tool%20Checked%20Word%20Problems/README.md) | Where does tool-check help on Exp66 — reading vs compute? | Word tool **51%** vs raw 32%; direct/hard tool 100% — READ is bottleneck | **kill** (baseline for 69) |
| 69 | [Full Epoch Word Reasoning](Experiment%2069%20-%20Full%20Epoch%20Word%20Reasoning/README.md) | Does full-epoch SFT fix undertrained READING (Exp68 diagnosis)? | Raw word **62.5%**; tool **~98%** overall — **Because** 83× more token exposure on membrane | **promote** |
| 70 | [Comparative Logic Corpus](Experiment%2070%20-%20Comparative%20Logic%20Corpus/README.md) | Can logic corpus + SFT make a strong generator for gating probes? | easy **91%**, hard **83%** — **Because** dedicated logic data beats general pretrain | **promote** |
| 71 | [Body Compression Revival](Experiment%2071%20-%20Body%20Compression%20Revival/README.md) | Are h128-killed body compression lanes revivable at h256 after Exp70 SFT? | Hard **83→77%** (−6 pp) — **Because** compression still breaks logic generalization | **kill** |
| 73 | [RLVR Gaming Probe](Experiment%2073%20-%20RLVR%20Gaming%20Probe/README.md) | Does GRPO on verifiable rewards cause chain gaming on Exp69? | 93.3% acc, gamed_frac **0.0** — **Because** verifier-shaped rewards stay honest | **promote** |
| 74 | [Faithful DistIL Verified Logic](Experiment%2074%20-%20Faithful%20DistIL%20Verified%20Logic/README.md) | Can DistIL overlay learn verified logic traces from Exp70? | **Not run** — planned overlay on logic traces | **planned** |
| 75 | [NextLat Logic SFT](Experiment%2075%20-%20NextLat%20Logic%20SFT/README.md) | Does next_h latent loss improve hard logic vs Exp70 alone? | Best hard **75%** vs **83%** — **Because** latent aux loss hurt generalization | **kill** |

---

*Last indexed: **89** experiment READMEs. Missing folder: **76** only. Each row links to the canonical README for commands, artifacts, and decision rules.*
