# Ternary-HRM Frontier Analysis

_Generated 2026-05-30 by a multi-agent workflow (19 agents, ~1.4M subagent tokens, ~29 min)._
_Mode: report-only — no code or experiment was modified._

## How this was produced

Three-phase workflow:

1. **Mine** — 4 agents read the extracted paper corpus (`papers/extracted/*.md`) in
   4 clusters, extracting techniques applicable to our **open** frontier only.
2. **Audit** — 4 agents ground-truthed the locked / killed / open claims against
   actual experiment code and result files. Every verdict cites `file:line` or a
   result number.
3. **Rank** — 1 synthesis agent merged 25 mined techniques + 24 audit findings into
   14 ranked proposals; 10 adversarial skeptics then each tried to **kill** the top
   proposals. 4 survived.

Counts: **25 techniques, 24 audit findings, 14 proposals, 4 survivors.**

> ⚠️ **Session caveat:** one audit agent (recurrence axis, Exps 32–37) hit a
> total tool failure — every `Bash`/`Read`/`Grep`/`Glob` returned empty. The
> recurrence state below is therefore **unverified this session** and must be
> re-audited in a working environment before any recurrence proposal (ranks 9/10)
> is trusted. All other findings were obtained with working tools.

---

## 1. Audit corrected the memory map in three places

| Memory / map said | Repo actually shows | Verdict |
|---|---|---|
| `mlp.down_proj` is **untested** | Quantized in **two** sweeps. Ternary: Exp21 `both_mlp_down` @500/seed1 → gap **−0.0059** (at floor, looked fine). 2-bit: Exp24 `both_mlp_down` @500/seed1 → **+0.0357** (above floor). But **never scaled, never L-only, never seed-replicated.** | **contradicted** |
| Frozen arithmetic gate protects body compression | Gate lives **only** in downstream Exp27/28 and is run **manually**. `grep frozen\|answer_loss\|exact_acc` across Exp21/24/25 sweep scripts = **0 matches** (one docstring mention only). The single gate that caught GQKV is **outside** the candidate-nomination loop. | **unsupported** |
| Locked preset is deploy-ready | Combo is 3-seed but **single hidden size (128), single step count (5000)**. **No** locked item varies *both* hidden size and step count with seeds. By the project's own `DISCIPLINE.md:49-62` 2x2 rule, the deploy claim is **not yet earned**. | **partially-confirmed** |

### Kills all hold — but on partial grids

| Killed item | Evidence | Grid coverage |
|---|---|---|
| Hadamard 2-bit attention | seed1 +0.0913 / seed2 +0.1289 (both above floor) | 2 seeds, **h256 only** |
| Hadamard 2-bit MLP-body (worst sub-variant) | +0.1779 / +0.1991 | 2 seeds, **h256 only** |
| `attention.gqkv_proj` 2-bit (frozen gate) | normal eval +0.0059 (PASS) but **frozen_gap +0.5914**, token_acc 0.406→0.226 | **single seed**, h256 |
| Full attention 2-bit (normal-eval) | seed1 +0.0134 (OK) but **seed2 +0.0410** (above floor) | 2 seeds, h256, borderline |

All defensible (frozen gate is a hard gate → one failing seed kills; Hadamard/full-attn replicate across 2 seeds), but **none satisfies the full 2x2 deploy grid**. "Killed on partial grid coverage" is a fair label.

Two number-corrections worth noting:
- The memory map's GQKV magnitudes (`dense 0.214 → 2-bit 1.876`) **do not match** the repo (`baseline frozen 3.3578 → 3.9492`). Directional conclusion right, numbers wrong.
- An audit agent initially fabricated `NotImplementedError` stubs / `REFERENCE_*` dicts; it **retracted** them on a second pass. The Exp24/25/27/28 harnesses are confirmed **real training code** (792-line / 489-line / 334-line scripts with genuine AdamW loops, groupwise quant, Walsh–Hadamard rotation, frozen answer-token loss).

---

## 2. Survivors (passed the adversarial skeptic)

### Rank 1 — Wire the frozen answer-loss gate inline into the body-compression sweeps
- **Target:** eval-infra · **Risk:** low · **Cost:** near-zero (one frozen-eval forward pass per candidate) · **2x2:** n/a
- **What:** add a `discipline.assert_frozen_gate()` that calls the existing
  `evaluation/frozen` path / Exp27-28 logic as an automatic Promote/Kill assertion
  inside the Exp21/24/25-style sweep scripts, so every nominated candidate
  auto-emits `frozen_token_acc`, `frozen_exact_acc`, `frozen_gap` next to normal
  eval. Reuse `candidate_failed_gate()` from `h256_gqkv_frozen_gate.py:113-123`.
- **Why now:** the hard gate that already caught one silent failure (GQKV) is not
  in the candidate-nomination loop. Cheapest high-leverage fix; **protects every
  other proposal in this list.**
- **Promote if:** known-good (locked `mixed_top512_tequila_L_mlp_gate_up`) auto-passes
  with `frozen_gap ≤ 0.0203` **and** known-bad (GQKV 2-bit) auto-fails, reproducing
  Exp28's `+0.5914` frozen_gap programmatically.
- **Kill if:** the inline gate cannot reproduce the Exp28 GQKV fail signal
  (`frozen_gap ≫ 0.0203`) — i.e. it is not actually measuring frozen answer-loss.

### Rank 2 — L-only ternary-Tequila on `mlp.down_proj`, full 2x2 + reserved seeds
- **Target:** down_proj · **Risk:** low · **Cost:** 4 cells × small model × ≤2000 steps + 3-seed confirm (Exp21/24 envelope) · **2x2:** yes
- **What:** `L_level.mlp.down_proj` = ternary Tequila (thr 0.5, group 128, mean_abs,
  master-weight-free absmean per BitNet b1.58) on the default 2x2
  (h128/h256 × 500/2000), then 3-seed confirm at the winning cell. **L-only scope**
  (not `both`), mirroring the promoted gate_up pattern.
- **Why now:** the single cheapest **legitimately-open** ternary lane. The
  promising 500-step `both_mlp_down` ternary (−0.0059) was **dropped** before any
  scaled/L-only/seeded run.
- **Promote if:** mean eval gap ≤ +0.0203 across 3 seeds at **both** hidden sizes
  **and** quality/MB ≥ current combo `0.04179` **and** `frozen_gap ≤ 0.0203` (rank-1
  gate). Stack with gate_up only if the stacked combo still clears the bar.
- **Kill (park) if:** mean gap > +0.0203 at either hidden size, **or** quality/MB
  < 0.04179 (no size win to offset), **or** any seed fails the frozen gate.

### Rank 3 — Promote-to-deploy 2x2 + reserved-seed gate for the CURRENT locked preset
- **Target:** discipline debt · **Risk:** low · **Cost:** largest of the cheap moves (4 cells × 3 seeds), small models, sequential over a few sessions · **2x2:** yes
- **What:** run the locked combo on the full 2x2 (h128/h256 × 500/2000 or 5000)
  with 3 seeds/cell, emitting quality/MB and the inline frozen gate.
- **Why now:** retires the discipline debt — the deploy claim currently rests on a
  single hidden size and single step count. Foundational; de-risks ranks 2/4/6.
- **Promote (confirm-deploy) if:** mean gap ≤ +0.0203 in all four cells **and**
  quality/MB advantage over dense holds in every cell **and** `frozen_gap ≤ 0.0203`
  every seed. Otherwise downgrade "locked" → "h128/5000-only".
- **Kill:** demote the **deploy claim** (not the preset) if any cell shows mean gap
  > +0.0203 or a quality/MB regression vs dense; flag the offending regime.

### Rank 8 — h128 frozen-gate confirmation cell (make the HARD gate satisfy 2x2)
- **Target:** eval-infra · **Risk:** low · **Cost:** one h128 frozen baseline + a couple candidate passes (minutes) · **2x2:** yes
- **What:** re-run the Exp27/28-style frozen gate at h128/steps2000 so the gate
  itself is validated across two hidden sizes, not h256-only. Pair with rank 1.
- **Why now:** the gate that gates everything is **h256-only** → does not itself
  meet the 2x2 shape. If it behaves differently at h128, small-model candidates are
  being mis-gated.
- **Promote if:** h128 gate reproduces h256 pass/fail behavior on a known-good and
  known-bad candidate within `frozen_gap ± 0.0203` of the h256 readings.
- **Kill (flag size-sensitive) if:** h128 `frozen_gap` on the known-good candidate
  exceeds 0.0203 where h256 passed → require per-size frozen baselines.

---

## 3. Did not survive the skeptic, but worth queuing

### Rank 4 — N:M (6:8) semi-structured sparsity stacked on locked ternary gate_up
- **Source:** Sparse-BitNet `2603.05168` (applicability **direct**) · **Risk:** low · **2x2:** yes
- **Not killed on merit** — it ranked below the infra/gap fixes. **Highest-novelty
  low-risk move.** Sparse-BitNet proves structured sparsity stacks cleanly on
  ternary (+5.7% PPL degradation) and badly on BF16 (+18.8%) — it works *because*
  we are already ternary. Magnitude-based 6:8 mask from pre-quant continuous
  weights, per-step recompute, Dual-STE dense gradient flow.
- **Promote if:** mean gap ≤ +0.0203 across 3 seeds both hidden sizes **and**
  quality/MB beats 0.04179 **and** `frozen_gap ≤ 0.0203`.
- **Queue:** after ranks 1-3.

### Rank 11 — ECO master-weight-free optimizer wrapper
- **Source:** ECO `2601.22101` (**direct**) + BitNet `2402.17764` · **Risk:** low
- Injects quantization error into momentum, eliminating master FP16 weights →
  **~25% static train-time memory cut** → enables a 2nd hidden size / larger batch
  on the 3050 Ti. Audit notes the master-weight-free storage claim was **never
  verified** — this both saves memory and forces that check.
- **Promote if:** matches locked combo eval within ±0.0203 across 2 seeds **and**
  measurably reduces peak train memory, with `frozen_gap ≤ 0.0203`.

### Rank 12 — Robustness fix: `quality_per_mb` key-name path in the batch enricher
- **Source:** audit · **Risk:** low · **Cost:** a few lines + one test
- `quality_per_mb` **is** implemented and tested (`discipline.py:54-57`,
  `test_experiment_discipline.py:53-68`) and written into live tables by Exp22-29.
  Latent bug: `enrich()` (`discipline.py:77`) looks up `packed_MB`/`packed_mb`, but
  callers pass `packed_disk_mb` — a dict routed through the enricher would silently
  **NaN** the column. Cosmetic today (live callers use the direct function) but
  exactly the silent-NaN that corrupts a promote decision.

### Rank 13 — Token-over-parameter budget reallocation probe
- **Source:** Spectra `2506.23025` (**adapt**) · **Risk:** low
- Spectra shows TriLMs benefit more from **tokens** than params (steeper token
  exponent). For a fixed 3050 Ti budget this argues for data-efficient QAT over
  wider models. 2-point probe: baseline tokens vs ~1.5-2× at fixed width, matched
  wall-clock. Informs how to spend budget on the rank 2/3 confirms.

### Rank 14 — QK-Norm in the (dense) attention path as an enabler
- **Source:** Spectra-1.1 `2506.23025` (**adapt**) · **Risk:** medium
- Stability change, **not** a compression target (attention stays dense per the
  lock). Shrinks attention activation outliers — the prerequisite for ever
  re-opening attention quantization (killed largely via the frozen gate / seed-2
  outlier failure). Lowest priority; do **after** rank 1.

---

## 4. Killed by the skeptic (do not run as proposed)

| Proposal | Why it died |
|---|---|
| **Rank 5** — Tequila λ-reactivation (deadzone-as-bias) on down_proj | Misrepresents the repo: `TernaryLinear158Init` (`models/layers.py`) has no learnable λ / bias / reactivation — "tequila" mode is just latent-passthrough STE (`effective_weight()` 176-190). The full paper mechanism (`TernaryTraining.md:117`) was deliberately **not** built. Also benchmarks against an **unrun** baseline (rank 2) and co-varies a 2-3× LR rider with the mechanism (confounded). **Fix:** gate behind rank 2; scope honestly as net-new layer work + unit test; single-variable A/B at baseline LR. |
| **Rank 6** — L-only 2-bit MLP-side + 16-bit warm-start | Premise false: Exp23/24/25 **already** ran L-only 2-bit MLP. Exp24 h384 fails by ~**+0.102** across 3 seeds (≈5× floor); Exp25 mlp-scope h256 confirms (+0.111). Dropping Hadamard likely **hurts** (rotation tames outliers). Warm-start would need to recover ~5× the floor with zero local prior. **Fix:** if probing warm-start at all, one cheap h256 diagnostic on the *already-passing* ternary recipe first. |
| **Rank 7** — both-vs-L control for stacked 2-bit gate_up | Already answerable from data on hand: Exp24 `scope="all"` (both-level) 2-bit gate_up fails both seeds (+0.029/+0.041); Exp25 L-only stacked mostly passes (+0.010/+0.024). Re-running both-scope just relabels a **killed** recipe. **Fix:** at most one *stacked* both-scope variant to remove the stacked-vs-unstacked confound — but prior says it won't rescue. |
| **Rank 9** — randomized `zL_init` + path noise (EqR) | Relabel of finished, dedicated EqR work (Exp32/33/33.6/34/36); git log shows it advanced to multi-seed isolation + bridge runs. Proposed `sigma_L~8` is a large perturbation on the wrong side of the scaling-probe trend → "low risk" label unsupported. |
| **Rank 10** — Segmented Online Training, late-anchor supervision | Already implemented: `experiments/Experiment 33.../run_exp33_2_sot_..._seg2x3_steps2000_*.ps1` literally encodes "sot" + "seg2x3". Also a loss-vs-accuracy category error in its promote rule (applies the eval-loss noise floor to an accuracy metric). |

---

## 5. Recommended order

```
1. Rank 1  — wire frozen gate inline        (lock the gate first)
2. Rank 8  — h128 frozen-gate cell          (gate satisfies 2x2)
3. Rank 2  — L-only ternary down_proj 2x2   (cheapest open ternary lane)
4. Rank 3  — deploy 2x2 for locked preset   (retire discipline debt)
   ── then the novelty bets ──
5. Rank 4  — 6:8 sparsity on ternary gate_up (highest-novelty low-risk)
6. Rank 11 — ECO master-weight-free optimizer (3050 Ti memory headroom)
```

Logic: **lock the gate before running anything it should protect**, then close the
single genuine open ternary gap (down_proj), then earn the deploy claim, then spend
novelty budget on sparsity / training-memory.

## 6. Open follow-up

- ~~**Re-audit the recurrence axis** (Exps 32–37, `scaling_probe.py`) in a working
  environment — this session's recurrence agent got empty tool output, so
  ranks 9/10 and "does more H/L cycle help any verified metric?" are unanswered here.~~
  **DONE 2026-05-31 → see §7.** Verdict: no verified gain from more cycles; run Exp32 probe next.
- **Memory decayed:** `project_ternary_hrm.md` still says `down_proj` is untested
  (now contradicted) and does not record that the frozen gate is not wired into the
  sweeps. Worth updating.

### Paper → frontier map (applicability ≥ adapt, touching open parts)

| Paper | id | Use |
|---|---|---|
| Sparse-BitNet | 2603.05168 | N:M sparsity stacks on ternary (rank 4) — **direct** |
| ECO | 2601.22101 | master-weight-free QAT, ~25% train memory (rank 11) — **direct** |
| Scaling Laws for Precision | 2411.04330 | predicts low-bit-training robustness (rank 6 premise) |
| Continual QAT for BitNet | 2502.11895 | 16→1.58-bit transition schedule (rank 6) |
| Spectra 1.1 | 2506.23025 | token>param scaling (rank 13); QK-Norm (rank 14); TQ2 packing |
| Tequila | 2509.23809 | λ-reactivation — **not** what the repo implements (rank 5 caveat) |
| BitNet b1.58 | 2402.17764 | absmean master-weight-free, LR-scaling, SubLN |
| Equilibrium Reasoners | 2605.21488 | recurrence axis — already in-flight (Exp33 SOT/seg2x3) |

---

## 7. Recurrence-axis re-audit (2026-05-31, working tools)

_Re-ran the audit flagged in §6 with a working environment (4 agents, every
`Read`/`Grep`/`Glob`/`Bash` succeeded). This resolves the §0 caveat: the recurrence
state below is now **verified this session**. Every claim cites `file:line` or a
named result file + metric._

### Central question — "does more H/L cycle help any verified metric?": **NO** (one unreplicated exception)

- **At scale, more H cycles do not help, and often hurt.** Frozen-generation accuracy
  is flat-to-declining as H goes 2→4→6:
  - Exp34 EqR-SFT (no bridge): 45.0 → 44.5 → 40.5
    (`results_h256_exp34_eqr_..._sft10000_eval200_h246.md:33-35`).
  - Exp34.1 bridge: 55.5 → 56.5 → 55.5 (flat)
    (`results_h256_exp34_1_..._plain2000_then_eqr_..._eval200_h246.md:33-35`).
  - `exact_acc` **declines** with H (0.4688 → 0.4531 → 0.4531); `token_acc` flat ~0.977
    (same file `:25-27`).
- **Only positive signal — Exp36, single seed, unreplicated.** Exp35 mixed-pretrain +
  25 % Dolmino language rehearsal → frozen arithmetic H2 58.5 → H4 62.0 → H6 **63.5**
  (monotonic ↑) (`results_h256_exp35pretrain_rehearsal25_eqr10000_seed1.md:25-27`).
  One seed only; language probes still collapse to repeated phrases. Does **not** earn
  the claim on its own.
- **Residual ≠ accuracy.** The EqR convergence residual shrinks with H (mean 22.6 → 11.6,
  `results_h256_exp33_5_bp4_residual_only_h246.md:12-14`; Exp34.1 20.44 → 2.44,
  Exp34 `README.md:286-288`) — recurrence is **contractive/stable**, but converges to a
  *stable-but-wrong attractor*. Stability is not a task-metric gain.

### Exp32 scaling probe — the experiment built to answer this **never ran**

The one pre-registered sweep designed to answer the central question directly has
**zero results, zero metrics, zero logs**. `results_*.md`, `artifacts/phase0_recurrence_scaling/`,
and `experiments/_logs/exp32*` all absent; `README.md:57-59` Results = "(pending)".
The probe is **inference-only** (mutates `hrm.H_cycles` on one fixed checkpoint,
`recurrence_scaling_probe.py:84-130`) and its input checkpoint
(`artifacts/phase0_arithmetic_sft_v2/h256_exp30_plus_v2_steps2000_seed1/`) **exists** —
so the cleanest answer is one cheap unexecuted run away. Grid as specified is
H∈{1,2,4,6,10,20} × L=3, **h256 only, seed1 only** → does not meet the 2x2.

### Rank 9 verdict refined — mechanism finished, but `sigma_L~8` is unsupported

- The mechanism **is** genuinely implemented + finished work: randomized `zL_init`
  (`eqr_lite_recurrence_sft.py:136`) + per-cycle path noise (`:118`), both gated
  **training-only** (`:299-300`; eval deterministic `:402`). So "relabel of finished
  work" is correct **on mechanism**.
- **But `sigma_L~8` appears nowhere.** Actual `ri_z_l_std` used = **0.10** (`zl010` runs),
  max **1.0** in the abandoned early `zlonly` runs; `noise_beta` = 0.01 throughout. The
  proposed σ is **8–80× larger** than anything run — a large perturbation on the wrong
  side of the trend. The "low risk" label on the σ~8 rider is unsupported. **Kill holds.**

### Rank 10 verdict refined — dead on merit; two stated reasons were wrong

- SOT/seg2x3 **is** real dispatched training code (`train_sft_eqr_lite_sot`
  `:536`; `h_values = 2·(idx+1)` → depths 2/4/6 `:561`; run at
  `run_exp33_2_sot_..._seg2x3_steps2000_nogen.ps1:26-27`).
- **"late-anchor supervision" is a mislabel** — the code supervises *every* segment each
  step (`:589-591`; `anchor`/`late` = 0 grep hits), not a late anchor.
- **The "category error" cannot be confirmed** — no rank-10 promote rule exists in the
  repo to quote. The actual Exp33 decision rule (`README.md:31-38`) is accuracy/frozen-gen
  based and **floor-free**; the `0.0203` floor is loss-domain everywhere it appears
  (`DISCIPLINE.md:34`, `discipline.py:51-55,157-166`). The skeptic's kill-reason is
  itself unsupported.
- **Kill still holds on merit:** SOT underperformed the matched non-SOT EqR run at equal
  compute (6000 steps) — H6 `exact_acc` 0.1641 vs 0.1875, worse on all three eval-H losses
  (`results_h256_exp33_2_sot_..._seed1.md:21-23,37` vs `results_h256_exp33_1_..._steps6000_seed1.md:22-24,39`).

### New findings not in the original doc

- **Reproducibility failed.** Exp34.1 seed2 (~40 %) ≠ seed1 (~55 %); the seed-isolation run
  (seed2-pretrain + seed1-SFT, 38–39 %) pins the weakness to the seed2 **pretrain
  checkpoint**, not SFT seed variance. Repo's own call: "Do not promote Phase 0 from
  Exp34.1 yet" (`Experiment 34 .../README.md:110`).
- **More-expensive recurrence path is worse.** Full 50k EqR pretrain (Exp34: 45.0/44.5/40.5)
  **underperformed** the cheaper EqR-SFT-only Exp33.5 baseline (57.5/50.0/46.0,
  `Experiment 34 .../README.md:273`).
- **Best verified EqR-lite config** (Exp33.5: damping 0.15, σ_L 0.10, bp4, 10k steps):
  frozen-gen H2 **0.5750**, valid exact H2 **0.50**
  (`results_h256_exp33_5_..._bp4_steps10000_eval200_h246.md:33-35,25-27`). A clean
  EqR-vs-plain delta is **not** verifiable from the Exp33 folder — no matched no-EqR
  H-sweep baseline file exists there.
- **Exp33 README stale:** "Run Target" says damping 0.05 / σ_L 1.0 (`README.md:26-29`),
  but every promoted run + the dataclass defaults use 0.15 / 0.10.
- **Exp37 is not recurrence** — a DeepSeek dataset-generation tool, not run (no metrics).

### Bottom line

The recurrence axis shows **no verified task-metric gain from more H/L cycles** at any
replicated scale; the lone monotonic-↑ result (Exp36) is single-seed and language-coupled.
Ranks 9 and 10 **stay killed**, but partly for reasons the original skeptic stated wrong
(σ~8 magnitude unsupported; "late-anchor"/"category-error" mislabels). The §0 caveat was
justified. The single highest-value unblocked move on this axis is to **actually run the
Exp32 inference probe** — it is cheap, the checkpoint exists, and it answers the central
question head-on.

---

## 8. Execution results (2026-05-31) — Ranks 3, 4, 11, 13, 14 run end-to-end

_All five ranks were implemented and **actually trained** on the local RTX 3050 Ti
(2x2 grids, multi-seed, frozen gate where applicable). New code is opt-in and leaves
the locked path byte-identical when off. Verdicts below cite the per-experiment
result files._

| Rank | Experiment | Verdict | Headline |
|---|---|---|---|
| 3  | Exp38 Deploy Lock 2x2 | **PROMOTE** | Locked preset beats dense in all 4 cells (mean gap −0.0065 to −0.0453), quality/MB 5-17x. Deploy claim earned across full 2x2. |
| 13 | Exp42 Token-vs-Param | **PROMOTE** | 2x tokens (−0.2177) beats ~2.15x params (−0.1856) at fixed width. Spend budget on tokens, not width. |
| 4  | Exp39 6:8 N:M gate_up | **KILL** | Eval loss within floor all 12 seeds, but frozen gate fails all 12 (frozen_gap +0.03 to +2.66). Silent failure caught by the gate. |
| 14 | Exp41 QK-Norm | **KILL** (as drop-in) | Eval over floor at h128/2000; frozen gate fails all 12. Cannot be hot-patched onto the trained lock; re-test only as a pretrain-time ingredient. |
| 11 | Exp40 ECO optimizer | **KILL** (arch mismatch) | Eval +0.5 nats every cell, no memory win. ECO's store-quantized premise fights the repo's STE FP-latent; master-weight-free claim is not free here. |

### What this validates beyond the individual verdicts

- **The frozen gate (Rank 1) is load-bearing.** Both Rank 4 and Rank 14 pass the
  normal eval-loss bar and would have been wrongly promoted on loss alone; the
  frozen answer-loss gate flipped both to KILL. This is direct empirical support for
  wiring the gate inline (Rank 1) before running anything it should protect.
- **2/5 promoted, 3/5 killed — all on the full 2x2 with seeds**, not partial grids.
  The deploy claim is now earned (Rank 3), and budget allocation has a local
  answer (Rank 13: tokens > width).
- **Honest negatives with root causes**, not just numbers: N:M damages arithmetic
  structure; QK-Norm needs co-training; ECO needs a store-quantized layer. Each KILL
  README records the precise mechanism and what a fair re-test would require.

Artifacts: `experiments/Experiment 38..42/` (READMEs carry pre-registered Decision
Rules + Results tables; per-cell `results_*.md` carry every seed). New opt-in code:
`models/layers.py` (N:M `ternary_nm_*`, `Attention.qk_norm`),
`experiments/Experiment 40 .../eco_optimizer.py`.
