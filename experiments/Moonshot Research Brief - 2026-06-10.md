# Moonshot Research Brief — Bounded Post-Turing Verified Reasoning on Laptop Hardware

**Date:** 2026-06-10 · **Rev b (owner correction):** "1000×" redefined frontier-first — comparators are GPT-5.5-class / Opus-Fable-class measured via API on our own sets; same-VRAM 7B local demoted to non-headline sanity ladder (L1). §§1, 2, 4, 5 (Exp112), 6, 8, 9 and footer revised.
**Rev c (2026-06-13, renumber):** experiment IDs moved **Exp90–94 → Exp112–116** so the machine milestones sort *last*, matching execution order (the machine consumes every other brief — Verifier, Memory, Architecture, Training — so it runs after them, and its IDs should not read as "first"). Mapping: M1 BPTM-1 Exp90→**112** · BPTM-2/M3 Exp91→**113** · M2 TNM Exp92→**114** · M4 TAM Exp93→**115** · M5 MDB Exp94→**116**. No experiment had run under 90–94 (spec-only), so no result files break. Canonical run order: `EXECUTION_ORDER.md`.
**Scope:** Long-horizon system design. Research only; no code changed. Sequenced AFTER the performance lane (Exp79/81) and the parked training-efficiency lane — this brief informs direction, not this week's queue.
**⚠ Run order ≠ ID order.** These machine milestones (Exp112–116) are the *last* tier despite once carrying low numbers — they consume every other brief. Canonical schedule: [`EXECUTION_ORDER.md`](EXECUTION_ORDER.md).
**Companions:** `VISION.md` (phases, failure modes) · `Architecture Research Brief - 2026-06-10.md` (90-day bets C1–C5) · `Training Efficiency Brief - 2026-06-10.md` (envelope, parked) · `Experimental Log Summary.md` (ground truth). `papers/extracted/` (20+ extractions, ternary/low-VRAM corpus) skimmed; consistent with the papers/ summaries already cross-referenced.

---

## 1. Executive summary

- **Moonshot claim (falsifiable, dated):** by 2029-06, the integrated machine — ≤5 MB packed control unit + ≤64 MB tape + ephemeral fast store + strict halt oracles + tool coprocessors + verifier-gated self-edit, trained on RTX 3050 Ti only — achieves strict verifier-picked pass@1 **equal to or above frontier datacenter models** (GPT-5.5-class, Claude Opus/Fable-class, and successors) on **≥3 of 5 verifier domains** at stated test-time budget tiers, with every headline number ablated against a null machine and a model-only baseline.
- **"1000×" (owner intent, revised 2026-06-10):** not “beat a 7B that fits in 4 GB VRAM.” It is a **storage-density phase change**: ~**1000× less persistent weight bytes** (and orders-of-magnitude less train compute) than frontier, while matching or beating frontier on **strict programmatic verifiers** on our frozen/held-out sets. Early computing stored KB in a building; terabytes now fit in a palm — the SSD looked impossible until the medium changed. We target the same category shift for **reasoning procedure in ROM**, not a faster tape drive (§2.1).
- **Single highest-leverage integration:** the **verifier as halt oracle + selector inside a revise-loop** (BPTM-1, §5). It converts test-time compute — the one resource we have in surplus — into verified accuracy at zero packed bytes, and it is the substrate every other multiplier (tools, tape, self-edit) plugs into. PTRM's 87.4→98.75% Sudoku jump and the entire pass@k literature say this is the cheapest order of magnitude available.
- **Single hardest impossibility: the zero-support frontier.** Selection cannot create probability mass. K-sampling, verified filtering, and RLVR only harvest tasks where the model's prior puts p ≳ 1/K on a correct trajectory; tasks at p≈0 stay unsolved at any K. (This generalizes the RLVR-sharpens-but-doesn't-create result.) Every escape route is named in §4: curriculum-through-strata (the compounding bet), coprocessors (re-parameterize the task), tape (shift the prior with worked examples), representation (Abacus-class). If none move a domain's frontier, that domain's claim downgrades — pre-registered.
- **Three historical ideas that survive into the stack:** (1) **fast-weight programmers** (Schmidhuber 1991–92) → Exp77/C3 overlay, now trainable with STE+TTT objectives the 1990s lacked; (2) **Gödel-machine governed self-modification** (Schmidhuber 2003) → Phase 6, with the unimplementable proof obligation replaced by a *strict verifier + frozen held-out gate*; (3) **Kanerva's sparse distributed memory** (1988) → Phase 5 tape, now addressable with dense embeddings + ANN search.
- **The anachronism check passes:** what killed these in the 1990s was the absence of (a) sound cheap halt oracles, (b) abundant test-time search compute, (c) stable low-bit training recipes, (d) anti-collapse knowledge (replay mixing). We have all four. AlphaProof is the datacenter-scale existence proof that *verifier-gated self-training compounds* when the verifier is sound; our bet is the laptop-scale version with programmatic verifiers instead of Lean.
- **Two negative results define the design, not just the risks:** LLMs cannot intrinsically self-correct (Huang et al. 2023) — so revision must be verifier-driven, never self-judged; models collapse on recursive self-generated data (Shumailov et al., Nature 2024) — so the trace buffer is verifier-filtered and replay-mixed by construction (Exp79 already implements 0.25 replay).
- **Killer dependency, stated once and loudly: the entire tower stands on verifier soundness.** Every §4 multiplier — selection, revision, self-edit — routes through the halt oracle; a grading hole corrupts all of them *silently and compoundingly* (wrong picks at inference, wrong targets in training). Not hypothetical: the 2026-06-10 repo audit found real edge cases in `ArithmeticExactVerifier` (spaced negative numbers extracted sign-dropped; leading-zero acceptance) while 7/200 frozen answers are negative. Verifier hardening (audit task-plan T5/T6 lane, in flight with the implementation agent) is **moonshot critical path, upstream of M1** — and every new domain verifier ships with an adversarial wrong-candidate suite (VISION Phase 1 exit) before its domain enters the headline count.
- **TRM is the proof of existence for the parameter claim:** 7M params beating R1/o3-mini-class on ARC-AGI is a published 1000×-class result on one domain tier. We are not arguing possibility; we are arguing *system generality across verifier domains* — that is the actual moonshot.
- The machine spec (§3) is implementable as a state machine + oracle calls + self-edit rules without guessing neural internals; the control unit is swappable (today h256 HRM; C5 may shrink it).
- Multiplier math (§4): word-arithmetic-with-tools may already approach **frontier parity** on Tier B (Exp69 ~98% — must be confirmed vs F1/F2 on held-out, M1); logic reaches moonshot only if the SAT-coprocessor lane (M2) lands; strict in-weights arithmetic is **formally excluded** from headline moonshot count — honesty gauge (Tier C).
- **Self-edit is central but conditional.** BPTM-2 (M3) is the decisive experiment: 3 governed cycles, each gated on strict held-out non-degradation. Published evidence (STaR/V-STaR: +4–17 pp over 2–4 iterations, then plateau) supports *bounded* compounding, not unbounded. If two M3 attempts fail with different replay mixes, self-edit is demoted to "one-shot verified data engine" and the moonshot downgrades to: frontier parity only where already published (Tier E via TRM-class replication) plus saturated Tier B — everything else re-labeled "local win only" — stated now, in writing.
- Milestone ladder: **M1** machine loop + **frontier scoreboard** (F1/F2 API + N0; L1 optional) (≤3 mo) → **M2** tool-native logic (≤6 mo) → **M3** single-domain compounding (≤12 mo, flagship) → **M4** tape + puzzles + code entry (≤24 mo) → **M5** multi-domain bootstrap = Phase 6 exit (≤36 mo).
- Total moonshot compute is laptop-shaped: one self-edit cycle ≈ 1–2 GPU-hours at h256 (generate K=8 over ~1k tasks + verify + SFT + re-eval). The flagship M3 experiment costs less than the Exp29 pretrain did. The constraint is discipline, not FLOPs.
- **Primary comparators are frontier APIs** (F1/F2 in §2) on our frozen sets — same strict verifier, tool-parity rules documented. Local 7B/3B rows are **ladder/sanity only**, not the moonshot bar. M1 must include at least one frontier row per affordable domain.
- Everything here respects closed lanes: deploy compression stays closed (Exp71), plain-HRM depth stays dead (Exp33.5/34.1 + Misconceptions), in-weights multiply stays refused (Exp67), retrieval stays parked until M4 with the copy-gaming kill armed.
- What we refuse to build even for the moonshot: §8. Headline refusals — learned verifiers, ungoverned self-modification, Turing-completeness claims.

---

## 1b. Contradiction & assumption register (Pass 0 — surfaced, not smoothed over)

| # | Tension | Resolution / standing bet |
|---|---|---|
| 1 | Loose 55.5% vs strict 8.5% on the **same checkpoint** (Exp34.1 × Exp64) | any capability number without its verifier named is void; every moonshot number is strict-verifier or it does not exist |
| 2 | VISION Phase 4 "recurrence = scratchpad" vs Exp33.5 depth *hurts* + Misconceptions audit (H-module removable, 2–4 effective steps) vs CMM (attractor-trained depth works) | recurrence demoted to conditional multiplier (0–0.7 log), C4-gated, constraint domains only; machine default keeps R ≤ 8 |
| 3 | TRM frontier wins (Tier E spearhead) vs TRM's ~1000× augmentation + per-task training confound | asymmetry not hidden: per-task adaptation IS the machine paradigm (fast store, self-edit); in exchange §2.2 budget parity gives frontier its native extended-thinking mode |
| 4 | Exp73 "RLVR not gamed" vs literature "RLVR sharpens, doesn't create" | both true — honesty ≠ new capability; rlvr mode = sampler-sharpener; tool_supervised = the only new-signal mode; BPTM-2 leads with it |
| 5 | Exp69 98% tool path vs "frontier parity" | 98% is on OUR distribution, unverified vs F1/F2 until the M1 scoreboard; owner's saturated-parity rule applies only after that row exists |
| 6 | Moonshot M4 (Tier E pretrain) vs parked Training-efficiency lane | M4 transitively requires unparking Exp87/88 + the C5 spec — a scheduled debt declared in §7, not an ignored one |
| 7 | Breadth multiplier assumes sample diversity vs Exp34.1 seed-2 repro failure (~40%) hinting distributional fragility | load-bearing assumption; Exp81's K-curve is the direct test — a flat curve kills the cheapest multiplier and re-scopes BPTM-1 before anything is built |
| 8 | Verifier labeled "strict" vs audited grading holes (spaced negatives; leading zeros; 7/200 frozen answers negative) | hardening precedes M1 (critical path); verifier code + frozen sets are hash-pinned into every self-edit cycle log (§3.5) |

---

## 2. "1000×" definition & comparator table

**Resource class (ours):** packed persistent weights ≤ 5 MB · tape (retrieval index + work log) ≤ 64 MB · ephemeral fast store ≤ 50 MB, wiped per task · inference working set ≤ 4 GB VRAM · test-time budget tiers defined below. Train side: 3050 Ti only (Training brief envelope).

**Test-time budget tiers** (every reported number names its tier):

| Tier | Budget | Intent |
|---|---|---|
| B0 | 1 sample, ≤8 recurrence steps, 0 revise | single-shot honesty |
| B1 | K=16 samples × ≤2 revise rounds | standard machine operation |
| B2 | ≤256 rollouts, search allowed | frontier harvest / trace generation |

**Comparators (measured on OUR frozen/held-out sets; same strict verifier for all rows):**

| # | Comparator | Persistent store (order of magnitude) | Role | How measured |
|---|---|---|---|---|
| **F1** | GPT-5.5-class / latest OpenAI reasoning model | datacenter frontier (TB-scale train; GB–TB weights) | **primary moonshot bar** | API on frozen JSONL; tool-parity per §2.1 |
| **F2** | Claude Opus / Fable-class (latest Anthropic) | datacenter frontier | **primary moonshot bar** | API, same protocol |
| **F3** | Other published frontier (o3, Gemini Deep Think, etc.) | datacenter | secondary | API where affordable + literature |
| **L1** | Qwen2.5-7B / Llama-3.2-3B-class, 4-bit GGUF | ~2–4 GB packed | **ladder/sanity only** — not headline | local inference on 3050 Ti, optional |
| **P1** | TRM / PTRM (5–7M, puzzle specialist) | ~3–28 MB | proof-of-existence peer; Tier E | literature + replication at M4 |
| **N0** | **Null machine**: coprocessors + verifier + generator, **0 MB model** | 0 | integrity floor | local, every domain |

**Storage-ratio reporting (mandatory for "1000×" claims):**

```text
storage_ratio ≈ frontier_persistent_bytes / our_packed_MB     # target ~10³–10⁶
train_ratio   ≈ frontier_train_cost / ours (3050 Ti only)    # estimate bands; flag uncertainty
capability_d  = strict_pass@1 on domain d vs max(F1, F2)
moonshot_win  = capability_d ≥ frontier AND storage_ratio ≳ 1000× on headline domains
```

**Validity rules:** (a) same strict verifier grades all rows; (b) beat **N0** and model-only ablation; (c) invalid 0%, no held-out leakage; (d) budget tier stated; (e) **beating L1 alone is not moonshot** — label "local win only." **Beat-today vs must-build:** Exp69 ~98% word+tools is promising but must be confirmed vs **F1/F2** on held-out (M1); logic, puzzles, code require M2–M4.

### 2.1 Storage-density thesis (category shift, not compression)

Early computers: **kilobytes in a building**. Today: **terabytes in a palm**. That jump was not incremental tape improvement — it required a **new medium** (flash), **new addressing** (random access), **new controller** (FTL, ECC), and **new use patterns** (files, not spools). A 1960s engineer shown a palm SSD would say **impossible** because they were extrapolating inside the tape paradigm.

**BitNet-HRM maps the same story to reasoning:**

| SSD revolution | This stack |
|---|---|
| New medium | Ternary **ROM** — packed control program, not FP weights in HBM |
| New addressing | **Verifiers + tools** — halt oracles and coprocessors, not "hope logits are right" |
| New controller | **Post-Turing loop** — step, revise, wipe fast store, governed self-edit |
| New use pattern | **Verifiable tasks only** — not storing the internet in weights |

We are **not** building a smaller frontier LLM. We are building the **SSD of reasoning**: palm-sized persistent store, frontier-class outcomes on strict verifiers — if the encoding is right. Hardware horizon (`papers/ComputeWall.md`): mature-node ROM-heavy silicon, clock-speed-bound loop engine — not leading-edge litho for hoarded parameters.

### 2.2 Frontier eval protocol (F1/F2/F3) — binding requirements

1. **Same everything:** identical task JSONL, identical programmatic verifier, identical pass/fail for us and frontier. Frontier outputs go through the same extractor; extraction failures count against that row and are reported, never hand-fixed.
2. **Contamination:** our sets are synthetic, seeded, locally generated (Phase 0.5) — exact-item leakage ≈ nil today; task-*family* familiarity for frontier is certain (GSM8K-style word problems are in every frontier corpus). Report headline **with and without hardest strata**. **API-leak guard:** sending the 40 held-out IDs to a third-party API exposes them to provider logs — frontier rows therefore run on frozen train-visible + **fresh-seed probe sets drawn from the same generators**, never the held-out 40; probe sets are versioned and regenerated after each audit.
3. **Tool parity (2×2 panel):** if our machine uses calculator/SAT/sandbox, frontier gets equivalent tool access (code-interpreter / function calling). Report all four cells: {us, frontier} × {raw, tool-augmented}. A win that exists only when frontier is tool-starved is labeled **tool-asymmetric** and does not count toward the headline.
4. **Budget parity:** our B0/B1/B2 declared; frontier gets matched K attempts where the API allows, plus its native extended-thinking mode at B1 — we do not handicap frontier to its single cheapest call. Report K, revise budget, temperature, latency for both sides.
5. **Cost column (part of the asymmetry story):** every row carries cost — frontier: API $ per sweep (~500 tasks × 1–2k tok ≈ **$5–50 per model at B0/B1**; B2 at K=256 is priced out → literature/spot checks only, labeled); ours: laptop GPU-h at ≈$0 marginal; train side: 3050 Ti hours vs frontier ~10²⁵–10²⁶ FLOPs.

**Refined moonshot metric** (replaces the draft formula's FLOPs division — dividing by test-time compute punishes our cheapest asset): report a **Pareto panel**, not a scalar:

```text
For each domain d and budget tier B:  strict_pass@1_d(B) × difficulty_coverage_d
Persistent-store denominator only:    packed_MB + λ·retrieval_MB   (λ≈0.1; tape is cheap, disk-resident)
Ephemeral store + test-time FLOPs:    reported as budget-tier axes, never divided out
Validity gates (hard, not weighted):  invalid=0, no-leak, null-machine margin, model-only margin
Headline:  domains at frontier parity max(F1,F2) (count) + storage_ratio panel + full Pareto
```

---

## 3. Post-Turing machine specification (moonshot target state)

A reader should be able to implement this as a state machine + oracle calls + self-edit rules. Neural internals are a swappable implementation detail of the control unit.

### 3.1 Control unit (slow store / ROM)
Packed ternary HRM, ~4.64 MB deploy preset (h256-class; C5 may replace with a tied recursive block — the spec is agnostic). **Fixed at inference.** Its only jobs: READ (parse task into internal/tape representation), PLAN (choose next action: emit, call coprocessor, revise, halt-request), TRANSLATE (formalize for coprocessors — the membrane). It is *forbidden* from being the arithmetic ALU (Exp67 kill; two-lever thesis).

### 3.2 Step semantics — one CPU cycle
```text
CYCLE(state):
  1. READ    : attend over tape segment (input + retrieved frames + work-log tail)
  2. THINK   : ≤ R recurrence steps (HRM H/L; R from budget tier; plain-HRM evidence says R small)
  3. ACT     : emit one of
                 a) candidate answer/step        → goto 4
                 b) coprocessor call (typed ABI) → execute, write result frame to tape, goto 1
                 c) revise request               → re-enter 1 with verifier evidence frame appended
  4. CHECK   : halt oracle on candidate (strict verifier; partial-step verify where domain supports it)
  5. LOG     : append (state, action, verifier result) to work log — append-only, non-differentiable
  6. BRANCH  : pass → HALT(accept)
               fail & budget left → revise (verifier error string becomes tape frame) or resample
               budget exhausted   → HALT(fail)   # invalid never escapes as "answer"
```
Step budget, K, and revise rounds are owner-set constants (§9 Q1/Q7). No unbounded loops anywhere.

**Implementer's I/O contract (Pass 5 — no neural knowledge needed):** the control unit is a black box `f(context_string) → action_string`. READ = serialize tape frames into `context_string` in fixed order (T0 task, last-k T1 frames, retrieved T2 frames) under a context cap. ACT = parse `action_string` against a closed grammar — `ANSWER: <text>` | `CALL <tool_id> <json_args>` | `REVISE` — where the parser is a ~50-line host dispatcher, not the model; malformed actions are automatic CHECK-fails, never "interpreted." Everything else in the cycle — verifier call, tool execution, logging, branching, budget counters — is ordinary host code. Swapping the neural checkpoint changes nothing outside the black box; BPTM-1 is implementable from this section alone.

### 3.3 Tape hierarchy & persistence rules
| Layer | Contents | Writes | Persistence | Wipe rule |
|---|---|---|---|---|
| T0 input | task text | read-only | per-task | always |
| T1 working tape | coprocessor results, verifier evidence, revise frames | machine, append-only | per-task | wiped at halt |
| T2 retrieval index (Phase 5, M4) | worked examples, verified solutions, domain facts | offline build + governed ingest only | persistent, ≤64 MB | never written at inference; copy-gate guards reads |
| T3 work log / trace buffer (Phase 3) | full episode traces, success AND failure | machine at halt | persistent until ingested | held-out IDs refused at write (guard_rail) |
| Fast store (RAM) | overlay/TTT deltas (C3 arc) | per-task TTT rule | ephemeral | **wiped at halt, unconditionally** |

### 3.4 Halt policy
Legal halts: **verifier-pass** (accept) or **budget-exhaust** (fail, logged with best failed trace). The verifier is the *only* accept authority — the model never self-certifies (Huang 2023 makes self-judged revision a known failure). Revise-and-continue: verifier evidence (error string, failed-step index) is appended as a tape frame and the cycle re-enters; bounded by revise budget. Partial-step verification (tool_check_steps, LDT closure soundness) is used where the domain provides it — it collapses the search tree early, the single biggest search-efficiency lever rStar-Math demonstrates with process rewards (ours are exact, not learned).

### 3.5 Self-edit policy (Phase 6 — the post-Turing part)
The machine may update its own transition table (weights) **only** through this pipeline:
```text
traces (T3) → verifier-filter (strict pass for targets; failures kept for contrastive/DPO use only)
            → guard_rail held-out refusal (mandatory, fail-loud)
            → replay-mix SFT (≥25% original human/curriculum data — anti-collapse, Shumailov 2024)
            → re-eval: frozen sets + held-out strict
            → ACCEPT new weights iff strict held-out non-degrading AND no failure-mode flags
            → else ROLLBACK (previous checkpoint retained; cycle logged as failed)
```
**What may change:** control-unit weights (SFT), retrieval index contents (governed ingest of verified solutions), revise-policy hyperparameters (K, temperature schedule) via recorded experiments. **What may never change:** verifiers, frozen/held-out sets, the guard rail, the wipe rules, the coprocessor implementations.
**Forbidden ingest transitions (hard refusals, fail-loud):** (1) any trace whose task_id is held-out — guard_rail, mandatory; (2) gamed traces — Exp73's `gamed_frac` monitor: chain-validity falling while answer-accuracy rises quarantines that cycle's entire buffer; (3) verifier-failed candidates as positive targets — failures enter only as contrastive/DPO negatives; (4) any cycle where one domain exceeds the 50% dominance cap (MDB gate); (5) any trace generated while verifier code or frozen sets changed mid-cycle — both are **hash-pinned into the cycle log**, so a drifted checksum voids the cycle automatically. This is the Gödel-machine shape with the proof obligation replaced by an empirical gate: sound-but-incomplete, which is the correct trade — Gödel machines never ran because proofs of improvement are unobtainable; held-out verification is obtainable every cycle.

### 3.6 Coprocessor ABI
Typed syscall boundary: `call(tool_id, args) → result_frame` with tools = {exact calculator (Exp65), SAT/SMT solver (M2), code sandbox + hidden tests (M4), closure/lattice solver (Exp57)}. Rules: results are **trusted facts** written to T1 — the control unit may not overwrite or "correct" a coprocessor output; every call is logged with args (auditable); the model's score contribution is exactly the READ/PLAN/TRANSLATE work, enforced by the null-machine ablation (if the null machine matches the full machine, the model added nothing — kill or fix). COMPUTE in coprocessors, READ in weights — Exp68/69's two-lever result elevated to an ABI invariant.

### 3.7 Machine loop diagram
```text
            ┌─────────────────────────── BOUNDED POST-TURING MACHINE ──────────────────────────┐
            │                                                                                   │
 task ──► [T0 input]                                                                            │
            │        ┌──────────────┐   think ≤R steps   ┌─────────────┐                        │
            ├──────► │ CONTROL UNIT │ ─────────────────► │  ACT:       │── candidate ──► ┌──────────────┐
            │        │ ~4.6 MB ROM  │                    │ emit/call/  │                 │ HALT ORACLE  │
 [T2 retrieval] ───► │ (READ/PLAN/  │ ◄── fast store ──  │ revise      │── tool call ──► │ strict       │
  ≤64 MB, M4         │  TRANSLATE)  │     (RAM, wiped)   └─────────────┘       │         │ verifier     │
            │        └──────────────┘                                          ▼         └──────┬───────┘
            │               ▲                                          ┌──────────────┐  pass   │  fail+budget
            │               │ evidence frames                          │ COPROCESSORS │         │      │
            └── [T1 working tape] ◄────────────────────────────────────│ calc/SAT/    │  HALT   │   revise loop
                            ▲                                          │ sandbox      │ (accept)│  (bounded)
                            └── verifier error strings ◄───────────────└──────────────┘         │
                                                                                                │
   [T3 work log] ◄── every episode (success + failure) ◄────────────────────────────────────────┘
        │
        ▼  Phase 6 (offline, governed)
   verifier-filter → guard_rail → replay-mix SFT → frozen+held-out gate → ACCEPT/ROLLBACK weights
```

---

## 3b. Historical constraints canon → modern mapping

| Historical idea | Era | Constraint addressed | Repo analogue | Gap / opportunity |
|---|---|---|---|---|
| Fast weights / fast-weight programmers (Hinton & Plaut 1987; Schmidhuber 1991–92) | 1987–93 | short-term writable store without huge RNN state | Exp77 overlay + HyperBuilder; C3 arc | site+objective wrong in Exp77 (oracle 0/64); In-Place TTT supplies the corrected recipe; wipe rules now formal (§3.3) |
| Dynamic link architecture (von der Malsburg) — backward-chase from Hinton & Plaut | 1981 | binding via fast synaptic modulation, not unit activity | conceptual ancestor of the overlay: state lives in *connections*, transient per stimulus | confirms the wipe discipline: 1981's binding was per-episode transient — our unconditional per-task wipe (§3.3) is the same rule, now enforceable in code |
| Hypernetworks — slow net programs fast net | 1991→2016 | parameter reuse under memory limits | `models/fast_weight_overlay.py` HyperBuilder (ROM→RAM) | builder is trainable-side only; keep out of packed denominator (owner Q, §9.4) |
| Hopfield / attractor nets (Hopfield 1982) | 1982–90s | constraint solving via dynamics, no search hardware | C4 CMM attractor recurrence (gated) | 1980s capacity 0.14N + spurious minima → modern Hopfield (2020) exponential capacity; CMM training recipe makes depth pay on constraint domains only |
| Sparse distributed memory (Kanerva 1988) | 1988 | huge effective memory on tiny address hardware | Phase 5 tape (T2) | addressing solved by embeddings+ANN; new failure is copy-gaming — gate pre-registered in VISION Phase 5 exit |
| Turing's oracle machines (1939) | 1939 | computation relative to an oracle | verifiers + coprocessors = oracle calls | honest frame: machine computes in P^O; model's value = reducing tasks to oracle calls; null-machine ablation keeps it honest |
| Minimal machines — tag/register, 2-register universality (Minsky 1961/1967) | 1936–67 | tiny control + tape suffices; control complexity trades against steps/tape | 4.6 MB control + tape + step budget | supports tiny-control thesis: don't grow ROM, grow cycles/tape — exactly the §4 multiplier ordering |
| Gödel machine / self-modifying code (Schmidhuber 2003) | 2003 | provably-beneficial self-modification | Phase 6 self-edit (§3.5) | proof obligation → strict verifier + frozen held-out gate; rollback always retained |
| Blackboard systems — Hearsay-II (Erman et al. 1980) | 1970s–80 | multiple specialists share a workspace | T1 working tape + coprocessor frames | their knowledge sources couldn't learn; our control unit does — the missing piece was the trainable generator |
| Case-based reasoning (Kolodner 1992; Schank) | 1980s–93 | reuse solved episodes: retrieve→adapt→verify→store | Phase 3 traces + Phase 5 retrieval + Phase 6 loop — CBR's loop, literally | CBR died on weak adaptation; the neural control unit *is* the adapter; keep CBR's lesson: index by problem features, not surface text |
| Fixed-point / limited-precision nets (Höhfeld & Fahlman 1992; 1990s VLSI) | 1990–94 | train under bits-starved hardware | entire ternary lane (Tequila STE, Phase 0) | they lacked STE + scaling laws; stochastic-rounding insight survives in Tequila/QuEST lineage |
| NTM / DNC (Graves et al. 2014; 2016) | 2014–16 | differentiable external tape | T1–T3 tape hierarchy | the lesson is what NOT to do: differentiable tape was untrainably fragile — our tape is discrete, append-only, ingested via SFT, never backprop-through-memory |

---

## 4. Multiplier accounting

Factors as log₁₀ effective-parameter-equivalents, evidence-linked. "×-equivalent" = how many more packed params a plain single-shot dense model would need for the same strict score (anchored to published pass@k-vs-scale curves and our own lanes; ranges deliberately wide — these are planning numbers, the M1 scoreboard replaces them with measurements).

**Post-correction reframe (Pass 1):** under the frontier definition, the orders of magnitude live in the **denominator by construction** — storage_ratio is 10⁴–10⁶ the moment we ship ≤5 MB. The multipliers' job is to close a **capability gap measured in percentage points, not orders of magnitude**: from our raw pass@1 to the frontier band, per domain. That is why a product of modest factors (each ≤1.5 log) suffices — nothing below needs to manufacture 1000× capability, and any recipe pitched as if it did should be distrusted on sight.

| Factor | log₁₀ range | Evidence anchor | Failure mode (VISION) |
|---|---|---|---|
| Verifier-K breadth + revise (B1) | 1.0–1.5 | PTRM +11 pts no retrain; pass@k curves (small-model pass@16 ≈ 10–30× larger model pass@1); Exp81 will measure ours | trivial-true harvesting; flat-K mode collapse |
| Coprocessors (exact tools) | 1.0–2.0 (compute-bound domains); ~0 elsewhere | Exp69 62.5→98%; no dense model does exact 30-digit arithmetic in weights — unbounded in the limit, capped by READ quality | oracle laundering — null-machine ablation mandatory |
| Strict halt oracle vs none | qualitative gate, not a multiplier | Exp64: loose 55.5% vs strict 8.5% — verifiers don't add capability, they *reveal* truth and enable everything above | weak-verifier contamination |
| Recurrence cycles | 0–0.7, constraint domains only | plain HRM dead (Exp33.5); CMM 5M ≈ 27M+ HRM on Sudoku | overthinking; gated C4 |
| Fast store (TTT/overlay) | 0–0.8, unproven here | TTT-ARC: up to 6× accuracy on abstract reasoning; Exp77 currently 0/64 | overbuilt architecture — one-retry-then-kill (C3) |
| Tape / retrieval | 0–0.5 on procedure-dense domains | RAG lifts knowledge tasks ~10×; our domains are procedure-dense — modest until code/puzzles | retrieval copying — copy-gate at M4 entry |
| Self-edit cycles (governed) | 0.3–1.0 total, then plateau | STaR/V-STaR +4–17 pp over 2–4 iters; AlphaProof compounds with sound verifier + deep search; collapse bounded by replay mix | self-training collapse; gamed traces |
| Representation (Abacus-class) | 0–0.3 cross-domain (large but narrow) | 2405.17399: chance→99% on its task family | none if C1-gated |
| **Specialization dividend** (structural, not multiplied in) | — | comparator spends ≥99% of bytes on breadth (languages, world knowledge, chat) we refuse to compete on; this is *why* the products below are reachable | one-domain dominance if over-applied |

**Per-domain product (B1 budget, with the M2/M3 builds landed):**

| Domain (tier) | Factors in play | log₁₀ product | Claim class | Honest note |
|---|---|---|---|---|
| Word arithmetic (B) | K 1.0–1.5 · tools 1.5–2.0 · self-edit 0.3–0.7 | **2.8–4.2** | **moonshot candidate — confirm vs F1/F2** (Exp69 ~98% promising; not proven vs frontier until M1) | saturated task class; difficulty coverage must extend (multi-step, larger operands via tools) |
| Logic (A) | K 1.0–1.5 · SAT coproc 0.5–1.5 (formalizable subset) · CMM 0–0.7 · self-edit 0.3–0.7 | **1.8–4.4** | 100× now; **1000× conditional on M2** (SAT lane + READ/formalization holding) | the model's score = translation quality; null-machine margin is the integrity test |
| Puzzles (E) | architecture+recursion (TRM) 3.0–3.5 standalone · +K/verifier | **3.0–4.0** | **1000× already published by others (TRM/ARC)** — ours to replicate on ternary + breadth | TRM's data-augmentation confound noted; our replication uses matched protocols |
| Code (D) | K 1.0–1.5 (pass@k native to code) · sandbox oracle strong · generation floor unknown at 20M | **1.0–2.5** | 10–100× initial; grows with self-edit | needs M4 corpus + sandbox; hardest READ domain |
| Strict in-weights arithmetic (C) | repr 0–0.3 · tool-supervised self-edit 0.3 | **0.3–0.6** | **excluded from headline moonshot** | frontier likely wins in-weights; we refuse this fight (Exp67) and say so |

**Downgrade rules (owner-set, binding):**
1. Beat L1 (7B–70B local) but lose to max(F1, F2) on a domain → labeled **"local win only"**; zero contribution to the headline frontier-parity count.
2. Tier B saturation: if we sit ≥98% with tools AND frontier sits ≥99% with tools → **counts as frontier parity (success, per owner rule)** — the panel must still show the difficulty-coverage axis so saturation stays visible and coverage keeps extending.
3. Tier C stays excluded from the headline count unless the owner overrides (§9).
4. A win that survives only when frontier is denied tool access is tool-asymmetric (§2.2) and excluded from the headline.

**Worked example — Tier E (the ratio math; peer-published):** TRM-class 7M params ≈ 3–28 MB persistent vs DeepSeek V4-Pro's published 1.6T params ≈ 1.6 TB ⇒ storage_ratio ≈ 3.4×10⁵, at capability ≥ frontier on ARC-AGI (TRM 45% ARC-AGI-1; named frontier reasoning models lower — published). The moonshot_win condition is already satisfied by a peer system on this tier; our M4 job is replication on ternary (packed ~3 MB ⇒ ratio ~5×10⁵) under the §2.2 protocol, first-party.

**Worked example — Tier B (the pp math; ours):** raw strict pass@1 62.5% (Exp69, measured) → verifier-K at B1: +10–20 pp on mid-difficulty strata (planning band from pass@k-vs-scale curves; Exp81 measures the real curve) → coprocessor ABI: READ-limited ceiling ≈98% (Exp69, measured) → self-edit: +2–5 pp on residual READ errors (STaR-band, M3) ⇒ ≈98–99% vs tool-augmented frontier ≈99% ⇒ **parity within noise at storage_ratio ≈10⁵–10⁶**. The chain closes a ~36-pp gap, not a 1000× gap — that is the whole trick.

**Flagged guesses (Pass 4 honesty):** tape/retrieval 0–0.5 log is the weakest anchor in the factor table — knowledge-RAG numbers imported into a procedure-dense regime (guess until Exp115); fast-store 0–0.8 imports TTT-ARC results from 8B scale to 20M (guess until Exp82); verifier-K and tools are repo-measured; self-edit band is literature-cited (STaR/V-STaR) but unreplicated here until M3.

**The zero-support frontier (hardest problem, stated plainly).** All selection-based multipliers (K, verified filter, RLVR) re-weight existing support. For tasks where the control unit's prior assigns ≈0 mass to any verifier-passing trajectory, no budget tier helps. Escape routes, each falsifiable: (1) **strata curriculum** — traces from solved difficulty-n shift the prior over difficulty-n+1 (this is exactly the M3 compounding hypothesis; if Δstrict on the *next* stratum per cycle ≤ 0, compounding is dead, not just plateaued); (2) **tools** — re-parameterize so the residual READ task has support; (3) **tape** — worked examples in context shift the prior at inference (M4); (4) **representation** — make the unreachable trivially reachable (C1). A domain where all four fail is out of the moonshot, recorded as such.

---

## 5. Moonshot candidates — integrated system recipes

(≤7; each names historical + modern roots. Exp IDs **Exp112–116** — the last block in the ledger, because the machine runs last; see Rev c above and `EXECUTION_ORDER.md`.)

### BPTM-1 — Bounded post-Turing agent (self-edit OFF)
- **Mechanism:** the §3 machine loop at inference: K-sample + verifier halt + bounded revise with error-evidence feedback; weights frozen.
- **Roots:** Turing 1939 oracle machines + Hearsay-II blackboard (historical); PTRM + V-STaR-style verifier selection + Huang 2023 (revision must be external-signal-driven) (modern).
- **Wired:** control unit + T0/T1 tape + halt oracle + calc coprocessor. No self-edit, no retrieval.
- **Lift:** logic hard 83%→90%+ and word 62.5%→75%+ at B1 (prediction; Exp81 curve is the input). moonshot_score: first full Pareto panel + **frontier scoreboard** (F1/F2).
- **Cost:** 0 packed MB; inference-only GPU-hours; VRAM ~0.6–1 GB.
- **Failure modes:** trivial-true outputs; flat-K collapse.
- **Smallest experiment — Exp112 "Machine Loop v0 + Frontier Scoreboard":** wrap Exp69/70 checkpoints in the cycle loop (revise ≤2, K≤16); run **F1 or F2 API** + **N0** on frozen/held-out sets (L1 GGUF optional ladder). **Promote if** machine beats model-only by ≥10 pp strict on ≥2 domains AND beats null machine by ≥20 pp; **record frontier row** even if we lose (baseline for storage_ratio story). **Kill if** revise adds nothing over plain K (then BPTM-1 reduces to Exp81 breadth and the loop machinery is dropped).

### BPTM-2 — Governed self-edit machine (the moonshot engine)
- **Mechanism:** BPTM-1 + Phase 6 cycles per §3.5; compounding hypothesis: each cycle's traces shift the prior enough that the *next* stratum becomes harvestable.
- **Roots:** Gödel machine 2003 (historical); STaR/V-STaR + AlphaProof verifier-gated RL + Shumailov 2024 (replay-mix as anti-collapse) (modern).
- **Wired:** everything in BPTM-1 + T3 work log + self-edit pipeline + rollback.
- **Lift:** the only factor that raises *single-shot* (B0) capability — everything else is test-time. Target +0.3–1.0 log₁₀.
- **Cost:** ~1–2 GPU-h per cycle at h256 (generate K=8 × ~1k tasks ≈ 10–20 min; verify ≈ seconds; SFT 200–2k steps ≈ 10–60 min; re-eval ≈ minutes).
- **Failure modes:** collapse, gaming, one-domain dominance, plateau-at-cycle-1.
- **Smallest experiment — Exp113 "Three-Cycle Compounding"** (= M3, gated on Exp79 promote): word domain, 3 full cycles, difficulty-stratified reporting. **Promote if** strict held-out word rises monotonically across ≥3 cycles (each Δ ≥ +2 pp) AND next-stratum pass@K rises AND zero failure-mode flags. **Kill if** two attempts (different replay fractions) fail → self-edit demoted to one-shot data engine; moonshot downgrades per §1.

### VFA — Verifier-first revise policy (BPTM-1 inner ablation)
- **Mechanism:** generate → verify → feed the verifier's error evidence back as a tape frame → revise, vs blind resample.
- **Roots:** blackboard error-posting (historical); Huang 2023 — intrinsic self-correction fails, external signal works (modern).
- **Smallest experiment:** factor inside Exp112 (with/without evidence frames at matched budget). **Promote if** evidence-fed revise beats blind resample ≥5 pp at equal K×revise budget. **Kill if** no margin (then revise = resample and we save the machinery).

### TNM — Tool-native machine (SAT coprocessor for logic)
- **Mechanism:** extend the ABI with a SAT/SMT solver; control unit's job on Tier A becomes formalize-then-call, mirroring the calculator lane.
- **Roots:** oracle machines; Exp57 closure soundness (in-repo historical analogue) (historical); AlphaGeometry-2's neuro-symbolic split — LM proposes, symbolic engine grinds (modern).
- **Lift:** logic hard toward ≥95% on the formalizable subset → unlocks the 1000× column for Tier A.
- **Failure modes:** oracle laundering (null machine with naive enumeration into SAT may be strong here — the margin requirement is doing real work); formalization READ errors silently dropping tasks (count them).
- **Smallest experiment — Exp114 "Logic SAT Lane":** Exp70 checkpoint + pysat-class solver behind the ABI; compare {model-only, model+SAT, null+SAT}. **Promote if** model+SAT ≥ +8 pp over model-only AND ≥ +15 pp over null+SAT on hard split. **Kill if** null+SAT ≈ model+SAT (task family too SAT-trivial — replace task family, not the thesis).

### TAM — Tape-augmented machine (Phase 5 entry)
- **Mechanism:** retrieval over verified solved episodes (T2) into context; copy-detection gate per VISION Phase 5 exit (near-duplicate top-hits removed from headline).
- **Roots:** Kanerva SDM + CBR retrieve-adapt-verify-store (historical); retrieval-for-SLM literature, Mallen-style when-to-trust (modern).
- **Sequencing:** M4 — *after* compounding verdict; tape entering earlier confounds the M3 measurement.
- **Smallest experiment — Exp115:** worked-example retrieval on word+logic; headline on copy-clean subset. **Promote if** copy-clean strict +5 pp. **Kill if** lift exists only on near-duplicates (pure copying — park retrieval again, lesson recorded).

### FSM — Fast/slow store machine (C3 arc continued)
- **Mechanism:** ephemeral overlay at down_proj with next-token TTT objective, wiped per task (§3.3) — the machine's registers.
- **Roots:** Hinton & Plaut 1987; Schmidhuber FWP 1992 (historical); In-Place TTT 2026 + TTT-ARC 6× + FWP↔linear-attention bridge 2021 (modern).
- **Status:** inherits Architecture brief C3's one-retry-then-kill bar (Exp82). Not re-litigated here; if Exp82 promotes, FSM wires it into the cycle loop with wipe audits.
- **Kill:** Exp82's bar. If killed, the machine runs without registers — spec degrades gracefully (T1 tape absorbs the role).

### MDB — Multi-domain bootstrap (Phase 6 exit form)
- **Mechanism:** BPTM-2 across ≥3 domains with domain-balanced replay; cross-domain transfer measured (does logic self-edit lift word?); no-dominance gate: no domain >50% of any cycle's ingested traces.
- **Roots:** CBR cross-domain episode libraries (historical); multi-task verified RL / domain-balanced self-play (modern).
- **Smallest experiment — Exp116** (= M5 entry): 3 domains × 3 cycles. **Promote if** Σ strict held-out rises with *no domain falling* >2 pp and ≥1 positive cross-domain transfer measured. **Kill if** dominance gate trips twice → fall back to per-domain BPTM-2 with merged final SFT.

### Red-team summary (Pass 3 — every recipe vs its hardest precedent)

| Recipe | Hardest attack | Survives because | Residual risk |
|---|---|---|---|
| BPTM-1 | flat-K diversity collapse (Exp34.1 seed-2 fragility, register #7) | Exp81 measures the K-curve before anything is built; VFA ablation separates revise from resample | flat curve → machine reduces to plain breadth — pre-registered |
| BPTM-2 | Exp67 precedent: training on own outputs moved loss, not truth | targets are tool-corrected + strict-verified (external signal), never raw self-output; replay ≥25%; rollback every cycle | plateau-at-cycle-1 → §1 demotion path |
| VFA | Huang 2023: self-correction degrades performance | revision consumes *verifier* evidence frames, never self-judgment | evidence string may be too thin to steer a ~20M model — 5 pp kill bar catches it |
| TNM | oracle laundering: null+SAT may match model+SAT | ≥15 pp margin over null+SAT required to promote | task family may be SAT-trivial → one family swap, then lane dies |
| TAM | retrieval copying (VISION Phase 5 ban) | copy-clean subset is the only headline | tape may be worthless in procedure-dense domains (flagged guess, §4) |
| FSM | two strikes already: Exp77 oracle 0/64, Exp78 kill | exactly one corrected retry (down_proj site + next-token objective per In-Place TTT), then lane closes | inherits C3 bar — no special pleading |
| MDB | one-domain dominance + 3-domain wall-clock | ≤50% dominance cap; weekend-class GPU cost (§7) | cross-domain transfer may be ≈0 — fallback: per-domain BPTM-2, merged final SFT |

"Depth hurts" (Exp33.5) red-teams every recipe tempted to raise recurrence: none does — R stays ≤8 at B1; only C4 (gated in the Architecture brief, not a recipe here) may revisit depth, and only with CMM-style attractor training on constraint domains.

---

## 6. Milestone roadmap

```text
Phase A (IN FLIGHT, owner lane): Exp79 verdict → Exp81 breadth → Exp80/82 gated   [Architecture brief]
Phase B (PARKED): Exp72 gate, Exp85–88                                            [Training brief; unpark rules §7]
Phase C (THIS BRIEF):

M1  Machine loop + scoreboard      (≤3 mo)   Exp112 + VFA ablation
    entry: Exp81 curve recorded    exit: Pareto panel v1; F1/F2 frontier rows + N0 on frozen +
    GPU-h: ~10–20 + ~$20–100 API         fresh-seed probe sets (§2.2 — never the held-out 40);
                                         null/model-only margins established on 2 domains
    KILL: revise+K ≤ plain K → machine reduces to breadth; moonshot re-scoped to training-side

M2  Tool-native logic              (≤6 mo)   Exp114
    entry: M1 panel                exit: Tier A ≥90% hard with clean ablation margins
    GPU-h: ~10                     KILL: null+SAT parity → task family swap, one retry

M3  Compounding (FLAGSHIP)         (≤12 mo)  Exp113, 3 cycles, word domain
    entry: Exp79 promote + M1      exit: monotone strict held-out ×3 cycles + next-stratum gain
    GPU-h: ~5–10                   KILL: 2 failed attempts → self-edit = data engine;
                                         downgrade per §1: Tier E + saturated B only, rest "local win only"

M4  Tape + new domains             (≤24 mo)  Exp115 (retrieval) + Tier E puzzles (TRM-class replication,
    entry: M3 verdict either way         needs C5/Exp88 pretrain envelope) + Tier D code sandbox v0
    GPU-h: ~50–150 (incl. pretrain)  exit: 4-domain panel; copy-gate clean
    KILL per lane: copy-gaming → park tape; code floor too low → defer Tier D

M5  Multi-domain bootstrap         (≤36 mo)  Exp116 = VISION Phase 6 exit criteria
    exit: held-out verified pass@k strictly improves across ≥3 cycles, ≥3 domains,
          zero failure-mode flags → final moonshot audit vs F1/F2 + storage_ratio
```

12-month picture: M1–M3 done, verdict on compounding in hand. 24-month: M4 panel. 36-month: M5 audit — the dated claim in §1 stands or falls there.

---

## 7. Training on this GPU only

- **Cycle cost (measured-anchored):** generation dominates (sequential decode); at h256 with AMP (~5–10k tok/s inference-class), K=8 × 1k tasks × ~200 tok ≈ 1.6–3 M tok ≈ 10–25 min; verify: ms-class × 8k = seconds; SFT 200–2k steps ≈ 10–60 min (Exp70 anchor: 8k steps = 12.9 min); re-eval minutes. **One governed cycle ≈ 1–2 h** → M3 flagship ≈ an evening; M5 ≈ a weekend. The moonshot's training loop is cheap *because* the control unit is tiny — this is the constraint thesis paying us back.
- **Trace vs replay vs synthetic mixes:** per cycle at laptop scale: ~1–5k verified traces + ≥25% original replay + curriculum synthetic (Exp30-lineage generators). Buffer sizes trivially fit disk; guard_rail at every ingest.
- **Unpark triggers for the Training brief:** M3 → wants Exp85 (AMP gate) + Exp86 (chunked CE) to cut cycle wall-clock ~2–3× (nice, not blocking). M4 Tier E pretrain → **requires** unparking Exp87/88 (checkpointing + scale ladder) and consumes the C5/TRM spec. Nothing in M1–M2 needs the parked lane.
- **Budget honesty:** no milestone assumes >150 GPU-h total; nothing assumes cloud. If a step doesn't fit overnight-to-weekend granularity, it is re-scoped until it does.

---

## 8. Explicit non-goals (moonshot edition)

| Refusal | One line why |
|---|---|
| Learned verifiers / neural reward models as halt oracles | soundness is the moat; learnable judges reopen the gaming surface Exp73 was built to detect |
| RLVR-without-tools as a capability creator | literature: selection sharpens, doesn't create; tool_supervised is the new-signal mode |
| Giant laptop pretrain (>h512 ladder) | Training brief verdict: wall-clock-bound; tokens beat params here |
| Retrieval without copy-detection | copying is capability theater; VISION Phase 5 exit already bans it |
| In-weights multiplication / digit ALU | Exp67 kill; coprocessor ABI invariant |
| Conflating FlashAttention with fast weights | FlashAttention = IO-aware softmax kernel; fast weights = short-term synaptic store (Exp77's "Flash Grid" name is the latter — keep the disambiguation in every doc) |
| Turing-completeness / universal-computation claims | bounded tape, bounded steps, finite budgets — "post-Turing" here means *governed self-edit + oracles*, nothing more |
| Arbitrary / ungoverned self-modification | §3.5 pipeline only; verifiers, eval sets, guard rail, wipe rules are immutable |
| Training or deploying our stack on cloud/multi-GPU | violates laptop-train thesis; **frontier API eval (F1/F2) is in scope** as comparator only |
| Open-domain chat / memorization parity | no verifier, no claim — out of scope by definition |

---

## 9. Open questions for the owner (≤7)

1. **Test-time budget constants:** confirm B0/B1/B2 (§2) — K max 16 standard / 256 harvest, revise ≤2, recurrence ≤8, latency cap per task?
2. **Tape budget:** is 64 MB retrieval index the right ceiling (λ=0.1 in the metric), and is disk-resident index acceptable or must it be VRAM-resident?
3. **Code sandbox scope (Tier D, M4):** Python-only, no network, hidden-test harness — local container or bare subprocess with limits? Owner's security comfort sets this.
4. **"1000×" framing (locked):** storage-density vs **frontier F1/F2**, not vs local 7B. Public headline = "palm-sized control program, frontier parity on strict verifiers" — confirm domain count (3/5 vs 4/5) for M5 audit.
5. **Domain weights w_d** for the headline panel (equal weights default until M4 adds Tiers D/E?).
6. **Self-edit scope:** §3.5 proposes weights + retrieval-index contents + revise hyperparameters as editable, verifiers/evals/guards immutable. Approve, or restrict to weights-only for M3?
7. **Step budget:** max cycles per task before forced halt (proposal: 1 + revise budget at B1; 32 at B2 harvest) — confirm so HALT(fail) semantics are fixed before Exp112.

---

## 10. Citation index

### Historical (pre-2015)

| Topic | Work | Year | Link/ref |
|---|---|---|---|
| Oracle machines | Turing, *Systems of Logic Based on Ordinals* (O-machines) | 1939 | Proc. LMS |
| Minimal control + tape | Minsky, *Recursive unsolvability of Post's problem*; *Computation: Finite and Infinite Machines* (2-register universality) | 1961/1967 | Annals of Math.; Prentice-Hall |
| Attractor memory | Hopfield, *Neural networks and physical systems with emergent collective computational abilities* | 1982 | PNAS 79:2554 |
| Blackboard architecture | Erman, Hayes-Roth, Lesser, Reddy, *The Hearsay-II speech-understanding system* | 1980 | ACM Computing Surveys 12(2) |
| Binding via fast synaptic modulation (backward-chase) | von der Malsburg, *The Correlation Theory of Brain Function* | 1981 | MPI Biophysical Chemistry report 81-2 |
| Fast weights (origin) | Hinton & Plaut, *Using fast weights to deblur old memories* | 1987 | CogSci 1987 |
| Sparse distributed memory | Kanerva, *Sparse Distributed Memory* | 1988 | MIT Press |
| Fast-weight programmers | Schmidhuber, *Learning to control fast-weight memories* | 1991/1992 | Neural Computation 4(1):131–139 |
| Limited-precision training | Höhfeld & Fahlman, *Probabilistic rounding in neural network learning with limited precision* | 1992 | Neurocomputing 4(6) |
| Case-based reasoning | Kolodner, *An introduction to case-based reasoning* | 1992 | AI Review 6 |
| Governed self-modification | Schmidhuber, *Gödel machines: self-referential universal problem solvers* | 2003+ | arXiv:cs/0309048 |
| Differentiable tape (cautionary) | Graves, Wayne, Danihelka, *Neural Turing Machines* | 2014 | https://arxiv.org/abs/1410.5401 |

### Modern (2015+)

| Topic | Paper | Year | Link |
|---|---|---|---|
| FWP ↔ attention bridge | Schlag, Irie, Schmidhuber, *Linear Transformers Are Secretly Fast Weight Programmers* | 2021 | https://arxiv.org/abs/2102.11174 |
| Modern Hopfield capacity | Ramsauer et al., *Hopfield Networks is All You Need* | 2020 | https://arxiv.org/abs/2008.02217 |
| Differentiable memory (capstone) | Graves et al., DNC, *Hybrid computing using a neural network with dynamic external memory* | 2016 | Nature 538 |
| TTT for abstract reasoning | Akyürek et al., *The Surprising Effectiveness of Test-Time Training for Abstract Reasoning* | 2024 | https://arxiv.org/abs/2411.07279 |
| Fast-weight site/objective | *In-Place Test-Time Training* | 2026 | https://arxiv.org/abs/2604.06169 |
| Tiny recursive reasoning | Jolicoeur-Martineau, *Less is More: TRM* | 2025 | https://arxiv.org/abs/2510.04871 |
| Test-time breadth + selection | *Probabilistic TRM* | 2026 | https://arxiv.org/abs/2605.19943 |
| HRM component audit | *HRM Perspectives & Misconceptions* | 2025 | https://arxiv.org/abs/2510.00355 |
| Attractor-trained depth | *CMM / HRM Dynamical Systems Theory* | 2026 | https://arxiv.org/abs/2603.22871 |
| Self-training root / verifier iter | STaR; V-STaR | 2022; 2024 | https://arxiv.org/abs/2203.14465 · https://arxiv.org/abs/2402.06457 |
| SLM + process reward + search | rStar-Math (ICML) | 2025 | https://icml.cc/virtual/2025/poster/46400 |
| Verifier-gated RL compounds (existence proof) | DeepMind, AlphaProof / AlphaGeometry-2 (IMO silver) | 2024 | https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/ |
| Self-correction needs external signal | Huang et al., *LLMs Cannot Self-Correct Reasoning Yet* | 2023 | https://arxiv.org/abs/2310.01798 |
| Recursive-data collapse | Shumailov et al., *AI models collapse…* | 2024 | https://www.nature.com/articles/s41586-024-07566-y |
| pass@k honesty | *Don't Pass@k: Bayesian framework* | 2025 | https://arxiv.org/html/2510.04265v1 |
| Digit representation | *Transformers Can Do Arithmetic with the Right Embeddings* | 2024 | https://arxiv.org/abs/2405.17399 |
| SLMs need strong verifiers | *Small LMs Need Strong Verifiers to Self-Correct* | 2024 | https://arxiv.org/html/2404.17140v2 |

---

*Success-criteria self-check: (1) 1000× = storage-density vs frontier F1/F2 on strict verifiers, not local 7B parity — §2/§2.1/§4. (2) Machine spec §3, six subsections + diagram, implementable without neural guesswork. (3) M1–M5 each carry kill criteria — §6. (4) Refusals — §8. (5) Three-plus pre-2015 ideas with paths — §1/§3b. (6) Anachronism check — §1 bullet 7 and §3b gap column. (7) State-machine readability — §3.2/3.5 pseudocode. (8) Self-edit central-but-conditional with pre-registered downgrade — §1, §5 BPTM-2, §6 M3. (9) Storage-density thesis stated — §2.1, VISION.md.*

**Self-check (owner-mandated, one paragraph):** This brief targets **matching or beating frontier datacenter models — GPT-5.5-class (F1) and Claude Opus/Fable-class (F2) — on strict programmatic verifiers at ~10³–10⁶× less persistent storage and ~10⁶–10⁸× less train compute**, with frontier measured via API on our own task sets under the §2.2 same-verifier / tool-parity / budget-parity / cost protocol; the same-VRAM 7B local comparator survives only as the non-headline sanity ladder L1, beating it earns the "local win only" label and contributes nothing to the headline frontier-parity count, and §2/§4 ban claiming 1000× off it — the brief has not slipped back to 7B local. The raised bar's honest cost is recorded in §4: Tier E (puzzles) is the spearhead with a published frontier win by a TRM-class peer, Tier B counts as saturated parity once confirmed vs F1/F2 at M1, Tier A is the contested swing conditional on M2 surviving tool parity, Tiers C/D are excluded or long-shot — the ≥3-of-5 claim routes through E + B + A and dies visibly if any leg fails.

**BIG BRAIN self-audit (amendment checklist):** §3b = **12 historical rows**, gap column filled, one backward-chase added (von der Malsburg 1981 ← Hinton & Plaut 1987) ✓ · §4 carries **explicit worked math** — Tier E ratio chain (3.4×10⁵ at published capability ≥ frontier) and Tier B pp-chain (62.5 → ≈98–99 vs frontier ≈99), with the Pass-1 reframe that multipliers close a ~36-pp gap while the 10⁴–10⁶ lives in the denominator by construction; weakest anchors (tape, fast-store) flagged as guesses ✓ · BPTM-1 vs BPTM-2 differ exactly on the §3.5 self-edit policy (OFF vs governed cycles) ✓ · §3.5 enumerates **forbidden ingest transitions** — held-out (guard_rail), gamed (Exp73 `gamed_frac` monitor), verifier-failed positives, dominance cap, hash-drift voiding ✓ · M1 entry-gated on Exp81, M3 on Exp79 promote — Phase A runs first, training lane stays parked ✓ · citations: 12 pre-2015 + 18 modern, split tables ✓ · contradictions surfaced in the §1b register (8 rows, incl. the verifier-soundness audit finding as moonshot critical path); every downgrade pre-registered in §1/§4/§6 ✓ · anti-slop: no scale-data/RLHF filler, FlashAttention/fast-weights disambiguated (§8), Turing-completeness refused (§8) ✓.
