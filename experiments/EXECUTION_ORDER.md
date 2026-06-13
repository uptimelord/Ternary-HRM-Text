# EXECUTION_ORDER.md — Run order across all research briefs

**Date:** 2026-06-13 · **Authority:** this file is the single source of truth for *what runs when*.
**Audience:** any agent or human picking up work. Read this before scheduling an experiment.

---

## The one rule

> **Experiment ID numbers are identifiers (authoring order). They are NOT run order.**
> A low number does not mean "do this first." Schedule by the tiers below, not by the integer.

Why the mismatch exists: each brief grabbed the next free Exp-number block on the day it was written.
That bakes *authoring date* into the ID. Dependency order is different and is defined here.

---

## Brief → ID block ledger

| Brief | ID block | Owns |
|---|---|---|
| Architecture Research Brief (2026-06-10) | Exp80–84 | control-unit backbone bets (C1–C5) |
| Training Efficiency Brief (2026-06-10) | Exp85–89 | VRAM/throughput envelope |
| Universal Verifier Brief (2026-06-11) | Exp95–100 | sound halt oracles (the gate) |
| Taste Brief (2026-06-11) | Exp101–102 | corpus/ingest admission gate |
| Grounded TRM Brief (2026-06-13) | Exp90–93 | TRM tied-vocab, grounded head, game data, FJLT head (Track G, ~Tier 2–3) |
| Memory & Retrieval Brief (2026-06-13) | Exp103–111 | within-task context (LC) + across-task retrieval (RT) |
| Moonshot Brief (2026-06-10) | **Exp112–116** | the integrated machine, M1–M5 |

Exp94 is **retired** (vacated by the renumber, do not reuse). Verifier Exp96/97/99/100 run on **Track V** (§B);
Grounded TRM Exp90–93 on **Track G** (§B); both are CPU-parallel to the spine.

**Renumber history (2026-06-13):**
- Moonshot **Exp90–94 → Exp112–116** — moved to the end so the machine milestones sort *after* the
  organs they consume. Mapping: M1 BPTM-1 90→**112** · M3 BPTM-2 91→**113** · M2 TNM 92→**114** ·
  M4 TAM 93→**115** · M5 MDB 94→**116**. Spec-only at the time; **no result files broke.**
- This freed **90–93** for the Grounded TRM brief, which had been silently colliding with old Moonshot.
- The Memory & Retrieval brief's predecessor (Long Context draft) had claimed Exp95–99, colliding with the
  Verifier brief; renumbered to **103–111** the same day. Also unrun, no breakage.

Gaps in the integer sequence (e.g. 94 retired) are normal and intentional — do not "fill" them by
renumbering live experiments.

---

## Run order (the schedule)

Sequenced by dependency, not ID. Detailed rationale lives in **Memory & Retrieval Brief §7b**; this is the
canonical short form.

### A. Critical-path spine (Tier 0 → 5)

```text
TIER 0 — FOUNDATION (CPU, no GPU contention; can overlap the current Exp81/84 GPU queue)
  Verifier V0           Exp95    institutionalizes the halt-oracle gate everything routes through
  + audit fixes                  (verifier hardening is upstream of the machine — Moonshot §1)
  Plan-step plug-in     Exp98    formalize Exp65 step-check as a registered plug-in; upgrades Exp79 evidence
  LC0 instrument        Exp103   verified long-dependency benchmark; without it Part A is unmeasurable
  RT0 retrieval + gate  Exp108   corpus copy-gate; without it every retrieval number lies
  Taste conjunction     Exp101   the admission gate — defines what may enter the corpus / Phase-6 SFT
        → all are instruments/gates; later work is UNMEASURABLE or UNSAFE without them. Build rulers first.

TIER 1 — ENABLING THROUGHPUT (small GPU, switch-flips)
  8-bit Adam 8k gate    Exp72    finish the parity gate; unlocks T1 in every later stack
  AMP gate              Exp85    1.7× measured, flag already in runners
  chunked CE            Exp86    4× batch ceiling; HARD prereq for LC1's W=2k arm
        → the ONLY part of the Training brief on the critical path. The scale ladder (Exp87–89) stays Tier 5.

TIER 2 — MEMORY: WITHIN-TASK CONTEXT (model-quality organ, ahead of the scale ladder)
  LC1 window stretch    Exp104   512→2k; prereq for Phase-5 prepend AND Moonshot evidence frames
  LC2 streaming fold    Exp105   ┐ state-vs-tape comparative pair, same instrument
  LC3 input paging      Exp106   ┘
  LC4 verifier-gated decomposition  Exp107

TIER 3 — MEMORY: ACROSS-TASK RETRIEVAL (Phase 5 proper)
  RT1 addressing        Exp109   needs LC1 window room + Tier-0 admission gate
  RT2 copy-clean integ. Exp110   = Moonshot TAM (Exp115); ONE run, two viewing angles; the prior-shift lever

TIER 4 — MOONSHOT MACHINE (consumes everything above)
  BPTM-1 loop+scoreboard Exp112  M1; only makes sense once context delivery (T2) + verifier (T0) exist
  TNM SAT logic          Exp114  M2; needs SAT plug-in from the Verifier track
  BPTM-2 compounding     Exp113  M3 — FLAGSHIP; the make-or-break self-edit verdict; precedes RT3

TIER 5 — SELF-IMPROVING + SCALE (latest; gated on the M3 verdict)
  RT3 corpus growth      Exp111  self-improving store; joins Phase 6
  MDB multi-domain       Exp116  M5 = Phase 6 exit; needs the composer (Exp100)
  checkpointed cycles    Exp87   enables h512-class training; prereq for the scale probe
  scale ladder           Exp88   the Training brief's ONE sanctioned size probe, only if T2/T3 say size pays
  ECO spike              Exp89   only if Exp88 promotes AND a >100M ambition is live (else skip)
  Taste rung-4 calib.    Exp102  gated on debate machinery (does not exist yet)
```

### B. Parallel tracks (CPU/independent — run alongside the spine, gated by what they unblock)

```text
TRACK V — Verifier surface expansion (CPU; each is a sound plug-in, ms–s on the laptop)
  Code sandbox          Exp96    +5 task families on one trust decision; GATE for Moonshot code (M4)
                                 → must land before Exp115/M4 code domain. Build during Tier 2–3.
  Grid DSL              Exp97    Sudoku/Latin/KenKen rule engine; GATE for Tier-E puzzles (M4) AND
                                 the Grounded TRM game/grid data (Exp92). Build during Tier 2.
  Cross-domain composer Exp100   seq·all·any·reduce algebra; substrate for M5; build before Exp116.
  Template compiler     Exp99    GATED — rule-of-three, only after 95+96+97 stable. Optional, latest.

TRACK G — Grounded TRM backbone (the "fukano" recurrent solver; extends the Exp83 promote)
  Exp83.1 TRM mixed vocab head  — train-time mixed_top512_tequila on the vocab head; finalizes the
                                  TRM denominator (64 MB -> ~3.5 MB, q/mb ~18x) while holding logic.
                                  CPU wiring done; GPU retrain (~1 hr, 4 arms) waits for a gap. Run
                                  this before quoting Exp83 q/mb as a backbone-selection headline.
  TRM tied vocab        Exp90    backbone variant on verified worlds
  Grounded claim/head   Exp91    grounded claim/action head
  Game/grid/process data Exp92   needs Verifier grid DSL (Exp97) for its verified worlds
  FJLT / shortlist head Exp93    representation probe
        → backbone R&D track, parallel to memory work. Feeds Moonshot Tier-E puzzle replication (M4).
          Target: complete before Exp115/M4. Not yet owner-slotted to an exact tier — treat as ~Tier 2–3,
          gated on the training envelope (Tier 1) + Verifier grid (Exp97).
```

### C. Backlog (unscheduled; not on any critical path)

```text
  Exp32  recurrence sweep        — older probe, no current customer
  Exp37  DeepSeek rehearsal data — blocked on API key
  Exp74  (planned)               — not yet specced
  Exp94  RETIRED — do not reuse  — vacated by the Moonshot 90–94 → 112–116 renumber
```

**One-line summary:** Verifier + instruments + Taste-gate → throughput switch-flips → within-task memory →
retrieval → machine loop + compounding → self-improving + scale. **Moonshot is last, by construction.**
Two CPU tracks (Verifier surface expansion, Grounded TRM backbone) run in parallel and gate the M4/M5 domains.

---

## Standing constraints (apply at every tier)

- **One GPU, serialize-only.** Training runs own the card; inference/eval sweeps fill idle gaps. Parallel
  authoring does not mean parallel GPU.
- **Gated serial chains do not parallelize.** A self-edit cycle trains on the previous cycle's verified
  traces; M4 waits on the M3 verdict; promote/kill gates are hard barriers. Agents cannot jump a gate.
- **Held-out + frozen sets are reporting-only**, refused at train load (and, for the retrieval corpus, at
  *build* time — Memory & Retrieval §6b.2).
- **No commits unless the owner asks.**

---

## Current queue status (2026-06-13)

- **Decided:** Exp80 Abacus → **kill** · Exp82 Down-proj TTT → **kill** (closes Memory&Retrieval LC5) ·
  Exp83 Tied Recursive Block → **promote**.
- **Pending (Codex lane):** Exp81 Verified Breadth Sweep · Exp84 Attractor Logic Recurrence.
- **Tier 0 may start now** (CPU-only: LC0/RT0 generators, Verifier V0, Taste gate) — does not contend with
  the GPU queue.

*Update this file whenever a tier completes or the queue changes. It is meant to be read cold by the next agent.*
