# Project context — BitNet-HRM (for external model seats)

Compact brief injected into non-Claude model calls (Gemini scout, Codex refute) so
they answer *with* this repo's intent instead of blind. Codex also auto-reads
`AGENTS.md` at the repo root; Gemini gets this text prepended. Keep this short —
it is prepended to prompts. Canonical detail lives in the docs linked below.

## What this is
Compact **ternary HRM** research repo. North star: **reasoning density under
constraint** (RTX 3050 Ti / 4 GB, train-at-home) — NOT datacenter scale, NOT
parameter count. Optimize for **verified pass@k × difficulty coverage ÷ packed MB**.

## Non-negotiable invariants (do not propose anything that violates these)
1. **Strict verifiers decide.** A capability number without its strict verifier is
   void. Loose/soft metrics never become the headline (Exp64: loose 55.5% vs
   strict 8.5% on the same checkpoint).
2. **Never train on held-out / frozen IDs.** `evaluation/guard_rail.py` runs at
   every train load; held-out + frozen sets are **reporting-only**. For the
   retrieval corpus, held-out is refused at *build* time too.
3. **Kill means kill.** Every experiment pre-registers a Decision Rule
   (Promote if… / Kill if…). A no-effect result is a kill, not "inconclusive".
4. **Measure-twice.** Smallest fair diff, one question per experiment, 2 seeds,
   noise floor ±0.0203 @ 5k steps. No promoting on one seed.
5. **Run-order ≠ ID-order.** Experiment numbers are authoring order. The schedule
   is `experiments/EXECUTION_ORDER.md` (Tier 0→5; machine milestones Exp112–116 last).
6. **Serialize-only GPU.** One card; training owns it; inference sweeps fill gaps.
7. **No model-judge gates verifiable work.** Code (strict verifiers, builder_gate)
   decides commits/verdicts. Model panels only advise on *unverifiable* calls
   (planning, design). This is the Universal-Verifier-brief rule.

## Current state (2026-06-14)
- Verdicts: Exp80 abacus **kill**, Exp82 down-proj TTT **kill**, Exp83 TRM **promote**;
  Exp81 verified-breadth running (temp arm +9–12 pp over greedy, seed-1; seed-2 pending).
- Baselines: word raw ~62.5% heldout, word+calculator ~98%, logic Exp70, LDT Exp57.
- Deploy compression recipe: `mixed_top512_tequila_L_mlp_gate_up` (Phase 0 preset). No fixed packed-size target; deploy sizing is bounded by the 4 GB GPU envelope (peak <= 3,800 MiB), not a byte ceiling.

## Canonical docs (read these for detail; don't duplicate them)
- `CLAUDE.md` — intent stability, scope, voice.
- `experiments/EXECUTION_ORDER.md` — what runs when, across all briefs.
- `experiments/DISCIPLINE.md` — decision rules, noise floor, reporting shape.
- `experiments/Experimental Log Summary.md` — living index of every experiment.

## Your seat
- **Gemini = scout:** surface leads / prior art / gaps. Never assert a result,
  number, or verdict. Output is candidates for a separate judge to verify.
- **Codex = refute:** adversarially try to break a claim/plan. Hunt metric swaps,
  held-out leaks, gate-jumping, hidden costs, stitched-fragment results. You
  advise; a code gate sets the final verdict.
