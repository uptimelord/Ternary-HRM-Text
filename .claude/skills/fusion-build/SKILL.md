---
name: fusion-build
description: "Stress-test an existing PLAN before building it, via an adversarial model panel. Use right BEFORE executing/committing to a multi-step plan, before ExitPlanMode, or when the user says 'fusion-build', 'stress-test this plan', 'refute my plan', 'go/no-go on this', 'panel this plan'. The plan must already exist. Fans it to Codex (refute) + Gemini (scout), then Opus synthesizes blind-spots/contradictions and forces a GO/NO-GO Decision Rule. Pairs with the fusion-plan skill (which GENERATES the plan first)."
metadata:
  version: 1.0.0
---

# Fusion-Build — adversarial plan review (the go/no-go gate)

The **convergent** half of the fusion loop. A plan already exists; the panel
tries to BREAK it before you spend real work building it. Go/no-go on a plan is
**unverifiable** at decision-time (no code oracle), so this model-judge layer is
legitimate here — but it ADVISES; it never gates verifiable work (`builder_gate.py`
does that).

```
goal -> fusion-plan (generate) -> plan -> [fusion-build: refute] -> execute -> builder_gate
```

## When to run it
- Right before executing a multi-step / architectural / costly plan, or before `ExitPlanMode`.
- When the user invokes `/fusion-build` or asks to refute / stress-test / go-no-go a plan.
- Skip for trivial single-step plans (the panel costs N x tokens + slowest-panelist latency).

## How to run it
The plan must already be written. Then:
```bash
python scripts/fusion_build.py --file <plan.md>
python scripts/fusion_build.py "the plan text"
python scripts/fusion_build.py --refute-only "the plan text"   # Codex only, faster
```
Panelists (auto-skipped if their adapter is down):
- **Codex (refute)** — `scripts/ask_codex.py`, xhigh, repo-aware (reads AGENTS.md).
  Hunts dependency/ordering errors, hidden costs, gate-jumping, speculative steps.
- **Gemini (scout)** — `scripts/ask_gemini.py`, project-context injected. Surfaces
  what's missing / prior art. Never gives a verdict.

## Then YOU (orchestrator) synthesize
The script prints each critique + a scaffold. Fill it honestly — the panel advises:
1. **Blind spots** — what the panel caught that the plan missed.
2. **Contradictions** — where panelists disagree; say which is right and why.
3. **Cheaper path** — any lower-cost route surfaced.
4. **Decision Rule (REQUIRED)** — Promote if <signal> / Kill if <signal>.
5. **Verdict** — GO (execute) or NO-GO (revise / replan).

A plan without a Decision Rule is unfalsifiable — do not build it. The Decision
Rule is the *late verifier*: execution checks it.

## Discipline
- A refuted-and-survived plan is **not a verified** plan. The panel improves
  confidence; reality (execution) still gets the final vote.
- Never let this model-judge pattern leak into verifiable work — that's the
  LLM-as-judge the Universal Verifier brief kills. Code gates code.
- Attribute each synthesized point to the panelist that raised it.
