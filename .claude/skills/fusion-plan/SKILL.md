---
name: fusion-plan
description: "GENERATE a plan for a goal via an independent model panel, then synthesize the best plan. Use when starting a non-trivial task that needs a plan, when entering plan mode, or when the user says 'fusion-plan', 'plan this', 'make a plan for', 'how should we approach'. Fans the GOAL to Codex (risk-aware plan) + Gemini (options/prior-art); Opus writes its own plan and synthesizes consensus vs divergence into the best plan with a Decision Rule. Pairs with fusion-build, which REFUTES the plan before you execute it."
metadata:
  version: 1.0.0
---

# Fusion-Plan — panel-generated planning (the divergent half)

The **divergent** half of the fusion loop. No plan exists yet; the panel produces
independent material so the synthesized plan isn't one mind's blind spots. Planning
is **unverifiable** at plan-time (no code oracle), so a model panel is the right
tool — it informs; execution is the late verifier via the forced Decision Rule.

```
goal -> [fusion-plan: generate] -> plan -> fusion-build (refute) -> execute -> builder_gate
```

## When to run it
- Starting a non-trivial / architectural / multi-step task that needs a real plan.
- On `EnterPlanMode` for anything beyond a trivial change.
- When the user invokes `/fusion-plan` or asks to plan / approach a goal.
- Skip for obvious one-step tasks.

## How to run it
Give it the GOAL (not a plan — that's fusion-build):
```bash
python scripts/fusion_plan.py "goal: wire LC0 the long-dependency instrument"
python scripts/fusion_plan.py --file goal.md
python scripts/fusion_plan.py --codex-only "goal ..."   # skip Gemini options, faster
```
Panel (each plays to its strength; auto-skipped if its adapter is down):
- **Codex (plan)** — `scripts/ask_codex.py`, xhigh, repo-aware. Produces a concrete,
  risk-aware, ordered plan with dependencies + smoke tests + a Decision Rule.
- **Gemini (options)** — `scripts/ask_gemini.py`, context injected. Surfaces the
  option space / prior art / reuse — NOT a committed plan (its bias is optimism).

## Then YOU (orchestrator) write the plan
The panel material is INPUT, not the plan. Opus is the strongest planner seat, so:
1. Write your OWN primary plan.
2. **Consensus** — steps all independent plans share -> high confidence, keep.
3. **Divergence** — where they disagree -> real uncertainty; resolve it explicitly
   (pick a path and say why), don't average them.
4. **Reuse** — existing repo machinery the options surfaced (don't rebuild).
5. **The plan** — synthesized ordered steps, dependencies, smoke tests.
6. **Decision Rule (REQUIRED)** — Promote if <signal> / Kill if <signal>.

Then hand the plan to **fusion-build** to refute it before executing.

## Discipline
- A fused plan is **not a verified** plan — the panel improves quality, not
  correctness. Reality still gets the final vote.
- Weight Gemini for breadth of options, not feasibility claims.
- This model-judge pattern is legitimate ONLY here (no code verifier can score a
  plan at plan-time). Never let it gate verifiable work — code gates code.
