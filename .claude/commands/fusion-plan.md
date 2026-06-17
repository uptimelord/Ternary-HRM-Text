---
description: Generate a plan for a goal via an independent model panel, then synthesize the best plan.
---

Run the fusion-plan generation workflow for this goal:

$ARGUMENTS

Steps:
1. Run `python scripts/fusion_plan.py "<the goal above>"` (or `--file` if a goal
   file was given; add `--codex-only` if speed matters). This fans the goal to
   Codex (risk-aware plan) and Gemini (options/prior-art) in parallel.
2. Read the panel material. Then YOU write the plan per the synthesis scaffold:
   - your own primary plan, then Consensus / Divergence / Reuse / The Plan
   - a REQUIRED Decision Rule (Promote if <signal> / Kill if <signal>)
3. Present the synthesized plan. Offer to hand it to `/fusion-build` to refute
   before executing.

The panel ADVISES; you are the judge. A plan without a Decision Rule is
unfalsifiable — do not finalize it. See `.claude/skills/fusion-plan/SKILL.md`.
