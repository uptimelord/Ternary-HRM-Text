---
description: Stress-test an existing plan via an adversarial model panel (go/no-go before executing).
---

Run the fusion-build refutation workflow on this plan:

$ARGUMENTS

Steps:
1. The plan must already exist (the text above, or a `--file`). Run
   `python scripts/fusion_build.py "<the plan above>"` (add `--refute-only` for
   Codex-only / faster). This fans the plan to Codex (refute) and Gemini (scout).
2. Read the panel critiques. Then YOU synthesize per the scaffold:
   - Blind spots / Contradictions / Cheaper path
   - a REQUIRED Decision Rule (Promote if <signal> / Kill if <signal>)
   - a Verdict: GO (execute) or NO-GO (revise / replan)
3. Present the verdict. If GO, the plan is ready to execute; commits still pass
   through `python scripts/builder_gate.py`.

The panel ADVISES; you are the judge. A refuted-and-survived plan is not a
verified plan — reality (execution) gets the final vote. See
`.claude/skills/fusion-build/SKILL.md`.
