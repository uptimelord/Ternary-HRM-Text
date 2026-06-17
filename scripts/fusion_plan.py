#!/usr/bin/env python3
"""fusion_plan.py — generate a PLAN for a goal via an independent panel.

The DIVERGENT half of the fusion loop: no plan exists yet. The panel produces
independent material for one, each seat playing to its strength:
  - Codex (GPT-5.5, skeptical): a concrete, risk-aware, ordered plan.
  - Gemini (3.1 Pro, broad):    candidate approaches / prior art / options to
                                consider (its strength — NOT a committed plan).
Opus then writes its own primary plan and SYNTHESIZES all three into the best
plan, surfacing where the independent plans agree (high confidence) vs diverge
(real uncertainty to resolve), and attaches a Decision Rule.

  goal ─▶ [THIS: fusion-plan] ─▶ plan ─▶ fusion-build (refute) ─▶ execute ─▶ builder_gate

This is plan GENERATION (model-judge over an unverifiable task — legitimate,
because no code verifier can score a plan at plan-time). It informs; execution
is the late verifier via the forced Decision Rule.

Usage:
    python scripts/fusion_plan.py "goal: wire LC0 the long-dependency instrument"
    python scripts/fusion_plan.py --file goal.md
    python scripts/fusion_plan.py --codex-only "goal ..."   # skip Gemini options
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fusion_common import fan_out, print_panel  # noqa: E402


def _planner_prompt(goal: str) -> str:
    return (
        "Produce a concrete, ordered PLAN to achieve this GOAL in this repo "
        "(you have read access — consult AGENTS.md and the actual files). Be "
        "risk-aware and skeptical. Output:\n"
        "  - numbered steps, each with its dependency (what must exist first)\n"
        "  - the single RISKIEST assumption the plan rests on\n"
        "  - what to SMOKE-TEST to prove each step works (not just 'wired')\n"
        "  - a Decision Rule: Promote if <observable success signal> / "
        "Kill if <observable failure signal>\n"
        "Respect the repo invariants (strict verifiers, no held-out training, "
        "run-order != ID-order, measure-twice). Be specific to THIS goal.\n\n"
        f"GOAL:\n{goal}"
    )


def _options_prompt(goal: str) -> str:
    return (
        "For this GOAL, surface the OPTION SPACE (you are the scout, not the "
        "planner). List only: (a) candidate approaches/designs, (b) relevant "
        "prior art or existing repo machinery to reuse, (c) facts to check before "
        "choosing. Do NOT commit to one plan or claim one works — give a separate "
        "judge the menu of options and tradeoffs.\n\n"
        f"GOAL:\n{goal}"
    )


SCAFFOLD = """\
================ FUSION-PLAN SYNTHESIS (orchestrator writes the plan) ================
The panel material above is INPUT, not the plan. Opus now:

  1. Writes its OWN primary plan (Opus is the strongest planner seat).
  2. CONSENSUS   — steps all independent plans share -> high confidence, keep.
  3. DIVERGENCE  — where the plans disagree -> real uncertainty; resolve it
                   explicitly (say which path and why), don't average them.
  4. REUSE       — existing repo machinery the options surfaced (don't rebuild).
  5. THE PLAN    — the synthesized ordered steps with dependencies + smoke tests.
  6. DECISION RULE (REQUIRED) — Promote if <signal> / Kill if <signal>, so the
     late verifier (execution) can falsify the plan. Then hand the plan to
     fusion-build to refute before executing.
=====================================================================================
Note: Gemini's material is OPTIONS, not a verdict; weight it for breadth, not
feasibility claims (its known bias is optimism)."""


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a plan for a goal via an independent panel")
    ap.add_argument("goal", nargs="?", help="goal text, or '-' for stdin")
    ap.add_argument("--file", "-f", help="read goal from a file")
    ap.add_argument("--codex-only", action="store_true", help="skip Gemini options — faster")
    args = ap.parse_args()

    if args.file:
        goal = Path(args.file).read_text(encoding="utf-8")
    elif args.goal == "-" or (args.goal is None and not sys.stdin.isatty()):
        goal = sys.stdin.read()
    elif args.goal:
        goal = args.goal
    else:
        ap.error("provide a goal, --file, or pipe via stdin")

    print(f"=== fusion-plan: generating plans ({'codex only' if args.codex_only else 'codex plan + gemini options'}) ===",
          file=sys.stderr)
    results = fan_out(goal, codex_prompt=_planner_prompt, codex_label="codex/plan",
                      gemini_prompt=_options_prompt, gemini_label="gemini/options",
                      codex_only=args.codex_only)
    if not print_panel(results):
        print("\n(no panelists responded — Opus plans solo; no independent cross-check)",
              file=sys.stderr)
    print("\n" + SCAFFOLD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
