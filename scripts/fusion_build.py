#!/usr/bin/env python3
"""fusion_build.py — stress-test an existing PLAN/diff before you build it.

The CONVERGENT half of the fusion loop: a plan already exists; the panel tries to
BREAK it. Codex refutes (dependency/ordering/cost/gate-jumping), Gemini surfaces
what's missing. Opus then synthesizes blind-spots + contradictions and forces a
Decision Rule. This is the model-judge layer for the UNVERIFIABLE go/no-go on a
plan — it ADVISES; it never gates verifiable work (that stays builder_gate.py).

  goal ─▶ fusion-plan (generate) ─▶ plan ─▶ [THIS: fusion-build] ─▶ execute ─▶ builder_gate

Usage:
    python scripts/fusion_build.py "the plan text"
    python scripts/fusion_build.py --file plan.md
    python scripts/fusion_build.py --refute-only "the plan text"   # Codex only, faster
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fusion_common import fan_out, print_panel  # noqa: E402


def _refute_prompt(plan: str) -> str:
    return (
        "Adversarially REVIEW this PLAN (not code — a plan of action). Your job is to "
        "find why it FAILS, not to approve it. Check specifically:\n"
        "  - dependency/ordering errors (step N assumes step N+1's result)\n"
        "  - hidden costs or wasted work (building on an unverified assumption)\n"
        "  - things that look done but aren't (speculative / unverifiable steps)\n"
        "  - a cheaper or safer path that achieves the same goal\n"
        "Output: a short list of concrete objections, each tagged [BLOCKER]/[RISK]/"
        "[NIT], then one line: WEAK SPOT = <the single most likely way this plan goes "
        "wrong>. Be specific to THIS plan; no generic project advice.\n\n"
        f"PLAN:\n{plan}"
    )


def _scout_prompt(plan: str) -> str:
    return (
        "Review this PLAN for what's MISSING (you are the scout, not the judge). "
        "List only: (a) relevant prior art / known approaches the plan ignores, "
        "(b) options it didn't consider, (c) facts worth checking before committing. "
        "Do NOT give a verdict or say whether the plan is good — only surface leads "
        "and gaps for a separate judge to weigh.\n\n"
        f"PLAN:\n{plan}"
    )


SCAFFOLD = """\
================ FUSION-BUILD SYNTHESIS (orchestrator fills this) ================
The panel critiques above are ADVISORY. Opus now synthesizes them into:

  BLIND SPOTS    — what the panel surfaced that the plan missed
  CONTRADICTIONS — where panelists disagree (and which is right, with reason)
  CHEAPER PATH   — any lower-cost route a panelist found
  DECISION RULE  — REQUIRED before executing. Pre-register both:
       Promote if  : <observable signal by step K that the plan is working>
       Kill if     : <observable signal the plan is wrong — abort/replan>
  Verdict: GO (proceed to execute) or NO-GO (revise/replan). A plan without a
  Decision Rule is unfalsifiable and must not be built.
================================================================================"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Stress-test a plan via an adversarial panel")
    ap.add_argument("plan", nargs="?", help="plan text, or '-' for stdin")
    ap.add_argument("--file", "-f", help="read plan from a file")
    ap.add_argument("--refute-only", action="store_true", help="Codex only (skip Gemini) — faster")
    args = ap.parse_args()

    if args.file:
        plan = Path(args.file).read_text(encoding="utf-8")
    elif args.plan == "-" or (args.plan is None and not sys.stdin.isatty()):
        plan = sys.stdin.read()
    elif args.plan:
        plan = args.plan
    else:
        ap.error("provide a plan, --file, or pipe via stdin")

    print(f"=== fusion-build: refuting plan ({'codex only' if args.refute_only else 'refute + scout'}) ===",
          file=sys.stderr)
    results = fan_out(plan, codex_prompt=_refute_prompt, codex_label="codex/refute",
                      gemini_prompt=_scout_prompt, gemini_label="gemini/scout",
                      codex_only=args.refute_only)
    if not print_panel(results):
        print("\n(no panelists responded — plan goes through unreviewed; "
              "treat with extra skepticism)", file=sys.stderr)
    print("\n" + SCAFFOLD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
