"""_fusion_common.py — shared panel fan-out for the fusion-* tools.

Both fusion-plan (generate plans) and fusion-build (refute a plan) fan the same
input to the external model seats in parallel and collect their text. The only
difference is the prompt each seat gets and how the orchestrator (Opus) judges
the results — so that lives in the callers; the parallel-subprocess machinery
lives here.

Seats (each optional; missing adapters are skipped — graceful fallback):
  - Codex (GPT-5.5) via scripts/ask_codex.py, xhigh, repo-aware (cwd=repo so it
    reads AGENTS.md natively).
  - Gemini (3.1 Pro) via scripts/ask_gemini.py, project-context injected (it has
    no repo filesystem access here).
"""

from __future__ import annotations

import concurrent.futures as cf
import sys
from pathlib import Path
from typing import Callable

SCRIPTS = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))


def _run_codex(text: str, prompt_fn: Callable[[str], str], label: str) -> tuple[str, str | None]:
    try:
        import ask_codex
        return label, ask_codex.ask_codex(
            prompt_fn(text), role="raw", effort="xhigh",
            cwd=str(REPO_ROOT), timeout_s=480)
    except Exception as exc:
        return label, f"__ERROR__ {exc}"


def _run_gemini(text: str, prompt_fn: Callable[[str], str], label: str) -> tuple[str, str | None]:
    try:
        import ask_gemini
        return label, ask_gemini.ask_gemini(
            prompt_fn(text), role="raw", project_context=True, timeout_s=300)
    except Exception as exc:
        return label, f"__ERROR__ {exc}"


def fan_out(text: str, *, codex_prompt: Callable[[str], str], codex_label: str,
            gemini_prompt: Callable[[str], str] | None, gemini_label: str,
            codex_only: bool = False) -> dict[str, str | None]:
    """Run the panel in parallel; return {label: response or None}."""
    jobs = [(lambda: _run_codex(text, codex_prompt, codex_label))]
    if gemini_prompt is not None and not codex_only:
        jobs.append(lambda: _run_gemini(text, gemini_prompt, gemini_label))
    results: dict[str, str | None] = {}
    with cf.ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        for f in cf.as_completed([ex.submit(j) for j in jobs]):
            name, out = f.result()
            results[name] = out
    return results


def print_panel(results: dict[str, str | None]) -> bool:
    """Print each panelist's output. Returns True if any panelist responded."""
    any_live = False
    for name, out in results.items():
        print(f"\n---------------- panelist: {name} ----------------")
        if out and out.startswith("__ERROR__"):
            print(f"(unavailable — {out[len('__ERROR__'):].strip()})")
        elif out:
            any_live = True
            print(out.strip())
        else:
            print("(unavailable — no response)")
    return any_live
