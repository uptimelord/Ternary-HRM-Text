#!/usr/bin/env python3
"""ask_gemini_window.py — fire a Gemini (`agy`) call in a VISIBLE Windows Terminal
window you can watch, while still capturing the response for the orchestrator.

Unlike ask_gemini.py (silent, transcript-scraped), this pops a real `wt` window
so you SEE Gemini think and answer live. The window runs `agy -p <prompt>`, the
response is also written to an output file, and (optionally) this process blocks
until that file appears and prints it — so an automated caller still gets the text.

Usage:
    # watch-only: pop a window, return immediately
    python scripts/ask_gemini_window.py "your prompt"

    # watch AND capture: pop a window, wait, print the response to stdout
    python scripts/ask_gemini_window.py --capture "your prompt"

    # interactive: open a full agy TUI you can type in
    python scripts/ask_gemini_window.py --interactive
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HOME = Path.home()
AGY_BIN = os.environ.get("AGY_BIN", str(HOME / "AppData" / "Local" / "agy" / "bin" / "agy.exe"))
WT = "wt.exe"
OUT_DIR = HOME / ".gemini" / "antigravity-cli" / "_window_out"


def _role_prompt(role: str) -> str:
    from importlib import import_module
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        ag = import_module("ask_gemini")
        return ag.ROLE_PROMPTS.get(role, "")
    except Exception:
        return ""


def launch(prompt: str, *, role: str = "raw", capture: bool = False,
           interactive: bool = False, timeout_s: int = 300) -> int:
    if interactive:
        # Full TUI you drive yourself, in a new tab/window.
        subprocess.Popen([WT, "new-tab", "--title", "Gemini (agy)", AGY_BIN])
        print("opened interactive agy window.")
        return 0

    sysp = _role_prompt(role)
    full = f"{sysp}\n\n{prompt}" if sysp else prompt

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = str(int(time.time()))
    out_file = OUT_DIR / f"resp_{stamp}.txt"
    prompt_file = OUT_DIR / f"prompt_{stamp}.txt"
    ps_file = OUT_DIR / f"run_{stamp}.ps1"

    # Write prompt + runner to disk so NOTHING goes through wt/PowerShell inline
    # quoting (which mangles ';' and quotes -> 0x80070002). The window just runs
    # a .ps1 file that reads the prompt from a file.
    prompt_file.write_text(full, encoding="utf-8")
    ps_body = (
        "$ErrorActionPreference='Continue'\n"
        f"$p = Get-Content -Raw -LiteralPath '{prompt_file}'\n"
        "Write-Host '=== Gemini (agy) ===' -ForegroundColor Cyan\n"
        "Write-Host $p -ForegroundColor DarkGray\n"
        "Write-Host '--- response ---' -ForegroundColor Cyan\n"
        f"& '{AGY_BIN}' -p $p | Tee-Object -FilePath '{out_file}'\n"
        "Write-Host '--- done (window stays open; close when ready) ---' -ForegroundColor Cyan\n"
    )
    ps_file.write_text(ps_body, encoding="utf-8")

    # File-based launch: no inline command, no quoting to mangle.
    subprocess.Popen([
        WT, "new-tab", "--title", f"Gemini scout [{role}]",
        "powershell", "-NoExit", "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", str(ps_file),
    ])
    print(f"opened Gemini window (role={role}); output -> {out_file}")

    if not capture:
        return 0

    # Block until the window's tee'd file has content, then print it.
    print("waiting for response (watch the window)...", file=sys.stderr)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if out_file.exists() and out_file.stat().st_size > 0:
            time.sleep(1.0)  # let the write settle
            text = out_file.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                print(text)
                return 0
        time.sleep(1.0)
    print("ask_gemini_window: timed out waiting for response", file=sys.stderr)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Gemini (agy) in a visible Windows Terminal window")
    ap.add_argument("prompt", nargs="?", help="prompt text")
    ap.add_argument("--role", default="raw", help="role contract (scout|raw); see ask_gemini.py")
    ap.add_argument("--capture", action="store_true", help="wait for + print the response")
    ap.add_argument("--interactive", action="store_true", help="open a full agy TUI instead")
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    if not args.interactive and not args.prompt:
        ap.error("provide a prompt, or use --interactive")
    return launch(args.prompt or "", role=args.role, capture=args.capture,
                  interactive=args.interactive, timeout_s=args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
