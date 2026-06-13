#!/usr/bin/env python3
"""ask_codex.py / ask_codex_window.py logic — non-interactive Codex (OpenAI Codex
CLI) adapter for the build stack's DEEP-IMPL / REFUTE seat.

Unlike `agy`, the Codex CLI's `exec` subcommand prints cleanly to stdout and
supports `-o/--output-last-message <FILE>`, so no transcript-scraping is needed.

Roles (see scripts/BLIND_EVAL_README.md):
  - refute : adversarial verifier. Paranoia weaponized — try to BREAK the claim.
             Hunt metric swaps, leaks, stitched fragments, hidden costs. This is
             Codex's best seat (skeptical bias is a feature here).
  - build  : deep implementation / hard debugging (the Codex-rescue role).
  - raw    : no role injection.

Two entry points:
  ask_codex(prompt, role=...)            -> headless, returns the final message
  python scripts/ask_codex.py ...        -> CLI; add --window to pop a visible
                                            Windows Terminal tab you can watch.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HOME = Path.home()


def _resolve_codex() -> str:
    """Find the codex launcher. Prefer CODEX_BIN, then PATH, then the npm-global
    .cmd shim (Windows has no codex.exe — it's a node shim)."""
    env = os.environ.get("CODEX_BIN")
    if env:
        return env
    from shutil import which
    for cand in ("codex", "codex.cmd"):
        hit = which(cand)
        if hit:
            return hit
    npm_cmd = HOME / "AppData" / "Roaming" / "npm" / "codex.cmd"
    if npm_cmd.exists():
        return str(npm_cmd)
    return "codex"  # last resort; will error clearly if missing


CODEX_BIN = _resolve_codex()
WT = "wt.exe"
OUT_DIR = HOME / ".codex" / "_window_out"

ROLE_PROMPTS = {
    "refute": (
        "You are the ADVERSARIAL VERIFIER in a verifier-gated build stack. Your job "
        "is to REFUTE the claim/result/diff below, not to agree with it. Default to "
        "'broken / not proven' if uncertain. Hunt specifically for: a metric swap "
        "(loose vs strict), held-out or reporting-only data used in training, "
        "stitched/partial results presented as a full run, hidden costs (e.g. an "
        "uncompressed head behind a quality-per-MB number), claims with no artifact, "
        "and verdicts 'nicer' than the evidence supports. Output a list of concrete "
        "objections with severity; end with REFUTED or HOLDS and one-line justification. "
        "You advise; you do NOT set the final verdict (a code gate does)."
    ),
    "build": (
        "You are the DEEP IMPLEMENTATION agent. Implement faithfully to the paper / "
        "spec. Do not 'paper-ish' approximate — match the method. Run what you can, "
        "report what you ran, and state honestly if a result is a kill."
    ),
    "raw": "",
}


def ask_codex(prompt: str, *, role: str = "raw", model: str | None = None,
              cwd: str | None = None, sandbox: str = "read-only",
              effort: str = "xhigh", timeout_s: int = 600) -> str:
    """Run `codex exec` non-interactively and return its final message.

    effort defaults to 'xhigh' (standing rule: codex always runs xhigh unless
    explicitly told otherwise; the repo config.toml default is 'low')."""
    sysp = ROLE_PROMPTS.get(role, "")
    full = f"{sysp}\n\n{prompt}" if sysp else prompt

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f"resp_{int(time.time())}.txt"

    cmd = [CODEX_BIN, "exec", "--skip-git-repo-check",
           "-c", f"model_reasoning_effort={effort}",
           "-s", sandbox, "-o", str(out_file)]
    if model:
        cmd += ["-m", model]
    if cwd:
        cmd += ["-C", cwd]
    cmd.append(full)

    try:
        # On Windows the codex launcher is a .cmd shim; subprocess needs shell=True
        # to resolve it. With shell=True pass the command as a list is unreliable,
        # so build a properly-quoted string via subprocess.list2cmdline.
        if CODEX_BIN.lower().endswith(".cmd"):
            run_cmd = subprocess.list2cmdline(cmd)
            subprocess.run(run_cmd, timeout=timeout_s, shell=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.run(cmd, timeout=timeout_s,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"codex exec timed out after {timeout_s}s") from exc
    except FileNotFoundError as exc:
        raise RuntimeError(f"codex binary not found ({CODEX_BIN}); set CODEX_BIN") from exc

    if not out_file.exists():
        raise RuntimeError("codex exec produced no output-last-message file")
    text = out_file.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        raise RuntimeError("codex exec produced an empty final message")
    return text


def launch_window(prompt: str, *, role: str = "raw", model: str | None = None,
                  cwd: str | None = None, sandbox: str = "read-only",
                  effort: str = "xhigh") -> int:
    """Pop a visible Windows Terminal tab running `codex exec` so you can watch it."""
    sysp = ROLE_PROMPTS.get(role, "")
    full = f"{sysp}\n\n{prompt}" if sysp else prompt

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = str(int(time.time()))
    prompt_file = OUT_DIR / f"prompt_{stamp}.txt"
    ps_file = OUT_DIR / f"run_{stamp}.ps1"
    out_file = OUT_DIR / f"resp_{stamp}.txt"
    prompt_file.write_text(full, encoding="utf-8")

    # File-based launch (never inline) to dodge wt/PowerShell quoting (0x80070002).
    model_arg = f"-m '{model}' " if model else ""
    cwd_arg = f"-C '{cwd}' " if cwd else ""
    ps_body = (
        "$ErrorActionPreference='Continue'\n"
        f"$p = Get-Content -Raw -LiteralPath '{prompt_file}'\n"
        "Write-Host '=== Codex exec (xhigh) ===' -ForegroundColor Green\n"
        f"& '{CODEX_BIN}' exec --skip-git-repo-check -c model_reasoning_effort={effort} "
        f"-s {sandbox} {model_arg}{cwd_arg}-o '{out_file}' $p\n"
        "Write-Host '--- done (window stays open) ---' -ForegroundColor Green\n"
    )
    ps_file.write_text(ps_body, encoding="utf-8")
    subprocess.Popen([
        WT, "new-tab", "--title", f"Codex [{role}] xhigh",
        "powershell", "-NoExit", "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", str(ps_file),
    ])
    print(f"opened Codex window (role={role}, effort={effort}); output -> {out_file}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Non-interactive Codex (deep-impl / refute seat)")
    ap.add_argument("prompt", nargs="?", help="prompt, or '-' for stdin")
    ap.add_argument("--file", "-f", help="read prompt from file")
    ap.add_argument("--role", default="raw", choices=sorted(ROLE_PROMPTS))
    ap.add_argument("--model", "-m", default=None)
    ap.add_argument("--cwd", "-C", default=None, help="run with this repo as working dir")
    ap.add_argument("--sandbox", "-s", default="read-only",
                    help="codex sandbox mode (read-only | workspace-write | danger-full-access)")
    ap.add_argument("--effort", default="xhigh",
                    help="reasoning effort (default xhigh; standing rule). low|medium|high|xhigh")
    ap.add_argument("--window", action="store_true", help="pop a visible wt tab to watch")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    if args.file:
        prompt = Path(args.file).read_text(encoding="utf-8")
    elif args.prompt == "-" or (args.prompt is None and not sys.stdin.isatty()):
        prompt = sys.stdin.read()
    elif args.prompt:
        prompt = args.prompt
    else:
        ap.error("provide a prompt, --file, or pipe via stdin")

    if args.window:
        return launch_window(prompt, role=args.role, model=args.model,
                             cwd=args.cwd, sandbox=args.sandbox, effort=args.effort)
    try:
        print(ask_codex(prompt, role=args.role, model=args.model, cwd=args.cwd,
                        sandbox=args.sandbox, effort=args.effort, timeout_s=args.timeout))
        return 0
    except RuntimeError as exc:
        print(f"ask_codex error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
