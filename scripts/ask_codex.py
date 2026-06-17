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


def _resolve_codex() -> list[str]:
    """Return the codex launcher as an argv prefix.

    Prefer `node codex.js` (no .cmd shim, no shell, stdin-safe) over the .cmd
    wrapper — the shim mangles multi-line args and breaks stdin on Windows."""
    env = os.environ.get("CODEX_BIN")
    if env:
        return [env]
    # node + codex.js entry (the .cmd shim just wraps this)
    from shutil import which
    node = which("node")
    codex_js = HOME / "AppData" / "Roaming" / "npm" / "node_modules" / "@openai" / "codex" / "bin" / "codex.js"
    if node and codex_js.exists():
        return [node, str(codex_js)]
    # fallbacks: PATH codex, then the .cmd shim
    for cand in ("codex", "codex.cmd"):
        hit = which(cand)
        if hit:
            return [hit]
    npm_cmd = HOME / "AppData" / "Roaming" / "npm" / "codex.cmd"
    if npm_cmd.exists():
        return [str(npm_cmd)]
    return ["codex"]


CODEX_ARGV = _resolve_codex()
CODEX_BIN = CODEX_ARGV[0]  # for diagnostics
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

PROJECT_CONTEXT_FILE = Path(__file__).resolve().parent / "project_context.md"


def _project_context() -> str:
    """Compact repo-invariants brief. Codex also auto-reads AGENTS.md from the
    repo root, so this is belt-and-suspenders (off by default for codex)."""
    try:
        return PROJECT_CONTEXT_FILE.read_text(encoding="utf-8")
    except OSError:
        return ""


def ask_codex(prompt: str, *, role: str = "raw", model: str | None = None,
              cwd: str | None = None, sandbox: str = "read-only",
              effort: str = "xhigh", project_context: bool = False,
              timeout_s: int = 600) -> str:
    """Run `codex exec` non-interactively and return its final message.

    effort defaults to 'xhigh' (standing rule: codex always runs xhigh unless
    explicitly told otherwise; the repo config.toml default is 'low').
    project_context=True prepends the invariants brief (codex also auto-reads
    AGENTS.md from cwd, so this is usually unnecessary)."""
    sysp = ROLE_PROMPTS.get(role, "")
    ctx = (_project_context() + "\n\n---\n\n") if project_context else ""
    head = f"{ctx}{sysp}".strip()
    full = f"{head}\n\n{prompt}" if head else prompt

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f"resp_{int(time.time())}.txt"

    # Pass the prompt via STDIN (codex exec '-' reads stdin), never as a CLI arg:
    # multi-line args get truncated at the first newline by cmd.exe on the .cmd
    # shim path. stdin is newline-safe and has no length limit.
    cmd = [*CODEX_ARGV, "exec", "--skip-git-repo-check",
           "-c", f"model_reasoning_effort={effort}",
           "-s", sandbox, "-o", str(out_file)]
    if model:
        cmd += ["-m", model]
    if cwd:
        cmd += ["-C", cwd]
    cmd.append("-")  # read prompt from stdin

    stderr_text = ""
    try:
        # CODEX_ARGV is node+codex.js (no shell needed). Only a bare .cmd fallback
        # would need shell=True; node path is a clean argv list. Stdin is passed as
        # UTF-8 BYTES (not text=) because Windows text mode would encode via cp1252
        # and codex requires valid UTF-8 on stdin. Capture stderr so a usage-limit /
        # auth error surfaces as the reason instead of a silent "no output".
        use_shell = CODEX_ARGV[0].lower().endswith(".cmd")
        proc = subprocess.run(cmd if not use_shell else subprocess.list2cmdline(cmd),
                              input=full.encode("utf-8"), timeout=timeout_s, shell=use_shell,
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        stderr_text = proc.stderr.decode("utf-8", "ignore") if proc.stderr else ""
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"codex exec timed out after {timeout_s}s") from exc
    except FileNotFoundError as exc:
        raise RuntimeError(f"codex binary not found ({CODEX_BIN}); set CODEX_BIN") from exc

    def _reason() -> str:
        low = stderr_text.lower()
        if "usage limit" in low or "credits" in low:
            # pull the codex usage-limit line for a precise reason
            for line in stderr_text.splitlines():
                if "usage limit" in line.lower():
                    return line.strip()
            return "usage limit reached"
        if "auth" in low or "login" in low or "401" in low:
            return "auth error (run `codex login`)"
        tail = " ".join(stderr_text.split())[-200:]
        return tail or "no stderr"

    if not out_file.exists():
        raise RuntimeError(f"codex produced no output — {_reason()}")
    text = out_file.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        raise RuntimeError(f"codex produced an empty final message — {_reason()}")
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
