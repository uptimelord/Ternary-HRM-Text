#!/usr/bin/env python3
"""ask_gemini.py — non-interactive Gemini (Antigravity `agy` CLI) adapter.

The `agy` CLI runs Gemini 3.1 Pro (High) via Google Cloud Code Assist OAuth (no
API key needed), but in `--print` mode it writes the response to the Windows
console handle, which a piped/redirected parent process cannot capture. This
adapter works around that: it fires `agy -p`, then reads the response from the
per-conversation `transcript.jsonl` the CLI writes under its `brain/` dir.

Role in the model stack (see scripts/BLIND_EVAL_README.md / builder gate design):
Gemini is the SCOUT / retrieval seat — optimism is safe here because the output
is *leads and citations*, never a verdict, number, or claim. Do NOT use it to
summarize results or decide promote/kill.

Usage:
    python scripts/ask_gemini.py "what papers cover ternary KV-cache compression?"
    python scripts/ask_gemini.py --role scout --file prompt.md
    echo "..." | python scripts/ask_gemini.py -

Exit code 0 on a captured response, 1 on failure (empty / timeout / agy error).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# agy writes conversations + transcripts under the user's .gemini dir.
AGY_HOME = Path(os.environ.get("AGY_HOME", Path.home() / ".gemini" / "antigravity-cli"))
BRAIN_DIR = AGY_HOME / "brain"
AGY_BIN = os.environ.get("AGY_BIN", str(Path.home() / "AppData" / "Local" / "agy" / "bin" / "agy.exe"))

# Role system prompts — the contract that keeps each model's bias in its safe seat.
ROLE_PROMPTS = {
    "scout": (
        "You are the RETRIEVAL/SCOUT agent in a verifier-gated build stack. Your job "
        "is to surface leads: relevant papers (with arXiv ids), prior art, and "
        "locations in a codebase. Output ONLY candidates for a separate builder to "
        "verify. NEVER assert that something works, NEVER state a result or metric, "
        "NEVER give a promote/kill verdict. Optimism is fine; unverified claims are not. "
        "Format as a short list of leads."
    ),
    "raw": "",  # no role injection
}


def _transcripts_snapshot() -> dict[Path, float]:
    """Map of transcript.jsonl -> mtime, for detecting the one this run creates."""
    out: dict[Path, float] = {}
    if not BRAIN_DIR.exists():
        return out
    for t in BRAIN_DIR.glob("*/.system_generated/logs/transcript.jsonl"):
        try:
            out[t] = t.stat().st_mtime
        except OSError:
            pass
    return out


def _newest_transcript_since(before: dict[Path, float]) -> Path | None:
    """The transcript created/updated by the just-finished run."""
    after = _transcripts_snapshot()
    candidates: list[tuple[float, Path]] = []
    for t, mt in after.items():
        if t not in before or mt > before[t]:
            candidates.append((mt, t))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def _extract_response(transcript: Path) -> str:
    """Return the last PLANNER_RESPONSE content from an agy transcript.jsonl.

    Schema (observed): each line is a step object with `type` and `content`
    (older builds used `role`/`text` — both are handled)."""
    responses: list[str] = []
    for line in transcript.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = obj.get("type") or obj.get("role") or obj.get("kind") or ""
        if kind != "PLANNER_RESPONSE":
            continue
        txt = obj.get("content") or obj.get("text") or obj.get("message") or ""
        if isinstance(txt, (dict, list)):
            txt = json.dumps(txt)
        if txt:
            responses.append(str(txt))
    return responses[-1].strip() if responses else ""


def ask_gemini(prompt: str, *, role: str = "raw", thinking: str = "high",
               timeout_s: int = 300, model: str | None = None) -> str:
    """Fire agy -p and recover the response from its transcript. Raises on failure."""
    sys_prompt = ROLE_PROMPTS.get(role, "")
    full_prompt = f"{sys_prompt}\n\n{prompt}" if sys_prompt else prompt

    before = _transcripts_snapshot()
    # Keep the invocation minimal: `agy -p <prompt>`. Extra flags before the
    # positional prompt confused agy's parser in testing; permissions aren't
    # needed because scout/refute run with no tools (pure text generation).
    cmd = [AGY_BIN, "-p"]
    if model:
        cmd += ["--model", model]
    cmd.append(full_prompt)

    try:
        subprocess.run(cmd, timeout=timeout_s + 30,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"agy timed out after {timeout_s}s") from exc
    except FileNotFoundError as exc:
        raise RuntimeError(f"agy binary not found at {AGY_BIN} (set AGY_BIN)") from exc

    # The transcript is written as the run completes; give the FS a moment.
    transcript = None
    for _ in range(10):
        transcript = _newest_transcript_since(before)
        if transcript is not None:
            break
        time.sleep(0.5)
    if transcript is None:
        raise RuntimeError("no new agy transcript found after run (did agy actually run?)")

    text = _extract_response(transcript)
    if not text:
        raise RuntimeError(f"agy produced no PLANNER_RESPONSE in {transcript}")
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description="Non-interactive Gemini via the agy CLI")
    ap.add_argument("prompt", nargs="?", help="prompt text, or '-' to read stdin")
    ap.add_argument("--file", "-f", help="read prompt from a file")
    ap.add_argument("--role", default="raw", choices=sorted(ROLE_PROMPTS),
                    help="role contract to prepend (default: raw)")
    ap.add_argument("--thinking", default="high")
    ap.add_argument("--model", default=None, help="override model (default: agy's configured one)")
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    if args.file:
        prompt = Path(args.file).read_text(encoding="utf-8")
    elif args.prompt == "-" or (args.prompt is None and not sys.stdin.isatty()):
        prompt = sys.stdin.read()
    elif args.prompt:
        prompt = args.prompt
    else:
        ap.error("provide a prompt, --file, or pipe via stdin")

    try:
        print(ask_gemini(prompt, role=args.role, thinking=args.thinking,
                         timeout_s=args.timeout, model=args.model))
        return 0
    except RuntimeError as exc:
        print(f"ask_gemini error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
