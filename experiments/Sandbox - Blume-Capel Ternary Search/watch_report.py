"""Emit loop wake when a report file appears (for gpu_queue follow-up)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: watch_report.py <relative-report-path> [poll_seconds]", file=sys.stderr)
        return 1
    report = REPO_ROOT / sys.argv[1]
    poll_s = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    prompt = (
        "Exp 4.4 CUDA finished: read report.json, update README verdict, "
        "run gpu_queue run-next for exp4_5 hybrid then exp4_6, continue loop."
    )
    while not report.exists():
        time.sleep(poll_s)
    print(f'AGENT_LOOP_WAKE_blume_capel {json.dumps({"prompt": prompt})}', flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
