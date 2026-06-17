"""Serialize-only GPU job queue for Blume-Capel sandbox CUDA runs."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SANDBOX = Path(__file__).resolve().parent
REPO_ROOT = SANDBOX.parents[1]
QUEUE_PATH = SANDBOX / "gpu_queue.json"
LOCK_PATH = SANDBOX / "gpu.lock"
STALE_LOCK_S = 3 * 3600


def _load_queue() -> dict[str, Any]:
    if not QUEUE_PATH.exists():
        return {"jobs": [], "history": []}
    return json.loads(QUEUE_PATH.read_text(encoding="utf-8"))


def _save_queue(data: dict[str, Any]) -> None:
    QUEUE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _lock_meta() -> dict[str, Any] | None:
    if not LOCK_PATH.exists():
        return None
    try:
        return json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def gpu_busy() -> bool:
    meta = _lock_meta()
    if meta is None:
        return False
    if time.time() - float(meta.get("started", 0)) > STALE_LOCK_S:
        LOCK_PATH.unlink(missing_ok=True)
        return False
    pid = int(meta.get("pid", -1))
    if pid > 0:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            LOCK_PATH.unlink(missing_ok=True)
            return False
    return True


def acquire_lock(job_id: str) -> bool:
    if gpu_busy():
        return False
    LOCK_PATH.write_text(
        json.dumps({"pid": os.getpid(), "started": time.time(), "job_id": job_id}),
        encoding="utf-8",
    )
    return True


def release_lock() -> None:
    LOCK_PATH.unlink(missing_ok=True)


def enqueue(job_id: str, cmd: str, *, goal: str = "", priority: int = 100) -> None:
    data = _load_queue()
    jobs = data.setdefault("jobs", [])
    if any(j.get("id") == job_id for j in jobs):
        raise ValueError(f"job id already queued: {job_id}")
    jobs.append(
        {
            "id": job_id,
            "cmd": cmd,
            "goal": goal,
            "priority": priority,
            "status": "pending",
            "enqueued_at": time.time(),
        }
    )
    jobs.sort(key=lambda j: (j.get("priority", 100), j.get("enqueued_at", 0)))
    _save_queue(data)
    print(f"enqueued {job_id}")


def status() -> dict[str, Any]:
    data = _load_queue()
    pending = [j for j in data.get("jobs", []) if j.get("status") == "pending"]
    return {
        "gpu_busy": gpu_busy(),
        "lock": _lock_meta(),
        "pending": len(pending),
        "jobs": data.get("jobs", []),
        "history": data.get("history", [])[-5:],
    }


def run_next(*, dry_run: bool = False) -> int:
    data = _load_queue()
    jobs = data.get("jobs", [])
    pending = [j for j in jobs if j.get("status") == "pending"]
    if not pending:
        print("queue empty")
        return 0
    if gpu_busy():
        print("gpu busy")
        return 2
    job = pending[0]
    if dry_run:
        print(json.dumps(job, indent=2))
        return 0
    if not acquire_lock(str(job["id"])):
        print("failed to acquire gpu lock")
        return 2
    job["status"] = "running"
    job["started_at"] = time.time()
    _save_queue(data)
    argv = job.get("argv")
    if argv is None:
        argv = shlex.split(str(job["cmd"]), posix=False)
    print(f"running {job['id']}: {argv}")
    rc = subprocess.call(argv, cwd=str(REPO_ROOT))
    job["status"] = "done" if rc == 0 else "failed"
    job["finished_at"] = time.time()
    job["exit_code"] = rc
    data.setdefault("history", []).append(dict(job))
    data["jobs"] = [j for j in jobs if j.get("id") != job["id"]]
    _save_queue(data)
    release_lock()
    print(f"finished {job['id']} rc={rc}")
    return rc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    enq = sub.add_parser("enqueue")
    enq.add_argument("--id", required=True)
    enq.add_argument("--command", nargs=argparse.REMAINDER, required=True)
    enq.add_argument("--goal", default="")
    enq.add_argument("--priority", type=int, default=100)

    sub.add_parser("status")
    run = sub.add_parser("run-next")
    run.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    if args.cmd == "enqueue":
        cmd = " ".join(str(part) for part in args.command if str(part) != "--").strip()
        if not cmd:
            raise SystemExit("enqueue requires a non-empty --command")
        enqueue(args.id, cmd, goal=args.goal, priority=args.priority)
        return 0
    if args.cmd == "status":
        print(json.dumps(status(), indent=2))
        return 0
    if args.cmd == "run-next":
        return run_next(dry_run=args.dry_run)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
