#!/usr/bin/env python3
"""builder_gate.py — the verifier-gated commit gate for the multi-agent build stack.

This is the LOAD-BEARING piece (Layer 1 of the build-stack design). Everything
else — which model fills the scout/refute/impl seat — is swappable around this.
The gate is pure code: no model vote decides anything. It answers one question:

    "Is this diff safe to commit?"  -> PASS / FAIL + structured reasons

It composes the repo's existing, trusted machinery (nothing new is invented):

  1. DISCIPLINE   any touched experiment README has a pre-registered '## Decision
                  Rule' (experiments.discipline.assert_preregistered_readme).
  2. GUARD_RAIL   no held-out / reporting-only split is referenced by touched
                  training code (evaluation.guard_rail + a source scan).
  3. TESTS        pytest for the touched experiments' test files is green.
  4. REGRESSION   the known-verdict blind-eval battery did not flip (a builder
                  change must not make Exp83 stop promoting / Exp82 stop killing).
  5. HYGIENE      no obvious "claimed green without running" / loose-metric tells
                  in touched result files.

This is the Moonshot §3.5 "held-out non-degradation" rule applied to the BUILDER,
not the model: an agent's change is rejected if it regresses a known verdict.
It is also the LCF/Universal-Verifier invariant — the gate is the kernel; agents
(scout/impl/refute) are untrusted tactics that propose; only this code disposes.

Usage:
    python scripts/builder_gate.py                 # gate the working-tree diff vs HEAD
    python scripts/builder_gate.py --base HEAD~1   # gate vs another ref
    python scripts/builder_gate.py --files a.py b.py  # gate an explicit file set
    python scripts/builder_gate.py --skip-tests    # fast checks only (no pytest)
    python scripts/builder_gate.py --json          # machine-readable result

Exit code 0 = PASS (safe to commit), 1 = FAIL, 2 = gate error.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

EXPERIMENTS_DIR = REPO_ROOT / "experiments"
TESTS_DIR = REPO_ROOT / "tests"
ANSWER_KEY = REPO_ROOT / "scripts" / "blind_eval_answer_key.json"


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    skipped: bool = False


@dataclass
class GateResult:
    passed: bool
    checks: list[CheckResult] = field(default_factory=list)

    def add(self, c: CheckResult) -> None:
        self.checks.append(c)
        if not c.passed and not c.skipped:
            self.passed = False


# --------------------------------------------------------------------------- diff


def changed_files(base: str | None) -> list[Path]:
    """Files changed vs base (default: working tree + index + untracked vs HEAD)."""
    out: set[str] = set()
    try:
        if base:
            r = subprocess.run(["git", "-C", str(REPO_ROOT), "diff", "--name-only", base],
                               capture_output=True, text=True, check=True)
            out.update(r.stdout.split())
        else:
            for args in (["diff", "--name-only", "HEAD"],
                         ["diff", "--name-only", "--cached"],
                         ["ls-files", "--others", "--exclude-standard"]):
                r = subprocess.run(["git", "-C", str(REPO_ROOT), *args],
                                   capture_output=True, text=True, check=True)
                out.update(r.stdout.split())
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"git diff failed: {exc.stderr}") from exc
    return [REPO_ROOT / f for f in sorted(out) if f]


def _experiment_of(path: Path) -> Path | None:
    """If path is under experiments/Experiment NN - .../, return that dir."""
    try:
        rel = path.resolve().relative_to(EXPERIMENTS_DIR)
    except ValueError:
        return None
    top = rel.parts[0] if rel.parts else ""
    if top.startswith("Experiment "):
        return EXPERIMENTS_DIR / top
    return None


# --------------------------------------------------------------------------- checks


def check_discipline(files: list[Path]) -> CheckResult:
    """Every touched experiment README must carry a pre-registered Decision Rule."""
    from experiments.discipline import extract_preregistered_decision_rule

    readmes = {e / "README.md" for f in files if (e := _experiment_of(f)) is not None}
    readmes = {r for r in readmes if r.exists()}
    if not readmes:
        return CheckResult("discipline", True, "no experiment README touched", skipped=True)

    failures = []
    for r in sorted(readmes):
        try:
            rule = extract_preregistered_decision_rule(r.read_text(encoding="utf-8"))
            if not rule.promote or not rule.kill:
                failures.append(f"{r.parent.name}: Decision Rule missing Promote/Kill line")
        except Exception as exc:
            failures.append(f"{r.parent.name}: {exc}")
    if failures:
        return CheckResult("discipline", False, "; ".join(failures))
    return CheckResult("discipline", True, f"{len(readmes)} README(s) carry a Decision Rule")


def check_guard_rail(files: list[Path]) -> CheckResult:
    """No touched training code may reference a held-out / reporting-only split.

    Tooling under scripts/ is excluded: the gate and harness *describe* the leak
    pattern in their own source, which must not count as a leak."""
    from evaluation.guard_rail import is_held_out_file

    py = [f for f in files if f.suffix == ".py" and f.exists()
          and "scripts" not in f.resolve().parts          # don't scan the scanners
          and "tests" not in f.resolve().parts]            # tests reference leaks on purpose
    suspects = []
    train_ctx = re.compile(r"\b(train|sft|fit|ingest)\b", re.I)
    heldout_ref = re.compile(r"held[_-]?out|heldout_|frozen_arithmetic_200|reporting[_-]?only", re.I)
    for f in py:
        for i, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):  # comments are not executable leaks
                continue
            if heldout_ref.search(line) and train_ctx.search(line):
                suspects.append(f"{f.name}:{i}: {stripped[:80]}")
            for tok in re.findall(r"[\"']([^\"']+\.jsonl)[\"']", line):
                if is_held_out_file(tok) and train_ctx.search(line):
                    suspects.append(f"{f.name}:{i}: held-out file in train ctx -> {tok}")
    if suspects:
        return CheckResult("guard_rail", False, "LEAK SUSPECTED: " + " | ".join(suspects[:6]))
    return CheckResult("guard_rail", True, f"{len(py)} py file(s) scanned, no leak pattern")


def check_tests(files: list[Path], *, skip: bool) -> CheckResult:
    """Run pytest for touched experiments' test files."""
    if skip:
        return CheckResult("tests", True, "skipped (--skip-tests)", skipped=True)
    # Map touched experiments -> their test file(s).
    exp_ids = set()
    for f in files:
        e = _experiment_of(f)
        if e:
            m = re.match(r"Experiment (\d+(?:\.\d+)?)", e.name)
            if m:
                exp_ids.add(m.group(1).replace(".", "_"))
    test_files = []
    for eid in exp_ids:
        for t in TESTS_DIR.glob(f"test_exp{eid}*.py"):
            test_files.append(str(t))
    # Also include any directly-touched test files.
    test_files += [str(f) for f in files if f.parent == TESTS_DIR and f.suffix == ".py" and f.exists()]
    test_files = sorted(set(test_files))
    if not test_files:
        return CheckResult("tests", True, "no mapped test files for touched experiments", skipped=True)
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", *test_files],
                       cwd=str(REPO_ROOT), capture_output=True, text=True)
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip()[:200]
    if r.returncode != 0:
        return CheckResult("tests", False, f"pytest FAILED: {tail}")
    return CheckResult("tests", True, f"{len(test_files)} test file(s): {tail}")


def _verdict_token(section: str) -> str:
    """Decide promote/kill from a Verdict section. The real verdict is stated in
    the FIRST sentence; later prose ('+5pp promote bar', 'within 2pp') is noise.
    So weight the first line, fall back to a whole-section count only if the first
    line is silent."""
    low = section.lower().strip()
    # first non-empty line carries the decision
    first = ""
    for ln in low.splitlines():
        if ln.strip():
            first = ln.strip()
            break
    # explicit phrasings, checked on the first line
    if re.search(r"\bkill\b|do not promote|no effect|no lift|lane closes", first):
        return "kill"
    if re.search(r"\bpromote\b", first):
        return "promote"
    # fall back to whole-section majority
    promote = len(re.findall(r"\bpromote\b", low))
    kill = len(re.findall(r"\bkill\b|do not promote|no effect|no lift", low))
    return "promote" if promote > kill else "kill" if kill > promote else "ambiguous"


def check_known_verdicts() -> CheckResult:
    """Regression: the known-verdict battery must still hold its ground truth.

    Reads the blind-eval answer key and confirms each experiment's CURRENT README
    Verdict section still matches the recorded ground truth. A builder change that
    silently flips Exp83 promote -> kill (or vice versa) is rejected."""
    if not ANSWER_KEY.exists():
        return CheckResult("known_verdicts", True, "no answer key present", skipped=True)
    key = json.loads(ANSWER_KEY.read_text(encoding="utf-8"))["experiments"]
    mismatches = []
    checked = 0
    for eid, truth in key.items():
        dirs = sorted(EXPERIMENTS_DIR.glob(f"Experiment {eid} - *"))
        if not dirs:
            continue
        readme = dirs[0] / "README.md"
        if not readme.exists():
            continue
        text = readme.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"##\s*Verdict\s*(.*?)(?:\n##\s|\Z)", text, re.DOTALL | re.IGNORECASE)
        section = m.group(1) if m else text
        gt = truth["ground_truth_verdict"]
        current = _verdict_token(section)
        checked += 1
        if current != gt:
            mismatches.append(f"Exp{eid}: README says {current!r}, ground truth {gt!r}")
    if mismatches:
        return CheckResult("known_verdicts", False,
                           "VERDICT REGRESSION: " + " | ".join(mismatches))
    return CheckResult("known_verdicts", True, f"{checked} known verdict(s) intact")


def check_hygiene(files: list[Path]) -> CheckResult:
    """Cheap 'claimed-green-without-running' / loose-metric tells in result files."""
    warnings = []
    for f in files:
        if f.suffix != ".md" or not f.exists():
            continue
        if not (f.name.startswith("results_") or f.name == "README.md"):
            continue
        low = f.read_text(encoding="utf-8", errors="ignore").lower()
        # promote claimed but no strict-metric language anywhere
        if "promote" in low and not re.search(r"strict|pass@1|exact|held-?out|frozen", low):
            warnings.append(f"{f.parent.name}/{f.name}: promote without strict-metric language")
    if warnings:
        # hygiene is advisory: it warns but does not hard-fail the gate
        return CheckResult("hygiene", True, "ADVISORY: " + " | ".join(warnings[:4]))
    return CheckResult("hygiene", True, "no hygiene tells")


# --------------------------------------------------------------------------- run


def run_gate(*, base: str | None, explicit_files: list[str] | None,
             skip_tests: bool) -> GateResult:
    files = ([REPO_ROOT / f for f in explicit_files] if explicit_files
             else changed_files(base))
    result = GateResult(passed=True)
    result.add(check_discipline(files))
    result.add(check_guard_rail(files))
    result.add(check_tests(files, skip=skip_tests))
    result.add(check_known_verdicts())
    result.add(check_hygiene(files))
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Verifier-gated commit gate for the build stack")
    ap.add_argument("--base", default=None, help="git ref to diff against (default: working tree vs HEAD)")
    ap.add_argument("--files", nargs="*", help="explicit file list instead of git diff")
    ap.add_argument("--skip-tests", action="store_true", help="skip pytest (fast checks only)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    try:
        result = run_gate(base=args.base, explicit_files=args.files, skip_tests=args.skip_tests)
    except RuntimeError as exc:
        print(f"builder_gate error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(asdict(result), indent=2))
    else:
        print("=== builder_gate ===")
        for c in result.checks:
            mark = "SKIP" if c.skipped else ("PASS" if c.passed else "FAIL")
            print(f"  [{mark}] {c.name}: {c.detail}")
        print(f"\n{'PASS — safe to commit' if result.passed else 'FAIL — do not commit'}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
