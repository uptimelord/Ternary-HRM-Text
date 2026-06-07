"""
Experiment 66 - Word Problem Reasoning Corpus (DeepSeek wording, Python truth)

Generates a lean, reasoning-dense corpus for the Phase-0 HRM. Core safety property
(Codex): DeepSeek writes the STORY, Python writes the TRUTH, the verifier FILTERS.
DeepSeek never computes -> it cannot poison the labels.

Pipeline per example:
  1. Python samples a spec (operands, ops) and computes the EXACT answer + step trace.
  2. DeepSeek-Flash dresses the spec in words (4 styles: direct / word / trace / noisy).
  3. Verify: the generated story must encode EXACTLY the spec's numbers + operations
     (no missing/extra numbers that change the math), AND the solver recomputed from
     the spec equals Python's answer. Else discard.
  4. Held-out guard: generated rows must not reuse a held-out spec signature.

This module is split so the TRUTH core (step 1) and the FILTER (step 3) are fully
unit-testable WITHOUT any API call. The DeepSeek call (step 2) reuses the existing
helper scripts/generate_deepseek_custom_dataset.py (urllib, env DEEPSEEK_API_KEY,
model deepseek-v4-flash, retry+backoff).

Mix target (Codex): 40k direct / 30k word / 20k trace / 10k noisy = 100k train.
Held-out: 1k direct / 1k word / 500 hard-multistep, disjoint specs + phrasings.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEEPSEEK_HELPER_PATH = REPO_ROOT / "scripts" / "generate_deepseek_custom_dataset.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ----------------------- STEP 1: PYTHON TRUTH -----------------------

Style = Literal["direct", "word", "trace", "noisy"]
OPS = ("+", "-", "*")


@dataclass(frozen=True)
class Spec:
    """A problem spec. Python owns this and the answer — DeepSeek never sees the answer."""
    kind: str                 # "binary" | "add_sub" (two-op)
    operands: tuple[int, ...]
    ops: tuple[str, ...]
    answer: int
    steps: tuple[str, ...]    # exact step trace, Python-computed

    def signature(self) -> str:
        return f"{self.kind}|{','.join(map(str, self.operands))}|{''.join(self.ops)}"


def _compute(a: int, op: str, b: int) -> int:
    return {"+": a + b, "-": a - b, "*": a * b}[op]


def sample_spec(rng: random.Random, *, hard: bool = False) -> Spec:
    """Sample a bounded arithmetic spec and compute its exact answer + trace.

    Easy: binary  a op b.   Hard: (a + b) - c  (two-step, carry-coupled).
    Bounds keep answers in a small range so the solver + verifier stay exact.

    Physicality: subtraction is constrained so stories stay sensible (you cannot
    drain more than you have). For binary '-': b <= a. For (a+b)-c: c <= a+b.
    This avoids absurd negative-quantity word problems ("ate 40 of 10 cookies").
    """
    if hard:
        a = rng.randint(10, 99)
        b = rng.randint(10, 99)
        r1 = a + b
        c = rng.randint(1, r1)  # c <= a+b  -> non-negative, physical
        ans = r1 - c
        steps = (f"{a} + {b} = {r1}", f"{r1} - {c} = {ans}")
        return Spec(kind="add_sub", operands=(a, b, c), ops=("+", "-"), answer=ans, steps=steps)
    op = rng.choice(OPS)
    if op == "*":
        a = rng.randint(2, 99)
        b = rng.randint(2, 99)
    elif op == "-":
        a = rng.randint(1, 99)
        b = rng.randint(0, a)  # b <= a -> non-negative, physical
    else:  # "+"
        a = rng.randint(1, 99)
        b = rng.randint(1, 99)
    ans = _compute(a, op, b)
    steps = (f"{a} {op} {b} = {ans}",)
    return Spec(kind="binary", operands=(a, b), ops=(op,), answer=ans, steps=steps)


# ----------------------- STEP 3: STORY-MATCH VERIFIER (no API) -----------------------

_NUM_RE = re.compile(r"-?\d+")


def numbers_in(text: str) -> list[int]:
    return [int(t) for t in _NUM_RE.findall(text)]


def story_matches_spec(story: str, spec: Spec, *, allow_answer_in_story: bool = False) -> tuple[bool, str]:
    """Does the DeepSeek story encode EXACTLY the spec's operand numbers?

    Reject if: a spec operand is missing, OR an extra number appears that is not
    the answer/an intermediate (which would change the math / signal a mis-generation).
    This stops WORDING poisoning even when the Python answer is correct.
    """
    nums = numbers_in(story)
    spec_nums = list(spec.operands)
    allowed = set(spec_nums)
    if allow_answer_in_story:
        allowed.add(spec.answer)
        for s in spec.steps:  # intermediate results may legitimately appear in trace style
            allowed.update(numbers_in(s))

    # every spec operand must be present
    for n in spec_nums:
        if n not in nums:
            return False, f"missing_operand:{n}"
    # no stray numbers that aren't allowed (distractor digits that change meaning)
    for n in nums:
        if n not in allowed:
            return False, f"unexpected_number:{n}"
    return True, "ok"


def verify_example(story: str, spec: Spec, style: Style) -> tuple[bool, str]:
    """Full filter: story encodes the spec AND solver recompute matches Python truth."""
    # Python owns the computed answer and step values. The prompt text should only
    # contain original operand numbers, even for trace style.
    ok, reason = story_matches_spec(story, spec, allow_answer_in_story=False)
    if not ok:
        return False, reason
    # solver recompute from the spec (NOT from the story) must equal Python's answer
    recomputed = _recompute_from_spec(spec)
    if recomputed != spec.answer:
        return False, f"solver_mismatch:{recomputed}!={spec.answer}"
    return True, "ok"


def _recompute_from_spec(spec: Spec) -> int:
    """Independent recompute of the spec answer (defense-in-depth vs sample bug)."""
    if spec.kind == "binary":
        a, b = spec.operands
        return _compute(a, spec.ops[0], b)
    if spec.kind == "add_sub":
        a, b, c = spec.operands
        return (a + b) - c
    raise ValueError(spec.kind)


# ----------------------- STEP 2: DEEPSEEK WORDING PROMPT -----------------------

def build_wording_prompt(specs: list[Spec], style: Style) -> str:
    """Ask DeepSeek to DRESS each spec in words. It must NOT compute or change numbers."""
    style_instr = {
        "direct": "Write each as a terse 'Compute ...' instruction using the exact numbers and operators.",
        "word": "Write each as a one-sentence everyday word problem (people, objects) using EXACTLY the given numbers and the given operation, in order. Do not add other numbers.",
        "trace": "Write each as a word problem followed by a short step plan, using EXACTLY the given original numbers. Do not compute. Do not state intermediate or final numeric results. Do not number steps with digits.",
        "noisy": "Write each as a word problem with ONE irrelevant distractor detail (a color, a name, a day) that contains NO number, using EXACTLY the given numbers and operation.",
    }[style]
    lines = []
    for i, s in enumerate(specs):
        opdesc = " then ".join(
            f"{s.operands[j]} {s.ops[j]} (next)" if j > 0 else f"{s.operands[0]} {s.ops[0]} {s.operands[1]}"
            for j in range(len(s.ops))
        )
        lines.append(f'{i}: numbers={list(s.operands)} ops={list(s.ops)} kind={s.kind}')
    spec_block = "\n".join(lines)
    return (
        "You are a wording engine. For each spec, produce ONLY the problem text. "
        "Do not compute. Do not state intermediate or final numeric results. "
        "Use EXACTLY the given numbers and operations, in order. Add NO other numbers.\n"
        f"STYLE: {style_instr}\n"
        "Return a JSON list of objects: [{\"i\": <index>, \"text\": <problem text>}].\n\n"
        f"SPECS:\n{spec_block}\n"
    )


def to_row(spec: Spec, story: str, style: Style, row_id: str) -> dict[str, Any]:
    """Final training row. answer + steps are PYTHON's; prompt is DeepSeek's wording."""
    return {
        "id": row_id,
        "style": style,
        "kind": spec.kind,
        "prompt": story.strip(),
        "answer": str(spec.answer),       # PYTHON TRUTH
        "steps": list(spec.steps),        # PYTHON TRUTH
        "spec_signature": spec.signature(),
    }


# ----------------------- DRIVER -----------------------

def offline_self_test(n: int = 200) -> dict[str, Any]:
    """No API: prove the truth core + verifier. Generates specs, fakes correct +
    poisoned stories, confirms verifier keeps good and drops bad."""
    rng = random.Random(66)
    kept = dropped = 0
    drop_reasons: dict[str, int] = {}
    for _ in range(n):
        hard = rng.random() < 0.3
        spec = sample_spec(rng, hard=hard)
        # a faithful "direct" story
        good = f"Compute {' '.join(_render_spec_tokens(spec))}."
        ok, _ = verify_example(good, spec, "direct")
        if ok:
            kept += 1
        else:
            dropped += 1
        # a poisoned story (swap one operand digit) MUST be dropped
        bad = good.replace(str(spec.operands[0]), str(spec.operands[0] + 10), 1)
        okb, rb = verify_example(bad, spec, "direct")
        if not okb:
            drop_reasons[rb.split(":")[0]] = drop_reasons.get(rb.split(":")[0], 0) + 1
    return {"n": n, "good_kept": kept, "good_dropped": dropped, "poison_drop_reasons": drop_reasons}


def _render_spec_tokens(spec: Spec) -> list[str]:
    if spec.kind == "binary":
        a, b = spec.operands
        return [str(a), spec.ops[0], str(b)]
    a, b, c = spec.operands
    return ["(", str(a), "+", str(b), ")", "-", str(c)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", help="offline: prove truth core + verifier, no API")
    ap.add_argument("--n", type=int, default=200, help="target kept rows")
    ap.add_argument("--style", choices=["direct", "word", "trace", "noisy"], default="word")
    ap.add_argument("--hard-frac", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=20, help="specs per DeepSeek call")
    ap.add_argument("--seed", type=int, default=66)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--api-key-env", type=str, default="DEEPSEEK_API_KEY")
    ap.add_argument("--exclude-sigs", type=str, action="append", default=[],
                    help="path to a .sigs file (one spec signature/line) to SKIP — held-out leak guard")
    ap.add_argument("--emit-sigs", type=str, default=None,
                    help="write the spec signatures of generated rows here (for use as a future exclude set)")
    ap.add_argument("--max-attempt-multiplier", type=float, default=6.0,
                    help="fail if target kept rows are not reached after n * this many generated specs")
    ap.add_argument("--max-spec-tries", type=int, default=1000,
                    help="tries for sampling a fresh non-excluded spec before failing")
    ap.add_argument("--base-url", type=str, default="https://api.deepseek.com")
    ap.add_argument("--model", type=str, default="deepseek-v4-flash")
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--sleep-seconds", type=float, default=0.15)
    args = ap.parse_args()

    if args.self_test:
        res = offline_self_test(args.n)
        print(json.dumps(res, indent=2))
        # sanity: every poisoned story must be dropped via a number-mismatch reason
        assert res["good_kept"] == res["n"], "verifier wrongly dropped faithful stories"
        assert sum(res["poison_drop_reasons"].values()) == res["n"], "verifier failed to drop poison"
        print("SELF-TEST PASS: truth core + verifier sound (keeps faithful, drops poison)")
        return

    deepseek = _load_module("exp66_deepseek_helper", DEEPSEEK_HELPER_PATH)
    # Load .env (DEEPSEEK_API_KEY) the same way the other generators do.
    env_file = REPO_ROOT / ".env"
    if hasattr(deepseek, "load_env_file") and env_file.exists():
        deepseek.load_env_file(env_file)
    api_key = os.environ[args.api_key_env]
    rng = random.Random(args.seed)

    # Leak/duplicate guard: load signatures to exclude, then reserve new sampled
    # signatures so one train file cannot repeat the same math spec.
    exclude_sigs: set[str] = set()
    for exclude_path in args.exclude_sigs:
        path = Path(exclude_path)
        if path.exists():
            before = len(exclude_sigs)
            exclude_sigs.update(l.strip() for l in path.open(encoding="utf-8") if l.strip())
            print(f"loaded {len(exclude_sigs) - before} excluded signatures from {exclude_path}", flush=True)
    reserved_sigs = set(exclude_sigs)

    def fresh_spec() -> Spec:
        for _ in range(args.max_spec_tries):
            s = sample_spec(rng, hard=(rng.random() < args.hard_frac))
            sig = s.signature()
            if sig not in reserved_sigs:
                reserved_sigs.add(sig)
                return s
        raise RuntimeError(
            f"could not sample a fresh spec after {args.max_spec_tries} tries; "
            f"reserved={len(reserved_sigs)}"
        )

    kept_rows: list[dict[str, Any]] = []
    dropped = 0
    drop_reasons: dict[str, int] = {}
    attempted = 0
    batch_idx = 0
    max_attempts = max(args.n, int(args.n * args.max_attempt_multiplier))
    while len(kept_rows) < args.n and attempted < max_attempts:
        want = min(args.batch, args.n - len(kept_rows), max_attempts - attempted)
        specs = [fresh_spec() for _ in range(want)]
        prompt = build_wording_prompt(specs, args.style)
        content, _usage = deepseek.call_deepseek(
            api_key=api_key, base_url=args.base_url, model=args.model,
            prompt=prompt, max_tokens=args.max_tokens, temperature=args.temperature,
            timeout=args.timeout, retries=args.retries,
        )
        attempted += want
        try:
            parsed = deepseek.parse_deepseek_json_batch(content)
        except Exception as exc:  # noqa
            print(f"batch {batch_idx}: parse failed ({exc}); skipping", flush=True)
            dropped += want
            drop_reasons["parse_failed"] = drop_reasons.get("parse_failed", 0) + want
            batch_idx += 1
            time.sleep(args.sleep_seconds)
            continue
        by_i = {int(o["i"]): str(o.get("text", "")) for o in parsed if "i" in o}
        for i, spec in enumerate(specs):
            story = by_i.get(i, "")
            if not story:
                dropped += 1; drop_reasons["no_story"] = drop_reasons.get("no_story", 0) + 1
                continue
            ok, reason = verify_example(story, spec, args.style)
            if ok:
                kept_rows.append(to_row(spec, story, args.style, f"wp_{args.seed}_{len(kept_rows)}"))
            else:
                dropped += 1; drop_reasons[reason.split(":")[0]] = drop_reasons.get(reason.split(":")[0], 0) + 1
        batch_idx += 1
        print(
            f"batch {batch_idx}: kept={len(kept_rows)} dropped={dropped} "
            f"({attempted}/{max_attempts} attempts, target={args.n})",
            flush=True,
        )
        time.sleep(args.sleep_seconds)

    discard_rate = dropped / max(1, dropped + len(kept_rows))
    print(f"\n=== DONE === kept={len(kept_rows)} dropped={dropped} discard_rate={discard_rate:.1%}")
    print(f"drop reasons: {drop_reasons}")
    if len(kept_rows) < args.n:
        raise RuntimeError(
            f"short generation: kept={len(kept_rows)} target={args.n} "
            f"attempted={attempted} max_attempts={max_attempts}"
        )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in kept_rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {len(kept_rows)} rows -> {args.out}")
    if args.emit_sigs:
        with open(args.emit_sigs, "w", encoding="utf-8") as f:
            for r in kept_rows:
                f.write(r["spec_signature"] + "\n")
        print(f"wrote {len(kept_rows)} signatures -> {args.emit_sigs}")


if __name__ == "__main__":
    main()
