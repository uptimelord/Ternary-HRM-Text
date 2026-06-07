from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
GEN_PATH = (
    REPO_ROOT
    / "experiments"
    / "Experiment 66 - Word Problem Reasoning Corpus"
    / "generate_word_problems.py"
)

spec = importlib.util.spec_from_file_location("exp66_word_problem_generator", GEN_PATH)
gen = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gen
spec.loader.exec_module(gen)


_SPEC_LINE_RE = re.compile(r"(\d+): numbers=\[([^\]]+)\]")


class FakeDeepSeek:
    def __init__(self, *, poison_first: int = 0):
        self.poison_first = poison_first

    def load_env_file(self, _path: Path) -> None:
        return None

    def call_deepseek(self, *, prompt: str, **_kwargs):
        rows = []
        for match in _SPEC_LINE_RE.finditer(prompt):
            idx = int(match.group(1))
            nums = [int(part.strip()) for part in match.group(2).split(",")]
            text = "Compute " + " ".join(str(n) for n in nums) + "."
            if self.poison_first > 0:
                text += " Extra 999."
                self.poison_first -= 1
            rows.append({"i": idx, "text": text})
        return json.dumps(rows), {}

    def parse_deepseek_json_batch(self, content: str):
        return json.loads(content)


def _run_main(monkeypatch, tmp_path: Path, fake: FakeDeepSeek, *extra_args: str) -> list[dict]:
    out = tmp_path / "rows.jsonl"
    sigs = tmp_path / "rows.sigs"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(gen, "_load_module", lambda _name, _path: fake)
    monkeypatch.setattr(gen.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_word_problems.py",
            "--n",
            "5",
            "--batch",
            "5",
            "--style",
            "direct",
            "--out",
            str(out),
            "--emit-sigs",
            str(sigs),
            *extra_args,
        ],
    )

    gen.main()
    return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]


def test_main_keeps_generating_until_target_kept_rows_after_drops(monkeypatch, tmp_path):
    rows = _run_main(monkeypatch, tmp_path, FakeDeepSeek(poison_first=2))

    assert len(rows) == 5
    assert all("999" not in row["prompt"] for row in rows)


def test_main_does_not_write_duplicate_train_signatures(monkeypatch, tmp_path):
    specs = iter(
        [
            gen.Spec("binary", (1, 2), ("+",), 3, ("1 + 2 = 3",)),
            gen.Spec("binary", (1, 2), ("+",), 3, ("1 + 2 = 3",)),
            gen.Spec("binary", (2, 3), ("+",), 5, ("2 + 3 = 5",)),
            gen.Spec("binary", (3, 4), ("+",), 7, ("3 + 4 = 7",)),
        ]
    )
    monkeypatch.setattr(gen, "sample_spec", lambda _rng, *, hard=False: next(specs))

    rows = _run_main(monkeypatch, tmp_path, FakeDeepSeek(), "--n", "3", "--batch", "3")
    signatures = [row["spec_signature"] for row in rows]

    assert len(rows) == 3
    assert len(signatures) == len(set(signatures))


def test_trace_verifier_rejects_computed_numbers_in_prompt_text():
    spec = gen.Spec(
        "add_sub",
        (64, 83, 43),
        ("+", "-"),
        104,
        ("64 + 83 = 147", "147 - 43 = 104"),
    )
    story = (
        "Mia had 64 cards and got 83 more, then gave away 43. "
        "First combine the starting cards and new cards. Then subtract the cards given away."
    )
    bad_story = (
        "Mia had 64 cards and got 83 more, then gave away 43. "
        "First get 147. Final answer 104."
    )

    assert gen.verify_example(story, spec, "trace") == (True, "ok")
    assert gen.verify_example(bad_story, spec, "trace") == (False, "unexpected_number:147")


def test_trace_prompt_tells_deepseek_not_to_compute_or_number_steps():
    prompt = gen.build_wording_prompt(
        [gen.Spec("binary", (12, 5), ("+",), 17, ("12 + 5 = 17",))],
        "trace",
    )

    assert "Do not compute" in prompt
    assert "Do not state intermediate or final numeric results" in prompt
    assert "Do not number steps with digits" in prompt
    assert "You may state intermediate results" not in prompt
