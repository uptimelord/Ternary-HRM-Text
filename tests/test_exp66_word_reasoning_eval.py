from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = (
    REPO_ROOT
    / "experiments"
    / "Experiment 66 - Word Problem Reasoning Corpus"
    / "eval_exp66_word_reasoning.py"
)

spec = importlib.util.spec_from_file_location("exp66_word_reasoning_eval", EVAL_PATH)
eval_mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = eval_mod
spec.loader.exec_module(eval_mod)


class FakeVerifier:
    def verify(self, task, candidate):
        expected = str(task["answer"])
        extracted = candidate.rsplit(" ", 1)[-1]
        passed = extracted == expected
        return {
            "passed": passed,
            "score": 1.0,
            "error": None if passed else "answer_mismatch",
            "runtime_s": 0.0,
            "evidence": {"extracted": extracted},
        }


def test_evaluate_rows_can_collect_every_generation_record():
    candidates = iter(["Answer: 5", "Answer: 9"])

    def fake_generate(*_args, **_kwargs):
        return next(candidates)

    exp30 = SimpleNamespace(
        greedy_generate_until_answer=fake_generate,
        get_hrm_net=lambda model: model,
    )
    model = SimpleNamespace(H_cycles=2)
    rows = [
        {
            "id": "d1",
            "prompt": "Compute 2 + 3",
            "answer": 5,
            "style": "direct",
            "kind": "binary",
            "ops": ["+"],
            "spec_signature": "binary|2,3|+|5",
        },
        {
            "id": "w1",
            "prompt": "Sam has 4 apples and buys 4. How many?",
            "answer": 8,
            "style": "word",
            "kind": "binary",
            "ops": ["+"],
            "spec_signature": "binary|4,4|+|8",
        },
    ]
    generation_records = []

    result = eval_mod.evaluate_rows(
        exp30,
        object(),
        model,
        rows,
        split="heldout_word",
        generation_records=generation_records,
        tokenizer=object(),
        verifier=FakeVerifier(),
        device=torch.device("cpu"),
        vocab_size=100,
        h=4,
        max_prefix_tokens=16,
        max_new_tokens=8,
        bp_steps=1,
    )

    assert result["n"] == 2
    assert len(generation_records) == 2
    assert generation_records[0]["split"] == "heldout_word"
    assert generation_records[0]["h"] == 4
    assert generation_records[0]["id"] == "d1"
    assert generation_records[0]["style"] == "direct"
    assert generation_records[0]["kind"] == "binary"
    assert generation_records[0]["ops"] == ["+"]
    assert generation_records[0]["gold"] == "5"
    assert generation_records[0]["generated"] == "Answer: 5"
    assert generation_records[0]["extracted"] == "5"
    assert generation_records[0]["passed"] is True
    assert generation_records[1]["error"] == "answer_mismatch"


def test_analyze_generation_records_counts_weak_spots():
    records = [
        {
            "split": "heldout_direct",
            "style": "direct",
            "kind": "binary",
            "ops": ["+"],
            "error": None,
            "passed": True,
        },
        {
            "split": "heldout_word",
            "style": "word",
            "kind": "binary",
            "ops": ["*"],
            "error": "answer_mismatch",
            "passed": False,
        },
        {
            "split": "heldout_word",
            "style": "word",
            "kind": "binary",
            "ops": ["*"],
            "error": "answer_mismatch",
            "passed": False,
        },
        {
            "split": "heldout_hard",
            "style": "trace",
            "kind": "add_sub",
            "ops": ["+", "-"],
            "error": "no_numeric_answer",
            "passed": False,
        },
    ]

    analysis = eval_mod.analyze_generation_records(records)

    assert analysis["overall"] == {"n": 4, "passed": 1, "failed": 3, "acc": 0.25}
    assert analysis["by_split"]["heldout_word"] == {
        "n": 2,
        "passed": 0,
        "failed": 2,
        "acc": 0.0,
    }
    assert analysis["by_style"]["word"]["failed"] == 2
    assert analysis["by_kind"]["binary"]["n"] == 3
    assert analysis["by_op"]["*"]["failed"] == 2
    assert analysis["by_error"]["answer_mismatch"]["n"] == 2


def test_generation_record_reads_ops_from_spec_signature_when_missing():
    record = eval_mod.make_generation_record(
        split="heldout_direct",
        h=4,
        row={
            "id": "x1",
            "prompt": "Compute 67 * 8",
            "answer": 536,
            "style": "direct",
            "kind": "binary",
            "spec_signature": "binary|67,8|*",
        },
        candidate="Answer: 50966",
        result={
            "passed": False,
            "error": "answer_mismatch",
            "evidence": {"extracted": "50966"},
        },
    )

    assert record["ops"] == ["*"]
