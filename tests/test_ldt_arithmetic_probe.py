from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from evaluation.arithmetic_lattice import N_DIGITS, N_DIGIT_VALUES, N_SIGN, parse_prompt

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = REPO_ROOT / "experiments" / "Experiment 43 - LDT Arithmetic Probe" / "ldt_arithmetic_probe.py"

spec = importlib.util.spec_from_file_location("ldt_arithmetic_probe", PROBE_PATH)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_model_forward_shapes():
    model = probe.LDTArithmetic(d_model=16, layers=1, heads=2, recurrent_steps=2)
    parsed = [parse_prompt("Compute 24 + 14."), parse_prompt("Compute 99 * 99.")]
    device = torch.device("cpu")
    op_ids, operand_tokens = probe.encode_inputs(parsed, device)
    outputs = model(op_ids, operand_tokens)
    assert len(outputs) == 2  # recurrent_steps
    sign_logits, digit_logits = outputs[-1]
    assert sign_logits.shape == (2, N_SIGN)
    assert digit_logits.shape == (2, N_DIGITS, N_DIGIT_VALUES)


def test_encode_inputs_targets_consistency():
    parsed = [parse_prompt("Compute (74 + 35) - 7.")]
    device = torch.device("cpu")
    op_ids, operand_tokens = probe.encode_inputs(parsed, device)
    sign_t, digit_t = probe.encode_targets(parsed, device)
    assert op_ids.shape == (1,)
    assert sign_t.shape == (1,)
    assert digit_t.shape == (1, N_DIGITS)
    # answer 102 -> digits 0,1,0,2
    assert digit_t[0].tolist() == [0, 1, 0, 2]


def test_oracle_decode_gives_perfect_accuracy():
    """A model whose digit/sign argmax equals the true encoding must score acc=1.0.

    We bypass the net: build rows, then verify the verifier-backed eval returns
    acc=1.0 when predictions are the oracle answers.
    """
    rows = [
        {"prompt": "Compute 24 + 14.", "parsed": parse_prompt("Compute 24 + 14."), "answer": "38", "id": "t1"},
        {"prompt": "Compute 99 * 99.", "parsed": parse_prompt("Compute 99 * 99."), "answer": "9801", "id": "t2"},
    ]

    class OracleModel:
        def eval(self): ...
        def __call__(self, op_ids, operand_tokens):
            # return logits that argmax to the true answer for each row
            import torch.nn.functional as F  # noqa
            signs = []
            digits = []
            for p in self._parsed:
                from evaluation.arithmetic_lattice import encode_answer
                s, d = encode_answer(p.answer)
                signs.append(s)
                digits.append(list(d))
            b = len(self._parsed)
            sign_logits = torch.full((b, N_SIGN), -10.0)
            for i, s in enumerate(signs):
                sign_logits[i, s] = 10.0
            digit_logits = torch.full((b, N_DIGITS, N_DIGIT_VALUES), -10.0)
            for i, drow in enumerate(digits):
                for k, dv in enumerate(drow):
                    digit_logits[i, k, dv] = 10.0
            return [(sign_logits, digit_logits)]

    model = OracleModel()
    model._parsed = [r["parsed"] for r in rows]
    verifier = probe.ArithmeticExactVerifier()
    metrics = probe.verifier_eval(model, rows, torch.device("cpu"), verifier)
    assert metrics["n"] == 2
    assert metrics["acc"] == 1.0
    assert metrics["invalid"] == 0.0


def test_load_parsable_rows_skips_unparsable(tmp_path):
    p = tmp_path / "mix.jsonl"
    p.write_text(
        '{"prompt": "Compute 24 + 14.", "answer": "38", "id": "a"}\n'
        '{"prompt": "Compute 24 / 6.", "answer": "4", "id": "b"}\n'   # division, skipped
        '{"prompt": "Compute 200 * 200.", "answer": "40000", "id": "c"}\n'  # out of range, skipped
        '{"prompt": "Compute 31 - 25.", "answer": "6", "id": "d"}\n',
        encoding="utf-8",
    )
    rows = probe.load_parsable_rows(p)
    ids = {r["id"] for r in rows}
    assert ids == {"a", "d"}


def test_prompt_of_from_expression():
    row = {"expression": "57 + 38", "answer": "95", "id": "x"}
    assert probe._prompt_of(row) == "Compute 57 + 38."
