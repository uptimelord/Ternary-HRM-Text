import json
import importlib.util
import sys
from pathlib import Path

import pytest

from evaluation.guard_rail import check_no_held_out_leak


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_VISIBLE = REPO_ROOT / "evaluation" / "frozen" / "train_visible_arithmetic_160.jsonl"
HELD_OUT = REPO_ROOT / "evaluation" / "frozen" / "held_out_arithmetic_40.jsonl"


def _load_exp30_module():
    path = REPO_ROOT / "experiments" / "Experiment 30 - Arithmetic Reasoning SFT Pilot" / "arithmetic_sft_pilot.py"
    spec = importlib.util.spec_from_file_location("exp30_sft_for_guard_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_train_visible_jsonl_passes_guard():
    check_no_held_out_leak([TRAIN_VISIBLE])


def test_held_out_jsonl_path_fails_before_scanning_ids():
    with pytest.raises(RuntimeError, match="held-out file path"):
        check_no_held_out_leak([HELD_OUT])


def test_jsonl_containing_held_out_id_fails(tmp_path):
    leak_path = tmp_path / "candidate_traces.jsonl"
    leak_path.write_text(
        json.dumps({"id": "arith_0129", "text": "leaked trace"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="HELD-OUT LEAK DETECTED"):
        check_no_held_out_leak([leak_path])


def test_missing_jsonl_fails_by_default(tmp_path):
    missing = tmp_path / "missing.jsonl"

    with pytest.raises(FileNotFoundError, match="missing data path"):
        check_no_held_out_leak([missing])


def test_missing_optional_jsonl_can_be_skipped(tmp_path):
    missing = tmp_path / "optional.jsonl"

    check_no_held_out_leak([missing], optional_paths=[missing])


def test_malformed_jsonl_fails_with_file_and_line(tmp_path):
    bad_path = tmp_path / "bad.jsonl"
    bad_path.write_text('{"id": "ok"}\n{"id":\n', encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        check_no_held_out_leak([bad_path])

    message = str(excinfo.value)
    assert str(bad_path) in message
    assert "line 2" in message


def test_exp30_jsonl_loader_runs_guard_before_loading_training_rows():
    exp30 = _load_exp30_module()

    with pytest.raises(RuntimeError, match="held-out file path"):
        exp30.read_jsonl(HELD_OUT)
