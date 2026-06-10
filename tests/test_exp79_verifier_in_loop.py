"""Exp79 verifier-in-the-loop tests. Real behavior only, no mocks of the verifier."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training import verifier_loop as VL


def test_trace_record_roundtrips_jsonl_with_verifier():
    task = {"id": "wp_x_1", "answer": "63"}
    target, audit, vres = VL.build_tool_supervised_target(
        "If you have 81 apples and give away 18?", "Step 1: 81 - 18 = 70\nAnswer: 70", task
    )
    assert target == "Step 1: 81 - 18 = 63\nAnswer: 63"
    assert vres["passed"] is True
    rec = VL.TraceRecord(
        task_id="wp_x_1", domain="word", prompt="p", raw_generation="g",
        training_target=target, mode="tool_supervised", verifier=vres,
        tool_audit=audit, difficulty="word", seed=1, step=0,
    )
    line = rec.to_json()
    rt = VL.TraceRecord.from_json(line)
    assert rt.task_id == rec.task_id
    assert rt.training_target == target
    assert rt.verifier["passed"] is True
    assert rt.verifier["evidence"]["expected"] == "63"
    # valid JSON
    json.loads(line)


def test_held_out_id_in_buffer_is_refused():
    held = VL.load_held_out_ids()
    assert held, "manifest should have held-out ids"
    bad_id = sorted(held)[0]
    buf = VL.TraceBuffer()
    buf.append(VL.TraceRecord(bad_id, "arithmetic", "p", "g", "t", "tool_supervised", {"passed": True}))
    with pytest.raises(VL.HeldOutLeakError):
        buf.refuse_held_out_ids()


def test_clean_buffer_passes_refusal():
    buf = VL.TraceBuffer()
    buf.append(VL.TraceRecord("wp_safe_99", "word", "p", "g", "t", "tool_supervised", {"passed": True}))
    buf.refuse_held_out_ids()  # must not raise


def test_missing_manifest_fails_loud(monkeypatch):
    def _boom():
        raise FileNotFoundError("missing manifest")

    monkeypatch.setattr(VL, "load_held_out_ids", _boom)
    buf = VL.TraceBuffer()
    with pytest.raises(FileNotFoundError, match="missing manifest"):
        buf.held_out_ids()


def test_unsafe_flag_skips_held_out_refusal(monkeypatch):
    def _boom():
        raise FileNotFoundError("missing manifest")

    monkeypatch.setattr(VL, "load_held_out_ids", _boom)
    buf = VL.TraceBuffer(allow_missing_manifest=True)
    buf.append(VL.TraceRecord("arith_0038", "arithmetic", "p", "g", "t", "tool_supervised", {"passed": True}))
    buf.refuse_held_out_ids()  # explicit unsafe opt-out


def test_tool_supervised_corrects_poisoned_generation():
    # model structure right, arithmetic wrong -> solver corrects -> verifier passes
    task = {"id": "wp_y_1", "answer": "8"}
    poisoned = "Step 1: 22 + 89 = 100\nStep 2: 100 - 103 = -3\nAnswer: -3"
    target, audit, vres = VL.build_tool_supervised_target("(22+89)-103?", poisoned, task)
    assert audit["final"] == 8
    assert target is not None
    assert target.endswith("Answer: 8")
    assert vres["passed"] is True


def test_tool_supervised_skips_unparsable_generation():
    task = {"id": "wp_z_1", "answer": "5"}
    target, audit, vres = VL.build_tool_supervised_target("p", "the answer is probably five", task)
    assert target is None
    assert vres["passed"] is False


def test_verified_filter_rejects_all_wrong():
    task = {"id": "wp_w_1", "answer": "63"}
    gens = ["Step 1: 81 - 18 = 70\nAnswer: 70", "Answer: 99", "nonsense"]
    winner, vres, results = VL.select_verified_filter_target("p", gens, task)
    assert winner is None
    assert vres["passed"] is False
    assert len(results) == 3


def test_verified_filter_keeps_a_passing_generation():
    task = {"id": "wp_w_2", "answer": "63"}
    gens = ["Answer: 70", "Step 1: 81 - 18 = 63\nAnswer: 63", "Answer: 12"]
    winner, vres, _ = VL.select_verified_filter_target("p", gens, task)
    assert winner is not None
    assert vres["passed"] is True


def test_replay_mix_blends_both_pools():
    rng = random.Random(0)
    orig = [VL.sft_row("op", "Answer: 1", "1", "o1")]
    trace = [VL.sft_row("tp", "Answer: 2", "2", "t1")]
    batch = VL.replay_mix(orig, trace, batch_size=8, replay_frac=0.25, rng=rng)
    assert len(batch) == 8
    ids = {r["id"] for r in batch}
    assert "o1" in ids and "t1" in ids  # both represented


def test_replay_mix_trace_only_when_no_original():
    rng = random.Random(0)
    trace = [VL.sft_row("tp", "Answer: 2", "2", "t1")]
    batch = VL.replay_mix([], trace, batch_size=4, replay_frac=0.25, rng=rng)
    assert len(batch) == 4
    assert all(r["id"] == "t1" for r in batch)


def test_rlvr_mode_exits_without_training():
    import importlib.util
    runner_path = REPO_ROOT / "experiments" / "Experiment 79 - Verifier In Loop Training" / "verifier_in_loop_train.py"
    spec = importlib.util.spec_from_file_location("exp79_runner_rlvr", runner_path)
    runner = importlib.util.module_from_spec(spec)
    sys.modules["exp79_runner_rlvr"] = runner
    spec.loader.exec_module(runner)

    argv = [
        "prog", "--mode", "smoke", "--train-mode", "rlvr",
        "--steps", "1", "--limit", "2", "--eval-limit", "2", "--device", "cpu",
    ]
    old = sys.argv
    sys.argv = argv
    try:
        rc = runner.main()
    finally:
        sys.argv = old
    assert rc == 1


def test_smoke_two_steps_finite_loss_nonempty_buffer():
    """End-to-end smoke via the runner on CPU, 2 steps, 4 tasks."""
    import importlib.util
    runner_path = REPO_ROOT / "experiments" / "Experiment 79 - Verifier In Loop Training" / "verifier_in_loop_train.py"
    spec = importlib.util.spec_from_file_location("exp79_runner_test", runner_path)
    runner = importlib.util.module_from_spec(spec)
    sys.modules["exp79_runner_test"] = runner
    spec.loader.exec_module(runner)

    ckpt = REPO_ROOT / "artifacts" / "phase0_exp69_fullepoch" / "h256_word100k_b16_s18000_seed1" / "checkpoint_fp32.pt"
    if not ckpt.exists():
        pytest.skip(f"word checkpoint absent: {ckpt}")

    out_dir = REPO_ROOT / "artifacts" / "exp79_test_smoke"
    argv = [
        "prog", "--mode", "smoke", "--train-mode", "tool_supervised",
        "--steps", "2", "--limit", "4", "--eval-limit", "4",
        "--device", "cpu", "--output-dir", str(out_dir),
    ]
    old = sys.argv
    sys.argv = argv
    try:
        rc = runner.main()
    finally:
        sys.argv = old
    assert rc == 0
    report = json.loads((out_dir / "report.json").read_text())
    mode = report["modes"]["tool_supervised"]
    assert mode["last_loss"] is None or mode["last_loss"] == mode["last_loss"]  # finite (not nan) or None
    trace = out_dir / "trace_tool_supervised.jsonl"
    assert trace.exists()
    assert sum(1 for _ in trace.open()) >= 1  # buffer non-empty
