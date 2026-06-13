from models.abacus_embedding import digit_positions_for_token_ids


def test_digit_positions_within_number():
    digit_ids = {10, 11, 12}  # pretend 0,1,2 tokens
    tokens = [99, 10, 11, 12, 99, 10]
    pos = digit_positions_for_token_ids(tokens, digit_token_ids=digit_ids, max_positions=8)
    assert pos[1] == 1 and pos[2] == 2 and pos[3] == 3
    assert pos[5] == 1
    assert pos[0] == 0 and pos[4] == 0


def test_abacus_on_the_fly_fallback_matches_explicit():
    import torch

    from models.abacus_embedding import digit_positions_for_token_ids

    digit_ids = {5, 6}
    tokens = [1, 5, 6, 2, 5]
    pos = digit_positions_for_token_ids(tokens, digit_token_ids=digit_ids, max_positions=8)
    assert pos == [0, 1, 2, 0, 1]


def test_digit_positions_lsd_order_counts_from_right():
    digit_ids = {10, 11, 12}
    tokens = [99, 10, 11, 12, 99]
    pos = digit_positions_for_token_ids(tokens, digit_token_ids=digit_ids, max_positions=8, digit_order="lsd")
    assert pos == [0, 3, 2, 1, 0]


def test_add_sub_frozen_metrics_filters_by_prompt():
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "experiments" / "Experiment 80 - Abacus Digit Embedding Probe" / "abacus_digit_probe.py"
    spec = importlib.util.spec_from_file_location("exp80_for_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    report = {
        "examples": [
            {"prompt": "Compute 2 + 3.", "passed": True},
            {"prompt": "Compute 9 - 4.", "passed": False},
            {"prompt": "Compute 2 * 3.", "passed": True},
        ]
    }
    metrics = mod.add_sub_frozen_metrics(report)
    assert metrics["n_add_sub"] == 2
    assert metrics["add_sub_pass@1"] == 0.5


def test_add_sub_frozen_metrics_uses_full_per_row_and_regex_operator():
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "experiments" / "Experiment 80 - Abacus Digit Embedding Probe" / "abacus_digit_probe.py"
    spec = importlib.util.spec_from_file_location("exp80_per_row_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    report = {
        "n": 4,
        "examples": [{"prompt": f"Compute {i} + 1.", "passed": False} for i in range(20)],
        "per_row": [
            {"id": "a", "prompt": "Compute -2 + 3.", "passed": True},
            {"id": "b", "prompt": "Compute 9 - -4.", "passed": False},
            {"id": "c", "prompt": "Compute 2 * -3.", "passed": True},
            {"id": "d", "prompt": "No arithmetic here.", "passed": True},
        ],
    }
    metrics = mod.add_sub_frozen_metrics(report)
    assert metrics["n_add_sub"] == 2
    assert metrics["add_sub_pass@1"] == 0.5
    assert mod.arithmetic_operator_from_prompt("Compute -2 + 3.") == "+"
    assert mod.arithmetic_operator_from_prompt("Compute 2 * -3.") == "*"


def test_exp80_defaults_are_two_seed_full_contract():
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "experiments" / "Experiment 80 - Abacus Digit Embedding Probe" / "abacus_digit_probe.py"
    spec = importlib.util.spec_from_file_location("exp80_contract_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    parser = mod.build_arg_parser()
    args = parser.parse_args([])
    assert args.seeds == "1,2"
    assert args.word_heldout.name == "heldout_word_1k.jsonl"
    assert args.steps == 2000
    assert args.digit_order == "msd"


def test_exp80_record_arm_result_upserts_by_seed_and_arm():
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "experiments" / "Experiment 80 - Abacus Digit Embedding Probe" / "abacus_digit_probe.py"
    spec = importlib.util.spec_from_file_location("exp80_resume_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    report = {"runs": []}
    mod.record_arm_result(report, seed=1, arm="control", metrics={"acc": 0.1})
    mod.record_arm_result(report, seed=1, arm="abacus", metrics={"acc": 0.2})
    mod.record_arm_result(report, seed=1, arm="control", metrics={"acc": 0.3})
    assert report["runs"] == [{"seed": 1, "control": {"acc": 0.3}, "abacus": {"acc": 0.2}}]
    assert mod.arm_completed(report, seed=1, arm="control") is True
    assert mod.arm_completed(report, seed=2, arm="control") is False


def test_exp80_resume_config_rejects_changed_run_settings():
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "experiments" / "Experiment 80 - Abacus Digit Embedding Probe" / "abacus_digit_probe.py"
    spec = importlib.util.spec_from_file_location("exp80_resume_config_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    old = {"run_config": {"steps": 30, "frozen_limit": 10}}
    assert mod.resume_config_matches(old, {"steps": 30, "frozen_limit": 10})
    assert not mod.resume_config_matches(old, {"steps": 60, "frozen_limit": 10})


def test_exp80_build_results_lines_reads_report_runs():
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "experiments" / "Experiment 80 - Abacus Digit Embedding Probe" / "abacus_digit_probe.py"
    spec = importlib.util.spec_from_file_location("exp80_lines_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    report = {
        "tokenizer_audit": {"abacus_ready": True},
        "seeds": [1],
        "runs": [
            {
                "seed": 1,
                "control": {"frozen": {"acc": 0.1}, "frozen_add_sub": {"add_sub_pass@1": 0.2}, "word_heldout": {"acc": 0.3}},
                "abacus": {"frozen": {"acc": 0.4}, "frozen_add_sub": {"add_sub_pass@1": 0.5}, "word_heldout": {"acc": 0.6}},
            }
        ],
    }
    lines = mod.build_results_lines(report)
    assert any("seed 1" in line for line in lines)
    assert any("abacus frozen pass@1: `0.400`" in line for line in lines)


def test_exp80_build_results_lines_marks_floor_inconclusive_and_invalid_rate():
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "experiments" / "Experiment 80 - Abacus Digit Embedding Probe" / "abacus_digit_probe.py"
    spec = importlib.util.spec_from_file_location("exp80_floor_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    report = {
        "tokenizer_audit": {"abacus_ready": True},
        "seeds": [1],
        "runs": [
            {
                "seed": 1,
                "control": {"frozen": {"acc": 0.0, "invalid": 0.1}, "frozen_add_sub": {"add_sub_pass@1": 0.0}},
                "abacus": {"frozen": {"acc": 0.0, "invalid": 0.2}, "frozen_add_sub": {"add_sub_pass@1": 0.0}},
            }
        ],
    }
    lines = mod.build_results_lines(report)
    assert any("verdict: `inconclusive_floor`" in line for line in lines)
    assert any("control invalid rate: `0.100`" in line for line in lines)
    assert any("abacus invalid rate: `0.200`" in line for line in lines)
