from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = (
    REPO_ROOT
    / "experiments"
    / "Experiment 44 - Arithmetic Latent Structure Probe"
    / "latent_arithmetic_probe.py"
)

spec = importlib.util.spec_from_file_location("latent_arithmetic_probe", PROBE_PATH)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_latent_targets_are_granular_not_final_answer_slots():
    assert probe.LATENT_TARGETS["add"] == ("ones_sum", "tens_sum")
    assert probe.LATENT_TARGETS["sub"] == ("ones_diff", "tens_diff")
    assert probe.LATENT_TARGETS["mul"] == ("ones_partial", "tens_partial")
    assert probe.LATENT_TARGETS["add_sub"] == ("add_ones_sum", "add_tens_sum")


def test_load_latent_rows_skips_unparsable_but_keeps_valid_rows(tmp_path):
    p = tmp_path / "mix.jsonl"
    p.write_text(
        '{"prompt": "Compute 24 + 14.", "answer": "38", "id": "a"}\n'
        '{"prompt": "Compute 24 / 6.", "answer": "4", "id": "b"}\n'
        '{"expression": "38 * 78", "answer": "2964", "id": "c"}\n',
        encoding="utf-8",
    )

    rows = probe.load_latent_rows(p)

    assert [r["id"] for r in rows] == ["a", "c"]
    assert rows[0]["latents"]["task"] == "add"
    assert rows[1]["latents"]["task"] == "mul"


def test_load_latent_rows_raises_on_answer_mismatch(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"prompt": "Compute 24 + 14.", "answer": "39", "id": "bad"}\n', encoding="utf-8")

    try:
        probe.load_latent_rows(p)
    except ValueError as exc:
        assert "answer mismatch" in str(exc)
    else:
        raise AssertionError("expected answer mismatch to raise")


def test_encode_operand_digits_for_three_operand_prompt():
    row = probe.row_from_task({"id": "x", "prompt": "Compute (74 + 35) - 7.", "answer": "102"})

    encoded = probe.encode_operand_digits([row], torch.device("cpu"))

    assert encoded.shape == (1, 6)
    assert encoded[0].tolist() == [7, 4, 3, 5, 0, 7]


def test_tiny_digit_classifier_forward_shape():
    model = probe.TinyDigitClassifier(n_classes=19, width=16)
    x = torch.tensor([[2, 4, 1, 4, 0, 0], [3, 8, 7, 8, 0, 0]])

    logits = model(x)

    assert logits.shape == (2, 19)


def test_encode_numeric_features_shape_and_values():
    row = probe.row_from_task({"id": "x", "prompt": "Compute (74 + 35) - 7.", "answer": "102"})

    encoded = probe.encode_numeric_features([row], torch.device("cpu"))

    assert encoded.shape == (1, 9)
    expected = torch.tensor([[74 / 99, 7 / 9, 4 / 9, 35 / 99, 3 / 9, 5 / 9, 7 / 99, 0.0, 7 / 9]])
    assert torch.allclose(encoded, expected, atol=1e-7)


def test_numeric_feature_classifier_forward_shape():
    model = probe.NumericFeatureClassifier(n_classes=19, width=16)
    x = torch.zeros(2, 9)

    logits = model(x)

    assert logits.shape == (2, 19)


def test_encode_product_features_adds_pairwise_numeric_terms():
    row = probe.row_from_task({"id": "x", "prompt": "Compute 38 * 78.", "answer": "2964"})

    encoded = probe.encode_product_features([row], torch.device("cpu"))

    assert encoded.shape == (1, probe.N_PRODUCT_FEATURES)
    # Numeric slots are [a_value, a_tens, a_ones, b_value, b_tens, b_ones, ...].
    assert abs(encoded[0, 0].item() - (38 / 99)) < 1e-6
    assert abs(encoded[0, 9 + 4].item() - ((38 / 99) * (7 / 9))) < 1e-6
    assert abs(encoded[0, 9 + 5].item() - ((38 / 99) * (8 / 9))) < 1e-6


def test_product_feature_classifier_forward_shape():
    model = probe.ProductFeatureClassifier(n_classes=19, width=16)
    x = torch.zeros(2, probe.N_PRODUCT_FEATURES)

    logits = model(x)

    assert logits.shape == (2, 19)


def test_oracle_latent_composition_scores_perfect_accuracy():
    rows = [
        probe.row_from_task({"id": "a", "prompt": "Compute 57 + 38.", "answer": "95"}),
        probe.row_from_task({"id": "s", "prompt": "Compute 74 - 93.", "answer": "-19"}),
        probe.row_from_task({"id": "m", "prompt": "Compute 38 * 78.", "answer": "2964"}),
        probe.row_from_task({"id": "as", "prompt": "Compute (43 + 22) - 98.", "answer": "-33"}),
    ]
    predictions = {
        row["id"]: {target: row["latents"][target] for target in probe.LATENT_TARGETS[row["task"]]}
        for row in rows
    }

    answer_strings = probe.compose_predicted_answer_strings(rows, predictions)
    metrics = probe.verifier_metrics_from_answer_strings(rows, answer_strings)

    assert answer_strings == ["95", "-19", "2964", "-33"]
    assert metrics == {"n": 4, "acc": 1.0, "invalid": 0.0}


def test_eval_task_heads_reuses_trained_heads_for_multiple_splits():
    rows = [
        probe.row_from_task({"id": "a1", "prompt": "Compute 57 + 38.", "answer": "95"}),
        probe.row_from_task({"id": "a2", "prompt": "Compute 31 + 20.", "answer": "51"}),
    ]
    predictions = {
        row["id"]: {target: row["latents"][target] for target in probe.LATENT_TARGETS[row["task"]]}
        for row in rows
    }

    class ConstantHead:
        def __init__(self, values):
            self.values = values

    heads = probe.TaskHeads(
        direct_answer=ConstantHead([95, 51]),
        latent_heads={"ones_sum": ConstantHead([15, 1]), "tens_sum": ConstantHead([80, 50])},
    )

    def fake_predict_values(head, eval_rows):
        return head.values[: len(eval_rows)]

    original = probe.predict_values
    probe.predict_values = fake_predict_values
    try:
        metrics_1 = probe.eval_task_heads(heads, rows, "add")
        metrics_2 = probe.eval_task_heads(heads, rows[:1], "add")
    finally:
        probe.predict_values = original

    assert predictions["a1"] == {"ones_sum": 15, "tens_sum": 80}
    assert metrics_1["direct_answer"]["acc"] == 1.0
    assert metrics_1["latent_composed"]["acc"] == 1.0
    assert metrics_2["direct_answer"]["n"] == 1
    assert metrics_2["latent_composed"]["target_acc"] == {"ones_sum": 1.0, "tens_sum": 1.0}


def test_arg_parser_accepts_multiple_seeds():
    args = probe.build_arg_parser().parse_args(["--seeds", "43", "44", "45"])

    assert args.seeds == [43, 44, 45]


def test_arg_parser_accepts_numeric_input_mode():
    args = probe.build_arg_parser().parse_args(["--input-mode", "numeric"])

    assert args.input_mode == "numeric"


def test_arg_parser_accepts_product_input_mode():
    args = probe.build_arg_parser().parse_args(["--input-mode", "product"])

    assert args.input_mode == "product"


def test_arg_parser_accepts_regression_prediction_mode():
    args = probe.build_arg_parser().parse_args(["--prediction-mode", "regress"])

    assert args.prediction_mode == "regress"


def test_arg_parser_accepts_sparse_rule_prediction_mode():
    args = probe.build_arg_parser().parse_args(["--prediction-mode", "sparse-rule"])

    assert args.prediction_mode == "sparse-rule"


def test_train_head_accepts_regression_prediction_mode():
    rows = [probe.row_from_task({"id": "m", "prompt": "Compute 38 * 78.", "answer": "2964"})]

    head = probe.train_head(
        rows,
        "ones_partial",
        steps=0,
        width=8,
        input_mode="product",
        prediction_mode="regress",
        device=torch.device("cpu"),
    )
    predictions = probe.predict_values(head, rows)

    assert head.prediction_mode == "regress"
    assert head.target_scale == 304
    assert isinstance(predictions[0], int)


def test_sparse_rule_selects_multiplication_rule_by_name():
    rows = [
        probe.row_from_task({"id": "a", "prompt": "Compute 38 * 78.", "answer": "2964"}),
        probe.row_from_task({"id": "b", "prompt": "Compute 66 * 34.", "answer": "2244"}),
        probe.row_from_task({"id": "c", "prompt": "Compute 29 * 56.", "answer": "1624"}),
    ]

    head = probe.train_head(
        rows,
        "ones_partial",
        input_mode="product",
        prediction_mode="sparse-rule",
        device=torch.device("cpu"),
    )
    predictions = probe.predict_values(head, rows)

    assert head.prediction_mode == "sparse-rule"
    assert head.selected_rule.rule.name == "a_times_b_ones"
    assert head.selected_rule.rule.expression == "a * ones(b)"
    assert predictions == [304, 264, 174]


def test_target_key_uses_input_piece_not_target_value():
    add = probe.row_from_task({"id": "add", "prompt": "Compute 57 + 38.", "answer": "95"})
    mul = probe.row_from_task({"id": "mul", "prompt": "Compute 38 * 78.", "answer": "2964"})

    assert probe.target_key(add, "ones_sum") == (7, 8)
    assert probe.target_key(add, "tens_sum") == (5, 3)
    assert probe.target_key(mul, "ones_partial") == (38, 8)
    assert probe.target_key(mul, "tens_partial") == (38, 7)


def test_build_target_key_ood_split_removes_held_key_but_keeps_target_value_seen():
    rows = [
        probe.row_from_task({"id": "a", "prompt": "Compute 57 + 38.", "answer": "95"}),  # ones 7+8=15
        probe.row_from_task({"id": "b", "prompt": "Compute 58 + 37.", "answer": "95"}),  # ones 8+7=15
        probe.row_from_task({"id": "c", "prompt": "Compute 64 + 21.", "answer": "85"}),  # ones 4+1=5
    ]

    split = probe.build_target_key_ood_split(rows, "ones_sum", held_keys={(7, 8)})

    assert [row["id"] for row in split.eval_rows] == ["a"]
    assert {row["id"] for row in split.train_rows} == {"b", "c"}
    assert split.held_keys == [(7, 8)]
    assert split.target == "ones_sum"


def test_build_target_key_ood_split_rejects_unseen_eval_target_value():
    rows = [
        probe.row_from_task({"id": "a", "prompt": "Compute 57 + 38.", "answer": "95"}),  # ones 15
        probe.row_from_task({"id": "b", "prompt": "Compute 64 + 21.", "answer": "85"}),  # ones 5
    ]

    try:
        probe.build_target_key_ood_split(rows, "ones_sum", held_keys={(7, 8)})
    except ValueError as exc:
        assert "target values unseen in train" in str(exc)
    else:
        raise AssertionError("expected unseen eval target value to raise")


def test_arg_parser_accepts_ood_stress_mode():
    args = probe.build_arg_parser().parse_args(["--ood-stress", "--ood-holdout-fraction", "0.2"])

    assert args.ood_stress is True
    assert args.ood_holdout_fraction == 0.2
