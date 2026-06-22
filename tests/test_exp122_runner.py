import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.stacked_reasoning import StackedReasoningModel


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "experiments" / "Experiment 122 - Stacked Reasoning Architecture" / "runner.py"
spec = importlib.util.spec_from_file_location("exp122_runner", RUNNER_PATH)
runner = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(runner)


def test_repaired_architecture_has_explicit_artifact_version():
    assert runner.ARCHITECTURE_VERSION == "exp122_v3_additive_positions_full_ternary_head"


def test_response_only_causal_labels():
    batch = runner.make_causal_batch(
        [[1, 2, 3, 4], [5, 6, 7]],
        prompt_lengths=[2, 1],
        max_length=5,
        pad_id=0,
        device=torch.device("cpu"),
    )
    assert batch["input_ids"].tolist() == [[1, 2, 3, 4, 0], [5, 6, 7, 0, 0]]
    assert batch["labels"].tolist() == [[-100, 3, 4, -100, -100], [6, 7, -100, -100, -100]]


def test_real_model_training_loss_is_finite():
    model = StackedReasoningModel(
        vocab_size=32,
        d_model=16,
        factor_dim=4,
        num_layers=1,
        d_state=4,
        kan_basis=4,
        max_positions=12,
    )
    batch = runner.make_causal_batch(
        [[1, 2, 3, 4], [5, 6, 7, 8]],
        prompt_lengths=[2, 2],
        max_length=8,
        pad_id=0,
        device=torch.device("cpu"),
    )
    loss = runner.causal_loss(model, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.embedding.token_factors.weight.grad is not None


def test_causal_accuracy_ignores_masked_tokens_and_reports_exact_rows():
    predictions = torch.tensor([[0, 2, 3, 4], [1, 2, 0, 0]])
    logits = torch.nn.functional.one_hot(predictions, num_classes=5).float()
    labels = torch.tensor([[-100, 2, 3, -100], [-100, 2, 4, -100]])
    token_acc, exact_acc = runner.causal_accuracy(logits, labels)
    assert token_acc == pytest.approx(0.75)
    assert exact_acc == pytest.approx(0.5)


def test_training_loader_calls_guard_before_read(monkeypatch, tmp_path):
    events = []
    path = tmp_path / "train.jsonl"
    path.write_text('{"id":"x"}\n', encoding="utf-8")
    monkeypatch.setattr(runner, "check_no_held_out_leak", lambda paths, verbose=False: events.append("guard"))
    rows = runner.load_jsonl(path, limit=1, training=True, event_hook=events.append)
    assert rows == [{"id": "x"}]
    assert events == ["guard", "read"]


def test_requested_cuda_does_not_fall_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA requested but unavailable"):
        runner.resolve_device("cuda")


def test_training_sequences_are_selected_from_sdm_admissions():
    sequences = [
        ([1, 2], 1, {"id": "a"}),
        ([3, 4], 1, {"id": "b"}),
        ([5, 6], 1, {"id": "c"}),
    ]
    replay = runner.select_sdm_replay_sequences(sequences, ["c", "a"])
    assert [item[2]["id"] for item in replay] == ["c", "a"]


def test_sdm_seeding_uses_hidden_only_encode(monkeypatch):
    class PassingVerifier:
        def verify(self, _row, _text):
            return {"passed": True}

    class HiddenOnlyModel:
        def __call__(self, _input_ids):
            raise AssertionError("SDM seeding must not materialize vocab logits")

        def encode(self, input_ids):
            hidden = torch.ones(input_ids.shape[0], input_ids.shape[1], 4)
            return None, hidden

        @staticmethod
        def sdm_address(trace, address_bits):
            return trace[:, :address_bits] >= 0

    monkeypatch.setattr(runner, "ArithmeticExactVerifier", PassingVerifier)
    monkeypatch.setattr(runner, "load_held_out_ids", lambda: set())
    args = SimpleNamespace(
        sdm_address_bits=2,
        d_model=4,
        sdm_locations=8,
        sdm_k_active=2,
        seed=1,
        sdm_seed_rows=1,
    )
    metrics, memory = runner.seed_sdm(
        HiddenOnlyModel(),
        tokenizer=None,
        sequences=[([1, 2, 3], 1, {"id": "a", "response": "ok"})],
        args=args,
        device=torch.device("cpu"),
        pad_id=0,
    )
    assert metrics["writes"] == memory.write_count == 1


def test_ternary_layers_import_without_flash_attention_runtime():
    completed = subprocess.run(
        [sys.executable, "-c", "from models.layers import TernaryLinear158Init; print(TernaryLinear158Init.__name__)"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_exp122_runner_is_greedy_only():
    destinations = {action.dest for action in runner.build_parser()._actions}
    assert hasattr(runner, "evaluate_greedy")
    assert not hasattr(runner, "evaluate_search")
    assert not any("mcts" in destination for destination in destinations)
    assert not (REPO_ROOT / "evaluation" / "mcts_psl_decoder.py").exists()
