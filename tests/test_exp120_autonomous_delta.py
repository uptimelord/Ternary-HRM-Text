"""Exp120 autonomous delta reachability checks."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXP_DIR = REPO_ROOT / "experiments" / "Experiment 120 - Autonomous Delta Reachability"
spec = importlib.util.spec_from_file_location("exp120", EXP_DIR / "autonomous_delta.py")
exp120 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(exp120)


def _chain_row(length=3, row_id="chain"):
    symbols = [chr(ord("A") + i) for i in range(length + 1)]
    return {
        "domain": "comparative_order",
        "symbols": symbols,
        "edges": list(zip(symbols, symbols[1:])),
        "facts": [],
        "query": None,
        "gold_bool": None,
        "gold_order": symbols,
        "query_type": "full_order",
        "id": row_id,
        "_text": " > ".join(symbols),
    }


def _model():
    return exp120.AutonomousDeltaReachability(
        vocab_size=200,
        width=32,
        heads=2,
        layers=1,
        max_len=16,
        use_checkpoint=False,
        factorized_emb_dim=0,
    )


def _inputs():
    ids = torch.randint(0, 200, (1, 8))
    text_mask = torch.ones(1, 8, dtype=torch.bool)
    tag = torch.zeros(1, 8, dtype=torch.long)
    sym_mask = torch.zeros(1, exp120.MAX_SYMBOLS, dtype=torch.bool)
    sym_mask[:, :4] = True
    dom = torch.zeros(1, dtype=torch.long)
    return ids, text_mask, tag, sym_mask, dom


def test_frontiers_partition_closure():
    rows = [_chain_row()]
    closures, frontiers, _ = exp120.precompute_targets(rows, torch.device("cpu"), 3)
    assert torch.allclose(sum(frontiers), closures[-1])
    assert frontiers[0][0, 0, 1] == 1
    assert frontiers[1][0, 0, 2] == 1
    assert frontiers[2][0, 0, 3] == 1


def test_model_has_distinct_frontier_and_cumulative_heads():
    model = _model()
    cumulative, frontier, states = model.forward_dual(*_inputs(), max_rounds=2)
    assert model.frontier_mlp is not model.pair_mlp
    assert len(cumulative) == len(frontier) == len(states) == 2
    assert cumulative[0].shape == frontier[0].shape == (1, exp120.MAX_SYMBOLS, exp120.MAX_SYMBOLS)
    assert cumulative[0].data_ptr() != frontier[0].data_ptr()


def test_later_round_loss_backpropagates_through_prior_predicted_state():
    model = _model()
    rows = [_chain_row()]
    cumulative, frontier, states, loss = exp120.autonomous_rounds(
        model, *_inputs(), rows, max_rounds=3, state_weight=0.5, use_checkpoint=False
    )
    states[0].retain_grad()
    loss.backward()
    assert states[0].grad is not None
    assert torch.count_nonzero(states[0].grad).item() > 0
    assert torch.isfinite(loss)


def test_model_ignores_domain_routing_id():
    model = _model().eval()
    ids, text_mask, tag, sym_mask, dom = _inputs()
    with torch.no_grad():
        cumulative_a, frontier_a, _ = model.forward_dual(
            ids, text_mask, tag, sym_mask, torch.zeros_like(dom), max_rounds=2
        )
        cumulative_b, frontier_b, _ = model.forward_dual(
            ids, text_mask, tag, sym_mask, torch.ones_like(dom), max_rounds=2
        )
    for left, right in zip(cumulative_a + frontier_a, cumulative_b + frontier_b):
        assert torch.equal(left, right)


def test_training_loader_runs_guard_before_read(monkeypatch, tmp_path):
    events = []
    train_path = tmp_path / "train.jsonl"
    train_path.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        exp120,
        "check_no_held_out_leak",
        lambda data_paths, verbose=False: events.append(("guard", list(data_paths))),
    )
    monkeypatch.setattr(
        exp120,
        "load_rows",
        lambda path, limit=0: events.append(("load", path)) or [_chain_row()],
    )

    rows = exp120.load_training_rows(train_path, limit=10, train_k=3)
    assert rows
    assert events == [("guard", [train_path]), ("load", train_path)]


def test_eval_slices_use_exact_depth_k():
    rows = [_chain_row(3, "K"), _chain_row(4, "K+1"), _chain_row(5, "K+2")]
    slices = exp120.eval_depth_slices(rows, train_k=3)
    assert [(label, [row["id"] for row in subset]) for label, subset in slices] == [
        ("K", ["K"]),
        ("K+1", ["K+1"]),
        ("K+2", ["K+2"]),
    ]


def test_requested_cuda_never_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA requested but unavailable"):
        exp120.resolve_device("cuda")


def test_only_one_experiment_120_directory_exists():
    exp120_dirs = sorted(REPO_ROOT.glob("experiments/Experiment 120 -*"))
    assert exp120_dirs == [EXP_DIR]
