"""Exp119 Stepwise Reachability — minimal self-checks (no framework)."""
import importlib.util
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

spec = importlib.util.spec_from_file_location(
    "exp119", REPO_ROOT / "experiments" / "Experiment 119 - Stepwise Reachability" / "stepwise_reachability.py"
)
exp119 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp119)


def test_k_hop_closure_depth_truncates():
    """1-hop target = stated edges only; 2-hop adds the transitive pair."""
    rows = [{
        "domain": "logic_rules", "symbols": ["A", "B", "C"],
        "edges": [("A", "B"), ("B", "C")], "facts": ["A"], "query": "C",
        "gold_bool": True, "gold_order": None, "query_type": None, "_text": "x", "id": "t1",
    }]
    t1, _ = exp119.k_hop_closure(rows, torch.device("cpu"), 1)
    t2, _ = exp119.k_hop_closure(rows, torch.device("cpu"), 2)
    idx = {s: i for i, s in enumerate(rows[0]["symbols"])}
    A, B, C = idx["A"], idx["B"], idx["C"]
    assert t1[0, A, B] == 1.0 and t1[0, B, C] == 1.0
    assert t1[0, A, C] == 0.0, "1-hop must NOT include A->C"
    assert t2[0, A, C] == 1.0, "2-hop must include A->C"


def test_reasoning_depth_cycle_safe():
    """A cycle in the rule graph must not infinite-loop, and returns a finite int."""
    cyclic = {
        "domain": "logic_rules", "symbols": ["A", "B", "C"],
        "edges": [("A", "B"), ("B", "C"), ("C", "A")],  # cycle
        "facts": [], "query": "A", "gold_bool": False, "id": "cyc",
    }
    d = exp119.reasoning_depth(cyclic)  # must return, not raise
    assert isinstance(d, int) and d >= 0
    # comparative chain of 4 items => 3 edges
    comp = {"domain": "comparative_order", "gold_order": ["x", "y", "z", "w"]}
    assert exp119.reasoning_depth(comp) == 3


def test_model_forward_returns_one_logit_per_round():
    m = exp119.StepwiseReachability(vocab_size=200, width=32, heads=2, layers=1, max_len=16, use_checkpoint=False)
    b, S, seq = 2, exp119.MAX_SYMBOLS, 16
    ids = torch.randint(0, 200, (b, seq)); mask = torch.ones(b, seq, dtype=torch.bool)
    tag = torch.zeros(b, seq, dtype=torch.long); sym_mask = torch.ones(b, S, dtype=torch.bool)
    dom = torch.zeros(b, dtype=torch.long)
    rl, trace = m(ids, mask, tag, sym_mask, dom, max_rounds=3, bp_steps=3, need_trace=True)
    assert len(rl) == 3 and rl[0].shape == (b, S, S)
    assert len(trace) == 3


def test_bp_steps_only_tail_has_grad():
    """bp_steps=1 on 3 rounds: only the last round's logits require grad."""
    m = exp119.StepwiseReachability(vocab_size=200, width=32, heads=2, layers=1, max_len=16, use_checkpoint=False)
    b, S, seq = 2, exp119.MAX_SYMBOLS, 16
    ids = torch.randint(0, 200, (b, seq)); mask = torch.ones(b, seq, dtype=torch.bool)
    tag = torch.zeros(b, seq, dtype=torch.long); sym_mask = torch.ones(b, S, dtype=torch.bool)
    dom = torch.zeros(b, dtype=torch.long)
    rl, _ = m(ids, mask, tag, sym_mask, dom, max_rounds=3, bp_steps=1)
    assert not rl[0].requires_grad and not rl[1].requires_grad
    assert rl[2].requires_grad, "last bp_steps rounds must keep grad"


def test_curriculum_loss_runs_and_finite():
    m = exp119.StepwiseReachability(vocab_size=200, width=32, heads=2, layers=1, max_len=16, use_checkpoint=False)
    rows = [{
        "domain": "logic_rules", "symbols": ["A", "B"], "edges": [("A", "B")],
        "facts": ["A"], "query": "B", "gold_bool": True, "id": "r",
        "gold_order": None, "query_type": None, "_text": "A implies B",
    }]
    ids = torch.randint(0, 200, (1, 8)); mask = torch.ones(1, 8, dtype=torch.bool)
    tag = torch.zeros(1, 8, dtype=torch.long); sym = torch.ones(1, exp119.MAX_SYMBOLS, dtype=torch.bool)
    dom = torch.zeros(1, dtype=torch.long)
    rl, _ = m(ids, mask, tag, sym, dom, max_rounds=3, bp_steps=3)
    loss = exp119.curriculum_loss(rl, rows, torch.device("cpu"))
    assert torch.isfinite(loss) and loss.item() >= 0


def test_factorized_embedding_smaller_and_forward_ok():
    """Factorized embedding has fewer params than dense and forwards the same shape."""
    V, H = 2000, 64
    dense = exp119.StepwiseReachability(vocab_size=V, width=H, heads=2, layers=1, max_len=16,
                                        use_checkpoint=False, factorized_emb_dim=0)
    fact = exp119.StepwiseReachability(vocab_size=V, width=H, heads=2, layers=1, max_len=16,
                                       use_checkpoint=False, factorized_emb_dim=16)
    emb_dense = sum(p.numel() for p in [dense.tok_emb_table.weight] + ([dense.tok_emb_proj.weight] if dense.tok_emb_proj else []))
    emb_fact = sum(p.numel() for p in [fact.tok_emb_table.weight, fact.tok_emb_proj.weight])
    assert emb_fact < emb_dense, f"factorized ({emb_fact}) should be smaller than dense ({emb_dense})"
    # forward still works and produces hidden width
    ids = torch.randint(0, V, (2, 12)); mask = torch.ones(2, 12, dtype=torch.bool)
    tag = torch.zeros(2, 12, dtype=torch.long); sym = torch.ones(2, exp119.MAX_SYMBOLS, dtype=torch.bool)
    dom = torch.zeros(2, dtype=torch.long)
    rl, _ = fact(ids, mask, tag, sym, dom, max_rounds=2, bp_steps=2)
    assert rl[0].shape == (2, exp119.MAX_SYMBOLS, exp119.MAX_SYMBOLS)
