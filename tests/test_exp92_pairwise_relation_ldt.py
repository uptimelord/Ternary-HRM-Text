import importlib.util
from pathlib import Path

import torch

from training.comparative_logic import convert_comparative_logic_row
from training.verified_grid_rows import compile_comparative_logic_vgr


def _load_exp92():
    path = Path("experiments/Experiment 92 - Pairwise Relation LDT/pairwise_relation_ldt.py")
    spec = importlib.util.spec_from_file_location("exp92_pairwise_relation_ldt_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _vgr(style="order"):
    answer = "Bob > Sue > Ann" if style == "order" else "Bob"
    row = convert_comparative_logic_row(
        {
            "id": f"row_{style}",
            "domain": "comparative_logic",
            "prompt": "Bob taller Sue. Sue taller Ann. Order them by height from greatest to least.",
            "answer": answer,
            "style": style,
            "dimension": "height",
            "order": ["Bob", "Sue", "Ann"],
        }
    )
    return compile_comparative_logic_vgr(row)


def test_transitive_closure_adds_indirect_relation():
    mod = _load_exp92()
    row = mod.row_from_vgr(_vgr("order"))

    closure = mod.transitive_closure(row)

    ann = row["candidates"].index("Ann")
    bob = row["candidates"].index("Bob")
    sue = row["candidates"].index("Sue")
    assert closure[bob, sue]
    assert closure[sue, ann]
    assert closure[bob, ann]
    assert not closure[ann, bob]


def test_encode_batch_builds_pairwise_targets():
    mod = _load_exp92()
    rows = [mod.row_from_vgr(_vgr("order"))]

    encoded = mod.encode_batch(rows, torch.device("cpu"))

    assert encoded["relation_target"].shape == (1, mod.MAX_ENTITIES, mod.MAX_ENTITIES)
    ann = rows[0]["candidates"].index("Ann")
    bob = rows[0]["candidates"].index("Bob")
    assert encoded["relation_target"][0, bob, ann]
    assert not encoded["relation_target"][0, ann, bob]
    assert not encoded["relation_target"][0].diagonal().any()


def test_pairwise_ldt_forward_matches_relation_shape():
    mod = _load_exp92()
    rows = [mod.row_from_vgr(_vgr("order")), mod.row_from_vgr(_vgr("tallest"))]
    encoded = mod.encode_batch(rows, torch.device("cpu"))
    state = mod.direct_relation_state(encoded)
    model = mod.PairwiseRelationLDT(width=16, layers=1, heads=2, internal_iters=2)

    outputs = model(encoded, state, threshold=0.5)

    assert len(outputs) == 2
    relation_logits, conflict_logits, step_state = outputs[-1]
    assert relation_logits.shape == (2, mod.MAX_ENTITIES, mod.MAX_ENTITIES)
    assert conflict_logits.shape == (2,)
    assert step_state.shape == (2, mod.MAX_ENTITIES, mod.MAX_ENTITIES)


def test_neural_order_from_relations_sorts_by_pair_scores():
    mod = _load_exp92()
    row = {"candidates": ["Ann", "Bob", "Sue"], "edges": []}
    logits = torch.zeros(1, mod.MAX_ENTITIES, mod.MAX_ENTITIES)
    ann, bob, sue = 0, 1, 2
    logits[0, bob, sue] = 5.0
    logits[0, bob, ann] = 5.0
    logits[0, sue, ann] = 5.0

    orders = mod.neural_orders_from_relations(logits, [row])

    assert orders[0] == ["Bob", "Sue", "Ann"]


def test_constrained_order_rejects_edge_violations():
    mod = _load_exp92()
    row = {
        "candidates": ["Ann", "Bob", "Sue"],
        "edges": [("Bob", "Sue"), ("Sue", "Ann")],
    }
    logits = torch.zeros(1, mod.MAX_ENTITIES, mod.MAX_ENTITIES)
    ann, bob, sue = 0, 1, 2
    logits[0, ann, bob] = 99.0
    logits[0, bob, sue] = 1.0
    logits[0, sue, ann] = 1.0

    orders = mod.constrained_orders_from_relations(logits, [row])

    assert orders[0] == ["Bob", "Sue", "Ann"]
