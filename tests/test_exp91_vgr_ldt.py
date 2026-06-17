import importlib.util
from pathlib import Path

import torch

from training.comparative_logic import convert_comparative_logic_row
from training.verified_grid_rows import compile_comparative_logic_vgr


def _load_exp91():
    path = Path("experiments/Experiment 91 - VGR LDT Comparative Logic/vgr_ldt_comparative.py")
    spec = importlib.util.spec_from_file_location("exp91_vgr_ldt_test", path)
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


def test_row_from_vgr_uses_unordered_candidates_and_gold_slots():
    mod = _load_exp91()

    row = mod.row_from_vgr(_vgr("order"))

    assert row["candidates"] == ["Ann", "Bob", "Sue"]
    assert row["target_indices"] == [1, 2, 0]
    assert row["edges"] == [("Bob", "Sue"), ("Sue", "Ann")]


def test_encode_batch_builds_lattice_targets():
    mod = _load_exp91()
    rows = [mod.row_from_vgr(_vgr("order"))]

    encoded = mod.encode_batch(rows, torch.device("cpu"))

    assert encoded["candidate_ids"].shape == (1, mod.MAX_ENTITIES)
    assert encoded["edge_matrix"].shape == (1, mod.MAX_ENTITIES, mod.MAX_ENTITIES)
    assert encoded["target_indices"][0, :3].tolist() == [1, 2, 0]
    assert encoded["slot_mask"][0, :3].all()
    assert not encoded["slot_mask"][0, 3:].any()


def test_top_lattice_masks_padding_slots_and_candidates():
    mod = _load_exp91()
    rows = [mod.row_from_vgr(_vgr("order"))]
    encoded = mod.encode_batch(rows, torch.device("cpu"))

    alive = mod.top_lattice(encoded)

    assert alive.shape == (1, mod.MAX_ENTITIES, mod.MAX_ENTITIES)
    assert alive[0, :3, :3].all()
    assert not alive[0, 3:, :].any()
    assert not alive[0, :, 3:].any()


def test_vgr_ldt_forward_matches_lattice_shape():
    mod = _load_exp91()
    rows = [mod.row_from_vgr(_vgr("order")), mod.row_from_vgr(_vgr("tallest"))]
    encoded = mod.encode_batch(rows, torch.device("cpu"))
    alive = mod.top_lattice(encoded)
    model = mod.ComparativeLogicLDT(width=16, layers=1, heads=2, internal_iters=2)

    outputs = model(encoded, alive, threshold=0.5)

    assert len(outputs) == 2
    keep_logits, conflict_logits, step_alive = outputs[-1]
    assert keep_logits.shape == (2, mod.MAX_ENTITIES, mod.MAX_ENTITIES)
    assert conflict_logits.shape == (2,)
    assert step_alive.shape == (2, mod.MAX_ENTITIES, mod.MAX_ENTITIES)


def test_vgr_ldt_forward_updates_lattice_monotonically():
    mod = _load_exp91()
    rows = [mod.row_from_vgr(_vgr("order"))]
    encoded = mod.encode_batch(rows, torch.device("cpu"))
    alive = mod.top_lattice(encoded)
    model = mod.ComparativeLogicLDT(width=16, layers=1, heads=2, internal_iters=2)

    outputs = model(encoded, alive, threshold=1.0)

    first_alive = outputs[0][2]
    second_alive = outputs[1][2]
    assert int(first_alive.sum()) <= int(alive.sum())
    assert int(second_alive.sum()) <= int(first_alive.sum())


def test_greedy_permutation_indices_never_repeats_candidate():
    mod = _load_exp91()
    alive = torch.zeros(1, mod.MAX_ENTITIES, mod.MAX_ENTITIES, dtype=torch.bool)
    alive[0, :3, :3] = True
    logits = torch.zeros(1, mod.MAX_ENTITIES, mod.MAX_ENTITIES)
    logits[0, :3, 1] = 10.0
    logits[0, 1, 2] = 9.0
    logits[0, 2, 0] = 8.0

    picked = mod.greedy_permutation_indices(logits, alive, [3])

    assert picked[0][:3] == [1, 2, 0]


def test_greedy_permutation_indices_falls_back_to_unused_candidate():
    mod = _load_exp91()
    alive = torch.zeros(1, mod.MAX_ENTITIES, mod.MAX_ENTITIES, dtype=torch.bool)
    alive[0, 0, 0] = True
    alive[0, 1, 0] = True
    alive[0, 2, 0] = True
    logits = torch.zeros(1, mod.MAX_ENTITIES, mod.MAX_ENTITIES)
    logits[0, :3, 0] = 10.0

    picked = mod.greedy_permutation_indices(logits, alive, [3])

    assert sorted(picked[0]) == [0, 1, 2]


def test_constrained_permutation_indices_rejects_edge_violations():
    mod = _load_exp91()
    row = {
        "candidates": ["Ann", "Bob", "Sue"],
        "edges": [("Bob", "Sue"), ("Sue", "Ann")],
    }
    alive = torch.ones(1, mod.MAX_ENTITIES, mod.MAX_ENTITIES, dtype=torch.bool)
    alive[0, 3:, :] = False
    alive[0, :, 3:] = False
    logits = torch.zeros(1, mod.MAX_ENTITIES, mod.MAX_ENTITIES)
    logits[0, 0, 0] = 99.0
    logits[0, 1, 1] = 98.0
    logits[0, 2, 2] = 97.0

    picked = mod.constrained_permutation_indices(logits, alive, [row])

    assert picked[0][:3] == [1, 2, 0]


def test_lattice_loss_skips_ce_for_pruned_gold_candidate():
    mod = _load_exp91()
    rows = [mod.row_from_vgr(_vgr("order"))]
    encoded = mod.encode_batch(rows, torch.device("cpu"))
    step_alive = mod.top_lattice(encoded)
    step_alive[0, 0, encoded["target_indices"][0, 0]] = False
    keep_logits = torch.zeros(1, mod.MAX_ENTITIES, mod.MAX_ENTITIES)
    keep_logits = keep_logits.masked_fill(~step_alive, -1.0e9)

    loss = mod.lattice_loss([(keep_logits, torch.zeros(1), step_alive)], encoded)

    assert torch.isfinite(loss)
    assert loss.item() < 100.0
