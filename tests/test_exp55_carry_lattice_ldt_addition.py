from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = (
    REPO_ROOT
    / "experiments"
    / "Experiment 55 - Carry Lattice LDT Addition"
    / "carry_lattice_ldt_addition.py"
)

spec = importlib.util.spec_from_file_location("carry_lattice_ldt_addition", PROBE_PATH)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_addition_cells_include_digits_and_carries():
    assert probe.CELL_NAMES == ("ones", "carry0", "tens", "carry1", "hundreds")
    assert probe.CELL_CANDIDATE_COUNTS == (10, 2, 10, 2, 2)


def test_row_from_addition_prompt_computes_carry_lattice_solution():
    row = probe.row_from_prompt("x", "Compute 57 + 68.", "125")
    assert row["cell_values"] == (5, 1, 2, 1, 1)
    assert row["answer"] == "125"


def test_top_lattice_masks_invalid_carry_candidates():
    alive = probe.top_lattice(batch_size=1, device=torch.device("cpu"))
    assert alive.shape == (1, probe.N_CELLS, probe.MAX_CANDIDATES)
    assert alive[0, probe.CARRY0, :2].tolist() == [True, True]
    assert alive[0, probe.CARRY0, 2:].sum().item() == 0
    assert alive[0, probe.HUNDREDS, :2].tolist() == [True, True]
    assert alive[0, probe.HUNDREDS, 2:].sum().item() == 0


def test_alpha_target_singletons_when_true_path_alive():
    row = probe.row_from_prompt("x", "Compute 57 + 68.", "125")
    encoded = probe.encode_batch([row], torch.device("cpu"))
    alive = probe.top_lattice(batch_size=1, device=torch.device("cpu"))
    target, conflict = probe.alpha_targets(encoded["cell_values"], alive)

    assert conflict.tolist() == [False]
    for cell, value in enumerate(row["cell_values"]):
        assert target[0, cell].sum().item() == 1.0
        assert target[0, cell, value].item() == 1.0


def test_alpha_target_conflict_when_carry_path_eliminated():
    row = probe.row_from_prompt("x", "Compute 57 + 68.", "125")
    encoded = probe.encode_batch([row], torch.device("cpu"))
    alive = probe.top_lattice(batch_size=1, device=torch.device("cpu"))
    alive[0, probe.CARRY0, 1] = False
    target, conflict = probe.alpha_targets(encoded["cell_values"], alive)

    assert conflict.tolist() == [True]
    assert target[0, probe.CARRY0, 1].item() == 0.0


def test_exact_column_deduction_propagates_carry():
    row = probe.row_from_prompt("x", "Compute 57 + 68.", "125")
    alive = probe.top_lattice(batch_size=1, device=torch.device("cpu"))
    deduced, conflict = probe.exact_addition_deduction([row], alive)

    assert conflict.tolist() == [False]
    assert deduced[0, probe.ONES].nonzero(as_tuple=False).flatten().tolist() == [5]
    assert deduced[0, probe.CARRY0].nonzero(as_tuple=False).flatten().tolist() == [1]
    assert deduced[0, probe.TENS].nonzero(as_tuple=False).flatten().tolist() == [2]
    assert deduced[0, probe.CARRY1].nonzero(as_tuple=False).flatten().tolist() == [1]
    assert deduced[0, probe.HUNDREDS].nonzero(as_tuple=False).flatten().tolist() == [1]


def test_threshold_step_is_monotone():
    alive = probe.top_lattice(batch_size=1, device=torch.device("cpu"))
    logits = torch.full((1, probe.N_CELLS, probe.MAX_CANDIDATES), -10.0)
    logits[0, probe.ONES, 5] = 10.0
    next_alive = probe.threshold_eliminate(alive, logits, threshold=0.5)

    assert torch.logical_or(next_alive == alive, next_alive == torch.zeros_like(alive)).all()
    assert next_alive[0, probe.ONES].sum().item() == 1
    assert next_alive[0, probe.ONES, 5].item() is True


def test_branch_pins_one_unresolved_cell():
    alive = probe.top_lattice(batch_size=1, device=torch.device("cpu"))
    logits = torch.zeros((1, probe.N_CELLS, probe.MAX_CANDIDATES))
    logits[0, probe.ONES, 7] = 10.0
    pinned = probe.branch_pin(alive, logits)

    assert pinned[0, probe.ONES].sum().item() == 1
    assert pinned[0, probe.ONES, 7].item() is True


def test_model_forward_shapes():
    model = probe.CarryLatticeLDT(width=16, layers=1, heads=2, internal_iters=3)
    rows = [
        probe.row_from_prompt("a", "Compute 24 + 14.", "38"),
        probe.row_from_prompt("b", "Compute 57 + 68.", "125"),
    ]
    encoded = probe.encode_batch(rows, torch.device("cpu"))
    alive = probe.top_lattice(batch_size=2, device=torch.device("cpu"))
    outputs = model(encoded, alive)

    assert len(outputs) == 3
    keep_logits, conflict_logits = outputs[-1]
    assert keep_logits.shape == (2, probe.N_CELLS, probe.MAX_CANDIDATES)
    assert conflict_logits.shape == (2,)


def test_oracle_solver_returns_verified_addition_answer():
    row = probe.row_from_prompt("x", "Compute 57 + 68.", "125")

    class Oracle:
        def eval(self): ...

        def __call__(self, encoded, alive):
            logits = torch.full_like(alive.float(), -10.0)
            for cell, value in enumerate(row["cell_values"]):
                logits[0, cell, value] = 10.0
            conflict = torch.full((1,), -10.0)
            return [(logits, conflict)]

    report = probe.solve_rows(
        Oracle(),
        [row],
        torch.device("cpu"),
        threshold=0.5,
        cls_threshold=0.6,
        max_solve_steps=1,
        branch=False,
    )
    assert report["returned_correct"] == 1
    assert report["returned_wrong"] == 0
    assert report["abstained"] == 0


def test_run_probe_smoke_schema():
    args = argparse.Namespace(
        steps=2,
        batch_size=4,
        width=16,
        layers=1,
        heads=2,
        internal_iters=2,
        lr=1e-3,
        eval_every=1,
        eval_limit=4,
        train_limit=16,
        seed=55,
        device="cpu",
        threshold=0.1,
        cls_threshold=0.6,
        max_solve_steps=2,
        branch=False,
        on_policy_steps=0,
        smoke=True,
        out=None,
    )
    result = probe.run_probe(args)

    assert result["config"]["smoke"] is True
    assert result["train"]["n"] > 0
    assert "frozen_add" in result["report"]
