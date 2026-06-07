from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = (
    REPO_ROOT
    / "experiments"
    / "Experiment 57 - Closure Led Elimination"
    / "closure_led_elimination_probe.py"
)

spec = importlib.util.spec_from_file_location("closure_led_elimination_probe", PROBE_PATH)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_sound_closure_solves_from_top_by_carry_equations():
    row = probe.row_from_prompt("x", "Compute 84 + 19.", "103")
    alive = probe.top_lattice(batch_size=1, device=torch.device("cpu"))
    closed, conflict = probe.sound_carry_closure([row], alive)

    assert conflict.tolist() == [False]
    assert probe.cells_from_singleton(closed[0]) == (3, 1, 0, 1, 1)
    assert probe.decode_cells(probe.cells_from_singleton(closed[0])) == 103


def test_sound_closure_rejects_impossible_returned_state():
    row = probe.row_from_prompt("x", "Compute 84 + 19.", "103")
    alive = torch.zeros_like(probe.top_lattice(batch_size=1, device=torch.device("cpu")))
    # This is the bad Exp55-style return: 114 for 84 + 19.
    for cell, value in enumerate((4, 1, 1, 1, 1)):
        alive[0, cell, value] = True

    closed, conflict = probe.sound_carry_closure([row], alive)

    assert conflict.tolist() == [True]
    assert closed[0, probe.ONES].sum().item() == 0


def test_solver_does_not_return_wrong_oracle_singleton():
    row = probe.row_from_prompt("x", "Compute 84 + 19.", "103")

    class BadOracle:
        def eval(self): ...

        def __call__(self, encoded, alive):
            logits = torch.full_like(alive.float(), -10.0)
            for cell, value in enumerate((4, 1, 1, 1, 1)):
                logits[0, cell, value] = 10.0
            conflict = torch.full((1,), -10.0)
            return [(logits, conflict)]

    report = probe.solve_rows(
        BadOracle(),
        [row],
        torch.device("cpu"),
        threshold=0.5,
        cls_threshold=0.6,
        max_solve_steps=1,
        branch=False,
    )

    # Closure leads, so the bad 114 oracle cannot force a wrong return.
    assert report["returned_wrong"] == 0


def test_solver_returns_correct_after_closure_even_from_broad_oracle():
    row = probe.row_from_prompt("x", "Compute 84 + 19.", "103")

    class BroadOracle:
        def eval(self): ...

        def __call__(self, encoded, alive):
            logits = torch.full_like(alive.float(), 10.0)
            conflict = torch.full((1,), -10.0)
            return [(logits, conflict)]

    report = probe.solve_rows(
        BroadOracle(),
        [row],
        torch.device("cpu"),
        threshold=0.5,
        cls_threshold=0.6,
        max_solve_steps=1,
        branch=False,
    )

    assert report["returned_correct"] == 1
    assert report["returned_wrong"] == 0


def test_closure_veto_protects_unique_survivor_from_neural_elimination():
    # Closure on 84 + 19 pins every cell to a unique survivor (103).
    row = probe.row_from_prompt("x", "Compute 84 + 19.", "103")
    alive = probe.top_lattice(batch_size=1, device=torch.device("cpu"))
    closed, conflict = probe.sound_carry_closure([row], alive)
    assert conflict.tolist() == [False]
    assert probe.cells_from_singleton(closed[0]) == (3, 1, 0, 1, 1)

    # Neural threshold tries to eliminate EVERY candidate (would empty all cells).
    kill_all_logits = torch.full_like(closed.float(), -10.0)
    result = probe.closure_led_eliminate(closed, kill_all_logits, threshold=0.5)

    # Veto: unique survivors are protected, nothing emptied, lattice unchanged.
    assert torch.equal(result, closed)
    assert probe.cells_from_singleton(result[0]) == (3, 1, 0, 1, 1)


def test_impossible_114_becomes_conflict_at_closure_level():
    # The impossible Exp55-style 114 singleton for 84 + 19, fed straight to the
    # sound closure, must still collapse to a conflict (empty cell).
    row = probe.row_from_prompt("x", "Compute 84 + 19.", "103")
    alive = torch.zeros_like(probe.top_lattice(batch_size=1, device=torch.device("cpu")))
    for cell, value in enumerate((4, 1, 1, 1, 1)):
        alive[0, cell, value] = True

    closed, conflict = probe.sound_carry_closure([row], alive)

    assert conflict.tolist() == [True]
    assert int(closed[0].sum(dim=1).min().item()) == 0


def test_closure_led_loop_never_returns_impossible_114():
    # Under closure-led ordering, a bad oracle that wants 114 cannot force it:
    # closure leads and pins the true path, so 114 is never a wrong return.
    row = probe.row_from_prompt("x", "Compute 84 + 19.", "103")

    class BadOracle:
        def eval(self): ...

        def __call__(self, encoded, alive):
            logits = torch.full_like(alive.float(), -10.0)
            for cell, value in enumerate((4, 1, 1, 1, 1)):
                logits[0, cell, value] = 10.0
            conflict = torch.full((1,), -10.0)
            return [(logits, conflict)]

    report = probe.solve_rows(
        BadOracle(),
        [row],
        torch.device("cpu"),
        threshold=0.5,
        cls_threshold=0.6,
        max_solve_steps=4,
        branch=True,
    )

    assert report["returned_wrong"] == 0
    for example in report["wrong_examples"]:
        assert example["returned"] != "114"


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
        seed=57,
        device="cpu",
        threshold=0.1,
        cls_threshold=0.6,
        max_solve_steps=2,
        branch=False,
        on_policy_steps=1,
        smoke=True,
        out=None,
    )
    result = probe.run_probe(args)

    assert result["config"]["sound_carry_closure"] is True
    assert result["config"]["closure_led_elimination"] is True
    assert result["train"]["n"] > 0
    assert "frozen_add" in result["report"]
