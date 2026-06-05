from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = (
    REPO_ROOT
    / "experiments"
    / "Experiment 60 - Closure Controller Policy"
    / "closure_controller_policy_probe.py"
)

spec = importlib.util.spec_from_file_location("closure_controller_policy_probe", PROBE_PATH)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_closure_solves_fixed_addition_before_policy_can_act():
    row = probe.row_from_prompt("x", "Compute 84 + 19.", "103")

    class ExplodingPolicy:
        def eval(self): ...

        def __call__(self, encoded, alive):
            raise AssertionError("policy should not run after closure solves")

    report = probe.solve_rows(
        ExplodingPolicy(),
        [row],
        torch.device("cpu"),
        max_solve_steps=4,
    )

    assert report["returned_correct"] == 1
    assert report["returned_wrong"] == 0
    assert report["conflicts"] == 0
    assert report["policy_calls"] == 0
    assert report["mean_branches"] == 0.0


def test_controller_branch_only_pins_one_cell_and_never_bulk_eliminates():
    alive = probe.top_lattice(batch_size=1, device=torch.device("cpu"))
    cell_logits = torch.full((1, probe.N_CELLS), -10.0)
    value_logits = torch.full((1, probe.N_CELLS, probe.MAX_CANDIDATES), -10.0)
    cell_logits[0, probe.TENS] = 10.0
    value_logits[0, probe.TENS, 7] = 10.0

    pinned, branches = probe.controller_branch_pin(alive, cell_logits, value_logits)

    assert branches == 1
    assert pinned[0, probe.TENS].sum().item() == 1
    assert pinned[0, probe.TENS, 7].item() is True
    for cell in (probe.ONES, probe.CARRY0, probe.CARRY1, probe.HUNDREDS):
        assert torch.equal(pinned[0, cell], alive[0, cell])


def test_wrong_policy_pin_becomes_conflict_not_wrong_answer():
    row = probe.row_from_prompt("x", "Compute 84 + 19.", "103")
    alive = probe.top_lattice(batch_size=1, device=torch.device("cpu"))
    cell_logits = torch.full((1, probe.N_CELLS), -10.0)
    value_logits = torch.full((1, probe.N_CELLS, probe.MAX_CANDIDATES), -10.0)
    cell_logits[0, probe.ONES] = 10.0
    value_logits[0, probe.ONES, 4] = 10.0

    pinned, branches = probe.controller_branch_pin(alive, cell_logits, value_logits)
    closed, conflict = probe.sound_carry_closure([row], pinned)

    assert branches == 1
    assert conflict.tolist() == [True]
    assert closed[0, probe.ONES].sum().item() == 0


def test_run_probe_schema_shows_controller_has_no_work_on_fixed_addition():
    args = argparse.Namespace(
        seed=60,
        device="cpu",
        eval_limit=8,
        train_limit=32,
        max_solve_steps=4,
        smoke=True,
        out=None,
    )

    result = probe.run_probe(args)

    assert result["config"]["closure_only_eliminator"] is True
    assert result["config"]["neural_elimination"] is False
    assert result["config"]["controller_policy"] == "branch_only"
    frozen = result["report"]["frozen_add"]["solver"]
    assert frozen["returned_wrong"] == 0
    assert frozen["policy_calls"] == 0
    assert frozen["mean_branches"] == 0.0
