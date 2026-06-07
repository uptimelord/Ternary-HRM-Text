from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = (
    REPO_ROOT
    / "experiments"
    / "Experiment 59 - Soft Elimination Threshold"
    / "soft_elimination_threshold_probe.py"
)

spec = importlib.util.spec_from_file_location("soft_elimination_threshold_probe", PROBE_PATH)
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

    assert report["returned_correct"] == 0
    assert report["returned_wrong"] == 0
    assert report["conflicts"] == 1


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


def test_neural_elim_off_removes_nothing_in_threshold_step():
    alive = probe.top_lattice(batch_size=1, device=torch.device("cpu"))
    # Logits that would eliminate every candidate if neural elim were on.
    keep_logits = torch.full_like(alive.float(), -10.0)

    on = probe.threshold_eliminate(alive, keep_logits, threshold=0.1, neural_elim=True)
    off = probe.threshold_eliminate(alive, keep_logits, threshold=0.1, neural_elim=False)

    # On with strongly-negative logits kills everything in candidate domain.
    assert on.sum().item() == 0
    # Off removes nothing: equals the candidate-domain mask of the input lattice.
    expected = alive & probe.cell_candidate_mask(alive.device).unsqueeze(0)
    assert torch.equal(off, expected)
    assert off.sum().item() == expected.sum().item()
    assert off.sum().item() > 0


def test_neural_elim_off_stays_sound_against_bad_oracle():
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
        neural_elim=False,
    )

    # With elimination off the bad oracle cannot kill the true path; closure
    # narrows the lattice and never returns a wrong answer.
    assert report["returned_wrong"] == 0


def test_impossible_114_becomes_conflict_with_neural_elim_off():
    row = probe.row_from_prompt("x", "Compute 84 + 19.", "103")
    alive = torch.zeros_like(probe.top_lattice(batch_size=1, device=torch.device("cpu")))
    # Force the impossible 114 singleton state.
    for cell, value in enumerate((4, 1, 1, 1, 1)):
        alive[0, cell, value] = True

    # neural-elim off only re-masks to candidate domain; it must not rescue 114.
    masked = probe.threshold_eliminate(
        alive, torch.zeros_like(alive.float()), threshold=0.02, neural_elim=False
    )
    closed, conflict = probe.sound_carry_closure([row], masked)

    assert conflict.tolist() == [True]
    assert closed[0, probe.ONES].sum().item() == 0


def test_run_probe_smoke_schema_neural_elim_in_config():
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
        seed=56,
        device="cpu",
        threshold=0.02,
        cls_threshold=0.6,
        max_solve_steps=2,
        branch=False,
        neural_elim="off",
        on_policy_steps=1,
        smoke=True,
        out=None,
    )
    result = probe.run_probe(args)

    assert result["config"]["sound_carry_closure"] is True
    assert result["config"]["neural_elim"] == "off"
    assert result["train"]["n"] > 0
    assert "frozen_add" in result["report"]
    # Soundness invariant: pure-deduction baseline never returns wrong.
    assert result["report"]["frozen_add"]["solver"]["returned_wrong"] == 0
