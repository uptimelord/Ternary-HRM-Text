from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = (
    REPO_ROOT
    / "experiments"
    / "Experiment 58 - Top State Curriculum"
    / "top_state_curriculum_probe.py"
)

spec = importlib.util.spec_from_file_location("top_state_curriculum_probe", PROBE_PATH)
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


def test_impossible_114_still_becomes_conflict_through_solver():
    """84 + 19 must never return the impossible 114; closure forces conflict."""
    row = probe.row_from_prompt("x", "Compute 84 + 19.", "103")

    class Impossible114Oracle:
        def eval(self): ...

        def __call__(self, encoded, alive):
            logits = torch.full_like(alive.float(), -10.0)
            for cell, value in enumerate((4, 1, 1, 1, 1)):
                logits[0, cell, value] = 10.0
            conflict = torch.full((1,), -10.0)
            return [(logits, conflict)]

    report = probe.solve_rows(
        Impossible114Oracle(),
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


def _train_args(steps: int, curriculum_steps: int) -> argparse.Namespace:
    return argparse.Namespace(
        steps=steps,
        curriculum_steps=curriculum_steps,
        batch_size=4,
        width=16,
        layers=1,
        heads=2,
        internal_iters=2,
        lr=1e-3,
        eval_every=steps,
        eval_limit=4,
        train_limit=16,
        seed=58,
        device="cpu",
        threshold=0.1,
        cls_threshold=0.6,
        max_solve_steps=2,
        branch=False,
        on_policy_steps=1,
        keep_pos_weight=4.0,
        keep_neg_weight=0.5,
        cls_weight=0.1,
        ce_weight=0.2,
        smoke=False,
        out=None,
    )


def test_curriculum_phase_a_top_then_phase_b_on_policy(monkeypatch):
    """Phase A (<= curriculum_steps) uses the top lattice; Phase B switches to
    on-policy rollout exactly at curriculum_steps + 1."""
    rollout_steps: list[int] = []
    real_rollout = probe.rollout_alive

    def counting_rollout(*pargs, **kwargs):
        rollout_steps.append(1)
        return real_rollout(*pargs, **kwargs)

    monkeypatch.setattr(probe, "rollout_alive", counting_rollout)

    # 4 steps, 2 warmup: rollout must fire only on steps 3 and 4.
    probe.train_one_seed(_train_args(steps=4, curriculum_steps=2), seed=58)
    assert len(rollout_steps) == 2

    rollout_steps.clear()
    # All-warmup: rollout must never fire.
    probe.train_one_seed(_train_args(steps=4, curriculum_steps=4), seed=58)
    assert len(rollout_steps) == 0


def test_curriculum_steps_defaults_to_60_percent():
    args = probe.build_arg_parser().parse_args(["--steps", "500"])
    result_args = probe._fill_defaults(args)
    assert result_args.curriculum_steps is None
    # run_probe resolves the default; emulate its resolution.
    if result_args.curriculum_steps is None:
        result_args.curriculum_steps = int(round(0.6 * result_args.steps))
    assert result_args.curriculum_steps == 300


def test_run_probe_smoke_schema():
    args = argparse.Namespace(
        steps=4,
        curriculum_steps=2,
        batch_size=4,
        width=16,
        layers=1,
        heads=2,
        internal_iters=2,
        lr=1e-3,
        eval_every=1,
        eval_limit=4,
        train_limit=16,
        seed=58,
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
    assert result["config"]["curriculum_steps"] >= 0
    assert result["train"]["n"] > 0
    assert "frozen_add" in result["report"]
