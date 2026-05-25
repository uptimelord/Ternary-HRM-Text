from experiments import scaling_probe


def test_scaling_probe_builds_trajectory_grid_jobs():
    jobs = scaling_probe.build_jobs(
        variants=["mixed_top512_tequila_L_mlp_gate_up"],
        hidden_sizes=[128, 256],
        step_checkpoints=[500, 2000],
        seeds=[1, 2],
    )

    assert len(jobs) == 8
    assert jobs[0].variant == "mixed_top512_tequila_L_mlp_gate_up"
    assert jobs[0].hidden_size == 128
    assert jobs[0].steps == 500
    assert jobs[0].seed == 1
    assert jobs[-1].hidden_size == 256
    assert jobs[-1].steps == 2000
    assert jobs[-1].seed == 2


def test_scaling_probe_requires_at_least_two_steps_and_hidden_sizes():
    try:
        scaling_probe.build_jobs(
            variants=["x"],
            hidden_sizes=[128],
            step_checkpoints=[500, 2000],
            seeds=[1, 2],
        )
    except ValueError as exc:
        assert "two hidden sizes" in str(exc)
    else:
        raise AssertionError("expected hidden size validation")

    try:
        scaling_probe.build_jobs(
            variants=["x"],
            hidden_sizes=[128, 256],
            step_checkpoints=[500],
            seeds=[1, 2],
        )
    except ValueError as exc:
        assert "two step checkpoints" in str(exc)
    else:
        raise AssertionError("expected step checkpoint validation")


def test_scaling_probe_writes_trajectory_grid(tmp_path):
    jobs = scaling_probe.build_jobs(
        variants=["combo"],
        hidden_sizes=[128, 256],
        step_checkpoints=[500, 2000],
        seeds=[1, 2],
    )
    grid = tmp_path / "grid.md"

    scaling_probe.write_trajectory_grid(jobs, output_dir=tmp_path, grid_md=grid)

    text = grid.read_text(encoding="utf-8")
    assert "| `combo` | 128 | 500 | 1,2 |" in text
    assert "combo_h128_s1_steps500.md" in text
    assert "combo_h256_s2_steps2000.md" in text
