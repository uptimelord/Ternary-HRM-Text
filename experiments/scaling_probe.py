"""Trajectory-grid launcher for BitNet-HRM experiments.

Default shape follows the project discipline: at least two hidden sizes and at
least two step checkpoints, with two seeds for routine trend checks.
"""

from __future__ import annotations

import argparse
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from experiments import discipline


DEFAULT_HIDDEN_SIZES = [128, 256]
DEFAULT_STEP_CHECKPOINTS = [500, 2000, 5000]
DEFAULT_SEEDS = [1, 2]


@dataclass(frozen=True)
class ProbeJob:
    variant: str
    hidden_size: int
    steps: int
    seed: int


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_jobs(
    *,
    variants: list[str],
    hidden_sizes: list[int],
    step_checkpoints: list[int],
    seeds: list[int],
) -> list[ProbeJob]:
    if len(hidden_sizes) < 2:
        raise ValueError("scaling probes require at least two hidden sizes")
    if len(step_checkpoints) < 2:
        raise ValueError("scaling probes require at least two step checkpoints")
    if not variants:
        raise ValueError("scaling probes require at least one variant")
    if not seeds:
        raise ValueError("scaling probes require at least one seed")

    return [
        ProbeJob(variant=variant, hidden_size=hidden, steps=steps, seed=seed)
        for hidden in hidden_sizes
        for steps in step_checkpoints
        for seed in seeds
        for variant in variants
    ]


def planned_result_path(job: ProbeJob, output_dir: Path) -> Path:
    return output_dir / f"{job.variant}_h{job.hidden_size}_s{job.seed}_steps{job.steps}.md"


def command_for_job(
    job: ProbeJob,
    *,
    runner: Path,
    device: str,
    output_dir: Path,
) -> list[str]:
    out = planned_result_path(job, output_dir)
    return [
        "python",
        "-u",
        str(runner),
        "--steps",
        str(job.steps),
        "--seeds",
        str(job.seed),
        "--variants",
        job.variant,
        "--hidden-size",
        str(job.hidden_size),
        "--device",
        device,
        "--append-md",
        str(out),
    ]


def write_trajectory_grid(jobs: list[ProbeJob], *, output_dir: Path, grid_md: Path) -> None:
    grouped: dict[tuple[str, int, int], list[ProbeJob]] = defaultdict(list)
    for job in jobs:
        grouped[(job.variant, job.hidden_size, job.steps)].append(job)

    lines = [
        "# Scaling Probe Trajectory Grid",
        "",
        "| variant | hidden_size | steps | seeds | result files |",
        "|---|---:|---:|---|---|",
    ]
    for (variant, hidden_size, steps), group in sorted(grouped.items()):
        seeds = ",".join(str(job.seed) for job in sorted(group, key=lambda item: item.seed))
        result_files = "<br>".join(
            str(planned_result_path(job, output_dir).as_posix())
            for job in sorted(group, key=lambda item: item.seed)
        )
        lines.append(f"| `{variant}` | {hidden_size} | {steps} | {seeds} | {result_files} |")

    grid_md.parent.mkdir(parents=True, exist_ok=True)
    grid_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a hidden-size x step-count trajectory grid")
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path("experiments/Experiment 22 - Vocab Body Combo Confirmation/vocab_body_combo.py"),
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=None,
        help="Experiment README with the pre-registered decision rule. Defaults to runner sibling README.md.",
    )
    parser.add_argument("--variants", default="mixed_top512_tequila_L_mlp_gate_up")
    parser.add_argument("--hidden-sizes", default=",".join(str(x) for x in DEFAULT_HIDDEN_SIZES))
    parser.add_argument("--step-checkpoints", default=",".join(str(x) for x in DEFAULT_STEP_CHECKPOINTS))
    parser.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/scaling_probe_results"))
    parser.add_argument(
        "--grid-md",
        type=Path,
        default=None,
        help="Markdown trajectory grid to write. Defaults to output-dir/trajectory_grid.md.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    jobs = build_jobs(
        variants=parse_str_list(args.variants),
        hidden_sizes=parse_int_list(args.hidden_sizes),
        step_checkpoints=parse_int_list(args.step_checkpoints),
        seeds=parse_int_list(args.seeds),
    )
    readme = args.readme or (args.runner.parent / "README.md")
    if not readme.exists():
        raise ValueError(f"scaling probe requires a pre-registered README: {readme}")
    rule = discipline.assert_preregistered_readme(readme)
    print(rule.promote)
    print(rule.kill)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid_md = args.grid_md or (args.output_dir / "trajectory_grid.md")
    write_trajectory_grid(jobs, output_dir=args.output_dir, grid_md=grid_md)
    print(f"grid={grid_md}")

    for job in jobs:
        cmd = command_for_job(job, runner=args.runner, device=args.device, output_dir=args.output_dir)
        print(" ".join(f'"{part}"' if " " in part else part for part in cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
