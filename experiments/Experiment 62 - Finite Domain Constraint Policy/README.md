# Experiment 62 - Finite Domain Constraint Policy

## Question

Can we stop building one math family at a time?

## Setup

DeepSeek generates structured constraint tasks. Python validates and solves them.
The model trains on search states, not final answers.

Task shape:

```json
{
  "variables": ["x", "y"],
  "domains": {"x": [0, 99], "y": [0, 99]},
  "constraints": [
    {"type": "sum_eq", "vars": ["x", "y"], "value": 103},
    {"type": "gt_const", "var": "x", "value": 40},
    {"type": "mod_eq", "var": "y", "mod": 2, "value": 0}
  ]
}
```

Allowed constraints include:

- sum, difference, product equality
- var/constant comparisons
- var/var comparisons
- range
- modulo
- parity
- not-equal constant

Solver backend:

- `python`: pure Python finite-domain enumeration
- `numba`: optional, used only if Numba is installed
- `auto`: picks Numba when available, else Python

In this local run, Numba was not installed, so `auto` fell back to Python.

## Command

```powershell
rtk python "experiments/Experiment 62 - Finite Domain Constraint Policy/finite_domain_constraint_policy_probe.py" --device cpu --dataset-source deepseek --backend auto --train-tasks 100 --eval-tasks 40 --deepseek-batch-size 20 --epochs 100 --width 96 --max-solve-steps 8 --out "experiments/Experiment 62 - Finite Domain Constraint Policy/results_seed62_deepseek_100.json" --dataset-out "experiments/Experiment 62 - Finite Domain Constraint Policy/deepseek_tasks_seed62.jsonl" --states-out "experiments/Experiment 62 - Finite Domain Constraint Policy/search_states_seed62.jsonl"
```

## Decision Rule

Promote if learned policy beats first/random on mean branches with `wrong == 0`
on eval tasks where oracle needs **≥3** branches and first/random need **≥5**.

Kill if all policies tie at **100%** coverage with `wrong == 0` on easy tasks, or
Python backend cannot scale to **1k+** tasks without Numba.

## Results

- train tasks: 100
- eval tasks: 40
- search states: 100
- backend: Python fallback
- Numba available: false

| Policy | correct | wrong | coverage | mean branches | max branches |
|---|---:|---:|---:|---:|---:|
| first | 40/40 | 0 | 100% | 1.025 | 2 |
| random | 40/40 | 0 | 100% | 1.025 | 2 |
| oracle | 40/40 | 0 | 100% | 1.000 | 1 |
| learned | 40/40 | 0 | 100% | 1.025 | 2 |

## Read

Good:

- The broad finite-domain harness works.
- DeepSeek rows are structured and exactly validated.
- Unsupported/unsolved rows fail closed.
- The solver stayed sound: `wrong=0`.
- The training target is search control, not answer imitation.

Hard truth:

- The generated tasks are still too easy.
- First/random/learned are basically tied.
- Oracle barely improves because most tasks need only one branch.
- Python fallback is too slow for 10k scale.

## Read

Keep Exp62 as the reusable math harness.

Do not run 10k until one of these is true:

- Numba is installed and `backend=auto` picks it.
- Tasks are made harder so oracle clearly beats first/random.

Next useful knob: require DeepSeek to generate tasks with larger solution sets
and force at least 3-5 branch steps under first/random before accepting the row.
