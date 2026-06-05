# Experiment 61 - DeepSeek Two Unknown Addition Policy

## Question

Can we train the tiny neural side on search states instead of final answers?

## Setup

- DeepSeek generates puzzle prompts and target sums.
- Python verifies every DeepSeek row before use.
- Task: find two integers `x,y` in `[0,99]` such that `x + y = target_sum`.
- Lattice cells: `x_tens`, `x_ones`, `y_tens`, `y_ones`, `carry0`, `carry1`.
- Closure is the only eliminator.
- Neural policy only chooses one branch/pin.

Training rows are:

```text
current lattice state -> best next branch
```

Not:

```text
prompt -> final answer
```

## Command

```powershell
rtk python "experiments/Experiment 61 - DeepSeek Two Unknown Addition Policy/two_unknown_addition_policy_probe.py" --device cpu --dataset-source deepseek --train-puzzles 32 --eval-puzzles 12 --deepseek-batch-size 12 --epochs 80 --width 48 --out "experiments/Experiment 61 - DeepSeek Two Unknown Addition Policy/results_seed61_deepseek_small.json" --dataset-out "experiments/Experiment 61 - DeepSeek Two Unknown Addition Policy/deepseek_puzzles_seed61.jsonl" --states-out "experiments/Experiment 61 - DeepSeek Two Unknown Addition Policy/search_states_seed61.jsonl"
```

## Result

DeepSeek rows:

- train puzzles: 32
- eval puzzles: 12
- accepted after exact validation
- search states: 53

| Policy | correct | wrong | coverage | mean branches | max branches |
|---|---:|---:|---:|---:|---:|
| first | 12/12 | 0 | 100% | 1.83 | 2 |
| random | 12/12 | 0 | 100% | 2.00 | 3 |
| oracle | 12/12 | 0 | 100% | 1.75 | 2 |
| learned | 12/12 | 0 | 100% | 1.83 | 2 |

## Read

Good:

- DeepSeek dataset path works.
- LLM rows are not trusted blindly; Python validates them.
- The solver stays sound: `wrong=0` for every policy.
- The tiny policy trains on branch states, not answers.

Hard truth:

- This domain is still easy.
- First-cell is already strong.
- Learned policy matched first-cell but did not beat it.
- Oracle only has a small advantage, so there is not much room to learn.

## Decision

Keep the harness. Do not claim a learned-controller win yet.

Next useful step: make the branch problem harder with extra constraints, like:

- `x + y = target`
- `x` must be even
- `y` must be greater than `x`
- `x` cannot use digit 7
- one addend must be in a range

That should create real search pressure where policy quality matters.
