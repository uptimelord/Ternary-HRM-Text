# Experiment 63 - Hard Constraint Scale

## What This Does

Build the dataset first.

The task shape is four small numbers:

```text
x, y, z, w are each 0..9
find values that obey the rules
```

Rules can be:

- add to a number
- multiply to a number
- greater than / less than
- even / odd
- inside a range
- all different
- equal or not equal to a number

The code checks every row. The bucket is not trusted from the row text.

## Files

- `hard_tasks_seed63_10k_verified.jsonl`
  - 10,000 verified local rows
  - 3,334 easy
  - 3,333 average
  - 3,333 difficult

- `deepseek_hard_tasks_seed63_300.jsonl`
  - 260 accepted DeepSeek rows before timeout
  - kept as a real DeepSeek sample

- `results_seed63_dataset_10k_verified.json`
  - summary for the 10k dataset

## Why The 10k File Is Local

Direct DeepSeek was too slow here.

It saved 260 accepted rows, but did not finish the 300-row check inside the
time limit. At that speed, 10k rows would take many hours.

So the full 10k dataset was made with the exact local task maker. It still uses
the same checker, same rule shape, and same bucket rules.

## Result

```text
total: 10,000
easy: 3,334
average: 3,333
difficult: 3,333
wrong/truth labels: none from model, all checked by code
training: not run
```

## Next

Train on this dataset only after this file is reviewed.

The model should learn:

```text
current open choices -> best next choice
```

It should not learn:

```text
task -> answer
```
