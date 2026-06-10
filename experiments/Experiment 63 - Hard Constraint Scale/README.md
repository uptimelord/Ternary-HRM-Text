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
  - 277 accepted DeepSeek rows before timeout
  - 100 easy
  - 100 average
  - 77 difficult
  - kept as a real DeepSeek sample

- `results_seed63_dataset_10k_verified.json`
  - summary for the 10k dataset

- `results_seed63_train_10k.json`
  - 10k train run from the saved dataset file

## Why The 10k File Is Local

Direct DeepSeek was too slow here.

It saved 277 accepted rows, but did not finish the 300-row check inside the
time limit. At that speed, 10k rows would take many hours.

So the full 10k dataset was made with the exact local task maker. It still uses
the same checker, same rule shape, and same bucket rules.

## Decision Rule

Promote if learned policy beats first/random on difficult-bucket eval with
`returned_wrong == 0` and mean branches at least **2** fewer than first/random.

Kill if first and random also solve **100%** of eval at equal branch counts —
dataset lacks search pressure for learned control.

## Results

```text
total: 10,000
easy: 3,334
average: 3,333
difficult: 3,333
wrong/truth labels: none from model, all checked by code
training: run
device: cuda
train/eval: 8,000 / 2,000
search states: 17,922
learned verified acc: 100%
first verified acc: 100%
random verified acc: 100%
oracle verified acc: 100%
returned wrong: 0 for all policies
```

## Next

This run proves the saved-file training path works.

It does not prove the tiny branch chooser is smart yet, because the simple
first and random rules also solve every eval task. The next dataset needs
choices where a weak branch rule takes more steps or gets stuck.

The model should learn this:

```text
current open choices -> best next choice
```

It should not learn this:

```text
task -> answer
```
