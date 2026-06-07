# Experiment 67 - Multiplication Repair

## Goal

Use the Exp66 strict generation dump to find the weak math spot, then try a short focused repair run.

## Failure Map From Exp66 Checkpoint

Command:

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File "experiments/Experiment 66 - Word Problem Reasoning Corpus/start_exp67_failure_dump_limit200.ps1"
```

Files:

- `experiments/Experiment 66 - Word Problem Reasoning Corpus/results_exp67_failure_dump_limit200.json`
- `experiments/Experiment 66 - Word Problem Reasoning Corpus/generations_exp67_failure_dump_limit200.jsonl`

Strict result, 200 rows per split:

| slice | acc | passed / n |
|---|---:|---:|
| overall | 41.7% | 250 / 600 |
| heldout_direct | 39.5% | 79 / 200 |
| heldout_word | 32.0% | 64 / 200 |
| heldout_hard | 53.5% | 107 / 200 |

By operation:

| op | acc | passed / n |
|---|---:|---:|
| `*` | 0.8% | 1 / 129 |
| `+` | 59.9% | 197 / 329 |
| `-` | 46.5% | 159 / 342 |

Read: the model does not mainly fail by junk output. It gives clean answer-shaped text, but the arithmetic is wrong. Multiplication is the biggest hole.

## Repair Data

Generated:

```powershell
rtk python "experiments/Experiment 67 - Multiplication Repair/generate_exp67_repair_sft.py" --train-rows 30000 --valid-rows 2000 --seed 1 --out-dir "data/exp67_mul_repair_sft/v1"
```

Files:

- `data/exp67_mul_repair_sft/v1/train.jsonl`
- `data/exp67_mul_repair_sft/v1/valid.jsonl`
- `data/exp67_mul_repair_sft/v1/manifest.json`

Data check:

- train rows: 30,000
- valid rows: 2,000
- multiplication train rows: 22,513
- excluded held-out signatures: 2,369
- duplicate train signatures: 0
- duplicate valid signatures: 0
- train/valid signature overlap: 0

## Repair Train

Command:

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File "experiments/Experiment 67 - Multiplication Repair/run_exp67_mul_repair_sft1000.ps1"
```

Checkpoint:

- `artifacts/phase0_exp67_mul_repair/h256_exp66_word_sft2000_mulrepair_sft1000_seed1/checkpoint_fp32.pt`

Training result:

| metric | value |
|---|---:|
| steps | 1,000 |
| valid loss before | 0.7532 |
| valid loss after | 0.2338 |
| valid token acc after | 91.7% |
| valid exact acc after | 5.5% |
| frozen generation acc | 7.0% |
| peak VRAM | 788.5 MB |

Strict smoke on held-out, 20 rows per split:

| slice | acc | passed / n |
|---|---:|---:|
| overall | 3.3% | 2 / 60 |
| heldout_direct | 5.0% | 1 / 20 |
| heldout_word | 0.0% | 0 / 20 |
| heldout_hard | 5.0% | 1 / 20 |
| `*` | 0.0% | 0 / 13 |

Files:

- `experiments/Experiment 67 - Multiplication Repair/results_exp67_mulrepair_strict_eval_limit20.json`
- `experiments/Experiment 67 - Multiplication Repair/generations_exp67_mulrepair_strict_eval_limit20.jsonl`

## Decision

No promote.

Keep using the Exp66 checkpoint as the last usable word-reasoning checkpoint. The Exp67 repair checkpoint makes exact arithmetic worse, even though token loss improves.

Next repair should use a smaller gate before long training: train a tiny 2-digit-only multiplication mix, keep a strong slice of the old Exp66 data, then run the strict 20-row gate before any bigger run.
