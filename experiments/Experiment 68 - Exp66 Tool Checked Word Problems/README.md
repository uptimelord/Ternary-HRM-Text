# Experiment 68 - Exp66 Tool Checked Word Problems

## Goal

Run the Exp66 word-reasoning checkpoint through the Exp65 calculator loop.

The model writes steps. The tool ignores the model's computed numbers, recomputes each step exactly, then the strict verifier checks the final answer.

## Files

- runner: `experiments/Experiment 68 - Exp66 Tool Checked Word Problems/exp66_tool_checked_word_problems.py`
- result: `experiments/Experiment 68 - Exp66 Tool Checked Word Problems/results_exp68_tool_checked_limit200.json`
- records: `experiments/Experiment 68 - Exp66 Tool Checked Word Problems/records_exp68_tool_checked_limit200.jsonl`

## Command

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File "experiments/Experiment 68 - Exp66 Tool Checked Word Problems/run_exp68_tool_checked_limit200.ps1"
```

## Result

Strict eval, 200 rows per split.

| split | raw | tool-checked | read/plan valid |
|---|---:|---:|---:|
| overall | 41.7% | 83.7% | 83.7% |
| direct arithmetic | 39.5% | 100.0% | 100.0% |
| word binary | 32.0% | 51.0% | 51.0% |
| word add/sub chains | 53.5% | 100.0% | 100.0% |

Buckets:

| bucket | count |
|---|---:|
| raw correct | 250 |
| tool fixed compute error | 253 |
| wrong plan/read | 97 |

By operation:

| op | raw | tool-checked |
|---|---:|---:|
| `*` | 0.8% | 65.9% |
| `+` | 59.9% | 96.0% |
| `-` | 46.5% | 88.0% |

Word-only operation result:

| op | raw | tool-checked |
|---|---:|---:|
| `*` | 1.8% | 22.8% |
| `+` | 61.4% | 81.4% |
| `-` | 27.4% | 43.8% |

## Plain Read

This is the payoff run.

The model often knows what math step to write, but it computes the number wrong. The calculator fixes that.

Direct arithmetic and two-step add/sub word chains hit 100% once tool-checked.

The hard remaining problem is word-binary reading:

- "rows with chairs each" should become multiplication
- "give away" should become subtraction
- one-step word prompts sometimes become fake two-step plans

So Exp68 says: do not train the model to be a calculator. Use the exact solver for numbers. Train or parse the model for reading/planning.

## Decision

Promote the student-plus-calculator direction.

Do not promote Exp67 repair checkpoint. Keep Exp66 checkpoint plus Exp68 tool-check loop as the best current path.

Next work should target plan/read errors on word-binary prompts, especially word multiplication.
