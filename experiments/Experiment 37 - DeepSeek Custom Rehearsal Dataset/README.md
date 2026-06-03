# Experiment 37 - DeepSeek Custom Rehearsal Dataset

> **Status: generator ready, API key needed.** Exp37 builds the custom dataset
> for the next SFT run: verified arithmetic plus DeepSeek-generated language
> rehearsal rows.

## Dataset Size

Full v1 target:

| Split | Rows |
|---|---:|
| train | 100,000 |
| valid | 4,000 |

Mix:

| Category | Share | Source |
|---|---:|---|
| arithmetic_cot | 50% | Python verified |
| simple_qa | 25% | DeepSeek V4 Flash |
| continuation | 15% | DeepSeek V4 Flash |
| anti_collapse_qa | 10% | DeepSeek V4 Flash |

The point is not to spend model credits on arithmetic we can verify ourselves.
DeepSeek is used where we need better language shape.

## Output

```text
data/deepseek_custom_rehearsal/v1/train.jsonl
data/deepseek_custom_rehearsal/v1/valid.jsonl
data/deepseek_custom_rehearsal/v1/manifest.json
```

## Run

Put the API key in the repo-local `.env` file:

```text
DEEPSEEK_API_KEY=...
```

Then run:

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File `
  "experiments/Experiment 37 - DeepSeek Custom Rehearsal Dataset/start_exp37_generate_dataset.ps1"
```

Direct runner:

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File `
  "experiments/Experiment 37 - DeepSeek Custom Rehearsal Dataset/run_exp37_generate_dataset.ps1"
```

## Safety Checks

The generator rejects:

- invalid JSON
- non-ASCII rows
- repeated phrase loops
- `"The New York Times"` style leakage
- arithmetic/`Step` leakage in normal QA rows
- duplicate prompts inside the generated file

Arithmetic rows are generated locally and avoid the frozen eval expressions.
