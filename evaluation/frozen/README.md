# Frozen Micro Benchmarks

These files are held-out eval slices. Do not train on them, rewrite them, or
refresh them after seeing model behavior.

## `frozen_arithmetic_200.jsonl`

- 200 deterministic arithmetic prompts
- one JSON object per line
- fields: `id`, `prompt`, `answer`, `split`, `version`
- intended as a cheap Phase 0.5 smoke benchmark, not a headline benchmark

Run it through the normal evaluation stack with:

```powershell
rtk python -m evaluation.main config=evaluation/config/frozen_arithmetic.yaml
```
