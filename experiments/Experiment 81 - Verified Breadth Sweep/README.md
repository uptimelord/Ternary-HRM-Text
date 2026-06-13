# Experiment 81 - Verified Breadth Sweep

> Architecture brief **C2**: measure verified pass@k and label-free picked pass@1 at zero packed MB.

## Question

Does sampling K diverse candidates and label-free selection lift effective pass@1 on word/logic held-out splits?

## Method

```text
Exp69/Exp70 checkpoints
  -> K rollouts per task (temperature or z-noise on TRM zL_init)
  -> strict oracle scoring for pass@k
  -> label-free picker for deployable pass@1
```

`oracle_any_pass@k` is the honest name for "any sampled answer matches the gold
label". The old `verifier_picked_pass@1` field stays in JSON for compatibility
but is not a deployable selector.

Label-free pickers:

- word/arithmetic: Exp65 `tool_check_steps` recomputes the candidate chain and
  accepts the first self-consistent final answer.
- comparative logic: prompt premises are parsed and topologically sorted, then
  the first matching candidate is picked.

`z_noise` diversity is greedy (`temperature=0`) plus recurrent-state noise, so
it is not confounded with temperature sampling.

## Decision Rule

Promote if `solver_picked_pass@1` or `derived_picked_pass@1` at K=8 beats
`single_sample_pass@1` by at least 5 pp on heldout word or logic hard, invalid 0%.

Kill if pass@k curve is flat (diversity collapse, all samples wrong the same way).

## Run

```powershell
python "experiments/Experiment 81 - Verified Breadth Sweep/verified_breadth_sweep.py" --mode smoke --device cpu
python "experiments/Experiment 81 - Verified Breadth Sweep/verified_breadth_sweep.py" --mode full --device cuda --k-max 16
```

Decision-grade run (n=200 per domain, seeds 1,2, both diversity arms; rerun
stale logic rows `0000-0039` with this code before reading the curve):

```powershell
python "experiments/Experiment 81 - Verified Breadth Sweep/verified_breadth_sweep.py" --mode full --device cuda --domains word,logic --limit 200 --k-max 16 --k-values 1,2,4,8,16 --diversities temp,z_noise --seeds 1,2
```

For long K=16 logic runs, use chunks and merge them:

```powershell
python "experiments/Experiment 81 - Verified Breadth Sweep/verified_breadth_sweep.py" --mode full --device cuda --domains logic --row-offset 0 --limit 1 --k-max 16 --k-values 1,2,4,8,16 --diversities temp,z_noise --seeds 1,2 --max-new-tokens 64 --output-dir artifacts/exp81_breadth_sweep/logic_chunk_0000
python "experiments/Experiment 81 - Verified Breadth Sweep/verified_breadth_sweep.py" --merge-jsons "artifacts/exp81_breadth_sweep/logic_chunk_*/breadth_full_seed1.json" --merge-output-json artifacts/exp81_breadth_sweep/logic_merged/breadth_merged.json
```

`--row-offset` and `--limit` select the held-out rows. The merge command
combines chunk means using each chunk's `n_tasks`, so the final curve is not a
plain average of chunks.

Temperature sampling is batched across K candidates. z-noise sampling is still
serial because each candidate needs its own recurrent-state noise.

When batching multiple rows for temperature sampling, the runner groups rows by
equal prompt token length. The Windows SDPA fallback assumes equal-length packed
sequences, so mixed prompt lengths are rejected instead of silently producing a
bad shape.

`single_sample_pass@1` is the first sample's pass rate for every K bucket.

## Results

Canonical result: `results_smoke_seed1.md`.

Superseded run logs: `results_*_codex.md` files in this folder are historical
run logs. Logic rows `0000-0039` must be rerun with current metric semantics;
rowbatched chunk `0040-0059` already used the honest single-sample path.

## Verdict

Awaiting decision-grade CUDA rerun with corrected metrics.
