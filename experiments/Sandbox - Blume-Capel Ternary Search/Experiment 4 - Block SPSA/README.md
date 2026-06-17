# Experiment 4 — Block SPSA

## Question

Does a zeroth-order SPSA gradient estimate improve Metropolis proposals vs global blind search?

## Method

Three arms per seed (shared init): blind Metropolis, SPSA-directed block proposals.
Same forward budget; energy = train CE. MeZO/SPSA family (Malladi 2023 / sparse ZO).

## Decision Rule

- Promote if SPSA beats blind on >= 2/3 seeds, mean edge >= +0.0203 CE,
  VRAM <= 1.25 GB, no backprop.
- Kill if SPSA <= blind within noise or compute/accept-rate pathology.

## Results

See `report.json`, `results_block_spsa.md`.

| Summary | Value |
|---------|-------|
| Seeds | 3 |
| SPSA beats blind | **0/3** |
| Mean edge vs blind | **-0.256 CE** |
| Verdict | **Kill** — ZO direction failed; do not invest in SPSA variants here |

```powershell
python "experiments/Sandbox - Blume-Capel Ternary Search/Experiment 4 - Block SPSA/run_exp4.py" --device cuda --seeds 1,2,3 --K 20 --cycles 20
```
