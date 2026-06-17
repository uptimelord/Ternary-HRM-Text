# Experiment 4.1 — SPSA Block Thermometer

## Question

Does SPSA with a thermometer / probe-batch variant beat **random-focus** Metropolis
(the stronger thermodynamic baseline)?

## Method

Arms: blind, random-focus (fixed module per cycle), SPSA thermometer.
Same K×cycles forward budget as Exp 4.2 family.

## Decision Rule

- Promote if SPSA beats random-focus on >= 2/3 seeds, mean edge >= +0.0203 CE.
- Kill if edge within noise or negative vs random-focus.

## Results

See `report.json`, `results_block_spsa.md`.

| Summary | Value |
|---------|-------|
| Seeds | 3 |
| SPSA beats random-focus | **0/3** |
| Mean edge vs random-focus | **-0.221 CE** |
| Verdict | **Kill** — thermometer did not fix ZO failure mode |

```powershell
python "experiments/Sandbox - Blume-Capel Ternary Search/Experiment 4.1 - SPSA Block Thermometer/run_exp4_1.py" --device cuda --seeds 1,2,3 --K 20 --cycles 20
```
