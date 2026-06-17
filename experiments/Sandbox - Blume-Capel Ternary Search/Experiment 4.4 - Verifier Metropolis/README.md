# Experiment 4.4 — Verifier Metropolis

## Question

If Metropolis energy = **strict comparative-logic verifier failures** (not train CE
alone), can we improve verifier pass rate without CE regression — escaping the
train-CE overfitting trap seen in Exp 4.3?

## Method

- **Proposal policy:** random-focus (same as Exp 4.2 — current thermodynamic anchor).
- **Train energy:** `failure_rate` on `verif_sample` train-visible rows per step
  (greedy generate + `comparative_logic_answer_pass`). Optional `ce_weight` for hybrid.
- **Eval:** eval CE + eval verifier pass rate vs random-focus arm from shared init.
- **Arms per seed:** blind CE, random-focus CE, verifier Metropolis.
- Shared helpers: `../sandbox_verifier_energy.py`.

## Decision Rule

- Promote if verifier Metropolis beats random-focus verifier pass on >= 2/3 seeds
  **and** max eval CE regression vs random-focus <= 0.02, **or** beats random-focus
  eval CE on >= 2/3 seeds by >= +0.0203 CE; VRAM <= 1.25 GB; ternary flips > 0; no backprop.
- Kill if no verifier or CE edge beyond noise, hidden gradients, held-out leak,
  or compute cost unusable on 4 GB GPU.

## Results

**CUDA 3-seed (kill):** Verifier Metropolis loses to random-focus on both gates. Mean CE edge **−0.71** vs random-focus (worse). Max CE regression **+0.93** vs random-focus. Verifier pass edge 0/3 seeds. ~4.5 h wall-clock.

Verdict: **kill** (confirms DFO pivot; verifier energy alone does not help). See `report.json`.

```powershell
# CPU smoke
python "experiments/Sandbox - Blume-Capel Ternary Search/Experiment 4.4 - Verifier Metropolis/run_exp4_4.py" --device cpu --seeds 1 --K 1 --cycles 1 --verif-sample 1

# CUDA full (also in gpu_queue.json)
python "experiments/Sandbox - Blume-Capel Ternary Search/Experiment 4.4 - Verifier Metropolis/run_exp4_4.py" --device cuda --seeds 1,2,3 --K 20 --cycles 20 --verif-sample 4
```

**Queued follow-up:** `exp4_5_hybrid_cuda` — same runner with `--ce-weight 0.01`.
