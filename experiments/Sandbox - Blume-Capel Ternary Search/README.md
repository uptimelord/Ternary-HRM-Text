# Sandbox — Blume-Capel Ternary Search

No-backprop lane for a tiny ternary TRM body on Exp70 comparative-logic SFT data.
**Forbidden in method arms:** `loss.backward()`, `optimizer.step()`, AdamW/SGD, autograd
gradients. **Allowed:** forward passes, CE energy, population search (CEM/ES),
ternary quantize, verifier scores as secondary metrics.

AdamW appears only as a **baseline to beat**, not inside no-backprop arms.

## Active lane (DFO pivot)

Metropolis / verifier / beam / credit-map arms are **historical** (killed or abandoned).
Current work: **zero-order DFO** vs AdamW under matched wall-clock.

| Priority | ID | Method | Gate |
|----------|-----|--------|------|
| 1 | [4.7 — CEM](Experiment%204.7%20-%20Cross%20Entropy%20Method/README.md) | Block Cross-Entropy Method | Beat AdamW 2/3 seeds, +0.05 CE |
| 2 | [4.8 — ES](Experiment%204.8%20-%20Evolution%20Strategies/README.md) | Evolution Strategies | Beat AdamW 2/3 seeds, +0.05 CE |
| 3 | [4.9 — LGL](Experiment%204.9%20-%20Layer%20Wise%20LGL/README.md) | Layer-wise Greedy Local Learning | Beat AdamW 2/3 seeds, +0.05 CE |
| 4 | [4.10 — EP](Experiment%204.10%20-%20Equilibrium%20Propagation/README.md) | Equilibrium Propagation | Beat AdamW 2/3 seeds, +0.05 CE |
| 5 | [4.11 — PC](Experiment%204.11%20-%20Predictive%20Coding%20Gibbs/README.md) | Predictive Coding Gibbs | Beat AdamW 2/3 seeds, +0.05 CE |
| 6 | [4.12 — RPF](Experiment%204.12%20-%20RPF%20Ternary%20Greedy/README.md) | RPF + ternary greedy (full model) | Beat AdamW; ~12 MB overhead at scale |
| 7 | [4.13 — Local LM](Experiment%204.13%20-%20Layerwise%20Local%20LM/README.md) | Layer-wise low-rank local LM | Beat AdamW; ~250 MB/block at scale |
| backlog | 4.14 | Simulated Bifurcation | After scalable lane verdict |

## Shared setup

| Item | Value |
|------|-------|
| Body | `build_trm_lmhead` h=64, L=2, `ternary_body=True`, weights in `{-1,0,1}` |
| Train data | Exp70 `train_30k_sft.jsonl` (train-visible only; `guard_held_out=True`) |
| Eval metric (headline) | Eval CE on held-aside batch from same train file slice |
| Secondary | Strict comparative-logic verifier pass rate (Exp 4.4+) |
| GPU | Serialize-only; one RTX 3050 Ti class card — use `gpu_queue.py` |
| Noise floor | +/- 0.0203 eval CE (5000-step discipline) |

## Experiment map

| ID | Question | Verdict | Notes |
|----|----------|---------|-------|
| [1 — PoC](Experiment%201%20-%20PoC%20Measurements/README.md) | Does blind Metropolis beat random flips on CPU? | **Promote PoC** | 5/5 seeds beat random |
| [2 — CUDA Scale](Experiment%202%20-%20CUDA%20Scale/README.md) | Same at CUDA scale? | **Promote scale** | 3/3 beat random; ~1 GB VRAM |
| [3 — AdamW Gate](Experiment%203%20-%20AdamW%20Gate/README.md) | Can Metropolis beat AdamW CE? | **Kill lane** | AdamW -2.44 CE mean edge |
| [4 — Block SPSA](Experiment%204%20-%20Block%20SPSA/README.md) | SPSA direction vs blind? | **Kill** | -0.26 CE vs blind |
| [4.1 — SPSA Thermometer](Experiment%204.1%20-%20SPSA%20Block%20Thermometer/README.md) | SPSA vs random-focus? | **Kill** | -0.22 CE |
| [4.2 — Credit Map](Experiment%204.2%20-%20Credit%20Map/README.md) | Bandit block credit vs random-focus? | **Kill** | -0.0003 CE mean edge |
| [4.3 — Beam Metropolis](Experiment%204.3%20-%20Beam%20Metropolis/README.md) | Beam shift vs random-focus? | **Kill** | -0.36 CE; train-greedy overfits |
| [4.4 — Verifier Metro](Experiment%204.4%20-%20Verifier%20Metropolis/README.md) | Verifier failure energy? | **Kill** | 0/3; CE −0.71 vs random-focus; ~4.5 h |
| [4.6 — FF Lite](Experiment%204.6%20-%20Forward%20Forward%20Lite/README.md) | Forward-Forward goodness? | **Kill (CPU)** | 0/3 vs random-focus |
| [4.7 — CEM](Experiment%204.7%20-%20Cross%20Entropy%20Method/README.md) | CEM vs AdamW? | **Kill** | 0/3 seeds; mean +8.55 CE vs AdamW |
| [4.8 — ES](Experiment%204.8%20-%20Evolution%20Strategies/README.md) | ES vs AdamW? | **Kill** | 0/3 seeds; mean +8.94 CE vs AdamW |
| [4.9 — LGL](Experiment%204.9%20-%20Layer%20Wise%20LGL/README.md) | InfoPro layer-wise local CE vs AdamW? | **Kill** | 0/3; mean +9.57 CE vs AdamW |
| [4.10 — EP](Experiment%204.10%20-%20Equilibrium%20Propagation/README.md) | EP BOP ternary vs AdamW? | **Kill** | 0/3; mean +9.40 CE; **0 flips** |
| [4.11 — PC](Experiment%204.11%20-%20Predictive%20Coding%20Gibbs/README.md) | Local PC Gibbs vs AdamW? | **Kill** | 0/3; mean +9.71 CE; ~250 accepts/seed |
| [4.12 — RPF](Experiment%204.12%20-%20RPF%20Ternary%20Greedy/README.md) | RPF full-model greedy vs AdamW? | **Kill** | 0/3; mean +9.10 CE; ~240 flips/seed |
| [4.13 — Local LM](Experiment%204.13%20-%20Layerwise%20Local%20LM/README.md) | Layer-wise local LM vs AdamW? | **Kill** | 0/3; mean +9.72 CE; 1 block only |

**Historical anchor:** random-focus Metropolis (~+0.07 CE vs global blind). AdamW still
leads Metropolis by ~-2.44 CE (Exp 3). DFO must close gap vs AdamW, not just Metropolis.

## DFO backlog

| Method | Lane ID | Status |
|--------|---------|--------|
| Cross-Entropy Method | **4.7** | **Kill** (CUDA) |
| Evolution Strategies | **4.8** | **Kill** (CUDA) |
| Layer-wise LGL (InfoPro) | **4.9** | Implemented; CUDA queued |
| Equilibrium Propagation | **4.10** | **Kill** — 0 flips (BOP never fired) |
| Predictive Coding Gibbs | **4.11** | **Kill** — local target ≠ LM CE |
| Simulated Bifurcation | 4.12 | Not implemented |
| Direct Feedback Alignment | — | Needs shadow weights |
| Layer-wise HSIC / local head | — | Local credit assignment |

## Commands

CPU smoke (any experiment runner supports `--device cpu` + tiny `--K` / `--cycles`):

```powershell
python -m pytest tests/test_sandbox_blume_capel_exp*.py -q
```

CUDA via serialize-only queue:

```powershell
python "experiments/Sandbox - Blume-Capel Ternary Search/gpu_queue.py" status
python "experiments/Sandbox - Blume-Capel Ternary Search/gpu_queue.py" run-next
```

## Files

| Path | Role |
|------|------|
| `gpu_queue.py` / `gpu_queue.json` | Serialize-only job queue + agent goal backlog |
| `sandbox_cem.py` | CEM helpers (Exp 4.7) |
| `sandbox_es.py` | ES helpers (Exp 4.8) |
| `sandbox_lgl.py` | LGL / InfoPro helpers (Exp 4.9) |
| `sandbox_ep.py` | Equilibrium Propagation helpers (Exp 4.10) |
| `sandbox_pc.py` | Predictive Coding Gibbs helpers (Exp 4.11) |
| `sandbox_dfo_common.py` | AdamW baseline loader + promote gate |
| `sandbox_lgl.py` | Layer-wise InfoPro/LGL local CEM (Exp 4.9) |
| `sandbox_ep.py` | Equilibrium Propagation + BOP flips (Exp 4.10) |
| `sandbox_pc.py` | Predictive coding local MH (Exp 4.11) |
| `sandbox_rpf.py` | Random projection feedback (Exp 4.12) |
| `sandbox_layerwise_lm.py` | Low-rank local LM heads (Exp 4.13) |
| `report.json` (root) | Exp 1 CPU smoke aggregate |
