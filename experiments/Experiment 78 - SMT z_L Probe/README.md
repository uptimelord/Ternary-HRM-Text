# Experiment 78 — SMT z_L Probe

> **Status: smoke + fair comparison completed; do not promote SMT for deploy.**
> Paper-faithful SMT+DMT wiring works, but matched-step fair eval shows BPTT
> beats SMT+DMT on rollout readout CE.

Faithful probe of **Supervised Memory Training** (Kumar & Isola,
[arxiv:2606.06479](https://arxiv.org/abs/2606.06479)).

## Question

Does paper-faithful SMT (encoder + decoder + RNN memory transition, Eq. 2–6)
beat a matched-step BPTT baseline on readout CE for Phase-0-scale pretrain text —
and does DMT mitigate memory drift?

Paper memory `m_t` is **per-sequence state across token time**. HRM `z_L` today
is a **depth scratchpad** within one forward. Promoting SMT for deploy requires
a temporal z_L carry redesign, not just swapping losses on the current HRM loop.

## Method

Implements paper Equations 2–6 and Appendix B.1 architectures:

| Component | Paper | Code |
|-----------|-------|------|
| Encoder `E_phi` | Bidirectional Transformer + memory registers | `SMTEncoder` |
| Decoder `D_psi` | Prefix-LM over memory + future `x` | `SMTDecoder` |
| RNN `f_theta` | Transformer memory transition | `SMTRNNDynamics` |
| `L_dec` | Future CE (Eq. 2) | `decode_loss` |
| `L_dyn` | MSE one-step memory (Eq. 3) | `dynamics_loss` |
| `L_unif` | Uniformity anti-collapse (Eq. 4) | `uniformity_loss` |
| `L_smt` | Joint objective (Eq. 5) | `smt_loss` |
| DMT | On-policy `MSE(m_hat_t, m_t)` (Eq. 6) | `forward_dmt` |

Default hyperparameters match paper Appendix B.3 (scaled for 3050 Ti):
`lambda_dec=1.0`, `lambda_dyn=0.1`, `lambda_unif=0.001`, sampled timestep `t`
per step. Optional `--rnn-backbone hrm_l` uses frozen checkpoint `L_level` as
`f_theta` with `M=1`.

Runner: `smt_zl_probe.py`. Shared training helpers in `models/smt_memory_training.py`.

| Mode | Purpose |
|------|---------|
| `smoke` | Finite losses, SMT + DMT + BPTT wiring |
| `smt` | Joint SMT pretrain (Eq. 5) |
| `dmt` | DMT finetune — encoder/decoder frozen (Eq. 6) |
| `smt_dmt` | Full recipe: SMT then DMT |
| `bptt` | BPTT baseline (same RNN + readout) |
| `fair` | Matched optimizer steps: SMT+DMT vs BPTT |
| `eval` | Readout CE + rollout drift |

## Decision Rule

Promote if `fair` mode (or matched-step `smt_dmt` vs `bptt`) shows
`smt_dmt_rollout_ce` at least **0.5** below `bptt_rollout_ce` **and** `l_dmt`
drops versus SMT-only (drift mitigated), with smoke passing.

Kill if `l_dyn` stays flat and rollout readout CE ≥ BPTT at matched steps —
teacher memory labels do not help temporal memory on this corpus.

## Results

### Smoke + smoke_pretrain

`results_smoke_seed1.md` / `results_smoke_pretrain_v2.md`:

- `smoke.pass == true`, all losses finite.
- Pretrain v2: rollout CE **11.33 → 4.77** after short SMT+DMT (wiring sanity only).

### SMT → DMT (toy scale, seed 1)

`results_smt_dmt_seed1.md`, M=8, d_model=128, 300 SMT + 150 DMT steps:

| Phase | last loss | peak VRAM |
|---|---:|---:|
| SMT `l_smt` | 0.030 | 44.4 MB |
| DMT `l_dmt` | 0.028 | 58.0 MB |
| readout CE | 9.40 | — |

### BPTT baseline (toy scale)

`results_bptt_seed1.md`, 300 steps:

| Metric | Value |
|---|---:|
| `l_bptt` (last) | 2.515 |
| readout CE | **6.13** |
| peak VRAM | 84.0 MB |

BPTT readout CE beats toy-scale SMT+DMT on this checkpoint slice.

### Fair matched-step comparison (seed 1, pretrain corpus)

`results_fair_v2_seed1.md`, vocab 65530, params ~26.2M, matched opt steps 1050:

| Recipe | rollout readout CE | peak VRAM |
|---|---:|---:|
| **BPTT** | **1.917** | 647.5 MB |
| SMT+DMT | 4.117 | 516.6 MB (SMT) / 385.3 MB (DMT) |

Fair verdict (`fair_verdict`):

```text
smt_dmt_wins: false
rollout_ce_gap: -2.201  (BPTT better)
```

DMT `l_dmt` drops (0.138 last) versus SMT-only, but readout CE does not beat BPTT.

## Read

Wiring is faithful and tests pass. At matched training budget, **BPTT wins rollout
readout CE** on the fair pretrain slice. SMT memory training does not yet justify
replacing BPTT or redesigning HRM z_L for deploy. Keep as reference implementation;
next work needs harder temporal carry integration, not more loss tuning alone.

## Commands

```powershell
python -u "experiments/Experiment 78 - SMT z_L Probe/smt_zl_probe.py" --mode smoke --device cuda

python -u "experiments/Experiment 78 - SMT z_L Probe/smt_zl_probe.py" `
  --mode smt_dmt --smt-steps 300 --dmt-steps 150 --device cuda `
  --output-dir artifacts/exp78_smt_probe/smt_dmt_seed1 `
  --append-md "experiments/Experiment 78 - SMT z_L Probe/results_smt_dmt_seed1.md"

pytest tests/test_exp78_smt_zl_probe.py -q
```
