# Experiment 34 - EqR Full Pretrain Then SFT

> **Status: Exp34.1 is the LOCKED Phase 0 candidate.** See [Locked Result](#locked-result)
> for the frozen eval200 / diagnostics, [Exp34.1 Bridge Test](#exp341-bridge-test) for the
> rationale, and [Command](#command) for the Exp34.1 run + diagnostic commands.

## Question

What happens when EqR-style recurrence dynamics are present from the beginning
of Phase 0 pretraining, instead of being added only during SFT?

This is the long counterpart to Experiment 33.

## Run Target

| Setting | Value |
|---|---:|
| base hidden size | 256 |
| params | ~19.79M |
| start point | fresh random init |
| pretrain steps | 50,000 |
| export calibration steps | 3,000 |
| SFT steps | 10,000 |
| pretrain tokens processed | 25.6M |
| export calibration tokens | 1.536M |
| SFT tokens processed | 5.12M |
| train H values | 2, 4, 6 |
| eval H values | 2, 4, 6 |
| pretrain bp steps | 4 fixed |
| SFT bp steps | 4 fixed |
| damping lambda | 0.15 |
| noise beta | 0.01 |
| RI zH std | 0.0 |
| RI zL std | 0.10 |

## Decision Rule

Promote if the final SFT checkpoint is stable at H=4 and H=6, and frozen
generation does not collapse versus H=2.

Strong promote if frozen generation improves as H increases.

Kill if full EqR pretraining produces worse H=2 behavior than Exp31 and still
collapses at H=4/H=6.

## Exp34.1 Bridge Test

Exp34 underperformed Exp33.5 after full EqR pretraining. The clean next test is
to add the plain arithmetic bridge that Exp33.5 had:

```text
Exp34 EqR pretrain -> Exp30-style plain arithmetic SFT -> Exp33.5-style EqR SFT
```

This does not rerun the 50k-token pretrain. It starts from:

```text
artifacts/phase0_eqr_full/h256_exp34_eqr_d015_zl010_h246_bp4_steps50000_sft10000_seed1/pretrain/checkpoint_fp32.pt
```

Promote Exp34.1 if the plain bridge recovers the raw arithmetic behavior and
the final EqR SFT closes most of the gap versus Exp33.5.

Kill if the plain bridge still leaves the raw model near zero or the final EqR
checkpoint remains clearly below Exp34.

## Exp34.2 Four-Stage Bridge Test

Exp34.2 keeps the locked Exp34.1 EqR settings, but strengthens the plain
arithmetic bridge before EqR SFT:

```text
Exp34 EqR pretrain -> plain v1 SFT -> plain v2 SFT -> EqR SFT
```

This isolates one question: does better raw arithmetic lift the balanced EqR
curve, without changing recurrence settings?

Promote Exp34.2 if H=2/H=4/H=6 stay balanced and move clearly above Exp34.1's
55-56% frozen eval200 band.

Kill if the extra plain v2 bridge improves raw arithmetic but breaks the stable
H=2/H=4/H=6 EqR curve.

Result: Exp34.2 stayed balanced, but did not clear the promote bar. Frozen
eval200 landed at H=2 54.5%, H=4 55.5%, H=6 54.5%, invalid 0.0%. Keep Exp34.1
locked.

## Command

Run in the background:

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File `
  "experiments/Experiment 34 - EqR Full Pretrain Then SFT/start_exp34_h256_eqr_full.ps1"
```

Run in the current terminal:

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File `
  "experiments/Experiment 34 - EqR Full Pretrain Then SFT/run_exp34_h256_eqr_full.ps1"
```

Run Exp34.1 in the background:

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File `
  "experiments/Experiment 34 - EqR Full Pretrain Then SFT/start_exp34_1_plain_sft_then_eqr_sft.ps1"
```

Run Exp34.1 eval200 after the SFT checkpoint is written:

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File `
  "experiments/Experiment 34 - EqR Full Pretrain Then SFT/start_exp34_1_eval200_h246.ps1"
```

Run the Exp34.1 lock diagnostics:

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File `
  "experiments/Experiment 34 - EqR Full Pretrain Then SFT/run_exp34_1_plain_eval200.ps1"

rtk powershell -NoProfile -ExecutionPolicy Bypass -File `
  "experiments/Experiment 34 - EqR Full Pretrain Then SFT/run_exp34_1_residual_h246.ps1"
```

Run Exp34.2 in the background:

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File `
  "experiments/Experiment 34 - EqR Full Pretrain Then SFT/start_exp34_2_plain_v1_v2_then_eqr_sft.ps1"
```

Run Exp34.2 scout eval50 after the SFT checkpoint is written:

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File `
  "experiments/Experiment 34 - EqR Full Pretrain Then SFT/start_exp34_2_eval50_h246.ps1"
```

Run Exp34.2 full eval200 only if the scout curve is worth locking:

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File `
  "experiments/Experiment 34 - EqR Full Pretrain Then SFT/start_exp34_2_eval200_h246.ps1"
```

## Artifacts

The runner writes:

```text
artifacts/phase0_eqr_full/h256_exp34_eqr_d015_zl010_h246_bp4_steps50000_sft10000_seed1/pretrain/checkpoint_fp32.pt
artifacts/phase0_eqr_full/h256_exp34_eqr_d015_zl010_h246_bp4_steps50000_sft10000_seed1/pretrain/checkpoint_packed.pt
artifacts/phase0_eqr_full/h256_exp34_eqr_d015_zl010_h246_bp4_steps50000_sft10000_seed1/pretrain/metrics.json
artifacts/phase0_eqr_full/h256_exp34_eqr_d015_zl010_h246_bp4_steps50000_sft10000_seed1/sft/checkpoint_fp32.pt
artifacts/phase0_eqr_full/h256_exp34_eqr_d015_zl010_h246_bp4_steps50000_sft10000_seed1/sft/checkpoint_packed.pt
artifacts/phase0_eqr_full/h256_exp34_eqr_d015_zl010_h246_bp4_steps50000_sft10000_seed1/sft/metrics.json
experiments/Experiment 34 - EqR Full Pretrain Then SFT/results_h256_exp34_eqr_d015_zl010_h246_bp4_steps50000_sft10000_seed1.md
```

## Notes

This is full Phase 0 pretraining with EqR dynamics, not the full EqR paper
stack. It does not yet include ACT, residual-based breadth selection, or learned
halting.

The RI is zL-only because this repo uses zH as the input-backed token embedding
state. Randomizing zH made Exp33 learn structured but incorrect arithmetic-like
outputs.

## Locked Result

Exp34.1 is the current locked Phase 0 candidate.

It restores the missing arithmetic bridge before EqR SFT:

```text
EqR pretrain -> plain arithmetic SFT -> EqR SFT
```

Frozen eval200:

| Run | H=2 | H=4 | H=6 |
|---|---:|---:|---:|
| Exp33.5 | 57.5% | 50.0% | 46.0% |
| Exp34 | 45.0% | 44.5% | 40.5% |
| Exp34.1 | 55.5% | 56.5% | 55.5% |
| Exp34.2 | 54.5% | 55.5% | 54.5% |

Lock diagnostics:

| Diagnostic | Result |
|---|---:|
| Exp34.1 plain/non-EqR frozen eval200 | 28.0% |
| Plain invalid rate | 0.0% |
| EqR residual H=2 | 20.4424 |
| EqR residual H=4 | 3.7097 |
| EqR residual H=6 | 2.4359 |

Interpretation: Exp34.1 is not just memorizing one recurrence depth. The frozen
curve is balanced across H=2/4/6, and the residual shrinks with more recurrence.
Plain mode is weaker than EqR mode, but it is no longer the near-zero raw model
seen in Exp34.
