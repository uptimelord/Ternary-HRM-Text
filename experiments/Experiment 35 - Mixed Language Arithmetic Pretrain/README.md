# Experiment 35 - Mixed Language Arithmetic Pretrain

> **Status: completed.** Exp35 improves arithmetic over Exp34.1, but does not
> solve language yet. Final frozen eval200: H=2 65.0%, H=4 67.0%, H=6 64.0%,
> invalid 0.0%. Language probes after mixed pretrain no longer collapse into
> pure `1:` repetition, but are still nonsensical; final arithmetic SFT pulls
> generations back toward arithmetic fragments.

## Question

Exp34.1 gave a stable arithmetic specialist, but the language probe collapsed
into repeated junk. Exp35 tests whether the same h256 EqR recipe can learn basic
language behavior by changing the pretraining diet before the arithmetic bridge.

## Recipe

```text
h256 EqR mixed pretrain from scratch
  -> plain arithmetic bridge on v1
  -> EqR arithmetic SFT on v2 frozen-like data
  -> eval200 H=2/4/6 + language probes
```

## Data Mix

The mixed token cache is built at:

```text
data/exp35_mixed_language_arithmetic/tokens_flat.npy
```

Default cache size is 8M unique tokens:

| Source | Target Share |
|---|---:|
| allenai/dolma3_dolmino_mix-10B-1025 text | 80% |
| synthetic arithmetic CoT v2 | 15% |
| synthetic arithmetic answer-only v2 | 5% |

The 50M-token run cycles over that cache, similar to Exp34 cycling over the
local 5M-token slice.

## Run

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File `
  "experiments/Experiment 35 - Mixed Language Arithmetic Pretrain/start_exp35_h256_mixed_full.ps1"
```

Direct runner:

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File `
  "experiments/Experiment 35 - Mixed Language Arithmetic Pretrain/run_exp35_h256_mixed_full.ps1"
```

## Decision Rule

Promote if the final EqR SFT checkpoint keeps frozen arithmetic balanced at
H=2/4/6, invalid rate stays at 0%, and language probes no longer collapse into
the old repeated `1:` pattern.

Kill if mixed pretraining damages the Exp34.1 recurrence shape, arithmetic falls
far below the Exp34.1/Exp34.2 55% band, or language probes still show obvious
repeated-token collapse.

## Results

Exp35 is a strong arithmetic result, but not a Phase 0 language solve.

| Run | H=2 | H=4 | H=6 | invalid |
|---|---:|---:|---:|---:|
| Exp34.1 locked | 55.5% | 56.5% | 55.5% | 0.0% |
| Exp35 mixed | 65.0% | 67.0% | 64.0% | 0.0% |

Pretrain loss by H stayed balanced after hard export:

| H | final loss | hard-export loss |
|---:|---:|---:|
| 2 | 5.0144 | 5.0617 |
| 4 | 4.9375 | 4.9812 |
| 6 | 4.9599 | 5.0061 |

Language probe read:

- After mixed pretrain: no `1: 1: 1` collapse, but outputs orbit repeated
  Dolmino phrases such as `"The New York Times"` and are not instruction-useful.
- After final EqR SFT: arithmetic improves, but language drifts back into
  number/step fragments.

Full result:

```text
experiments/Experiment 35 - Mixed Language Arithmetic Pretrain/results_h256_exp35_mixed50m_plain2000_eqr10000_seed1.md
```

## Expected Size / Runtime

| Item | Value |
|---|---:|
| params | ~19.79M |
| pretrain token exposures | 49,999,872 |
| plain bridge token exposures | 1,024,000 |
| EqR SFT token exposures | 5,120,000 |
| packed checkpoint | ~13.82 MB |

Based on Exp34 throughput, the full run should be roughly 4-5 hours including
data prep, pretrain, bridge, SFT, and eval.
