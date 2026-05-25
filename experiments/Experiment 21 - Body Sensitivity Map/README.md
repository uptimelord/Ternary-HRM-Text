# Experiment 21 - Body Sensitivity Map

## Goal

Find which HRM body weights can tolerate ternary training before spending more
time on long body runs.

Experiment 16 showed `mlp_gate_up_tequila` can still look competitive at 5000
steps, but it barely moves packed size. This experiment maps the safer and
larger body targets separately.

## Variants

Variant names have this shape:

```text
dense
both_<target>
H_<target>
L_<target>
```

Supported targets come from the current transformer implementation:

```text
mlp_gate_up
mlp_down
mlp
attention_gqkv
attention_o
attention
mlp_no_down
attention_no_o
body
```

`H_` ternarizes only the slow HRM level. `L_` ternarizes only the fast HRM level.
`both_` ternarizes both levels.

## First Command

Cheap sensitivity map:

```powershell
rtk python -u "experiments/Experiment 21 - Body Sensitivity Map/body_sensitivity_map.py" `
  --steps 500 `
  --warmup-steps 2 `
  --seeds 1 `
  --device cuda `
  --append-md "experiments/Experiment 21 - Body Sensitivity Map/results_500.md"
```

## Read

Good candidate:

```text
gap <= +0.02 at 500 steps
packed size moves meaningfully below dense
tokens/sec is not catastrophic
```

If only tiny targets work, body ternary is a quality probe, not a deploy-size
win. If larger targets hold, run the best 2-3 lanes at 5000 steps.

## Results - 500 steps, seed 1

Command:

```powershell
rtk python -u "experiments/Experiment 21 - Body Sensitivity Map/body_sensitivity_map.py" `
  --steps 500 `
  --warmup-steps 2 `
  --seeds 1 `
  --device cuda `
  --append-md "experiments/Experiment 21 - Body Sensitivity Map/results_500.md"
```

Machine:

- NVIDIA GeForce RTX 3050 Ti Laptop GPU
- train tokens: 3,999,951
- eval tokens: 999,987
- body preset: `threshold=0.5`, `group_size=128`, `scale=mean_abs`, `ste=tequila`
- vocab: untied dense

| Variant | Eval | Gap vs dense | Ternary params | Packed size | Compression | Tok/s |
|---|---:|---:|---:|---:|---:|---:|
| `dense` | 6.0066 | +0.0000 | 0.0% | 66.76 MB | 1.00x | 1644 |
| `both_mlp_gate_up` | 5.9831 | -0.0235 | 1.5% | 65.81 MB | 1.01x | 1911 |
| `H_mlp_gate_up` | 6.0157 | +0.0091 | 0.7% | 66.28 MB | 1.01x | 1723 |
| `L_mlp_gate_up` | 5.9719 | **-0.0347** | 0.7% | 66.28 MB | 1.01x | 1299 |
| `both_mlp_down` | 6.0007 | -0.0059 | 0.7% | 66.28 MB | 1.01x | 1388 |
| `both_attention_o` | 6.0109 | +0.0043 | 0.4% | 66.52 MB | 1.00x | 1545 |
| `both_attention_gqkv` | 6.0325 | +0.0258 | 1.5% | 65.81 MB | 1.01x | 1564 |

Full live log: [`results_500.md`](results_500.md).

## Read

The best 500-step quality signal is `L_mlp_gate_up`, not H-level gate/up.

```text
L_mlp_gate_up      -0.0347
both_mlp_gate_up   -0.0235
H_mlp_gate_up      +0.0091
```

So the gate/up benefit appears to come mainly from the fast L-level. This is
useful because it gives a more specific body target than the earlier
`both_mlp_gate_up` lane.

The attention lanes are not attractive:

```text
both_attention_o      +0.0043
both_attention_gqkv   +0.0258
```

`attention_gqkv` is especially weak because it costs the same ternary share as
`both_mlp_gate_up` while producing worse loss.

## Decision

Promote `L_mlp_gate_up` to the next body-lane confirmation candidate.

Next long check:

```text
dense vs L_mlp_gate_up vs both_mlp_gate_up
5000 steps
seeds 2,3 or 1,2,3 if time allows
```

Do not spend more time on attention body ternary yet. It is not the blocker we
want.
