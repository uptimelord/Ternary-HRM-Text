# Experiment 11 - Selective Body Ternary

## Goal

Find which body projections tolerate ternary weights.

Prior results:

```text
ternary_mlp    small gap
ternary_body   real gap
```

This experiment splits the body into smaller pieces instead of flipping the
whole body at once.

## Run

```bash
python "experiments/Experiment 11 - Selective Body Ternary/selective_body.py" \
  --steps 500 \
  --variants dense,mlp,mlp_gate_up,mlp_down,attention_gqkv,attention_o,mlp_no_down,attention_no_o \
  --device cuda
```

## Read Template

Safe candidates have a small quality gap and meaningful ternary params.

## Results

Run:

```bash
python "experiments/Experiment 11 - Selective Body Ternary/selective_body.py" \
  --steps 500 \
  --variants dense,mlp,mlp_gate_up,mlp_down,attention_gqkv,attention_o,mlp_no_down,attention_no_o \
  --device cuda
```

Machine:

- NVIDIA GeForce RTX 3050 Ti Laptop GPU
- train tokens: 3,999,951
- eval tokens: 999,987
- seed: 1
- body ternary preset: `threshold=0.5`, `group_size=128`

| Variant | Eval | Gap | Ternary params | Notes |
| --- | ---: | ---: | ---: | --- |
| `dense` | 6.0066 | 0.0000 | 0.00% | baseline |
| `mlp` | 6.0053 | -0.0013 | 2.25% | safe |
| `mlp_gate_up` | 5.9905 | -0.0161 | 1.50% | best selective body result |
| `mlp_down` | 6.0076 | +0.0010 | 0.75% | safe but small payoff |
| `attention_gqkv` | 6.0317 | +0.0251 | 1.50% | avoid for now |
| `attention_o` | 6.0087 | +0.0021 | 0.37% | safe but tiny payoff |
| `mlp_no_down` | 5.9905 | -0.0161 | 1.50% | same as `mlp_gate_up` |
| `attention_no_o` | 6.0317 | +0.0251 | 1.50% | same as `attention_gqkv` |

## Read

MLP ternary is safe at this scale. The best selective piece is
`mlp_gate_up`.

Attention QKV is the sensitive piece. It costs quality even though the ternary
fraction is not huge.

## Decision

If we add body ternary back into the main line, start with:

```text
target=mlp_gate_up
threshold=0.5
group_size=128
```

Do not ternarize attention QKV by default.
