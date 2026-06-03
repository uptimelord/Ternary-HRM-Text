# Phase 0 Forward-Only Latency Benchmark

| hsize | variant | mean ms/forward | std ms | tokens/s | slowdown vs dense | pass <= limit |
|---:|---|---:|---:|---:|---:|:--:|
| 128 | dense_tied_vocab | 36.436 | 6.498 | 14052 | 1.00 | yes |
| 128 | mixed_top512_tequila_L_mlp_gate_up | 42.092 | 3.578 | 12164 | 1.16 | yes |
| 256 | dense_tied_vocab | 46.724 | 11.444 | 10958 | 1.00 | yes |
| 256 | mixed_top512_tequila_L_mlp_gate_up | 56.411 | 3.882 | 9076 | 1.21 | yes |

Strict pass rule: every measured hidden-size cell slowdown <= 1.50x.
Strict result: PASS.
