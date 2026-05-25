# Experiment 21 live results

steps=500, seeds=[1], variants=['dense', 'both_mlp_gate_up', 'H_mlp_gate_up', 'L_mlp_gate_up', 'both_mlp_down', 'both_attention_o', 'both_attention_gqkv']
body: threshold=0.5, group_size=128, ste=tequila
vocab: untied dense

| variant | seed | first_eval | final_eval | gap_vs_dense | last_train | params | tern% | packed_MB | compr | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 1 | 11.7466 | 6.0066 | +0.0000 | 6.3446 | 17,498,112 | 0.0% | 66.76 | 1.00x | 1644 |
| both_mlp_gate_up | 1 | 11.7749 | 5.9831 | -0.0235 | 6.4146 | 17,498,112 | 1.5% | 65.81 | 1.01x | 1911 |
| H_mlp_gate_up | 1 | 11.7382 | 6.0157 | +0.0091 | 6.3785 | 17,498,112 | 0.7% | 66.28 | 1.01x | 1723 |
| L_mlp_gate_up | 1 | 11.7449 | 5.9719 | -0.0347 | 6.3531 | 17,498,112 | 0.7% | 66.28 | 1.01x | 1299 |
| both_mlp_down | 1 | 11.7646 | 6.0007 | -0.0059 | 6.3870 | 17,498,112 | 0.7% | 66.28 | 1.01x | 1388 |
| both_attention_o | 1 | 11.7533 | 6.0109 | +0.0043 | 6.3400 | 17,498,112 | 0.4% | 66.52 | 1.00x | 1545 |
| both_attention_gqkv | 1 | 11.7384 | 6.0325 | +0.0258 | 6.3503 | 17,498,112 | 1.5% | 65.81 | 1.01x | 1564 |
