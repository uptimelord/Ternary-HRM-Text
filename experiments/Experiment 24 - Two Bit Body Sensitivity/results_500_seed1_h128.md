# Experiment 24 live results

steps=500, hidden_size=128, seeds=[1], variants=['dense', 'both_mlp_gate_up', 'H_mlp_gate_up', 'L_mlp_gate_up', 'both_mlp_down', 'both_attention_o', 'both_attention_gqkv']
body: 2-bit, threshold=1.0, group_size=128
noise_floor=0.0203
vocab: untied dense

| variant | seed | first_eval | final_eval | gap_vs_dense | noise_floor | gap_read | quality_per_mb | last_train | params | two_bit% | packed_MB | compr | tok/s |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 1 | 11.7466 | 6.0066 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.00249 | 6.3446 | 17,498,112 | 0.0% | 66.76 | 1.00x | 2158 |
| both_mlp_gate_up | 1 | 11.6482 | 5.9687 | -0.0380 | 0.0203 | -0.0380 +/- 0.0203 (above noise floor) | 0.00255 | 6.3160 | 17,498,112 | 1.5% | 65.82 | 1.01x | 1785 |
| H_mlp_gate_up | 1 | 11.5237 | 6.0360 | +0.0294 | 0.0203 | +0.0294 +/- 0.0203 (above noise floor) | 0.00250 | 6.3405 | 17,498,112 | 0.7% | 66.29 | 1.01x | 1978 |
| L_mlp_gate_up | 1 | 11.5601 | 6.0116 | +0.0050 | 0.0203 | +0.0050 +/- 0.0203 (at noise floor) | 0.00251 | 6.3088 | 17,498,112 | 0.7% | 66.29 | 1.01x | 1846 |
| both_mlp_down | 1 | 11.6037 | 6.0423 | +0.0357 | 0.0203 | +0.0357 +/- 0.0203 (above noise floor) | 0.00250 | 6.3718 | 17,498,112 | 0.7% | 66.29 | 1.01x | 1710 |
| both_attention_o | 1 | 11.5291 | 5.9570 | -0.0496 | 0.0203 | -0.0496 +/- 0.0203 (above noise floor) | 0.00252 | 6.3256 | 17,498,112 | 0.4% | 66.52 | 1.00x | 1736 |
| both_attention_gqkv | 1 | 11.6343 | 5.9406 | -0.0660 | 0.0203 | -0.0660 +/- 0.0203 (above noise floor) | 0.00256 | 6.2752 | 17,498,112 | 1.5% | 65.82 | 1.01x | 1716 |
