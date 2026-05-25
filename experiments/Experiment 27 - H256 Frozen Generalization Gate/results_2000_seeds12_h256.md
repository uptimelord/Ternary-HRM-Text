# Experiment 27 live results

steps=2000, hidden_size=256, seeds=[1, 2], variants=['combo_baseline', 'combo_2bit_attention']
frozen_path=C:/Users/Dos/Documents/GRAM/BitNet-HRM/evaluation/frozen/frozen_arithmetic_200.jsonl
noise_floor=0.0203

| variant | seed | eval | eval_gap | frozen_loss | frozen_gap | frozen_token_acc | frozen_exact_acc | quality_per_mb | packed_MB | size_delta_MB | compr | tok/s |
|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| combo_baseline | 1 | 5.2270 | +0.0000 +/- 0.0203 (at noise floor) | 3.3578 | +0.0000 +/- 0.0203 (at noise floor) | 0.4061 | 0.0050 | 0.01384 | 13.82 | +0.00 | 5.46x | 2495 |
| combo_2bit_attention | 1 | 5.2405 | +0.0134 +/- 0.0203 (at noise floor) | 3.1190 | -0.2388 +/- 0.0203 (above noise floor) | 0.3908 | 0.0050 | 0.02085 | 9.15 | -4.67 | 8.25x | 1832 |
| combo_baseline | 2 | 5.1930 | +0.0000 +/- 0.0203 (at noise floor) | 3.1546 | +0.0000 +/- 0.0203 (at noise floor) | 0.4015 | 0.0050 | 0.01393 | 13.82 | +0.00 | 5.46x | 2020 |
| combo_2bit_attention | 2 | 5.2341 | +0.0410 +/- 0.0203 (above noise floor) | 3.1489 | -0.0057 +/- 0.0203 (at noise floor) | 0.4122 | 0.0150 | 0.02087 | 9.15 | -4.67 | 8.25x | 1465 |
