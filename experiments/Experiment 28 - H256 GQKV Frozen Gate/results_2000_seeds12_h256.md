# Experiment 28 live results

steps=2000, hidden_size=256, seeds=[1, 2], variants=['combo_baseline', 'combo_2bit_attention_gqkv']
frozen_path=C:/Users/Dos/Documents/GRAM/BitNet-HRM/evaluation/frozen/frozen_arithmetic_200.jsonl
noise_floor=0.0203

| variant | seed | eval | eval_gap | frozen_loss | frozen_gap | frozen_token_acc | frozen_exact_acc | quality_per_mb | packed_MB | size_delta_MB | compr | tok/s |
|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| combo_baseline | 1 | 5.2270 | +0.0000 +/- 0.0203 (at noise floor) | 3.3578 | +0.0000 +/- 0.0203 (at noise floor) | 0.4061 | 0.0050 | 0.01384 | 13.82 | +0.00 | 5.46x | 1825 |
| combo_2bit_attention_gqkv | 1 | 5.2329 | +0.0059 +/- 0.0203 (at noise floor) | 3.9492 | +0.5914 +/- 0.0203 (above noise floor) | 0.2260 | 0.0000 | 0.01895 | 10.09 | -3.73 | 7.49x | 1641 |

## Early stop

Stopped after seed 1 because the pre-registered kill rule fired. Normal eval
looked fine, but frozen answer-loss gap was `+0.5914 +/- 0.0203`, far above the
noise floor. Seed 2 was not run because this candidate can no longer pass the
gate.

## Decision

FAIL - do not promote `combo_2bit_attention_gqkv`.
