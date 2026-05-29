# Experiment 29 live result

recipe=mixed_top512_tequila_L_mlp_gate_up
steps=50000, seed=1, hidden_size=256
train_token_exposures=25,600,000
unique_tokens_total=4,999,938, train_split_tokens=3,999,951, eval_split_tokens=999,987

| metric | value |
|---|---:|
| first_eval | 11.9256 |
| final_eval | 3.9321 |
| hard_export_eval | 4.0664 |
| hard_export_gap | +0.1343 |
| last_train_loss | 3.3379 |
| params | 19,791,872 |
| ternary_pct | 87.4% |
| fp32_MB | 75.51 |
| packed_MB | 13.82 |
| compression | 5.46x |
| quality_per_mb | 0.01840 |
| peak_vram_MB | 953.8 |
| tok/s | 3313 |
| wall_time_min | 128.8 |
| frozen_loss | 2.1213 |
| frozen_token_acc | 0.3511 |
| frozen_exact_acc | 0.0000 |
| generation_acc | 0.0000 |
| generation_invalid | 0.0000 |

## Artifacts

- fp32 checkpoint: `artifacts\phase0_first_pretrain\h256_steps50000_seed1\checkpoint_fp32.pt`
- packed checkpoint: `artifacts\phase0_first_pretrain\h256_steps50000_seed1\checkpoint_packed.pt`
- metrics: `artifacts\phase0_first_pretrain\h256_steps50000_seed1\metrics.json`
- generation examples: `artifacts\phase0_first_pretrain\h256_steps50000_seed1\generation_examples.jsonl`
