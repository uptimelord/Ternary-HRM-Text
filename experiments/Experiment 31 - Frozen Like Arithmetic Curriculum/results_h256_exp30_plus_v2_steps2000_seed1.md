# Experiment 30 live result

base_checkpoint=artifacts\phase0_arithmetic_sft_pilot\h256_steps2000_seed1\checkpoint_fp32.pt
train_jsonl=data\synthetic_arithmetic_reasoning\v2_frozen_like\train.jsonl
steps=2000, seed=1, batch_size=4, total_len=128
token_exposures=1,024,000

| metric | value |
|---|---:|
| valid_loss_before | 0.2489 |
| valid_loss_after | 0.1019 |
| valid_token_acc_after | 0.9616 |
| valid_exact_acc_after | 0.2656 |
| hard_export_valid_loss | 0.0989 |
| hard_export_gap | -0.0030 |
| last_train_loss | 0.1213 |
| last_train_token_acc | 0.9560 |
| last_train_exact_acc | 0.2500 |
| frozen_chain_generation_acc | 0.0800 |
| frozen_chain_generation_invalid | 0.0000 |
| packed_MB | 13.82 |
| peak_vram_MB | 739.4 |
| tok/s | 1111 |
| wall_time_min | 15.4 |

## Artifacts

- fp32 checkpoint: `artifacts\phase0_arithmetic_sft_v2\h256_exp30_plus_v2_steps2000_seed1\checkpoint_fp32.pt`
- packed checkpoint: `artifacts\phase0_arithmetic_sft_v2\h256_exp30_plus_v2_steps2000_seed1\checkpoint_packed.pt`
- metrics: `artifacts\phase0_arithmetic_sft_v2\h256_exp30_plus_v2_steps2000_seed1\metrics.json`
- generation examples: `artifacts\phase0_arithmetic_sft_v2\h256_exp30_plus_v2_steps2000_seed1\generation_examples.jsonl`
