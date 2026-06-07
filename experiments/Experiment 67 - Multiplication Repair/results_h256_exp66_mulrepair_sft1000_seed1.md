# Experiment 30 live result

base_checkpoint=artifacts\phase0_exp66_word_reasoning\h256_exp34_1_word100k_sft2000_seed1\checkpoint_fp32.pt
train_jsonl=data\exp67_mul_repair_sft\v1\train.jsonl
steps=1000, seed=1, batch_size=4, total_len=128
token_exposures=512,000

| metric | value |
|---|---:|
| valid_loss_before | 0.7532 |
| valid_loss_after | 0.2338 |
| valid_token_acc_after | 0.9167 |
| valid_exact_acc_after | 0.0547 |
| hard_export_valid_loss | 0.2294 |
| hard_export_gap | -0.0043 |
| last_train_loss | 0.2508 |
| last_train_token_acc | 0.8962 |
| last_train_exact_acc | 0.0000 |
| frozen_chain_generation_acc | 0.0700 |
| frozen_chain_generation_invalid | 0.0000 |
| packed_MB | 13.82 |
| peak_vram_MB | 788.5 |
| tok/s | 2801 |
| wall_time_min | 3.0 |

## Artifacts

- fp32 checkpoint: `artifacts\phase0_exp67_mul_repair\h256_exp66_word_sft2000_mulrepair_sft1000_seed1\checkpoint_fp32.pt`
- packed checkpoint: `artifacts\phase0_exp67_mul_repair\h256_exp66_word_sft2000_mulrepair_sft1000_seed1\checkpoint_packed.pt`
- metrics: `artifacts\phase0_exp67_mul_repair\h256_exp66_word_sft2000_mulrepair_sft1000_seed1\metrics.json`
- generation examples: `artifacts\phase0_exp67_mul_repair\h256_exp66_word_sft2000_mulrepair_sft1000_seed1\generation_examples.jsonl`
