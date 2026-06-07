# Experiment 30 live result

base_checkpoint=artifacts\phase0_eqr_full\h256_exp34_1_eqrpretrain_plain2000_then_eqr_d015_zl010_h246_bp4_steps10000_eval200_h246\checkpoint_fp32.pt
train_jsonl=data\exp66_word_reasoning_sft\v1\train.jsonl
steps=18000, seed=1, batch_size=16, total_len=128
token_exposures=36,864,000

| metric | value |
|---|---:|
| valid_loss_before | 1.7693 |
| valid_loss_after | 0.0124 |
| valid_token_acc_after | 0.9953 |
| valid_exact_acc_after | 0.9004 |
| hard_export_valid_loss | 0.0121 |
| hard_export_gap | -0.0003 |
| last_train_loss | 0.0035 |
| last_train_token_acc | 0.9982 |
| last_train_exact_acc | 0.9375 |
| frozen_chain_generation_acc | 0.7200 |
| frozen_chain_generation_invalid | 0.0000 |
| packed_MB | 13.82 |
| peak_vram_MB | 2221.3 |
| tok/s | 6387 |
| wall_time_min | 96.2 |

## Artifacts

- fp32 checkpoint: `artifacts\phase0_exp69_fullepoch\h256_word100k_b16_s18000_seed1\checkpoint_fp32.pt`
- packed checkpoint: `artifacts\phase0_exp69_fullepoch\h256_word100k_b16_s18000_seed1\checkpoint_packed.pt`
- metrics: `artifacts\phase0_exp69_fullepoch\h256_word100k_b16_s18000_seed1\metrics.json`
- generation examples: `artifacts\phase0_exp69_fullepoch\h256_word100k_b16_s18000_seed1\generation_examples.jsonl`
