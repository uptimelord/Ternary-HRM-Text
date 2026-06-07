# Experiment 30 live result

base_checkpoint=artifacts\phase0_eqr_full\h256_exp34_1_eqrpretrain_plain2000_then_eqr_d015_zl010_h246_bp4_steps10000_eval200_h246\checkpoint_fp32.pt
train_jsonl=data\exp66_word_reasoning_sft\v1\train.jsonl
steps=2000, seed=1, batch_size=4, total_len=128
token_exposures=1,024,000

| metric | value |
|---|---:|
| valid_loss_before | 1.7669 |
| valid_loss_after | 0.0643 |
| valid_token_acc_after | 0.9777 |
| valid_exact_acc_after | 0.4688 |
| hard_export_valid_loss | 0.0615 |
| hard_export_gap | -0.0029 |
| last_train_loss | 0.0318 |
| last_train_token_acc | 0.9855 |
| last_train_exact_acc | 0.5000 |
| frozen_chain_generation_acc | 0.4250 |
| frozen_chain_generation_invalid | 0.0000 |
| packed_MB | 13.82 |
| peak_vram_MB | 788.5 |
| tok/s | 3012 |
| wall_time_min | 5.7 |

## Artifacts

- fp32 checkpoint: `artifacts\phase0_exp66_word_reasoning\h256_exp34_1_word100k_sft2000_seed1\checkpoint_fp32.pt`
- packed checkpoint: `artifacts\phase0_exp66_word_reasoning\h256_exp34_1_word100k_sft2000_seed1\checkpoint_packed.pt`
- metrics: `artifacts\phase0_exp66_word_reasoning\h256_exp34_1_word100k_sft2000_seed1\metrics.json`
- generation examples: `artifacts\phase0_exp66_word_reasoning\h256_exp34_1_word100k_sft2000_seed1\generation_examples.jsonl`
