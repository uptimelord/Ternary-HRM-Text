# Experiment 30 live result

base_checkpoint=artifacts\phase0_eqr_full\h256_exp34_1_eqrpretrain_plain2000_then_eqr_d015_zl010_h246_bp4_steps10000_seed1\eqr_sft\checkpoint_fp32.pt
train_jsonl=data\synthetic_arithmetic_reasoning\v2_frozen_like\train.jsonl
steps=0, seed=1, batch_size=4, total_len=128
token_exposures=0

| metric | value |
|---|---:|
| valid_loss_before | 0.1542 |
| valid_loss_after | 0.1542 |
| valid_token_acc_after | 0.9493 |
| valid_exact_acc_after | 0.0312 |
| hard_export_valid_loss | 0.1661 |
| hard_export_gap | +0.0118 |
| last_train_loss | 0.0000 |
| last_train_token_acc | 0.0000 |
| last_train_exact_acc | 0.0000 |
| frozen_chain_generation_acc | 0.2800 |
| frozen_chain_generation_invalid | 0.0000 |
| packed_MB | 13.82 |
| peak_vram_MB | 83.8 |
| tok/s | 0 |
| wall_time_min | 0.0 |

## Artifacts

- fp32 checkpoint: `artifacts\phase0_eqr_full\h256_exp34_1_plain_eval200\checkpoint_fp32.pt`
- packed checkpoint: `artifacts\phase0_eqr_full\h256_exp34_1_plain_eval200\checkpoint_packed.pt`
- metrics: `artifacts\phase0_eqr_full\h256_exp34_1_plain_eval200\metrics.json`
- generation examples: `artifacts\phase0_eqr_full\h256_exp34_1_plain_eval200\generation_examples.jsonl`
