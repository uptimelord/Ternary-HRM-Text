# Experiment 30 live result

base_checkpoint=artifacts\phase0_eqr_full\h256_exp34_eqr_d015_zl010_h246_bp4_steps50000_seed2\pretrain\checkpoint_fp32.pt
train_jsonl=data\synthetic_arithmetic_reasoning\v1\train.jsonl
steps=2000, seed=2, batch_size=4, total_len=128
token_exposures=1,024,000

| metric | value |
|---|---:|
| valid_loss_before | 3.2232 |
| valid_loss_after | 0.2425 |
| valid_token_acc_after | 0.9130 |
| valid_exact_acc_after | 0.0000 |
| hard_export_valid_loss | 0.2382 |
| hard_export_gap | -0.0043 |
| last_train_loss | 0.1957 |
| last_train_token_acc | 0.9296 |
| last_train_exact_acc | 0.0000 |
| frozen_chain_generation_acc | 0.0000 |
| frozen_chain_generation_invalid | 0.0000 |
| packed_MB | 13.82 |
| peak_vram_MB | 739.4 |
| tok/s | 6369 |
| wall_time_min | 2.7 |

## Artifacts

- fp32 checkpoint: `artifacts\phase0_eqr_full\h256_exp34_1_eqrpretrain_seed2_plain2000_then_eqr_d015_zl010_h246_bp4_steps10000_seed2\plain_sft\checkpoint_fp32.pt`
- packed checkpoint: `artifacts\phase0_eqr_full\h256_exp34_1_eqrpretrain_seed2_plain2000_then_eqr_d015_zl010_h246_bp4_steps10000_seed2\plain_sft\checkpoint_packed.pt`
- metrics: `artifacts\phase0_eqr_full\h256_exp34_1_eqrpretrain_seed2_plain2000_then_eqr_d015_zl010_h246_bp4_steps10000_seed2\plain_sft\metrics.json`
- generation examples: `artifacts\phase0_eqr_full\h256_exp34_1_eqrpretrain_seed2_plain2000_then_eqr_d015_zl010_h246_bp4_steps10000_seed2\plain_sft\generation_examples.jsonl`
