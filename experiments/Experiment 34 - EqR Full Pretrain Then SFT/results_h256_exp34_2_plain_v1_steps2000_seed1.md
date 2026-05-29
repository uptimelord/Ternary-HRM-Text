# Experiment 30 live result

base_checkpoint=artifacts\phase0_eqr_full\h256_exp34_eqr_d015_zl010_h246_bp4_steps50000_sft10000_seed1\pretrain\checkpoint_fp32.pt
train_jsonl=data\synthetic_arithmetic_reasoning\v1\train.jsonl
steps=2000, seed=1, batch_size=4, total_len=128
token_exposures=1,024,000

| metric | value |
|---|---:|
| valid_loss_before | 3.3021 |
| valid_loss_after | 0.2299 |
| valid_token_acc_after | 0.9156 |
| valid_exact_acc_after | 0.0078 |
| hard_export_valid_loss | 0.2264 |
| hard_export_gap | -0.0035 |
| last_train_loss | 0.2518 |
| last_train_token_acc | 0.9091 |
| last_train_exact_acc | 0.0000 |
| frozen_chain_generation_acc | nan |
| frozen_chain_generation_invalid | nan |
| packed_MB | 13.82 |
| peak_vram_MB | 739.4 |
| tok/s | 4588 |
| wall_time_min | 3.7 |

## Artifacts

- fp32 checkpoint: `artifacts\phase0_eqr_full\h256_exp34_2_eqrpretrain_plainv1_2000_plainv2_2000_then_eqr_d015_zl010_h246_bp4_steps10000_seed1\plain_v1_sft\checkpoint_fp32.pt`
- packed checkpoint: `artifacts\phase0_eqr_full\h256_exp34_2_eqrpretrain_plainv1_2000_plainv2_2000_then_eqr_d015_zl010_h246_bp4_steps10000_seed1\plain_v1_sft\checkpoint_packed.pt`
- metrics: `artifacts\phase0_eqr_full\h256_exp34_2_eqrpretrain_plainv1_2000_plainv2_2000_then_eqr_d015_zl010_h246_bp4_steps10000_seed1\plain_v1_sft\metrics.json`
