# Experiment 30 live result

base_checkpoint=artifacts\phase0_eqr_lite_recurrence\h256_exp33_zlonly_steps2000_seed1\checkpoint_fp32.pt
train_jsonl=data\synthetic_arithmetic_reasoning\v2_frozen_like\train.jsonl
steps=0, seed=1, batch_size=4, total_len=128
token_exposures=0

| metric | value |
|---|---:|
| valid_loss_before | 0.2172 |
| valid_loss_after | 0.2172 |
| valid_token_acc_after | 0.9247 |
| valid_exact_acc_after | 0.0156 |
| hard_export_valid_loss | 0.2136 |
| hard_export_gap | -0.0036 |
| last_train_loss | 0.0000 |
| last_train_token_acc | 0.0000 |
| last_train_exact_acc | 0.0000 |
| frozen_chain_generation_acc | 0.0200 |
| frozen_chain_generation_invalid | 0.0000 |
| packed_MB | 13.82 |
| peak_vram_MB | 83.8 |
| tok/s | 0 |
| wall_time_min | 0.0 |

## Artifacts

- fp32 checkpoint: `artifacts\phase0_eqr_lite_recurrence\debug_exp33_checkpoint_plain_eval50\checkpoint_fp32.pt`
- packed checkpoint: `artifacts\phase0_eqr_lite_recurrence\debug_exp33_checkpoint_plain_eval50\checkpoint_packed.pt`
- metrics: `artifacts\phase0_eqr_lite_recurrence\debug_exp33_checkpoint_plain_eval50\metrics.json`
- generation examples: `artifacts\phase0_eqr_lite_recurrence\debug_exp33_checkpoint_plain_eval50\generation_examples.jsonl`
