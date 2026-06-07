# Experiment 30 live result

base_checkpoint=artifacts\phase0_eqr_full\h256_exp34_1_eqrpretrain_plain2000_then_eqr_d015_zl010_h246_bp4_steps10000_eval200_h246\checkpoint_fp32.pt
train_jsonl=data\exp66_word_reasoning_sft\v1\train.jsonl
steps=100, seed=1, batch_size=16, total_len=128
token_exposures=204,800

| metric | value |
|---|---:|
| valid_loss_before | 0.0000 |
| valid_loss_after | 0.0000 |
| valid_token_acc_after | 0.0000 |
| valid_exact_acc_after | 0.0000 |
| hard_export_valid_loss | 0.0000 |
| hard_export_gap | +0.0000 |
| last_train_loss | 0.1411 |
| last_train_token_acc | 0.9556 |
| last_train_exact_acc | 0.1875 |
| frozen_chain_generation_acc | nan |
| frozen_chain_generation_invalid | nan |
| packed_MB | 13.82 |
| peak_vram_MB | 2026.8 |
| tok/s | 4795 |
| wall_time_min | 0.7 |

## Artifacts

- fp32 checkpoint: `artifacts\_speedtest_amp\checkpoint_fp32.pt`
- packed checkpoint: `artifacts\_speedtest_amp\checkpoint_packed.pt`
- metrics: `artifacts\_speedtest_amp\metrics.json`
