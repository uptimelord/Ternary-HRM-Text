# Experiment 29 live result

recipe=mixed_top512_tequila_L_mlp_gate_up
steps=0, seed=1, hidden_size=256
train_token_exposures=0
export_calibration_steps=3000, export_calibration_token_exposures=1,536,000
unique_tokens_total=4,999,938, train_split_tokens=3,999,951, eval_split_tokens=999,987

| metric | value |
|---|---:|
| first_eval | 3.9321 |
| final_eval | 3.9236 |
| hard_export_eval | 3.9468 |
| hard_export_gap | +0.0231 |
| last_train_loss | 2.3407 |
| params | 19,791,872 |
| ternary_pct | 87.4% |
| fp32_MB | 75.51 |
| packed_MB | 13.82 |
| compression | 5.46x |
| quality_per_mb | 0.01844 |
| peak_vram_MB | 813.8 |
| tok/s | 3872 |
| wall_time_min | 6.6 |
| frozen_loss | 2.2888 |
| frozen_token_acc | 0.3588 |
| frozen_exact_acc | 0.0150 |
| generation_acc | 0.0000 |
| generation_invalid | 0.0250 |

## Artifacts

- fp32 checkpoint: `artifacts\phase0_first_pretrain\h256_steps50000_seed1_exportcalib3000\checkpoint_fp32.pt`
- packed checkpoint: `artifacts\phase0_first_pretrain\h256_steps50000_seed1_exportcalib3000\checkpoint_packed.pt`
- metrics: `artifacts\phase0_first_pretrain\h256_steps50000_seed1_exportcalib3000\metrics.json`
- generation examples: `artifacts\phase0_first_pretrain\h256_steps50000_seed1_exportcalib3000\generation_examples.jsonl`
