# Exp83 Tied Recursive Block

- train source: `C:\Users\Dos\Documents\GRAM\BitNet-HRM\experiments\Experiment 70 - Comparative Logic Corpus\train_30k_sft.jsonl`
- eval source: `C:\Users\Dos\Documents\GRAM\BitNet-HRM\experiments\Experiment 70 - Comparative Logic Corpus\heldout_hard_1k.jsonl`
- train n: `30000`
- eval n: `200`
- ternary body: `True`
- peak_vram_mb: `462.0`
- packed_exact: `True`
- verdict: `promote` (q/mb beats HRM and strict logic stays within 2 pp)

## seed 1
- hrm: params=17498112 body=720896 head=16777216 fp32_mb=66.76 packed_mb=66.28 q/mb=0.0058 q/body_mb=0.1696 pretrain_loss=2.5820 strict_pass@1=0.205
- trm: params=17137664 body=360448 head=16777216 fp32_mb=65.38 packed_mb=64.08 q/mb=0.0423 q/body_mb=33.8085 pretrain_loss=0.3692 strict_pass@1=0.190

## seed 2
- hrm: params=17498112 body=720896 head=16777216 fp32_mb=66.76 packed_mb=66.28 q/mb=0.0073 q/body_mb=0.2106 pretrain_loss=2.0791 strict_pass@1=0.180
- trm: params=17137664 body=360448 head=16777216 fp32_mb=65.38 packed_mb=64.08 q/mb=0.0421 q/body_mb=33.6548 pretrain_loss=0.3708 strict_pass@1=0.220
