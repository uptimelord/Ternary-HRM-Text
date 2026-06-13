# Exp83 Tied Recursive Block

- train source: `C:\Users\Dos\Documents\GRAM\BitNet-HRM\experiments\Experiment 70 - Comparative Logic Corpus\train_30k_sft.jsonl`
- eval source: `C:\Users\Dos\Documents\GRAM\BitNet-HRM\experiments\Experiment 70 - Comparative Logic Corpus\heldout_hard_1k.jsonl`
- train n: `40`
- eval n: `8`
- ternary body: `True`

## seed 1
- hrm: params=17498112 body=720896 head=16777216 fp32_mb=66.76 packed_mb=66.28 q/mb=0.0013 q/body_mb=0.0375 pretrain_loss=11.6792 logic_pass@1=0.000
- trm: params=17137664 body=360448 head=16777216 fp32_mb=65.38 packed_mb=64.08 q/mb=0.0013 q/body_mb=1.0742 pretrain_loss=11.6186 logic_pass@1=0.000
