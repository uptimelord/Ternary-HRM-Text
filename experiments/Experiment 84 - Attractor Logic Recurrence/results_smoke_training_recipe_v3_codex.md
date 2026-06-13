# Exp84 Attractor Logic Recurrence

- train source: `C:\Users\Dos\Documents\GRAM\BitNet-HRM\experiments\Experiment 70 - Comparative Logic Corpus\train_30k_sft.jsonl`
- eval source: `C:\Users\Dos\Documents\GRAM\BitNet-HRM\experiments\Experiment 70 - Comparative Logic Corpus\heldout_hard_1k.jsonl`
- train n: `32`
- eval n: `4`

## seed 1
- cmm=False backbone=trm block=mlp_mixer depth=shallow H=2 L=2 loss=stablemax3 opt=adam_atan2 b=2x1 alggradnorm=False pass@1=0.000
- cmm=False backbone=trm block=mlp_mixer depth=deep H=4 L=3 loss=stablemax3 opt=adam_atan2 b=2x1 alggradnorm=False pass@1=0.000
- cmm=True backbone=trm block=mlp_mixer depth=shallow H=2 L=2 loss=stablemax3 opt=adam_atan2 b=2x1 alggradnorm=True pass@1=0.000
- cmm=True backbone=trm block=mlp_mixer depth=deep H=4 L=3 loss=stablemax3 opt=adam_atan2 b=2x1 alggradnorm=True pass@1=0.000
