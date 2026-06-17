# Exp84 Attractor Logic Recurrence

- report json: `C:\Users\Dos\Documents\GRAM\BitNet-HRM\artifacts\exp84_cmm_logic\report_full.json`
- mode: `full`
- device: `cuda`
- steps: `1`
- train source: `C:\Users\Dos\Documents\GRAM\BitNet-HRM\experiments\Experiment 70 - Comparative Logic Corpus\train_30k_sft.jsonl`
- eval source: `C:\Users\Dos\Documents\GRAM\BitNet-HRM\experiments\Experiment 70 - Comparative Logic Corpus\heldout_hard_1k.jsonl`
- train n: `30000`
- eval n: `200`
- control recipe: `paper`
- deep cycles: `l6`
- auto resume: `True`
- checkpoint every steps: `1`
- progress jsonl: `C:\Users\Dos\Documents\GRAM\BitNet-HRM\artifacts\exp84_cmm_logic\progress_full.jsonl`
- elapsed_s: `1145.0`

## seed 1
- arm=base_recipe cmm=False backbone=trm block=mlp_mixer depth=shallow H=2 L=2 loss=stablemax3 opt=adam_atan2 b=2x1 n_super=16 n_accum=16 halt_bce=True:0.5 amp=True alggradnorm=False pass@1=0.000 elapsed_s=157.7 peak_vram_mb=2111.5
- arm=base_recipe cmm=False backbone=trm block=mlp_mixer depth=deep H=2 L=6 loss=stablemax3 opt=adam_atan2 b=2x1 n_super=16 n_accum=16 halt_bce=True:0.5 amp=True alggradnorm=False pass@1=0.000 elapsed_s=309.4 peak_vram_mb=2112.2
- arm=cmm cmm=True backbone=trm block=mlp_mixer depth=shallow H=2 L=2 loss=stablemax3 opt=adam_atan2 b=2x1 n_super=16 n_accum=16 halt_bce=True:0.5 amp=True alggradnorm=True pass@1=0.000 elapsed_s=256.4 peak_vram_mb=2248.7
- arm=cmm cmm=True backbone=trm block=mlp_mixer depth=deep H=2 L=6 loss=stablemax3 opt=adam_atan2 b=2x1 n_super=16 n_accum=16 halt_bce=True:0.5 amp=True alggradnorm=True pass@1=0.000 elapsed_s=419.6 peak_vram_mb=2249.4
