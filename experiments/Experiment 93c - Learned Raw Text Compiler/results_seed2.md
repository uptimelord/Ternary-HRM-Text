# Exp93c Learned Raw Text Compiler

- train source: `experiments\Experiment 70 - Comparative Logic Corpus\train_30k_vgr.jsonl`
- eval source: `C:\Users\Dos\Documents\GRAM\BitNet-HRM\experiments\Experiment 70 - Comparative Logic Corpus\heldout_hard_1k_vgr.jsonl`
- train n: `30000`
- eval n: `200`
- steps: `3000`
- seed: `2`
- compiler_arch: `trm_tequila`
- backbone_recipe: `exp83_1_mixed_top512_tequila`
- head_dense_k: `512`
- vocab size: `47`
- compile coverage: `1.000`
- verified_pass@1: `0.880`
- pair accuracy: `0.927`
- train loss: `0.1157`
- params: `161793`
- ternary params: `158785`
- ternary param fraction: `0.981`
- fp32_mb: `0.62`
- packed_mb: `0.05`
- packed exact: `True`
- peak_vram_mb: `226.9`
- elapsed_s: `513.4`
- checkpoint: `artifacts\exp93c_trm_tequila_train30k_seed2\checkpoint.pt`

## examples
- `cl_heldout_hard_000000` pass=True gen="Answer: Dan > Eve > Ben > Bob."
- `cl_heldout_hard_000001` pass=True gen="Answer: Mia > Sue > Zoe > Eve > Leo > Tom."
- `cl_heldout_hard_000002` pass=True gen="Answer: Max."
- `cl_heldout_hard_000003` pass=True gen="Answer: Sam."
- `cl_heldout_hard_000004` pass=True gen="Answer: Ann > Lily > Tom > Kai."
- `cl_heldout_hard_000005` pass=True gen="Answer: Sue > Joy > Mia > Zoe."
- `cl_heldout_hard_000006` pass=True gen="Answer: Kai."
- `cl_heldout_hard_000007` pass=True gen="Answer: Lily."
- `cl_heldout_hard_000008` pass=True gen="Answer: Mia > Zoe > Tom > Ben > Joy."
- `cl_heldout_hard_000009` pass=True gen="Answer: Dan."