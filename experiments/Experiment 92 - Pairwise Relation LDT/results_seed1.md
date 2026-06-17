# Exp92 Pairwise Relation LDT

- train source: `experiments\Experiment 70 - Comparative Logic Corpus\train_1k_vgr_deepseek.jsonl`
- eval source: `experiments\Experiment 70 - Comparative Logic Corpus\heldout_hard_1k_vgr.jsonl`
- train n: `1000`
- eval n: `200`
- steps: `3000`
- internal iters: `4`
- threshold: `0.5`
- threshold warmup steps: `1000`
- lr: `0.0003`
- neural strict_pass@1: `0.810`
- constrained strict_pass@1: `1.000`
- train loss: `0.0000`
- params: `227618`
- fp32_mb: `0.87`
- peak_vram_mb: `152.5`
- checkpoint: `artifacts\exp90_1_lattice_seed1\checkpoint.pt`

## examples
- `cl_heldout_hard_000000` neural=True "Answer: Dan > Eve > Ben > Bob." constrained=True "Answer: Dan > Eve > Ben > Bob."
- `cl_heldout_hard_000001` neural=True "Answer: Mia > Sue > Zoe > Eve > Leo > Tom." constrained=True "Answer: Mia > Sue > Zoe > Eve > Leo > Tom."
- `cl_heldout_hard_000002` neural=True "Answer: Max." constrained=True "Answer: Max."
- `cl_heldout_hard_000003` neural=True "Answer: Sam." constrained=True "Answer: Sam."
- `cl_heldout_hard_000004` neural=True "Answer: Ann > Lily > Tom > Kai." constrained=True "Answer: Ann > Lily > Tom > Kai."
- `cl_heldout_hard_000005` neural=True "Answer: Sue > Joy > Mia > Zoe." constrained=True "Answer: Sue > Joy > Mia > Zoe."
- `cl_heldout_hard_000006` neural=True "Answer: Kai." constrained=True "Answer: Kai."
- `cl_heldout_hard_000007` neural=True "Answer: Lily." constrained=True "Answer: Lily."
- `cl_heldout_hard_000008` neural=True "Answer: Mia > Zoe > Tom > Ben > Joy." constrained=True "Answer: Mia > Zoe > Tom > Ben > Joy."
- `cl_heldout_hard_000009` neural=True "Answer: Dan." constrained=True "Answer: Dan."