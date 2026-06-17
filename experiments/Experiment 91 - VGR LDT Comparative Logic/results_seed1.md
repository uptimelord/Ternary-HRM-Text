# Exp91 VGR LDT Comparative Logic

- train source: `C:\Users\Dos\Documents\GRAM\BitNet-HRM\experiments\Experiment 70 - Comparative Logic Corpus\train_1k_vgr_deepseek.jsonl`
- eval source: `C:\Users\Dos\Documents\GRAM\BitNet-HRM\experiments\Experiment 70 - Comparative Logic Corpus\heldout_hard_1k_vgr.jsonl`
- train n: `64`
- eval n: `200`
- steps: `10`
- internal iters: `4`
- threshold: `0.5`
- threshold warmup steps: `10`
- lr: `0.0003`
- strict_pass@1: `1.000`
- train loss: `2.1913`
- params: `229831`
- fp32_mb: `0.88`
- peak_vram_mb: `31.0`
- checkpoint: `artifacts\exp91_vgr_ldt_constrained_smoke\checkpoint.pt`

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