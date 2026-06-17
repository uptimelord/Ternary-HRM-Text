# Exp93 Compiler Solver MoE

- eval source: `C:\Users\Dos\Documents\GRAM\BitNet-HRM\experiments\Experiment 70 - Comparative Logic Corpus\heldout_hard_1k_vgr.jsonl`
- eval n: `200`
- vgr route counts: `{"comparative_ldt": 200}`
- verified_n: `200`
- verified_pass@1: `1.000`
- unsupported probe n: `2`
- unsupported false LDT routes: `0`
- elapsed_s: `0.203`

## examples
- `cl_heldout_hard_000000` route=comparative_ldt verified=True gen="Answer: Dan > Eve > Ben > Bob."
- `cl_heldout_hard_000001` route=comparative_ldt verified=True gen="Answer: Mia > Sue > Zoe > Eve > Leo > Tom."
- `cl_heldout_hard_000002` route=comparative_ldt verified=True gen="Answer: Max."
- `cl_heldout_hard_000003` route=comparative_ldt verified=True gen="Answer: Sam."
- `cl_heldout_hard_000004` route=comparative_ldt verified=True gen="Answer: Ann > Lily > Tom > Kai."
- `cl_heldout_hard_000005` route=comparative_ldt verified=True gen="Answer: Sue > Joy > Mia > Zoe."
- `cl_heldout_hard_000006` route=comparative_ldt verified=True gen="Answer: Kai."
- `cl_heldout_hard_000007` route=comparative_ldt verified=True gen="Answer: Lily."
- `cl_heldout_hard_000008` route=comparative_ldt verified=True gen="Answer: Mia > Zoe > Tom > Ben > Joy."
- `cl_heldout_hard_000009` route=comparative_ldt verified=True gen="Answer: Dan."

## probes
- `raw_comparative_probe` route=trm_lm reason=raw_comparative_needs_schema_inference
- `free_text_probe` route=abstain reason=unsupported_or_low_confidence