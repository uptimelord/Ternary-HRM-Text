# Exp90.3 Shared Reachability Machine - Solver-Assisted Diagnostic (seed 2)

- train: `C:\Users\Dos\Documents\GRAM\BitNet-HRM\data\multidomain_schema\v2\train.jsonl`
- eval: `C:\Users\Dos\Documents\GRAM\BitNet-HRM\data\multidomain_schema\v2\heldout.jsonl`
- train n: `40000` | steps: `4000` | conf: `2.0`
- **combined strict@1: `0.995`**
- comparative strict@1: `1.000` (n=200)
- logic strict@1: `0.990` (n=200)
- params: `8869249` | fp32_mb: `35.48` | peak_vram_mb: `627.4`
- artifact: `artifacts\exp90_3_seed2\report.json`

## verdict

Diagnostic only. This exact-closure solver lane proves the fixed-rule readout can work, but it is not the resonance headline.
