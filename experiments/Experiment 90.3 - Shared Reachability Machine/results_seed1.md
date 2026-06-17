# Exp90.3 Shared Reachability Machine - Deep Resonance (seed 1)

- train: `C:\Users\Dos\Documents\GRAM\BitNet-HRM\data\multidomain_schema\v2\train.jsonl`
- eval: `C:\Users\Dos\Documents\GRAM\BitNet-HRM\data\multidomain_schema\v2\heldout.jsonl`
- train n: `40000` | steps: `8000` | conf: `2.0`
- **combined strict@1: `0.723`**
- comparative strict@1: `0.715` (n=200)
- logic strict@1: `0.730` (n=200)
- params: `8869249` | fp32_mb: `35.48` | peak_vram_mb: `2827.2`
- artifact: `artifacts\exp90_3_resonance_deep_seed1\report.json`

## verdict

Partial / continue. One seed clears both per-domain 0.70 floors but misses the combined 0.80 promote bar.
