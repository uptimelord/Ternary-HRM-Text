# Experiment 77 - Ternary Flash Grid Micro (distil)

- checkpoint: `C:\Users\Dos\Documents\GRAM\BitNet-HRM\artifacts\exp76_smoke\plain_sft\checkpoint_fp32.pt`
- rank: `8`
- builder params: `1634560`
- inject scale: `0.25`

```json
{
  "mode": "distil",
  "checkpoint": "C:\\Users\\Dos\\Documents\\GRAM\\BitNet-HRM\\artifacts\\exp76_smoke\\plain_sft\\checkpoint_fp32.pt",
  "rank": 8,
  "builder_params": 1634560,
  "inject_scale": 0.25,
  "distil": {
    "train_last": {
      "loss": 1.8228650093078613,
      "token_acc": 0.491150438785553,
      "exact_acc": 0.0
    },
    "valid_baseline": {
      "loss": 2.0528028731231163,
      "token_acc": 0.4690011481056257,
      "exact_acc": 0.0,
      "tokens": 1742.0,
      "examples": 32.0
    },
    "valid_overlay": {
      "loss": 1.756677077914215,
      "token_acc": 0.5103329506314581,
      "exact_acc": 0.0,
      "tokens": 1742.0,
      "examples": 32.0
    },
    "peak_vram_mb": 590.751953125,
    "builder_params": 1634560
  }
}
```
