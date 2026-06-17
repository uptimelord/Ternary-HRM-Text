# Sandbox Blume-Capel AdamW Gate Results

device=cuda, hidden_size=64, n_layers=2, steps=200, seeds=1,2,3
train_batches=16, eval_batches=16, adamw_lr=0.0001

| seed | init CE | metro CE | adamw CE | edge vs AdamW | metro delta | AdamW delta | metro s | AdamW s | AdamW steps | metro VRAM | AdamW VRAM | beat AdamW |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 11.6494 | 11.2853 | 9.4746 | -1.8106 | 0.3641 | 2.1748 | 12.13 | 12.14 | 87 | 1066.0 | 1737.7 | False |
| 2 | 11.2864 | 10.8967 | 8.5644 | -2.3323 | 0.3896 | 2.7219 | 17.91 | 18.05 | 116 | 1074.7 | 1738.2 | False |
| 3 | 11.5592 | 11.0950 | 7.9266 | -3.1685 | 0.4641 | 3.6326 | 18.23 | 18.49 | 141 | 1075.2 | 1738.8 | False |

## Summary

- metro_beats_adamw: 0/3
- success_rate: 0.0000
- mean_edge_vs_adamw: -2.4371
- mean_metro_delta: 0.4060
- mean_adamw_delta: 2.8431
- mean_vram_saving_mb: 666.2
