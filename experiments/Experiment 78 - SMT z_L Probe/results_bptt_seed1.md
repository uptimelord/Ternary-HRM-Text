# Experiment 78 - SMT z_L Probe

- mode: `bptt`
- paper: [arxiv:2606.06479](https://arxiv.org/abs/2606.06479)
- rnn backbone: `transformer`
- memory tokens M: `8`

```json
{
  "mode": "bptt",
  "paper": "arxiv:2606.06479",
  "lambda_dec": 1.0,
  "lambda_dyn": 0.1,
  "lambda_unif": 0.001,
  "rnn_backbone": "transformer",
  "n_memory": 8,
  "d_model": 128,
  "params": 1328128,
  "train": {
    "steps": 300,
    "last": {
      "l_bptt": 2.5147926807403564
    },
    "seconds": 72.51318169996375,
    "peak_vram_mb": 84.03515625
  },
  "readout_ce": 6.126566767692566
}
```
