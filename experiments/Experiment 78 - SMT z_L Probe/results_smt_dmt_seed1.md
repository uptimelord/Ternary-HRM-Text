# Experiment 78 - SMT z_L Probe

- mode: `smt_dmt`
- paper: [arxiv:2606.06479](https://arxiv.org/abs/2606.06479)
- rnn backbone: `transformer`
- memory tokens M: `8`

```json
{
  "mode": "smt_dmt",
  "paper": "arxiv:2606.06479",
  "lambda_dec": 1.0,
  "lambda_dyn": 0.1,
  "lambda_unif": 0.001,
  "rnn_backbone": "transformer",
  "n_memory": 8,
  "d_model": 128,
  "params": 1328128,
  "smt": {
    "steps": 300,
    "last": {
      "l_dec": 0.03131658583879471,
      "l_dyn": 0.0037329834885895252,
      "l_unif": -1.4014469385147095,
      "l_smt": 0.030288439244031906
    },
    "seconds": 9.744637400028296,
    "peak_vram_mb": 44.361328125
  },
  "dmt": {
    "steps": 150,
    "last": {
      "l_dmt": 0.028436005115509033
    },
    "seconds": 25.038503000047058,
    "peak_vram_mb": 58.033203125
  },
  "readout_ce": 9.404270887374878
}
```
