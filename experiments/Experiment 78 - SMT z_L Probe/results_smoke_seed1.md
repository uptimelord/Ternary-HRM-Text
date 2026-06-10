# Experiment 78 - SMT z_L Probe

- mode: `eval`
- paper: [arxiv:2606.06479](https://arxiv.org/abs/2606.06479)
- rnn backbone: `transformer`
- memory tokens M: `8`

```json
{
  "mode": "eval",
  "paper": "arxiv:2606.06479",
  "lambda_dec": 1.0,
  "lambda_dyn": 0.1,
  "lambda_unif": 0.001,
  "rnn_backbone": "transformer",
  "n_memory": 8,
  "d_model": 128,
  "params": 1328128,
  "readout_ce_smt_dmt.pt": 9.320700109004974,
  "dmt_drift": {
    "l_dmt": 0.028713123872876167
  }
}
```
