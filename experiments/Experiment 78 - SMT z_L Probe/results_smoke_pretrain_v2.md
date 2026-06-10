# Experiment 78 - SMT z_L Probe

- mode: `smoke_pretrain`
- paper: [arxiv:2606.06479](https://arxiv.org/abs/2606.06479)
- rnn backbone: `transformer`
- memory tokens M: `8`

```json
{
  "mode": "smoke_pretrain",
  "paper": "arxiv:2606.06479",
  "lambda_dec": 1.0,
  "lambda_dyn": 0.1,
  "lambda_unif": 0.001,
  "lambda_readout": 1.0,
  "rnn_backbone": "transformer",
  "n_memory": 8,
  "d_model": 128,
  "vocab_size": 65509,
  "params": 26221440,
  "smoke_pretrain": {
    "rollout_ce_before": 11.330045104026794,
    "rollout_ce_after": 4.7717976570129395,
    "rollout_ce_delta": 6.558247447013855,
    "smt": {
      "steps": 100,
      "last": {
        "l_dec": 3.335800886154175,
        "l_dyn": 0.006686036475002766,
        "l_unif": -1.3816118240356445,
        "l_readout": 6.549323081970215,
        "l_smt": 9.884410858154297
      },
      "seconds": 5.619049400032964,
      "peak_vram_mb": 516.60693359375
    },
    "dmt": {
      "steps": 50,
      "last": {
        "l_dmt": 0.07575076818466187
      },
      "seconds": 27.011507800023537,
      "peak_vram_mb": 361.50146484375,
      "readout_steps": 50,
      "readout_last": {
        "l_readout": 5.866270065307617
      }
    },
    "seconds": 36.837499400018714,
    "peak_vram_mb": 361.50146484375,
    "pass": true
  }
}
```
