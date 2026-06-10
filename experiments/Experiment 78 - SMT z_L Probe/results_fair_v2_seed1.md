# Experiment 78 - SMT z_L Probe

- mode: `fair`
- paper: [arxiv:2606.06479](https://arxiv.org/abs/2606.06479)
- rnn backbone: `transformer`
- memory tokens M: `8`

```json
{
  "mode": "fair",
  "paper": "arxiv:2606.06479",
  "lambda_dec": 1.0,
  "lambda_dyn": 0.1,
  "lambda_unif": 0.001,
  "lambda_readout": 1.0,
  "rnn_backbone": "transformer",
  "n_memory": 8,
  "d_model": 128,
  "vocab_size": 65530,
  "params": 26229504,
  "smt": {
    "steps": 600,
    "last": {
      "l_dec": 0.8736370801925659,
      "l_dyn": 0.006583726964890957,
      "l_unif": -1.3646053075790405,
      "l_readout": 4.6906657218933105,
      "l_smt": 5.563596725463867
    },
    "seconds": 34.23519879998639,
    "peak_vram_mb": 516.62255859375
  },
  "dmt": {
    "steps": 300,
    "last": {
      "l_dmt": 0.1377987265586853
    },
    "seconds": 175.47161920001963,
    "peak_vram_mb": 385.27490234375,
    "readout_steps": 150,
    "readout_last": {
      "l_readout": 6.688621520996094
    }
  },
  "smt_dmt_rollout_ce": 4.117300987243652,
  "bptt": {
    "steps": 1050,
    "last": {
      "l_bptt": 1.3896055221557617
    },
    "seconds": 554.3020782000385,
    "peak_vram_mb": 647.45361328125
  },
  "bptt_rollout_ce": 1.916732132434845,
  "fair_verdict": {
    "smt_dmt_wins": false,
    "rollout_ce_gap": -2.2005688548088074,
    "matched_opt_steps": 1050
  }
}
```
