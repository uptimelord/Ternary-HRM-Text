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
  "rnn_backbone": "transformer",
  "n_memory": 8,
  "d_model": 128,
  "vocab_size": 65530,
  "params": 34617344,
  "smt": {
    "steps": 600,
    "last": {
      "l_dec": 0.7902301549911499,
      "l_dyn": 0.0010213709902018309,
      "l_unif": -1.2506240606307983,
      "l_smt": 0.7890816926956177
    },
    "seconds": 28.296973999997135,
    "peak_vram_mb": 544.56005859375
  },
  "dmt": {
    "steps": 300,
    "last": {
      "l_dmt": 0.00591602548956871
    },
    "seconds": 157.07887209998444,
    "peak_vram_mb": 545.27490234375,
    "readout_steps": 150,
    "readout_last": {
      "l_readout": 6.453319072723389
    }
  },
  "smt_dmt_rollout_ce": 4.698316603899002,
  "bptt": {
    "steps": 1050,
    "last": {
      "l_bptt": 1.357557773590088
    },
    "seconds": 474.13151560002007,
    "peak_vram_mb": 839.45361328125
  },
  "bptt_rollout_ce": 1.9393358305096626,
  "fair_verdict": {
    "smt_dmt_wins": false,
    "rollout_ce_gap": -2.7589807733893394,
    "matched_opt_steps": 1050
  }
}
```
