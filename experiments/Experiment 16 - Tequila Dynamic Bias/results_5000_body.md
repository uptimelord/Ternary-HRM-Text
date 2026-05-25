# Experiment 16 live results

steps=5000, seeds=[1], variants=['dense', 'mlp_gate_up_tequila']
body: target=mlp_gate_up, threshold=0.5, group_size=128
vocab: mixed_top512, threshold=0.25, group_size=32, scale=mean_abs

| variant | seed | first_eval | final_eval | gap_vs_dense | last_train | params | tern% | packed_MB | compr | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
