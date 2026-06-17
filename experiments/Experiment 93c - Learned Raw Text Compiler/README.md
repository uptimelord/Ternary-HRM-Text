# Experiment 93c - Learned Raw Text Compiler

## Goal

Replace the Exp93b deterministic relation parser with a small learned compiler:

```text
raw prompt -> learned pairwise relation logits -> order solver -> verifier
```

This is the first learned "eyes" test. It still uses deterministic entity scanning and style detection, so it is not full TRM hidden-rule inference yet.

## Decision Rule

Promote if two seeds average at least 0.900 verified strict_pass@1 on 200 heldout raw prompts, with each seed at least 0.850.

Kill if either seed is below 0.750 verified strict_pass@1 or compile coverage is below 1.000.

## Run

```powershell
rtk python "experiments/Experiment 93c - Learned Raw Text Compiler/learned_raw_text_compiler.py" --steps 3000 --train-limit 1000 --eval-limit 200 --batch-size 64 --device cuda --output-dir "artifacts/exp93c_learned_raw_text_compiler"
```

Seed 2:

```powershell
rtk python "experiments/Experiment 93c - Learned Raw Text Compiler/learned_raw_text_compiler.py" --steps 3000 --train-limit 1000 --eval-limit 200 --batch-size 64 --device cuda --seed 2 --output-dir "artifacts/exp93c_learned_raw_text_compiler_seed2"
```

Smoke:

```powershell
rtk python "experiments/Experiment 93c - Learned Raw Text Compiler/learned_raw_text_compiler.py" --steps 10 --train-limit 64 --eval-limit 20 --batch-size 16 --device cuda --output-dir "artifacts/exp93c_learned_raw_text_compiler_smoke"
```

Exp83.1 TRM + mixed_top512_tequila, ternary pair head:

```powershell
rtk python "experiments/Experiment 93c - Learned Raw Text Compiler/learned_raw_text_compiler.py" --compiler-arch trm_tequila --pair-head ternary --loss-recipe pair_bce --train "experiments/Experiment 70 - Comparative Logic Corpus/train_30k_vgr.jsonl" --steps 3000 --train-limit 30000 --eval-limit 200 --batch-size 64 --width 64 --layers 2 --heads 4 --device cuda --output-dir "artifacts/exp93c_trm_tequila_train30k" --log-interval 300
```

Seed 2:

```powershell
rtk python "experiments/Experiment 93c - Learned Raw Text Compiler/learned_raw_text_compiler.py" --compiler-arch trm_tequila --pair-head ternary --loss-recipe pair_bce --train "experiments/Experiment 70 - Comparative Logic Corpus/train_30k_vgr.jsonl" --steps 3000 --train-limit 30000 --eval-limit 200 --batch-size 64 --width 64 --layers 2 --heads 4 --device cuda --seed 2 --output-dir "artifacts/exp93c_trm_tequila_train30k_seed2" --log-interval 300
```

Exp83.1 TRM + mixed_top512_tequila, dense pair head + permutation margin:

```powershell
rtk python "experiments/Experiment 93c - Learned Raw Text Compiler/learned_raw_text_compiler.py" --compiler-arch trm_tequila --pair-head dense --loss-recipe pair_bce_perm_margin --perm-margin-weight 0.2 --train "experiments/Experiment 70 - Comparative Logic Corpus/train_30k_vgr.jsonl" --steps 3000 --train-limit 30000 --eval-limit 200 --batch-size 64 --width 64 --layers 2 --heads 4 --device cuda --output-dir "artifacts/exp93c_trm_tequila_dense_perm_train30k" --log-interval 300
```

Seed 2:

```powershell
rtk python "experiments/Experiment 93c - Learned Raw Text Compiler/learned_raw_text_compiler.py" --compiler-arch trm_tequila --pair-head dense --loss-recipe pair_bce_perm_margin --perm-margin-weight 0.2 --train "experiments/Experiment 70 - Comparative Logic Corpus/train_30k_vgr.jsonl" --steps 3000 --train-limit 30000 --eval-limit 200 --batch-size 64 --width 64 --layers 2 --heads 4 --device cuda --seed 2 --output-dir "artifacts/exp93c_trm_tequila_dense_perm_train30k_seed2" --log-interval 300
```

## Outputs

- `report.json`
- `checkpoint.pt`
- `experiments/Experiment 93c - Learned Raw Text Compiler/results_seed<seed>.md`
- non-default arch uses `results_<compiler_arch>_seed<seed>.md`

## Results

1k seed 1:

- train source: `artifacts/exp93c_learned_raw_text_compiler/report.json`
- compile coverage: `1.000`
- verified strict_pass@1 after exact decode: `0.780`
- pair accuracy: `0.905`
- verdict: under data breadth; do not promote this arm

30k seed 1:

- compile coverage: `1.000`
- verified strict_pass@1: `1.000`
- pair accuracy: `0.934`
- params: `125633`
- fp32_mb: `0.48`
- peak_vram_mb: `46.6`
- elapsed_s: `259.4`
- report: `artifacts/exp93c_learned_raw_text_compiler_30k/report.json`

30k seed 2:

- compile coverage: `1.000`
- verified strict_pass@1: `1.000`
- pair accuracy: `0.935`
- params: `125633`
- fp32_mb: `0.48`
- peak_vram_mb: `45.2`
- elapsed_s: `255.4`
- report: `artifacts/exp93c_learned_raw_text_compiler_30k_seed2/report.json`

Verdict: promote the 30k learned compiler arm. The learned pair model only reaches about `0.934` pair accuracy, but exact permutation decode recovers `1.000` verified strict_pass@1 on both seeds.

Exp83.1 TRM + mixed_top512_tequila, 30k seed 1:

- compile coverage: `1.000`
- verified strict_pass@1: `0.815`
- pair accuracy: `0.913`
- params: `161793`
- ternary params: `158785`
- ternary param fraction: `0.981`
- fp32_mb: `0.62`
- packed_mb: `0.05`
- peak_vram_mb: `226.9`
- elapsed_s: `477.8`
- report: `artifacts/exp93c_trm_tequila_train30k/report.json`

Exp83.1 TRM + mixed_top512_tequila, 30k seed 2:

- compile coverage: `1.000`
- verified strict_pass@1: `0.880`
- pair accuracy: `0.927`
- params: `161793`
- ternary params: `158785`
- ternary param fraction: `0.981`
- fp32_mb: `0.62`
- packed_mb: `0.05`
- peak_vram_mb: `226.9`
- elapsed_s: `513.4`
- report: `artifacts/exp93c_trm_tequila_train30k_seed2/report.json`

Verdict: do not promote this arm. Average strict_pass@1 is `0.848`, and seed 1 misses the per-seed `0.850` floor. Storage is excellent, but accuracy trails the dense learned compiler.

Exp83.1 TRM + mixed_top512_tequila, dense pair head + permutation margin, 30k seed 1:

- compile coverage: `1.000`
- verified strict_pass@1: `0.910`
- pair accuracy: `0.933`
- params: `161793`
- ternary params: `142272`
- ternary param fraction: `0.879`
- fp32_mb: `0.62`
- packed_mb: `0.11`
- peak_vram_mb: `228.6`
- elapsed_s: `804.0`
- report: `artifacts/exp93c_trm_tequila_dense_perm_train30k/report.json`

Exp83.1 TRM + mixed_top512_tequila, dense pair head + permutation margin, 30k seed 2:

- compile coverage: `1.000`
- verified strict_pass@1: `0.795`
- pair accuracy: `0.909`
- params: `161793`
- ternary params: `142272`
- ternary param fraction: `0.879`
- fp32_mb: `0.62`
- packed_mb: `0.11`
- peak_vram_mb: `228.3`
- elapsed_s: `947.3`
- report: `artifacts/exp93c_trm_tequila_dense_perm_train30k_seed2/report.json`

Verdict: do not promote this arm. Average strict_pass@1 is `0.853`, and seed 2 misses the per-seed `0.850` floor. Dense head fixed seed 1, but the permutation-margin recipe is unstable across seeds.
