# Exp93e Multidomain Schema Compiler

- train source: `C:\Users\Dos\AppData\Local\Temp\pytest-of-Dos\pytest-156\test_train_model_writes_report0\data\train.jsonl`
- eval source: `C:\Users\Dos\AppData\Local\Temp\pytest-of-Dos\pytest-156\test_train_model_writes_report0\data\heldout.jsonl`
- train n: `8`
- eval n: `4`
- steps: `1`
- seed: `1`
- compiler_arch: `exp83_1_mixed_top512_tequila`
- backbone_recipe: `exp83_1_mixed_top512_tequila`
- width: `16`
- layers: `1`
- heads arg: `2`
- h_cycles: `2`
- l_cycles: `3`
- head_dense_k: `512`
- target_surface: `pointer`
- vocab size: `64`
- eval domain counts: `{'comparative_order': 1, 'arithmetic': 1, 'maze': 1, 'logic_rules': 1}`
- format_valid@1: `0.000`
- json_valid@1: `0.000`
- domain_match@1: `0.000`
- schema_exact@1: `0.000`
- solver_verified@1: `0.000`
- train loss: `4.8670`
- params: `15616`
- ternary params: `14592`
- ternary param fraction: `0.934`
- fp32_mb: `0.06`
- packed_mb: `0.01`
- packed exact: `True`
- peak_vram_mb: `0.0`
- elapsed_s: `0.3`
- checkpoint: `C:\Users\Dos\AppData\Local\Temp\pytest-of-Dos\pytest-156\test_train_model_writes_report0\out\checkpoint.pt`

## Per Domain

| domain | format_valid@1 | schema_exact@1 | solver_verified@1 |
|---|---:|---:|---:|
| comparative_order | 0.000 | 0.000 | 0.000 |
| arithmetic | 0.000 | 0.000 | 0.000 |
| maze | 0.000 | 0.000 | 0.000 |
| logic_rules | 0.000 | 0.000 | 0.000 |

## Examples

- `mds_heldout_comparative_order_000000_ff9c0b16` `comparative_order` score=`{'format_valid': False, 'json_valid': False, 'domain_match': False, 'schema_exact': False, 'solver_verified': False}`
  - target: `@comparative_order
dimension=speed
query=full_order
object=$5
object=$2
object=$1
object=$4
object=$3
object=$0
rel=$0>$1
rel=$1>$2
rel=$2>$3
rel=$4>$0
rel=$3>$5`
  - pred: ``
- `mds_heldout_arithmetic_000000_cc68c77d` `arithmetic` score=`{'format_valid': False, 'json_valid': False, 'domain_match': False, 'schema_exact': False, 'solver_verified': False}`
  - target: `@arithmetic
op=-
a=$0
b=$1`
  - pred: ``
- `mds_heldout_maze_000000_c1ece5d9` `maze` score=`{'format_valid': False, 'json_valid': False, 'domain_match': False, 'schema_exact': False, 'solver_verified': False}`
  - target: `@maze
grid_ref=input
start=S
goal=G
query=shortest_path_length`
  - pred: ``
- `mds_heldout_logic_rules_000000_669d754d` `logic_rules` score=`{'format_valid': False, 'json_valid': False, 'domain_match': False, 'schema_exact': False, 'solver_verified': False}`
  - target: `@logic_rules
fact=$0
query=$2
rule=$1->$2
rule=$3->$4
rule=$5->$6
rule=$6->$7
rule=$2->$8
rule=$9->$3
rule=$8->$9
rule=$0->$1`
  - pred: ``
