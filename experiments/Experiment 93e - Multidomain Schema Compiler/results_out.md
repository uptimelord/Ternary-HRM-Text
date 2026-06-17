# Exp93e Multidomain Schema Compiler

- train source: `C:\Users\Dos\AppData\Local\Temp\pytest-of-Dos\pytest-205\test_train_model_writes_report0\data\train.jsonl`
- eval source: `C:\Users\Dos\AppData\Local\Temp\pytest-of-Dos\pytest-205\test_train_model_writes_report0\data\heldout.jsonl`
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
- relation_rank_weight: `0.0`
- target_surface: `pointer`
- decode_repair: `none`
- vocab size: `70`
- eval domain counts: `{'comparative_order': 1, 'arithmetic': 1, 'maze': 1, 'logic_rules': 1}`
- format_valid@1: `0.000`
- json_valid@1: `0.000`
- domain_match@1: `0.000`
- schema_exact@1: `0.000`
- semantic_schema_exact@1: `0.000`
- raw_solver_verified@1: `0.000`
- repaired_solver_verified@1: `1.000`
- oracle_solver_verified@1: `1.000`
- solver_verified@1: `0.000` raw compatibility alias
- repair_applied@1: `0.000`
- train loss: `5.0923`
- params: `15808`
- ternary params: `0`
- ternary param fraction: `0.000`
- fp32_mb: `0.06`
- packed_mb: `0.06`
- packed exact: `False`
- peak_vram_mb: `0.0`
- elapsed_s: `0.1`
- checkpoint: `C:\Users\Dos\AppData\Local\Temp\pytest-of-Dos\pytest-205\test_train_model_writes_report0\out\checkpoint.pt`

## Per Domain

| domain | format_valid@1 | schema_exact@1 | semantic_schema_exact@1 | raw_solver_verified@1 | repaired_solver_verified@1 | oracle_solver_verified@1 |
|---|---:|---:|---:|---:|---:|---:|
| comparative_order | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| arithmetic | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| maze | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| logic_rules | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |

## Component Metrics

| component | pass@1 |
|---|---:|
| arithmetic.a@1 | 0.000 |
| arithmetic.b@1 | 0.000 |
| arithmetic.domain@1 | 0.000 |
| arithmetic.operands@1 | 0.000 |
| arithmetic.operator@1 | 0.000 |
| comparative_order.dimension@1 | 0.000 |
| comparative_order.domain@1 | 0.000 |
| comparative_order.objects_order@1 | 0.000 |
| comparative_order.objects_set@1 | 0.000 |
| comparative_order.query@1 | 0.000 |
| comparative_order.relations_order@1 | 0.000 |
| comparative_order.relations_set@1 | 0.000 |
| logic_rules.domain@1 | 0.000 |
| logic_rules.facts_set@1 | 0.000 |
| logic_rules.query@1 | 0.000 |
| logic_rules.rules_order@1 | 0.000 |
| logic_rules.rules_set@1 | 0.000 |
| maze.domain@1 | 0.000 |
| maze.goal@1 | 0.000 |
| maze.grid_ref@1 | 0.000 |
| maze.query@1 | 0.000 |
| maze.start@1 | 0.000 |

## Examples

- `mds_heldout_comparative_order_000000_b5a50ed1` `comparative_order` score=`{'format_valid': False, 'json_valid': False, 'domain_match': False, 'schema_exact': False, 'semantic_schema_exact': False, 'solver_verified': False, 'repair_applied': False, 'raw_solver_verified': False, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
  - target: `@comparative_order
dimension=score
query=full_order
object=$4
object=$2
object=$3
object=$0
object=$1
rel=$0>$1
rel=$1>$2
rel=$2>$3
rel=$3>$4`
  - pred: ``
- `mds_heldout_arithmetic_000000_8d21cf07` `arithmetic` score=`{'format_valid': False, 'json_valid': False, 'domain_match': False, 'schema_exact': False, 'semantic_schema_exact': False, 'solver_verified': False, 'repair_applied': False, 'raw_solver_verified': False, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
  - target: `@arithmetic
op=-
a=$0
b=$1`
  - pred: ``
- `mds_heldout_maze_000000_423b8880` `maze` score=`{'format_valid': False, 'json_valid': False, 'domain_match': False, 'schema_exact': False, 'semantic_schema_exact': False, 'solver_verified': False, 'repair_applied': False, 'raw_solver_verified': False, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
  - target: `@maze
grid_ref=input
start=S
goal=G
query=shortest_path_length`
  - pred: ``
- `mds_heldout_logic_rules_000000_52beed33` `logic_rules` score=`{'format_valid': False, 'json_valid': False, 'domain_match': False, 'schema_exact': False, 'semantic_schema_exact': False, 'solver_verified': False, 'repair_applied': False, 'raw_solver_verified': False, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
  - target: `@logic_rules
fact=$0
query=$1
rule=$1->$2
rule=$0->$1
rule=$3->$4
rule=$5->$6
rule=$4->$7
rule=$2->$5`
  - pred: ``
