# Exp93e Multidomain Schema Compiler

- train source: `data\multidomain_schema\v2\train.jsonl`
- eval source: `data\multidomain_schema\v2\heldout.jsonl`
- train n: `32`
- eval n: `200`
- steps: `0`
- seed: `1`
- compiler_arch: `exp83_1_mixed_top512_tequila`
- backbone_recipe: `exp83_1_mixed_top512_tequila`
- width: `64`
- layers: `2`
- heads arg: `4`
- h_cycles: `2`
- l_cycles: `3`
- head_dense_k: `512`
- target_surface: `pointer`
- decode_repair: `input`
- vocab size: `79`
- eval domain counts: `{'comparative_order': 50, 'arithmetic': 50, 'maze': 50, 'logic_rules': 50}`
- format_valid@1: `1.000`
- json_valid@1: `1.000`
- domain_match@1: `1.000`
- schema_exact@1: `0.750`
- solver_verified@1: `1.000`
- repair_applied@1: `1.000`
- train loss: `0.0000`
- params: `149376`
- ternary params: `144320`
- ternary param fraction: `0.966`
- fp32_mb: `0.57`
- packed_mb: `0.06`
- packed exact: `True`
- peak_vram_mb: `0.0`
- elapsed_s: `0.0`
- checkpoint: `artifacts\exp93e_input_repair_v2_seed1\checkpoint.pt`

## Per Domain

| domain | format_valid@1 | schema_exact@1 | solver_verified@1 |
|---|---:|---:|---:|
| comparative_order | 1.000 | 0.000 | 1.000 |
| arithmetic | 1.000 | 1.000 | 1.000 |
| maze | 1.000 | 1.000 | 1.000 |
| logic_rules | 1.000 | 1.000 | 1.000 |

## Component Metrics

| component | pass@1 |
|---|---:|
| arithmetic.domain@1 | 1.000 |
| arithmetic.operands@1 | 1.000 |
| arithmetic.operator@1 | 1.000 |
| comparative_order.dimension@1 | 1.000 |
| comparative_order.domain@1 | 1.000 |
| comparative_order.objects_order@1 | 0.000 |
| comparative_order.objects_set@1 | 1.000 |
| comparative_order.query@1 | 1.000 |
| comparative_order.relations_order@1 | 1.000 |
| comparative_order.relations_set@1 | 1.000 |
| logic_rules.domain@1 | 1.000 |
| logic_rules.facts_set@1 | 1.000 |
| logic_rules.query@1 | 1.000 |
| logic_rules.rules_order@1 | 1.000 |
| logic_rules.rules_set@1 | 1.000 |
| maze.domain@1 | 1.000 |
| maze.goal@1 | 1.000 |
| maze.grid_ref@1 | 1.000 |
| maze.query@1 | 1.000 |
| maze.start@1 | 1.000 |

## Examples

- `mds_heldout_comparative_order_000000_1d66ff7c` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'solver_verified': True, 'repair_applied': True}`
  - target: `@comparative_order
dimension=wealth
query=full_order
object=$4
object=$3
object=$1
object=$5
object=$0
object=$2
rel=$0>$1
rel=$1>$2
rel=$2>$3
rel=$3>$4
rel=$5>$0`
  - pred: ``
- `mds_heldout_comparative_order_000001_abb549ba` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'solver_verified': True, 'repair_applied': True}`
  - target: `@comparative_order
dimension=score
query=full_order
object=$3
object=$4
object=$0
object=$2
object=$1
rel=$0>$1
rel=$2>$0
rel=$1>$3
rel=$3>$4`
  - pred: ``
- `mds_heldout_comparative_order_000002_4cc17ace` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'solver_verified': True, 'repair_applied': True}`
  - target: `@comparative_order
dimension=speed
query=argmax
object=$1
object=$2
object=$4
object=$3
object=$0
object=$5
rel=$0>$1
rel=$2>$3
rel=$4>$0
rel=$3>$4
rel=$1>$5`
  - pred: ``
- `mds_heldout_comparative_order_000003_41af5dad` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'solver_verified': True, 'repair_applied': True}`
  - target: `@comparative_order
dimension=height
query=full_order
object=$1
object=$4
object=$0
object=$3
object=$2
rel=$0>$1
rel=$1>$2
rel=$2>$3
rel=$3>$4`
  - pred: ``
- `mds_heldout_comparative_order_000004_fd1b3aca` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'solver_verified': True, 'repair_applied': True}`
  - target: `@comparative_order
dimension=speed
query=argmax
object=$4
object=$6
object=$5
object=$0
object=$2
object=$1
object=$3
rel=$0>$1
rel=$1>$2
rel=$2>$3
rel=$3>$4
rel=$4>$5
rel=$5>$6`
  - pred: ``
