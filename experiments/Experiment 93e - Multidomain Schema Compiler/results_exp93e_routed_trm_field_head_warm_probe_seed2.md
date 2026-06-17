# Exp93e Multidomain Schema Compiler

- train source: `data\multidomain_schema\v2\train.jsonl`
- eval source: `data\multidomain_schema\v2\heldout.jsonl`
- train n: `20000`
- eval n: `200`
- steps: `300`
- seed: `2`
- compiler_arch: `routed_trm_field_head`
- backbone_recipe: `routed_trm_field_head`
- width: `64`
- layers: `2`
- heads arg: `4`
- h_cycles: `2`
- l_cycles: `3`
- head_dense_k: `512`
- relation_rank_weight: `0.0`
- target_surface: `pointer`
- decode_repair: `none`
- vocab size: `79`
- eval domain counts: `{'comparative_order': 50, 'arithmetic': 50, 'maze': 50, 'logic_rules': 50}`
- format_valid@1: `1.000`
- json_valid@1: `1.000`
- domain_match@1: `1.000`
- schema_exact@1: `0.500`
- semantic_schema_exact@1: `1.000`
- raw_solver_verified@1: `1.000`
- repaired_solver_verified@1: `1.000`
- oracle_solver_verified@1: `1.000`
- solver_verified@1: `1.000` raw compatibility alias
- repair_applied@1: `0.000`
- train loss: `0.8533`
- params: `682711`
- ternary params: `557056`
- ternary param fraction: `0.816`
- fp32_mb: `2.60`
- packed_mb: `0.63`
- packed exact: `True`
- peak_vram_mb: `49.3`
- elapsed_s: `3.0`
- checkpoint: `artifacts\exp93e_routed_trm_field_head_warm_probe_seed2\checkpoint.pt`

## Per Domain

| domain | format_valid@1 | schema_exact@1 | semantic_schema_exact@1 | raw_solver_verified@1 | repaired_solver_verified@1 | oracle_solver_verified@1 |
|---|---:|---:|---:|---:|---:|---:|
| comparative_order | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| arithmetic | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| maze | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| logic_rules | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Component Metrics

| component | pass@1 |
|---|---:|
| arithmetic.a@1 | 1.000 |
| arithmetic.b@1 | 1.000 |
| arithmetic.domain@1 | 1.000 |
| arithmetic.operands@1 | 1.000 |
| arithmetic.operator@1 | 1.000 |
| comparative_order.dimension@1 | 1.000 |
| comparative_order.domain@1 | 1.000 |
| comparative_order.objects_order@1 | 0.000 |
| comparative_order.objects_set@1 | 1.000 |
| comparative_order.query@1 | 1.000 |
| comparative_order.relations_order@1 | 0.540 |
| comparative_order.relations_set@1 | 1.000 |
| logic_rules.domain@1 | 1.000 |
| logic_rules.facts_set@1 | 1.000 |
| logic_rules.query@1 | 1.000 |
| logic_rules.rules_order@1 | 0.000 |
| logic_rules.rules_set@1 | 1.000 |
| maze.domain@1 | 1.000 |
| maze.goal@1 | 1.000 |
| maze.grid_ref@1 | 1.000 |
| maze.query@1 | 1.000 |
| maze.start@1 | 1.000 |

## Examples

- `mds_heldout_comparative_order_000000_1d66ff7c` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'semantic_schema_exact': True, 'solver_verified': True, 'repair_applied': False, 'raw_solver_verified': True, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
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
  - pred: `@comparative_order
dimension=wealth
query=full_order
object=$0
object=$1
object=$2
object=$3
object=$4
object=$5
rel=$0>$1
rel=$1>$2
rel=$2>$3
rel=$3>$4
rel=$5>$0`
- `mds_heldout_comparative_order_000001_abb549ba` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'semantic_schema_exact': True, 'solver_verified': True, 'repair_applied': False, 'raw_solver_verified': True, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
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
  - pred: `@comparative_order
dimension=score
query=full_order
object=$0
object=$1
object=$2
object=$3
object=$4
rel=$0>$1
rel=$1>$3
rel=$2>$0
rel=$3>$4`
- `mds_heldout_comparative_order_000002_4cc17ace` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'semantic_schema_exact': True, 'solver_verified': True, 'repair_applied': False, 'raw_solver_verified': True, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
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
  - pred: `@comparative_order
dimension=speed
query=argmax
object=$0
object=$1
object=$2
object=$3
object=$4
object=$5
rel=$0>$1
rel=$1>$5
rel=$2>$3
rel=$3>$4
rel=$4>$0`
- `mds_heldout_comparative_order_000003_41af5dad` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'semantic_schema_exact': True, 'solver_verified': True, 'repair_applied': False, 'raw_solver_verified': True, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
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
  - pred: `@comparative_order
dimension=height
query=full_order
object=$0
object=$1
object=$2
object=$3
object=$4
rel=$0>$1
rel=$1>$2
rel=$2>$3
rel=$3>$4`
- `mds_heldout_comparative_order_000004_fd1b3aca` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'semantic_schema_exact': True, 'solver_verified': True, 'repair_applied': False, 'raw_solver_verified': True, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
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
  - pred: `@comparative_order
dimension=speed
query=argmax
object=$0
object=$1
object=$2
object=$3
object=$4
object=$5
object=$6
rel=$0>$1
rel=$1>$2
rel=$2>$3
rel=$3>$4
rel=$4>$5
rel=$5>$6`
