# Exp93e Multidomain Schema Compiler

- train source: `artifacts\exp93e_messy_all_domains\train.jsonl`
- eval source: `artifacts\exp93e_messy_all_domains\held_out.jsonl`
- train n: `20000`
- eval n: `200`
- steps: `300`
- seed: `1`
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
- vocab size: `83`
- eval domain counts: `{'comparative_order': 50, 'arithmetic': 50, 'maze': 50, 'logic_rules': 50}`
- format_valid@1: `1.000`
- json_valid@1: `1.000`
- domain_match@1: `1.000`
- schema_exact@1: `0.360`
- semantic_schema_exact@1: `0.360`
- raw_solver_verified@1: `0.495`
- repaired_solver_verified@1: `0.495`
- oracle_solver_verified@1: `1.000`
- solver_verified@1: `0.495` raw compatibility alias
- repair_applied@1: `0.000`
- train loss: `1.4431`
- params: `683991`
- ternary params: `557056`
- ternary param fraction: `0.814`
- fp32_mb: `2.61`
- packed_mb: `0.64`
- packed exact: `True`
- peak_vram_mb: `563.0`
- elapsed_s: `131.5`
- checkpoint: `artifacts\exp93e_routed_trm_field_head_messy_all_probe\checkpoint.pt`

## Per Domain

| domain | format_valid@1 | schema_exact@1 | semantic_schema_exact@1 | raw_solver_verified@1 | repaired_solver_verified@1 | oracle_solver_verified@1 |
|---|---:|---:|---:|---:|---:|---:|
| comparative_order | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| arithmetic | 1.000 | 0.440 | 0.440 | 0.440 | 0.440 | 1.000 |
| maze | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| logic_rules | 1.000 | 0.000 | 0.000 | 0.540 | 0.540 | 1.000 |

## Component Metrics

| component | pass@1 |
|---|---:|
| arithmetic.a@1 | 0.920 |
| arithmetic.b@1 | 0.920 |
| arithmetic.domain@1 | 1.000 |
| arithmetic.operands@1 | 0.920 |
| arithmetic.operator@1 | 0.460 |
| comparative_order.dimension@1 | 1.000 |
| comparative_order.domain@1 | 1.000 |
| comparative_order.objects_order@1 | 0.020 |
| comparative_order.objects_set@1 | 0.840 |
| comparative_order.query@1 | 1.000 |
| comparative_order.relations_order@1 | 0.000 |
| comparative_order.relations_set@1 | 0.000 |
| logic_rules.domain@1 | 1.000 |
| logic_rules.facts_set@1 | 1.000 |
| logic_rules.query@1 | 1.000 |
| logic_rules.rules_order@1 | 0.000 |
| logic_rules.rules_set@1 | 0.000 |
| maze.domain@1 | 1.000 |
| maze.goal@1 | 1.000 |
| maze.grid_ref@1 | 1.000 |
| maze.query@1 | 1.000 |
| maze.start@1 | 1.000 |

## Examples

- `mds_messy_heldout_comparative_order_000000_1d66ff7c` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'semantic_schema_exact': False, 'solver_verified': False, 'repair_applied': False, 'raw_solver_verified': False, 'repaired_solver_verified': False, 'oracle_solver_verified': True}`
  - target: `@comparative_order
dimension=wealth
query=full_order
object=$2
object=$1
object=$6
object=$4
object=$3
object=$5
rel=$3>$6
rel=$6>$5
rel=$5>$1
rel=$1>$2
rel=$4>$3`
  - pred: `@comparative_order
dimension=wealth
query=full_order
object=$1
object=$2
object=$3
object=$4
object=$5
object=$6
rel=$1>$2
rel=$1>$3
rel=$1>$4
rel=$1>$6
rel=$2>$1
rel=$2>$3
rel=$2>$4
rel=$3>$1
rel=$3>$2
rel=$3>$4
rel=$3>$5
rel=$3>$6
rel=$4>$1
rel=$4>$2
rel=$4>$3
rel=$4>$5
rel=$4>$6
rel=$5>$1
rel=$5>$3
rel=$5>$6
rel=$6>$1
rel=$6>$2
rel=$6>$3
rel=$6>$4
rel=$6>$5`
- `mds_messy_heldout_comparative_order_000004_abb549ba` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'semantic_schema_exact': False, 'solver_verified': False, 'repair_applied': False, 'raw_solver_verified': False, 'repaired_solver_verified': False, 'oracle_solver_verified': True}`
  - target: `@comparative_order
dimension=score
query=full_order
object=$1
object=$2
object=$3
object=$4
object=$5
rel=$3>$5
rel=$4>$3
rel=$5>$1
rel=$1>$2`
  - pred: `@comparative_order
dimension=score
query=full_order
object=$1
object=$2
object=$3
object=$4
object=$5
rel=$1>$2
rel=$1>$3
rel=$1>$4
rel=$1>$5
rel=$2>$3
rel=$2>$4
rel=$3>$4
rel=$3>$5
rel=$5>$3`
- `mds_messy_heldout_comparative_order_000008_4cc17ace` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'semantic_schema_exact': False, 'solver_verified': False, 'repair_applied': False, 'raw_solver_verified': False, 'repaired_solver_verified': False, 'oracle_solver_verified': True}`
  - target: `@comparative_order
dimension=speed
query=argmax
object=$2
object=$5
object=$3
object=$4
object=$1
object=$6
rel=$1>$2
rel=$5>$4
rel=$3>$1
rel=$4>$3
rel=$2>$6`
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
rel=$1>$2
rel=$1>$3
rel=$1>$4
rel=$2>$3
rel=$2>$4
rel=$2>$6
rel=$3>$1
rel=$3>$4
rel=$4>$2
rel=$4>$5
rel=$4>$6
rel=$5>$2
rel=$5>$6`
- `mds_messy_heldout_comparative_order_000012_41af5dad` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'semantic_schema_exact': False, 'solver_verified': False, 'repair_applied': False, 'raw_solver_verified': False, 'repaired_solver_verified': False, 'oracle_solver_verified': True}`
  - target: `@comparative_order
dimension=height
query=full_order
object=$0
object=$4
object=$2
object=$3
object=$1
rel=$2>$0
rel=$0>$1
rel=$1>$3
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
rel=$0>$2
rel=$1>$0
rel=$1>$2
rel=$1>$3
rel=$1>$4
rel=$3>$1
rel=$3>$4`
- `mds_messy_heldout_comparative_order_000016_fd1b3aca` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'semantic_schema_exact': False, 'solver_verified': False, 'repair_applied': False, 'raw_solver_verified': False, 'repaired_solver_verified': False, 'oracle_solver_verified': True}`
  - target: `@comparative_order
dimension=speed
query=argmax
object=$3
object=$5
object=$2
object=$4
object=$1
object=$0
object=$6
rel=$4>$0
rel=$0>$1
rel=$1>$6
rel=$6>$3
rel=$3>$2
rel=$2>$5`
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
rel=$0>$2
rel=$0>$3
rel=$0>$4
rel=$0>$5
rel=$1>$2
rel=$1>$3
rel=$1>$6
rel=$2>$3
rel=$2>$5
rel=$3>$6
rel=$4>$2
rel=$4>$5
rel=$6>$3`
