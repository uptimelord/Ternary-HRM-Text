# Exp93e Multidomain Schema Compiler

- train source: `artifacts\exp93e_messy_all_domains\train.jsonl`
- eval source: `artifacts\exp93e_messy_all_domains\held_out.jsonl`
- train n: `256`
- eval n: `40`
- steps: `5`
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
- eval domain counts: `{'comparative_order': 10, 'arithmetic': 10, 'maze': 10, 'logic_rules': 10}`
- format_valid@1: `1.000`
- json_valid@1: `1.000`
- domain_match@1: `0.475`
- schema_exact@1: `0.250`
- semantic_schema_exact@1: `0.250`
- raw_solver_verified@1: `0.250`
- repaired_solver_verified@1: `0.250`
- oracle_solver_verified@1: `1.000`
- solver_verified@1: `0.250` raw compatibility alias
- repair_applied@1: `0.000`
- train loss: `5.7803`
- params: `683991`
- ternary params: `557056`
- ternary param fraction: `0.814`
- fp32_mb: `2.61`
- packed_mb: `0.64`
- packed exact: `True`
- peak_vram_mb: `271.4`
- elapsed_s: `2.8`
- checkpoint: `artifacts\exp93e_routed_trm_field_head_messy_all_smoke\checkpoint.pt`

## Per Domain

| domain | format_valid@1 | schema_exact@1 | semantic_schema_exact@1 | raw_solver_verified@1 | repaired_solver_verified@1 | oracle_solver_verified@1 |
|---|---:|---:|---:|---:|---:|---:|
| comparative_order | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| arithmetic | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| maze | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| logic_rules | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |

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
| logic_rules.domain@1 | 0.900 |
| logic_rules.facts_set@1 | 0.200 |
| logic_rules.query@1 | 0.000 |
| logic_rules.rules_order@1 | 0.000 |
| logic_rules.rules_set@1 | 0.000 |
| maze.domain@1 | 1.000 |
| maze.goal@1 | 1.000 |
| maze.grid_ref@1 | 1.000 |
| maze.query@1 | 1.000 |
| maze.start@1 | 1.000 |

## Examples

- `mds_messy_heldout_comparative_order_000000_1d66ff7c` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': False, 'schema_exact': False, 'semantic_schema_exact': False, 'solver_verified': False, 'repair_applied': False, 'raw_solver_verified': False, 'repaired_solver_verified': False, 'oracle_solver_verified': True}`
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
  - pred: `{"domain":"logic_rules"}`
- `mds_messy_heldout_comparative_order_000004_abb549ba` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': False, 'schema_exact': False, 'semantic_schema_exact': False, 'solver_verified': False, 'repair_applied': False, 'raw_solver_verified': False, 'repaired_solver_verified': False, 'oracle_solver_verified': True}`
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
  - pred: `{"domain":"logic_rules"}`
- `mds_messy_heldout_comparative_order_000008_4cc17ace` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': False, 'schema_exact': False, 'semantic_schema_exact': False, 'solver_verified': False, 'repair_applied': False, 'raw_solver_verified': False, 'repaired_solver_verified': False, 'oracle_solver_verified': True}`
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
  - pred: `{"domain":"logic_rules"}`
- `mds_messy_heldout_comparative_order_000012_41af5dad` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': False, 'schema_exact': False, 'semantic_schema_exact': False, 'solver_verified': False, 'repair_applied': False, 'raw_solver_verified': False, 'repaired_solver_verified': False, 'oracle_solver_verified': True}`
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
  - pred: `{"domain":"arithmetic"}`
- `mds_messy_heldout_comparative_order_000016_fd1b3aca` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': False, 'schema_exact': False, 'semantic_schema_exact': False, 'solver_verified': False, 'repair_applied': False, 'raw_solver_verified': False, 'repaired_solver_verified': False, 'oracle_solver_verified': True}`
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
  - pred: `{"domain":"logic_rules"}`
