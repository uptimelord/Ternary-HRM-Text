# Exp93e Multidomain Schema Compiler

- train source: `artifacts\exp93e_domain_slices\comparative_order_train.jsonl`
- eval source: `artifacts\exp93e_domain_slices\comparative_order_heldout.jsonl`
- train n: `20000`
- eval n: `200`
- steps: `300`
- seed: `1`
- compiler_arch: `comparative_trm_field_head`
- backbone_recipe: `comparative_trm_field_head`
- width: `64`
- layers: `2`
- heads arg: `4`
- h_cycles: `2`
- l_cycles: `3`
- head_dense_k: `512`
- relation_rank_weight: `0.0`
- target_surface: `pointer`
- decode_repair: `none`
- vocab size: `57`
- eval domain counts: `{'comparative_order': 200, 'arithmetic': 0, 'maze': 0, 'logic_rules': 0}`
- format_valid@1: `1.000`
- json_valid@1: `1.000`
- domain_match@1: `1.000`
- schema_exact@1: `0.000`
- semantic_schema_exact@1: `0.000`
- raw_solver_verified@1: `0.150`
- repaired_solver_verified@1: `1.000`
- oracle_solver_verified@1: `1.000`
- solver_verified@1: `0.150` raw compatibility alias
- repair_applied@1: `0.000`
- train loss: `1.1451`
- params: `168201`
- ternary params: `139264`
- ternary param fraction: `0.828`
- fp32_mb: `0.64`
- packed_mb: `0.15`
- packed exact: `True`
- peak_vram_mb: `356.4`
- elapsed_s: `42.4`
- checkpoint: `artifacts\exp93e_comparative_trm_field_head_probe_default_after_rank\checkpoint.pt`

## Per Domain

| domain | format_valid@1 | schema_exact@1 | semantic_schema_exact@1 | raw_solver_verified@1 | repaired_solver_verified@1 | oracle_solver_verified@1 |
|---|---:|---:|---:|---:|---:|---:|
| comparative_order | 1.000 | 0.000 | 0.000 | 0.150 | 1.000 | 1.000 |
| arithmetic | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| maze | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| logic_rules | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Component Metrics

| component | pass@1 |
|---|---:|
| comparative_order.dimension@1 | 1.000 |
| comparative_order.domain@1 | 1.000 |
| comparative_order.objects_order@1 | 0.000 |
| comparative_order.objects_set@1 | 1.000 |
| comparative_order.query@1 | 1.000 |
| comparative_order.relations_order@1 | 0.000 |
| comparative_order.relations_set@1 | 0.000 |

## Examples

- `mds_heldout_comparative_order_000000_1d66ff7c` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'semantic_schema_exact': False, 'solver_verified': False, 'repair_applied': False, 'raw_solver_verified': False, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
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
object=$5`
- `mds_heldout_comparative_order_000001_abb549ba` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'semantic_schema_exact': False, 'solver_verified': False, 'repair_applied': False, 'raw_solver_verified': False, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
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
rel=$0>$2
rel=$0>$3
rel=$0>$4
rel=$1>$0
rel=$1>$2
rel=$1>$3
rel=$1>$4
rel=$2>$0
rel=$2>$1
rel=$2>$3
rel=$2>$4
rel=$3>$0
rel=$3>$1
rel=$3>$2
rel=$3>$4
rel=$4>$0
rel=$4>$1
rel=$4>$2
rel=$4>$3`
- `mds_heldout_comparative_order_000002_4cc17ace` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'semantic_schema_exact': False, 'solver_verified': False, 'repair_applied': False, 'raw_solver_verified': False, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
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
rel=$0>$4
rel=$1>$2
rel=$1>$4
rel=$2>$0
rel=$2>$1
rel=$2>$3
rel=$2>$4
rel=$3>$4
rel=$4>$0
rel=$4>$1
rel=$4>$2
rel=$4>$3`
- `mds_heldout_comparative_order_000003_41af5dad` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'semantic_schema_exact': False, 'solver_verified': False, 'repair_applied': False, 'raw_solver_verified': False, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
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
rel=$0>$2
rel=$0>$3
rel=$0>$4
rel=$1>$0
rel=$1>$2
rel=$1>$3
rel=$1>$4
rel=$2>$0
rel=$2>$1
rel=$2>$3
rel=$2>$4
rel=$3>$0
rel=$3>$1
rel=$3>$2
rel=$3>$4
rel=$4>$0
rel=$4>$1
rel=$4>$2
rel=$4>$3`
- `mds_heldout_comparative_order_000004_fd1b3aca` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'semantic_schema_exact': False, 'solver_verified': True, 'repair_applied': False, 'raw_solver_verified': True, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
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
object=$6`
