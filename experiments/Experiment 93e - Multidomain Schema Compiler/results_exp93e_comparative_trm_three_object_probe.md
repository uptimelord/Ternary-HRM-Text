# Exp93e Multidomain Schema Compiler

- train source: `artifacts\exp93e_synthetic_3_object\train.jsonl`
- eval source: `artifacts\exp93e_synthetic_3_object\heldout.jsonl`
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
- vocab size: `53`
- eval domain counts: `{'comparative_order': 200, 'arithmetic': 0, 'maze': 0, 'logic_rules': 0}`
- format_valid@1: `1.000`
- json_valid@1: `1.000`
- domain_match@1: `1.000`
- schema_exact@1: `0.000`
- semantic_schema_exact@1: `0.000`
- raw_solver_verified@1: `0.690`
- repaired_solver_verified@1: `1.000`
- oracle_solver_verified@1: `1.000`
- solver_verified@1: `0.690` raw compatibility alias
- repair_applied@1: `0.000`
- train loss: `0.7178`
- params: `167945`
- ternary params: `139264`
- ternary param fraction: `0.829`
- fp32_mb: `0.64`
- packed_mb: `0.15`
- packed exact: `True`
- peak_vram_mb: `197.8`
- elapsed_s: `26.3`
- checkpoint: `artifacts\exp93e_comparative_trm_three_object_probe\checkpoint.pt`

## Per Domain

| domain | format_valid@1 | schema_exact@1 | semantic_schema_exact@1 | raw_solver_verified@1 | repaired_solver_verified@1 | oracle_solver_verified@1 |
|---|---:|---:|---:|---:|---:|---:|
| comparative_order | 1.000 | 0.000 | 0.000 | 0.690 | 1.000 | 1.000 |
| arithmetic | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| maze | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| logic_rules | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Component Metrics

| component | pass@1 |
|---|---:|
| comparative_order.dimension@1 | 1.000 |
| comparative_order.domain@1 | 1.000 |
| comparative_order.objects_order@1 | 0.065 |
| comparative_order.objects_set@1 | 1.000 |
| comparative_order.query@1 | 1.000 |
| comparative_order.relations_order@1 | 0.000 |
| comparative_order.relations_set@1 | 0.000 |

## Examples

- `mds_heldout_comparative_order_000000_fe78618d` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'semantic_schema_exact': False, 'solver_verified': True, 'repair_applied': False, 'raw_solver_verified': True, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
  - target: `@comparative_order
dimension=height
query=argmax
object=$2
object=$0
object=$1
rel=$0>$1
rel=$1>$2`
  - pred: `@comparative_order
dimension=height
query=argmax
object=$0
object=$1
object=$2
rel=$0>$1
rel=$0>$2
rel=$1>$2`
- `mds_heldout_comparative_order_000001_f0845b19` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'semantic_schema_exact': False, 'solver_verified': False, 'repair_applied': False, 'raw_solver_verified': False, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
  - target: `@comparative_order
dimension=age
query=full_order
object=$0
object=$1
object=$2
rel=$0>$1
rel=$2>$0`
  - pred: `@comparative_order
dimension=age
query=full_order
object=$0
object=$1
object=$2`
- `mds_heldout_comparative_order_000002_c2358e4c` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'semantic_schema_exact': False, 'solver_verified': True, 'repair_applied': False, 'raw_solver_verified': True, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
  - target: `@comparative_order
dimension=speed
query=argmax
object=$0
object=$2
object=$1
rel=$0>$1
rel=$1>$2`
  - pred: `@comparative_order
dimension=speed
query=argmax
object=$0
object=$1
object=$2
rel=$0>$1
rel=$0>$2
rel=$1>$2`
- `mds_heldout_comparative_order_000003_893d7b1a` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'semantic_schema_exact': False, 'solver_verified': True, 'repair_applied': False, 'raw_solver_verified': True, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
  - target: `@comparative_order
dimension=wealth
query=full_order
object=$1
object=$0
object=$2
rel=$0>$1
rel=$1>$2`
  - pred: `@comparative_order
dimension=wealth
query=full_order
object=$0
object=$1
object=$2
rel=$0>$1
rel=$0>$2
rel=$1>$2`
- `mds_heldout_comparative_order_000004_dcfb567f` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'semantic_schema_exact': False, 'solver_verified': False, 'repair_applied': False, 'raw_solver_verified': False, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
  - target: `@comparative_order
dimension=score
query=argmax
object=$1
object=$0
object=$2
rel=$0>$1
rel=$1>$2`
  - pred: `@comparative_order
dimension=score
query=argmax
object=$0
object=$1
object=$2
rel=$0>$1
rel=$0>$2
rel=$1>$0
rel=$1>$2`
