# Exp93e Multidomain Schema Compiler

- train source: `data\multidomain_schema\v2\train.jsonl`
- eval source: `data\multidomain_schema\v2\heldout.jsonl`
- train n: `100000`
- eval n: `200`
- steps: `3000`
- seed: `2`
- compiler_arch: `exp83_1_mixed_top512_tequila`
- backbone_recipe: `exp83_1_mixed_top512_tequila`
- width: `64`
- layers: `2`
- heads arg: `4`
- h_cycles: `2`
- l_cycles: `3`
- head_dense_k: `512`
- target_surface: `pointer`
- vocab size: `79`
- eval domain counts: `{'comparative_order': 50, 'arithmetic': 50, 'maze': 50, 'logic_rules': 50}`
- format_valid@1: `0.940`
- json_valid@1: `0.940`
- domain_match@1: `0.940`
- schema_exact@1: `0.330`
- solver_verified@1: `0.330`
- train loss: `0.1732`
- params: `149376`
- ternary params: `144320`
- ternary param fraction: `0.966`
- fp32_mb: `0.57`
- packed_mb: `0.06`
- packed exact: `True`
- peak_vram_mb: `186.5`
- elapsed_s: `298.9`
- checkpoint: `artifacts\exp93e_exp831_pointer_v2_seed2\checkpoint.pt`

## Per Domain

| domain | format_valid@1 | schema_exact@1 | solver_verified@1 |
|---|---:|---:|---:|
| comparative_order | 0.800 | 0.000 | 0.000 |
| arithmetic | 1.000 | 0.320 | 0.320 |
| maze | 1.000 | 1.000 | 1.000 |
| logic_rules | 0.960 | 0.000 | 0.000 |

## Examples

- `mds_heldout_comparative_order_000000_1d66ff7c` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'solver_verified': False}`
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
dimension=speed
query=full_order
object=$0
object=$0
object=$0
object=$2
object=$1
object=$1
rel=$0>$1
rel=$2>$2
rel=$3>$4
rel=$4>$4
rel=$5>$5`
- `mds_heldout_comparative_order_000001_abb549ba` `comparative_order` score=`{'format_valid': False, 'json_valid': False, 'domain_match': False, 'schema_exact': False, 'solver_verified': False}`
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
object=$2
object=$2
object=$0
object=$2
object=$1
rel=$0>$1
rel=$2>$2
rel=$2>$2
rel=$3>$4
rel=$4>$4
rel=$5>$5`
- `mds_heldout_comparative_order_000002_4cc17ace` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'solver_verified': False}`
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
object=$2
object=$1
object=$2
object=$1
rel=$0>$1
rel=$2>$3
rel=$3>$0
rel=$4>$4`
- `mds_heldout_comparative_order_000003_41af5dad` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'solver_verified': False}`
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
object=$2
object=$2
object=$2
rel=$0>$1
rel=$2>$2
rel=$2>$2
rel=$4>$0
rel=$4>$4`
- `mds_heldout_comparative_order_000004_fd1b3aca` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'solver_verified': False}`
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
object=$1
object=$3
object=$3
object=$2
object=$1
object=$2
rel=$0>$1
rel=$2>$3
rel=$3>$4
rel=$4>$4
rel=$4>$5
rel=$5>$5
rel=$5>$5`
