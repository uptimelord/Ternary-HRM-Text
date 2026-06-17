# Exp93e Multidomain Schema Compiler

- train source: `artifacts\exp93e_domain_slices\maze_train.jsonl`
- eval source: `artifacts\exp93e_domain_slices\maze_heldout.jsonl`
- train n: `20000`
- eval n: `200`
- steps: `300`
- seed: `2`
- compiler_arch: `maze_trm_field_head`
- backbone_recipe: `maze_trm_field_head`
- width: `64`
- layers: `2`
- heads arg: `4`
- h_cycles: `2`
- l_cycles: `3`
- head_dense_k: `512`
- relation_rank_weight: `0.0`
- target_surface: `pointer`
- decode_repair: `none`
- vocab size: `34`
- eval domain counts: `{'comparative_order': 0, 'arithmetic': 0, 'maze': 200, 'logic_rules': 0}`
- format_valid@1: `1.000`
- json_valid@1: `1.000`
- domain_match@1: `1.000`
- schema_exact@1: `1.000`
- semantic_schema_exact@1: `1.000`
- raw_solver_verified@1: `1.000`
- repaired_solver_verified@1: `1.000`
- oracle_solver_verified@1: `1.000`
- solver_verified@1: `1.000` raw compatibility alias
- repair_applied@1: `0.000`
- train loss: `0.0007`
- params: `158082`
- ternary params: `139264`
- ternary param fraction: `0.881`
- fp32_mb: `0.60`
- packed_mb: `0.11`
- packed exact: `True`
- peak_vram_mb: `331.3`
- elapsed_s: `47.9`
- checkpoint: `artifacts\exp93e_maze_trm_field_head_probe_seed2\checkpoint.pt`

## Per Domain

| domain | format_valid@1 | schema_exact@1 | semantic_schema_exact@1 | raw_solver_verified@1 | repaired_solver_verified@1 | oracle_solver_verified@1 |
|---|---:|---:|---:|---:|---:|---:|
| comparative_order | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| arithmetic | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| maze | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| logic_rules | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Component Metrics

| component | pass@1 |
|---|---:|
| maze.domain@1 | 1.000 |
| maze.goal@1 | 1.000 |
| maze.grid_ref@1 | 1.000 |
| maze.query@1 | 1.000 |
| maze.start@1 | 1.000 |

## Examples

- `mds_heldout_maze_000000_49f6ed92` `maze` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': True, 'semantic_schema_exact': True, 'solver_verified': True, 'repair_applied': False, 'raw_solver_verified': True, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
  - target: `@maze
grid_ref=input
start=S
goal=G
query=shortest_path_length`
  - pred: `@maze
grid_ref=input
start=S
goal=G
query=shortest_path_length`
- `mds_heldout_maze_000001_162343d7` `maze` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': True, 'semantic_schema_exact': True, 'solver_verified': True, 'repair_applied': False, 'raw_solver_verified': True, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
  - target: `@maze
grid_ref=input
start=S
goal=G
query=shortest_path_length`
  - pred: `@maze
grid_ref=input
start=S
goal=G
query=shortest_path_length`
- `mds_heldout_maze_000002_68193bef` `maze` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': True, 'semantic_schema_exact': True, 'solver_verified': True, 'repair_applied': False, 'raw_solver_verified': True, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
  - target: `@maze
grid_ref=input
start=S
goal=G
query=shortest_path_length`
  - pred: `@maze
grid_ref=input
start=S
goal=G
query=shortest_path_length`
- `mds_heldout_maze_000003_c153d849` `maze` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': True, 'semantic_schema_exact': True, 'solver_verified': True, 'repair_applied': False, 'raw_solver_verified': True, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
  - target: `@maze
grid_ref=input
start=S
goal=G
query=shortest_path_length`
  - pred: `@maze
grid_ref=input
start=S
goal=G
query=shortest_path_length`
- `mds_heldout_maze_000004_33eef6f9` `maze` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': True, 'semantic_schema_exact': True, 'solver_verified': True, 'repair_applied': False, 'raw_solver_verified': True, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
  - target: `@maze
grid_ref=input
start=S
goal=G
query=shortest_path_length`
  - pred: `@maze
grid_ref=input
start=S
goal=G
query=shortest_path_length`
