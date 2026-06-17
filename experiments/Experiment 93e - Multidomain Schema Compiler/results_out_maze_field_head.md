# Exp93e Multidomain Schema Compiler

- train source: `C:\Users\Dos\AppData\Local\Temp\pytest-of-Dos\pytest-205\test_train_model_runs_maze_trm0\maze\train.jsonl`
- eval source: `C:\Users\Dos\AppData\Local\Temp\pytest-of-Dos\pytest-205\test_train_model_runs_maze_trm0\maze\heldout.jsonl`
- train n: `4`
- eval n: `2`
- steps: `1`
- seed: `1`
- compiler_arch: `maze_trm_field_head`
- backbone_recipe: `maze_trm_field_head`
- width: `16`
- layers: `1`
- heads arg: `2`
- h_cycles: `1`
- l_cycles: `1`
- head_dense_k: `512`
- relation_rank_weight: `0.0`
- target_surface: `pointer`
- decode_repair: `none`
- vocab size: `34`
- eval domain counts: `{'comparative_order': 0, 'arithmetic': 0, 'maze': 2, 'logic_rules': 0}`
- format_valid@1: `1.000`
- json_valid@1: `1.000`
- domain_match@1: `1.000`
- schema_exact@1: `0.000`
- semantic_schema_exact@1: `0.000`
- raw_solver_verified@1: `0.000`
- repaired_solver_verified@1: `1.000`
- oracle_solver_verified@1: `1.000`
- solver_verified@1: `0.000` raw compatibility alias
- repair_applied@1: `0.000`
- train loss: `8.0695`
- params: `15202`
- ternary params: `13568`
- ternary param fraction: `0.893`
- fp32_mb: `0.06`
- packed_mb: `0.02`
- packed exact: `True`
- peak_vram_mb: `0.0`
- elapsed_s: `0.0`
- checkpoint: `C:\Users\Dos\AppData\Local\Temp\pytest-of-Dos\pytest-205\test_train_model_runs_maze_trm0\out_maze_field_head\checkpoint.pt`

## Per Domain

| domain | format_valid@1 | schema_exact@1 | semantic_schema_exact@1 | raw_solver_verified@1 | repaired_solver_verified@1 | oracle_solver_verified@1 |
|---|---:|---:|---:|---:|---:|---:|
| comparative_order | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| arithmetic | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| maze | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| logic_rules | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Component Metrics

| component | pass@1 |
|---|---:|
| maze.domain@1 | 1.000 |
| maze.goal@1 | 0.000 |
| maze.grid_ref@1 | 1.000 |
| maze.query@1 | 1.000 |
| maze.start@1 | 0.000 |

## Examples

- `mds_heldout_maze_000000_33bcc7c1` `maze` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'semantic_schema_exact': False, 'solver_verified': False, 'repair_applied': False, 'raw_solver_verified': False, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
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
- `mds_heldout_maze_000001_7db5b870` `maze` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'semantic_schema_exact': False, 'solver_verified': False, 'repair_applied': False, 'raw_solver_verified': False, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
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
