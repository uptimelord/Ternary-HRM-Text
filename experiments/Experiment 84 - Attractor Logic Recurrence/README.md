# Experiment 84 - Attractor Logic Recurrence

Architecture brief C4: 2x2 `{base_recipe, CMM}` x `{shallow, deep}` on logic SFT.

## CMM Pieces Implemented

- StableMax3 loss by default; StableMax, StableMax3, StableMax5 are available.
- train-time NSDE additive noise, sigma `0.01`
- equilibrium pull for input state `x` and high recurrence state `z_H`
- Routh-Hurwitz trace pressure:
  - stable point at `z_H`
  - unstable point at `x`
- hyperspherical repulsion by sample for `x` and `z_H`
- AlgGradNorm adaptive loss balancing
- TRM halt/BCE head is wired with paper weight `0.5`
- TRM path uses an MLP-mixer recurrence block by default (`--backbone-block mlp_mixer`)
- Adam-Atan2 optimizer by default
- full-mode defaults: batch `250`, `--grad-accum-steps 4` micro-batches per outer step. Note: optimizer steps fire every `n_accum` segments *inside* each micro-batch, so micro-batches do not accumulate — effective batch per optimizer step is `250` distinct examples, not `250 x 4`
- supervised segment knobs are wired: `--n-super 16`, `--n-accum 2`; full mode runs all 16 segments and steps every 2 segments; a trailing partial group (when `n_super % n_accum != 0`) is loss-scaled by its true size
- TRM/CMM segment carry keeps `(z_H, z_L)` across supervised segments and detaches between segments
- Exp84 TRM initializes `z_L` to zero for paper-style segment runs
- `--deep-cycles h4l3` is the default deep arm; `--deep-cycles l6` runs the N_L=6 replication axis
- `--control-recipe paper` keeps the matched paper-style base recipe; `--control-recipe repo` runs the follow-up CE + AdamW + no-halt control
- embedding freeze schedule is wired with default step `2500`
- AMP is enabled by default on CUDA
- CUDA `torch.compile` flag is wired; CPU smoke records `compiled=false`
- identical recurrent layers are enabled by default
- named audit terms in JSON:
  - `equilibrium_x`
  - `equilibrium_z_h`
  - `rh_stable_z_h`
  - `rh_unstable_x`
  - `repulsion_x`
  - `repulsion_z_h`
  - `final_residual`

Still not proven: full default batch `250` on the local 4 GB RTX 3050 Ti. The paper's default batch shape is too large for this text-vocab head on this GPU.

Local GPU checks:

- `results_gpu_lowvram_recipe_step1_batch1_amp_no_compile_codex.md`: passes one-step full-mode CMM path with AMP, AlgGradNorm, `n_super=16`, `n_accum=2`, batch `1`, compile off.
- `results_gpu_compile_amp_no_alggradnorm_step1_codex.md`: passes one-step full-mode compile path with AMP and CMM, batch `1`, AlgGradNorm off.
- default batch `250` OOMs on 4 GB because logits are `32000 x vocab`. compile + AlgGradNorm still trips PyTorch autograd/compile limits in this environment.

## Decision Rule

Promote if CMM-deep beats base-recipe-shallow on logic hard by at least 5 pp, 2 seeds, and deep >= shallow under CMM.

Kill if depth is still flat or harmful under CMM.

## Run

```powershell
rtk python "experiments/Experiment 84 - Attractor Logic Recurrence/attractor_logic_recurrence.py" --mode smoke --device cpu
rtk python "experiments/Experiment 84 - Attractor Logic Recurrence/attractor_logic_recurrence.py" --mode full --device cuda --deep-cycles l6
```

Decision-grade run is **gated on Exp79/83 verdicts** (brief §5). When unblocked
(logic-hard n=200, seeds 1,2 are defaults; full mode does not raise `--steps`,
set it explicitly; on the 4 GB GPU the 32k-vocab head forces a small batch until
chunked CE lands — effective batch per optimizer step is `--batch-size`):

```powershell
rtk python "experiments/Experiment 84 - Attractor Logic Recurrence/attractor_logic_recurrence.py" --mode full --device cuda --steps 5000 --deep-cycles l6 --batch-size 8 --grad-accum-steps 1
```

## Results

Canonical result: `results_smoke_seed1.md`.

Superseded run logs: `results_*_codex.md` files in this folder (including the
GPU VRAM checks cited above) are historical run logs from before the segment
schedule and effective-batch fixes; do not use them for promote/kill decisions.

The smoke run trains only 5 steps on 32 rows and evaluates 4 rows. It clamps batch/accumulation to `2 x 1` and segment settings to `n_super=1`, `n_accum=1`; it does not trigger embedding freeze. It is a crash check, not an accuracy result.

## Verdict

Awaiting decision-grade CUDA run. Full default batch `250` is still a 4 GB VRAM
risk because of the 32k vocab head; chunked CE is a separate follow-up.
