# Experiment 29 - First Local Pretrain

## Question

Can the current Phase 0 recipe produce one useful local pretrained artifact on
the RTX 3050 Ti target?

This is not another compression search. Keep the recipe fixed:

```text
mixed_top512_tequila_L_mlp_gate_up
```

## Run Target

| Setting | Value |
|---|---:|
| hidden size | 256 |
| layers | 4 |
| heads | 4 |
| steps | 50,000 |
| tokens per step | 512 |
| token exposures | 25.6M |
| expected params | 19.79M |
| expected packed size | 13.82 MB |
| expected wall time | 3-4 hours |

The data slice is recycled. This run is a local Phase 0 artifact, not a broad
internet-scale pretrain.

## Decision Rule

Promote if the run finishes without OOM, saves both FP32 and packed checkpoints,
keeps hard-export eval close to train-mode eval, has pack/unpack roundtrip error
inside tolerance, and gives usable frozen/output eval evidence.

Kill if the run OOMs, fails to save artifacts, hard-export changes eval loss
materially, pack/unpack roundtrip fails, or generated frozen answers are mostly
invalid after training.

## Command

Run in the background:

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File `
  "experiments/Experiment 29 - First Local Pretrain/start_exp29_h256_50k.ps1"
```

Run in the current terminal:

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File `
  "experiments/Experiment 29 - First Local Pretrain/run_exp29_h256_50k.ps1"
```

Calibrate the hard-export checkpoint after the 50k run:

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File `
  "experiments/Experiment 29 - First Local Pretrain/run_exp29_h256_export_calibration.ps1"
```

## Artifacts

The runner writes:

```text
artifacts/phase0_first_pretrain/h256_steps50000_seed1/checkpoint_fp32.pt
artifacts/phase0_first_pretrain/h256_steps50000_seed1/checkpoint_packed.pt
artifacts/phase0_first_pretrain/h256_steps50000_seed1/metrics.json
artifacts/phase0_first_pretrain/h256_steps50000_seed1/generation_examples.jsonl
experiments/Experiment 29 - First Local Pretrain/results_h256_steps50000_seed1.md
```

The export-calibrated runner writes:

```text
artifacts/phase0_first_pretrain/h256_steps50000_seed1_exportcalib3000/checkpoint_fp32.pt
artifacts/phase0_first_pretrain/h256_steps50000_seed1_exportcalib3000/checkpoint_packed.pt
artifacts/phase0_first_pretrain/h256_steps50000_seed1_exportcalib3000/metrics.json
artifacts/phase0_first_pretrain/h256_steps50000_seed1_exportcalib3000/generation_examples.jsonl
experiments/Experiment 29 - First Local Pretrain/results_h256_steps50000_seed1_exportcalib3000.md
```

## Results

The 50k CUDA pretrain completed and saved both checkpoints. Peak VRAM was
953.8 MB, packed size was 13.82 MB, and wall time was 128.8 minutes.

The raw 50k checkpoint improved eval loss from 11.9256 to 3.9321, but hard
export moved eval loss to 4.0664, a +0.1343 gap. That was too large to promote
as the stable Phase 0 artifact.

The 3,000-step export calibration reduced the hard-export gap to +0.0231 while
keeping eval loss flat at 3.9236. It used 813.8 MB peak VRAM and took 6.6
minutes. This is the best checkpoint from this run, but it is still not a full
Phase 0 promote because output eval is weak: frozen exact accuracy is 1.5% and
generation accuracy is 0.0%.
