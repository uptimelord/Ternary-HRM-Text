# Experiment 93d - Multidomain Schema Corpus

## Goal

Build the first real compiler corpus:

```text
sentence/grid input -> schema -> exact solver -> verifier -> sentence output
```

Domains:

- comparative order
- arithmetic
- maze shortest path
- logic rules

Each row has `input_text`, `schema`, `solution`, `grid`, `output_text`, task targets, verifier result, and a checked negative.

## Decision Rule

Promote if the default build writes at least `100000` train rows and `4000` heldout rows, covers all 4 domains in both splits, has `positive_pass == train_rows + heldout_rows`, has `signature_overlap == false`, has no negative that passes verification, and passes the held-out guard on train.

Kill if any domain is missing, any positive row fails verification, any train/heldout signature overlaps, any checked negative passes, or held-out guard rejects train.

## Run

```powershell
rtk python -m training.multidomain_schema_rows --output-dir "datasets/multidomain_schema/v2" --train-per-domain 25000 --heldout-per-domain 1000 --seed 9304
```

Smoke:

```powershell
rtk python -m training.multidomain_schema_rows --output-dir "artifacts/exp93d_multidomain_schema_smoke" --train-per-domain 10 --heldout-per-domain 5 --seed 9304
```

## v2 curriculum fixes

Compared to v1:

- comparative schema shuffles `objects`; they no longer equal solved order
- maze schema uses `grid_ref: "input"` instead of embedding the full grid
- heldout samples the hard end of each train range instead of jumping outside it
- solver accepts maze `grid_ref` via row context (`grid.rows`)
- rows include `text_to_schema_pointer` targets with row-local refs (`$0`, `$1`, `S`, `G`)
- streamed JSONL writes round-robin domains, so small `--eval-limit` samples are multidomain

## Outputs

- `train.jsonl`
- `heldout.jsonl`
- `report.json`

## Results

v1 default build (superseded — curriculum flaws, keep as stress test only):

- train rows: `100000`
- heldout rows: `4000`
- positive_pass: `104000`
- signature_overlap: `false`
- report: `datasets/multidomain_schema/v1/report.json`

v1 audit warnings: comparative order leak, maze full-grid copy, heldout hard OOD in all domains.

v2 default build (current):

```powershell
rtk python -m training.multidomain_schema_rows --output-dir "datasets/multidomain_schema/v2" --train-per-domain 25000 --heldout-per-domain 1000 --seed 9304
rtk python -m training.multidomain_schema_audit --train "datasets/multidomain_schema/v2/train.jsonl" --heldout "datasets/multidomain_schema/v2/heldout.jsonl" --output "artifacts/exp93d_multidomain_schema_audit_v2/report.json"
```

Expected v2 audit:

- positive_fail: `0`
- negative_pass: `0`
- signature_overlap: `false`
- comparative order leak: `0`
- maze full-grid schema copy: `0`
- heldout_oob_* warnings: none

Verdict: promote v2 as the main compiler corpus. Keep v1 only as a stress-test set.
