# Experiment 93 - Compiler Solver MoE

## Goal

Test the sharp framing:

```text
raw task -> compiler -> VGR schema -> router -> solver -> verifier
```

This is not a TRM + LDT MoE yet. It is the first harness for the real split:

- compiler: turn clean VGR into a schema
- router: choose `comparative_ldt`, `trm_lm`, or `abstain`
- solver: execute fixed comparative lattice rules
- verifier: accept or reject final answer

TRM is represented only as a routed fallback in this experiment. Exp93 does not claim hidden-rule inference.

## Decision Rule

Promote if 200/200 heldout VGR rows route to `comparative_ldt`, verified strict_pass@1 is 1.000, and unsupported tasks never route to `comparative_ldt`.

Kill if any heldout VGR row abstains, any verified answer fails, or any unsupported task routes to `comparative_ldt`.

## Run

```powershell
rtk python "experiments/Experiment 93 - Compiler Solver MoE/compiler_solver_moe.py" --eval-limit 200 --output-dir "artifacts/exp93_compiler_solver_moe"
```

Smoke:

```powershell
rtk python "experiments/Experiment 93 - Compiler Solver MoE/compiler_solver_moe.py" --eval-limit 20 --output-dir "artifacts/exp93_compiler_solver_moe_smoke"
```

## Outputs

- `report.json`
- `experiments/Experiment 93 - Compiler Solver MoE/results.md`

## Results

- eval n: `200`
- VGR route counts: `{"comparative_ldt": 200}`
- verified_n: `200`
- verified strict_pass@1: `1.000`
- unsupported probe route counts: `{"trm_lm": 1, "abstain": 1}`
- unsupported false LDT routes: `0`
- elapsed_s: `0.203`
- report: `artifacts/exp93_compiler_solver_moe/report.json`

Verdict: promote for the clean fixed-rule path. Caveat: this proves routing and schema-backed solving for clean VGR. It does not prove TRM hidden-rule inference yet; raw comparative text is routed to `trm_lm` as an unimplemented fallback.
