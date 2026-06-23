"""Arm 4 Phase 1a diagnostic: is the arm-3 refit delta predictable at all?

Loads the (features, M_before, M_after) dumped at each refit and reports, per
layer and overall:
  1. SVD effective rank -- fraction of delta energy in top-1 / top-4 / top-16
     singular values (is the correction low-rank, as the PRISM-style proxy needs?).
  2. Consecutive-delta cosine -- is the correction direction stable across refits?
  3. Leave-one-out mean-delta predictor cosine -- does a trivial constant proxy
     (predict the mean of the other deltas) already clear the gate? This is the
     degenerate predictor; a features->low-rank predictor (Phase 1b) would have to
     beat it to justify the features.

Gate (preregistered): mean cosine > 0.3 useful, > 0.5 promising. Kill-if deltas
are full-rank AND direction-unstable.

Run: rtk python "experiments/Experiment 125 - No-BP Hard Ternary FPRM/arm4_phase1a_analyze.py" \
       "artifacts/phase0_fprm_exp125/arm4_phase1_refit_log_seed1"
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch


def _energy_frac(svals: torch.Tensor, top: int) -> float:
    e = svals.square()
    return float(e[:top].sum().cpu() / max(1e-30, float(e.sum().cpu())))


def main(log_dir: Path) -> int:
    files = sorted(log_dir.glob("refit_step*.pt"))
    if len(files) < 2:
        raise SystemExit(f"need >=2 refit logs, found {len(files)} in {log_dir}")
    print(f"loaded {len(files)} refit logs from {log_dir}", flush=True)
    payloads = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
    layers = list(payloads[0]["M_before"].keys())

    # Per-layer deltas D_k = M_after - M_before, and their flattened vectors.
    deltas: dict[str, list[torch.Tensor]] = {n: [] for n in layers}
    for p in payloads:
        for n in layers:
            d = (p["M_after"][n] - p["M_before"][n]).float()
            deltas[n].append(d)

    rows = []
    all_loo_cos = []
    all_consec_cos = []
    for n in layers:
        ds = deltas[n]
        vecs = [d.reshape(-1) for d in ds]
        # 1. SVD effective rank (mean energy fraction across refits).
        e1 = e4 = e16 = 0.0
        for d in ds:
            s = torch.linalg.svdvals(d)
            e1 += _energy_frac(s, 1)
            e4 += _energy_frac(s, 4)
            e16 += _energy_frac(s, 16)
        e1 /= len(ds); e4 /= len(ds); e16 /= len(ds)
        # 2. Consecutive cosine.
        consec = []
        for a, b in zip(vecs, vecs[1:]):
            denom = max(1e-30, float(a.norm().cpu()) * float(b.norm().cpu()))
            consec.append(float(torch.dot(a, b).cpu() / denom))
        consec_mean = sum(consec) / len(consec)
        # 3. Leave-one-out mean-delta predictor cosine.
        loo = []
        for k in range(len(vecs)):
            others = vecs[:k] + vecs[k + 1:]
            pred = torch.stack(others).mean(0)
            denom = max(1e-30, float(pred.norm().cpu()) * float(vecs[k].norm().cpu()))
            loo.append(float(torch.dot(pred, vecs[k]).cpu() / denom))
        loo_mean = sum(loo) / len(loo)
        all_loo_cos.append(loo_mean)
        all_consec_cos.append(consec_mean)
        rows.append((n, e1, e4, e16, consec_mean, loo_mean))

    rows.sort(key=lambda r: r[5])
    print(
        "\nper-layer (sorted by LOO mean-predictor cosine, worst first):",
        flush=True,
    )
    print(f"{'layer':<48} {'top1':>6} {'top4':>6} {'top16':>6} {'consec':>8} {'loo_mean':>9}",
          flush=True)
    for n, e1, e4, e16, c, loo in rows:
        print(f"{n:<48} {e1:6.3f} {e4:6.3f} {e16:6.3f} {c:8.3f} {loo:9.3f}", flush=True)

    n_layers = len(layers)
    overall_loo = sum(all_loo_cos) / n_layers
    overall_consec = sum(all_consec_cos) / n_layers
    print(
        f"\nOVERALL (19 layers, {len(files)} refits): "
        f"LOO mean-predictor cosine = {overall_loo:.3f} | consecutive cosine = {overall_consec:.3f}",
        flush=True,
    )

    # Gate (preregistered): mean cosine > 0.3 useful, > 0.5 promising.
    deep = [r for r in rows if "attn.qkv" in r[0]]
    deep_loo = sum(r[5] for r in deep) / max(1, len(deep))
    print(f"deep qkv layers ({len(deep)}): LOO mean cosine = {deep_loo:.3f}", flush=True)
    verdict = (
        "PROMISING (>0.5)" if overall_loo > 0.5
        else "USEFUL (>0.3)" if overall_loo > 0.3
        else "NO SIGNAL (<=0.3) -- proxy not viable at this refit cadence"
    )
    print(f"\nPhase 1a verdict: {verdict}", flush=True)
    if overall_loo <= 0.3:
        print(
            "Kill-if met: deltas are not predictable by a constant proxy; check "
            "per-layer -- if full-rank AND low consec cosine, no cheap proxy works.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    log_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "artifacts/phase0_fprm_exp125/arm4_phase1_refit_log_seed1"
    )
    raise SystemExit(main(log_dir))
