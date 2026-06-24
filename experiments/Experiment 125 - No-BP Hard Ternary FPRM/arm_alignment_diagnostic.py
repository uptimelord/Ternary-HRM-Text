"""Dual pretrain + SFT feedback-alignment diagnostic (free -- no training run).

Measures rho_l = cosine(true BP grad, DFA-predicted grad) per body layer, at the
arm-3 50k pretrain checkpoint, under TWO batch distributions:

  (1) PRETRAIN batches (scheduled LM batches on train_tokens) -- does pretrain
      alignment hold on the drifted 50k weights, or did it decay from the init
      0.37-0.85 residual arm 6/7 measured?
  (2) SFT batches (sample_sequences + make_fixed_sft_batch) -- is SFT alignment
      worse than pretrain (the fix-catalog's claim), and by how much?

For each: capture true BP grad_output per body layer + head hidden_error during a
bounded warmup (no optimizer step, weights unchanged), fit the linear feedback
matrix M by ridge LS (exactly as bp_warmup_seed_feedback does), then compute
per-sample cosine(g_true, M @ h) averaged per layer. Reports shallow / deep-qkv
/ other-body group means.

Gate (both measured):
  - pretrain body mean rho > 0.3 AND sft body mean rho > 0.3 -> alignment OK in
    both; the cause is elsewhere (partial-fit magnitude, not direction). Revise.
  - pretrain rho low, sft rho low -> alignment is the lever for BOTH; a richer
    (state-conditioned) predictor is justified for pretrain AND SFT.
  - pretrain rho OK, sft rho low -> SFT-specific alignment drop; predictor for
    SFT only, pretrain gap is a different (magnitude) problem.

Reuses nobp_hard._capture_warmup_pairs + local_update_targets. No training run,
no optimizer, no checkpoint written. ~10 min GPU.

Run: rtk python "experiments/Experiment 125 - No-BP Hard Ternary FPRM/arm_alignment_diagnostic.py"
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from training import nobp_hard as NOBP  # noqa: E402
from training.sft_lib import make_fixed_sft_batch, read_jsonl, sample_sequences, tokenize_sft_rows  # noqa: E402

EXP125_DIR = REPO_ROOT / "experiments" / "Experiment 125 - No-BP Hard Ternary FPRM"
EXP123_DIR = REPO_ROOT / "experiments" / "Experiment 123 - Fixed-Point Reasoning Model"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


EXP123 = _load("exp123_for_align", EXP123_DIR / "fprm_full_pretrain_then_sft.py")
EXP125 = _load("exp125_for_align", EXP125_DIR / "exp125_nobp_hard.py")
EXP29 = EXP123.EXP29

CKPT = REPO_ROOT / "artifacts" / "phase0_fprm_exp125" / "arm3_full_nseq4_steps50000_seed1" / "pretrain" / "checkpoint_fp32.pt"
WARMUP = 50
RIDGE = 1e-3


def _fit_and_align(H, grad_rows, fit_targets, *, device):
    """Fit M by ridge LS (as the trainer does), then per-layer cosine(true, M@h).
    Returns {name: (rho_mean, residual_rel, n_samples)}."""
    eye = torch.eye(H.shape[1])
    HtH_inv = torch.linalg.inv(H.T @ H + RIDGE * eye)
    out = {}
    for name in fit_targets:
        G = torch.cat(grad_rows[name], 0)  # [N, out]
        M_T = HtH_inv @ (H.T @ G)  # [hidden, out], solves H @ M.T ~= G
        M = M_T.T.contiguous()  # [out, hidden]
        pred = H @ M_T  # [N, out], = (M @ h_i) per sample
        # per-sample cosine, averaged. Guard zero-norm rows.
        g_norm = G.norm(dim=1).clamp_min(1e-12)
        p_norm = pred.norm(dim=1).clamp_min(1e-12)
        cos = (G * pred).sum(1) / (g_norm * p_norm)
        rho = float(cos.mean().cpu())
        res = float(torch.linalg.norm(pred - G).cpu() / max(1e-12, float(torch.linalg.norm(G).cpu())))
        out[name] = (rho, res, int(G.shape[0]))
    return out


def _fit_and_audit(H, grad_rows, fit_targets, *, device, update_clip=1.0, core_lr=0.5):
    """Fit M by ridge LS, then the FULL per-layer gain/clip/boundary audit.
    For each layer computes:
      rho   = cosine(g, q)                          (direction, already measured)
      qnorm = mean |q|, gnorm = mean |g|            (magnitudes)
      ratio = mean |q|/|g|                          (is the proxy too big/small?)
      s*    = mean <g,q>/|q|^2 = rho*|g|/|q|         (least-squares gain to match BP magnitude)
      clip% = fraction of samples whose |q| exceeds update_clip
              (the trainer rescales per-layer gradient to update_clip -- saturation distorts ratios)
      eff_step = mean post-clip |core_lr*min(|q|,update_clip)|  (actual update magnitude applied)
    Returns {name: dict} + the fitted M (for boundary dist)."""
    eye = torch.eye(H.shape[1])
    HtH_inv = torch.linalg.inv(H.T @ H + RIDGE * eye)
    out = {}
    matrices = {}
    for name in fit_targets:
        G = torch.cat(grad_rows[name], 0)  # [N, out]
        M_T = HtH_inv @ (H.T @ G)
        M = M_T.T.contiguous()
        matrices[name] = M
        q = H @ M_T  # [N, out] proxy grad (= M @ h)
        g_norm = G.norm(dim=1).clamp_min(1e-12)
        q_norm = q.norm(dim=1).clamp_min(1e-12)
        cos = (G * q).sum(1) / (g_norm * q_norm)
        rho = float(cos.mean().cpu())
        s = (G * q).sum(1) / (q_norm ** 2).clamp_min(1e-12)  # per-sample LS gain
        clip_pct = float((q_norm > update_clip).float().mean().cpu())
        eff_step = float((core_lr * q_norm.clamp_max(update_clip)).mean().cpu())
        out[name] = {
            "rho": rho,
            "gnorm": float(g_norm.mean().cpu()),
            "qnorm": float(q_norm.mean().cpu()),
            "ratio": float((q_norm / g_norm).mean().cpu()),
            "s_star": float(s.mean().cpu()),
            "clip_pct": clip_pct,
            "eff_step": eff_step,
            "n": int(G.shape[0]),
        }
    return out, matrices


def _boundary_dist(modules_by_name, matrices, *, device):
    """Per-layer distance-to-ternary-flip-boundary (median, unitless: |w - hard_w|/scale).
    Smaller = elements sit near their quantized value, need a small update to flip;
    larger = far from flipping. Crude but monotonic per layer; the MEDIAN summarizes
    how flip-prone the layer is. Uses the live hard_ternary_weight + latent master."""
    out = {}
    for name, module in modules_by_name.items():
        with torch.no_grad():
            ternary, scale, _pad = module.ternary_components()
            scale = scale.reshape(-1)
            w = module.weight.detach().reshape(-1, module.weight.shape[1])
            hard = NOBP.hard_ternary_weight(module).reshape(-1, module.weight.shape[1])
            margin = (w - hard).abs().reshape(-1)
            gs = module.ternary_group_size
            scale_expanded = scale.repeat_interleave(gs)[: margin.numel()]
            ratio = (margin / scale_expanded.clamp_min(1e-12)).clamp_min(0)
            out[name] = float(ratio.median().cpu())
    return out


def _group_report(name, results, fit_names):
    """Print per-layer rho + group means; return group dict."""
    def grp(pred):
        return [results[n][0] for n in fit_names if pred(n)]
    shallow = [n for n in fit_names if "tape_reader" in n or "tape_writer" in n]
    qkv = [n for n in fit_names if "attn.qkv" in n]
    other = [n for n in fit_names if n not in shallow and n not in qkv]
    print(f"\n=== {name} ===", flush=True)
    print(f"{'layer':<48} {'rho':>8} {'residual':>9} {'n':>6}", flush=True)
    for n in fit_names:
        rho, res, nn = results[n]
        tag = " (qkv)" if n in qkv else (" (shallow)" if n in shallow else "")
        print(f"{n:<48}{rho:8.3f} {res:9.3f} {nn:6d}{tag}", flush=True)
    g = {
        "shallow": sum(results[n][0] for n in shallow) / max(1, len(shallow)),
        "deep_qkv": sum(results[n][0] for n in qkv) / max(1, len(qkv)),
        "other_body": sum(results[n][0] for n in other) / max(1, len(other)),
        "all_body": sum(results[n][0] for n in fit_names) / max(1, len(fit_names)),
    }
    print(f"\n  group means: shallow={g['shallow']:.3f}  deep_qkv={g['deep_qkv']:.3f}  "
          f"other_body={g['other_body']:.3f}  ALL_BODY={g['all_body']:.3f}", flush=True)
    return g


def _audit_report(name, audit, boundary, fit_names, *, core_lr):
    """Print the full gain/clip/boundary table + group means for the s* verdict."""
    shallow = [n for n in fit_names if "tape_reader" in n or "tape_writer" in n]
    qkv = [n for n in fit_names if "attn.qkv" in n]
    other = [n for n in fit_names if n not in shallow and n not in qkv]
    print(f"\n=== {name} ===", flush=True)
    # s* = <g,q>/|q|^2 = the per-layer scalar gain that makes the DFA proxy match
    # the true BP grad magnitude (1.0 = proxy already correctly scaled; <1 = proxy
    # too big, gain should shrink it; >1 = proxy too small). This is the number a
    # global core_lr cannot reach if it varies across layers.
    print(f"{'layer':<46}{'rho':>7}{'s*':>9}{'clip%':>7}{'eff_step':>10}{'boundary':>10}", flush=True)
    for n in fit_names:
        a = audit[n]
        tag = "q" if n in qkv else ("s" if n in shallow else "o")
        print(f"{n:<46}{a['rho']:7.3f}{a['s_star']:9.3f}"
              f"{a['clip_pct']*100:6.1f}%{a['eff_step']:10.4f}{boundary[n]:10.3f} {tag}", flush=True)
    def mean(key, group):
        return sum(audit[n][key] for n in group) / max(1, len(group))
    print(f"\n  group means:", flush=True)
    for gname, grp in (("shallow", shallow), ("deep_qkv", qkv), ("other_body", other), ("ALL_BODY", fit_names)):
        print(f"    {gname:<11} rho={mean('rho',grp):.3f}  s*={mean('s_star',grp):.3f}  "
              f"clip%={mean('clip_pct',grp)*100:.1f}%  "
              f"boundary={sum(boundary[n] for n in grp)/max(1,len(grp)):.3f}", flush=True)
    s_stars = [audit[n]["s_star"] for n in fit_names]
    s_min, s_max = min(s_stars), max(s_stars)
    print(f"\n  s* SPREAD: min={s_min:.3f}  max={s_max:.3f}  ratio={s_max/max(s_min,1e-9):.1f}x  "
          f"(global core_lr={core_lr} gives every layer 1x)", flush=True)
    return {"s_min": s_min, "s_max": s_max, "s_ratio": s_max / max(s_min, 1e-9)}


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  checkpoint={CKPT}", flush=True)
    print(f"warmup={WARMUP} steps per arm  ridge={RIDGE}", flush=True)

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    base = config.get("base_config", config)
    base.setdefault("max_seq_len", max(int(base.get("max_seq_len", 128)), 128))
    top_512_ids = ckpt["top_512_ids"].cpu()

    targets = NOBP.local_update_targets(
        # build a throwaway model just to enumerate target names (shape-independent)
        EXP125.build_exp125_model(base, dense_token_ids=top_512_ids, hard=False).to(device),
        "nobp-dfa-full-hard",
    )
    fit_names = [n for n in targets if n != "model.tape_writer.fc2"]
    print(f"DFA feedback layers: {len(fit_names)} ({sum(1 for n in fit_names if 'attn.qkv' in n)} deep qkv)", flush=True)

    # ---------- (1) PRETRAIN alignment ----------
    torch.manual_seed(1)
    model = EXP125.build_exp125_model(base, dense_token_ids=top_512_ids, hard=False).to(device)  # tequila
    model.load_state_dict({k: v.to(device) for k, v in ckpt["state_dict"].items()})
    tokens = EXP123.EXP29.load_tokens(EXP123.DEFAULT_TOKENS)
    train_tokens, _ = EXP123.EXP29.split_tokens(tokens, eval_fraction=0.2, min_eval_tokens=4096)

    def pretrain_batch(step):
        return EXP123.EXP29.EXP22.EXP9._scheduled_batch(
            train_tokens, step=step, numseqs=4, total_len=128, prefix_len=64, causal_len=64,
            device=device, vocab_size=65536,
        )

    fit_targets = {n: dict(model.named_modules())[n] for n in fit_names}
    H_pre, grad_pre = NOBP._capture_warmup_pairs(
        model, pretrain_batch, warmup_steps=WARMUP, fit_targets=fit_targets, bp_steps=4,
    )
    res_pre = _fit_and_align(H_pre, grad_pre, fit_names, device=device)
    g_pre = _group_report("PRETRAIN alignment (50k checkpoint, pretrain batches)", res_pre, fit_names)
    audit_pre, mats_pre = _fit_and_audit(H_pre, grad_pre, fit_names, device=device,
                                         update_clip=1.0, core_lr=0.5)
    pre_modules = {n: dict(model.named_modules())[n] for n in fit_names}
    bnd_pre = _boundary_dist(pre_modules, mats_pre, device=device)
    sp_pre = _audit_report("PRETRAIN gain/clip/boundary audit (update_clip=1.0, core_lr=0.5)",
                           audit_pre, bnd_pre, fit_names, core_lr=0.5)

    # ---------- (2) SFT alignment ----------
    # Same checkpoint, fresh model (clear any grad state), SFT batches.
    torch.manual_seed(1)
    model2 = EXP125.build_exp125_model(base, dense_token_ids=top_512_ids, hard=False).to(device)
    model2.load_state_dict({k: v.to(device) for k, v in ckpt["state_dict"].items()})
    tokenizer = Tokenizer.from_file(str(EXP123.DEFAULT_TOKENIZER))
    train_sequences = tokenize_sft_rows(
        read_jsonl(EXP123.DEFAULT_TRAIN_JSONL), tokenizer, max_prompt_tokens=48, max_response_tokens=80,
    )
    import random
    sft_rng = random.Random(1)

    def sft_batch(_step):
        return make_fixed_sft_batch(
            sample_sequences(train_sequences, rng=sft_rng, batch_size=4),
            device=device, vocab_size=65536, total_len=128,
        )

    fit_targets2 = {n: dict(model2.named_modules())[n] for n in fit_names}
    H_sft, grad_sft = NOBP._capture_warmup_pairs(
        model2, sft_batch, warmup_steps=WARMUP, fit_targets=fit_targets2, bp_steps=4,
    )
    res_sft = _fit_and_align(H_sft, grad_sft, fit_names, device=device)
    g_sft = _group_report("SFT alignment (50k checkpoint, SFT batches)", res_sft, fit_names)
    # SFT uses core_lr=0.5, beta=0.03 -> effective body scale 0.015; audit at that.
    audit_sft, mats_sft = _fit_and_audit(H_sft, grad_sft, fit_names, device=device,
                                         update_clip=1.0, core_lr=0.5)
    sft_modules = {n: dict(model2.named_modules())[n] for n in fit_names}
    bnd_sft = _boundary_dist(sft_modules, mats_sft, device=device)
    sp_sft = _audit_report("SFT gain/clip/boundary audit (update_clip=1.0, core_lr=0.5)",
                           audit_sft, bnd_sft, fit_names, core_lr=0.5)

    # ---------- verdict ----------
    print("\n=== VERDICT ===", flush=True)
    print(f"  pretrain ALL_BODY rho = {g_pre['all_body']:.3f}", flush=True)
    print(f"  sft     ALL_BODY rho = {g_sft['all_body']:.3f}", flush=True)
    print(f"  delta (sft - pretrain) = {g_sft['all_body'] - g_pre['all_body']:+.3f}", flush=True)
    gate = 0.3
    if g_pre["all_body"] > gate and g_sft["all_body"] > gate:
        v = f"alignment OK in both (>{gate}); cause is elsewhere (magnitude, not direction) -- REVISE"
    elif g_pre["all_body"] <= gate and g_sft["all_body"] <= gate:
        v = "alignment low in BOTH -> richer predictor justified for pretrain AND SFT"
    elif g_pre["all_body"] > gate and g_sft["all_body"] <= gate:
        v = "SFT-specific alignment drop -> predictor for SFT only; pretrain gap is magnitude"
    else:
        v = "pretrain alignment low, SFT OK (unexpected) -> predictor for pretrain only"
    print(f"  alignment gate (mean rho > {gate}): {v}", flush=True)
    print(f"\n  GAIN (the new question): per-layer s* spread", flush=True)
    print(f"    pretrain s* spread: min={sp_pre['s_min']:.3f} max={sp_pre['s_max']:.3f} "
          f"ratio={sp_pre['s_ratio']:.1f}x", flush=True)
    print(f"    sft     s* spread: min={sp_sft['s_min']:.3f} max={sp_sft['s_max']:.3f} "
          f"ratio={sp_sft['s_ratio']:.1f}x", flush=True)
    gain_gate = 3.0  # >3x spread = global knob provably can't calibrate
    if max(sp_pre["s_ratio"], sp_sft["s_ratio"]) > gain_gate:
        gv = (f"s* spread > {gain_gate}x in at least one arm -> global core_lr/beta CANNOT "
              f"calibrate per-layer gain; arm-8 per-layer scalar gains JUSTIFIED")
    else:
        gv = (f"s* spread <= {gain_gate}x -> per-layer gain mismatch is mild; "
              f"global knob is not the main bottleneck, REVISE")
    print(f"  gain gate (s* spread > {gain_gate}x): {gv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
