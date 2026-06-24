"""Arm 7 Phase 1 diagnostic: does a nonlinear (MLP) feedback predictor beat the
linear fit for the deep layers? FREE -- no training run.

The deep-qkv layers hit a ~0.85 linear-fit ceiling that arm 6 showed is a
NONLINEARITY limit (transpose composition made it worse, not better). Arm 7
tests whether a small per-layer MLP (hidden -> 256 -> out_l, GELU, Adam) trained
offline on (head_error, true_BP_grad_l) pairs beats the linear ridge fit on
HELD-OUT data. Frozen-MLP feedback (Phase 2) is only worth a training run if
nonlinearity helps where linear hit the ceiling.

Captures true BP grad_output per layer during a bounded 50-step warmup (no
optimizer step, reuse arm-6's hook approach), splits first 40 steps = train,
last 10 = test, fits per layer:
  (a) linear ridge LS (current arm-2/3 mechanism) -- held-out test residual
  (b) MLP 256->256->out_l GELU Adam -- held-out test residual
Gate to proceed: MLP test residual < linear test residual for the deep qkv
layers. Kill-if MLP >= linear for deep qkv (nonlinearity does not help; the
softmax-entangled grad is too complex for a small MLP).

Run: rtk python "experiments/Experiment 125 - No-BP Hard Ternary FPRM/arm7_phase1_diagnostic.py"
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from training import nobp_hard as NOBP  # noqa: E402

EXP125_DIR = REPO_ROOT / "experiments" / "Experiment 125 - No-BP Hard Ternary FPRM"
EXP123_DIR = REPO_ROOT / "experiments" / "Experiment 123 - Fixed-Point Reasoning Model"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


EXP123 = _load("exp123_for_arm7", EXP123_DIR / "fprm_full_pretrain_then_sft.py")
EXP125 = _load("exp125_for_arm7", EXP125_DIR / "exp125_nobp_hard.py")


# ponytail: fixed full-batch Adam schedule (300 ep, lr 1e-3). Upgrade to
# early-stopping / lr-decay if the test residual is still falling at epoch 300.
# wd=0 is the literal preregistered mechanism; wd=1e-2 controls the overfitting
# confound (the unregularized MLP beats linear in-sample but not out-of-sample).
MLP_EPOCHS = 300
MLP_LR = 1e-3
MLP_HIDDEN = 256
MLP_WD = 1e-2


def fit_mlp(H_train, G_train, H_test, G_test, *, device, weight_decay=0.0):
    """Train a small MLP (H -> MLP_HIDDEN -> out) with full-batch Adam; return
    (train_rel_res, test_rel_res). BOTH inputs and targets standardized on train
    so the MSE is well-conditioned (raw grad targets are ~1e-2; an unstandardized
    256->out head outputs ~O(10) and Adam diverges). Predictions are un-
    standardized before the residual so it is comparable to the linear fit.
    `weight_decay` > 0 controls overfitting (the unregularized MLP beats linear
    in-sample but not out-of-sample; wd gives nonlinearity its fair shot)."""
    w = H_train.shape[1]
    out = G_train.shape[1]
    h_mean = H_train.mean(0, keepdim=True)
    h_std = H_train.std(0, keepdim=True).clamp_min(1e-6)
    g_mean = G_train.mean(0, keepdim=True)
    g_std = G_train.std(0, keepdim=True).clamp_min(1e-6)
    Htr = ((H_train - h_mean) / h_std).to(device)
    Hte = ((H_test - h_mean) / h_std).to(device)
    Gtr_z = ((G_train - g_mean) / g_std).to(device)
    Gtr = G_train.to(device)
    Gte = G_test.to(device)
    g_mean_d = g_mean.to(device)
    g_std_d = g_std.to(device)
    mlp = nn.Sequential(nn.Linear(w, MLP_HIDDEN), nn.GELU(), nn.Linear(MLP_HIDDEN, out)).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=MLP_LR, weight_decay=weight_decay)
    for _ in range(MLP_EPOCHS):
        opt.zero_grad()
        loss = F.mse_loss(mlp(Htr), Gtr_z)
        loss.backward()
        opt.step()
    mlp.eval()
    with torch.no_grad():
        pred_tr = mlp(Htr) * g_std_d + g_mean_d
        pred_te = mlp(Hte) * g_std_d + g_mean_d
        tr_res = float(torch.linalg.norm(pred_tr - Gtr).cpu() / max(1e-12, float(torch.linalg.norm(Gtr).cpu())))
        te_res = float(torch.linalg.norm(pred_te - Gte).cpu() / max(1e-12, float(torch.linalg.norm(Gte).cpu())))
    return tr_res, te_res


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    torch.manual_seed(seed)
    print(f"seed {seed}", flush=True)
    config = {
        "hidden_size": 256, "num_attention_heads": 4, "n_layers": 4,
        "max_iters": 20, "tau": 0.1, "damping": 1.0, "damping_decay": 0.9,
        "patience": 3, "min_damping": 1e-3, "bp_steps": 4, "max_seq_len": 128,
        "vocab_size": 65536,
    }
    model = EXP125.build_exp125_model(config, hard=False).to(device)  # tequila for warmup
    tokens = EXP123.EXP29.load_tokens(EXP123.DEFAULT_TOKENS)
    train_tokens, _ = EXP123.EXP29.split_tokens(tokens, eval_fraction=0.2, min_eval_tokens=4096)
    total_len = 128
    def batch_fn(step):
        return EXP123.EXP29.EXP22.EXP9._scheduled_batch(
            train_tokens, step=step, numseqs=1, total_len=total_len,
            prefix_len=64, causal_len=64, device=device, vocab_size=65536)

    targets = NOBP.local_update_targets(model, "nobp-dfa-full-hard")
    fit_targets = {n: m for n, m in targets.items() if n != "model.tape_writer.fc2"}
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def hook_vocab(_m, grad_input, _grad_output):
        # First-fire-wins: backward fires the FINAL forward call first, matching
        # the no-BP loop's last-write-wins activation capture.
        if "hidden" not in captured:
            hd = grad_input[0]
            captured["hidden"] = (hd[0] if isinstance(hd, tuple) else hd).detach()
    handles.append(model.tied_vocab.register_full_backward_hook(hook_vocab))

    for name, module in fit_targets.items():
        def make(n):
            def h(_m, _gi, go):
                if n not in captured:
                    g = go[0] if isinstance(go, tuple) else go
                    captured[n] = g.detach()
            return h
        handles.append(module.register_full_backward_hook(make(name)))

    hidden_rows: list[torch.Tensor] = []
    grad_rows: dict[str, list[torch.Tensor]] = {n: [] for n in fit_targets}
    WARMUP = 50
    TRAIN_STEPS = 40
    for step in range(WARMUP):
        batch = batch_fn(step)
        model.zero_grad(set_to_none=True)
        captured.clear()
        shared = model._shared_weight()
        emb = model.embed_scale * F.embedding(batch["inputs"], shared)
        _c, hidden = model.model(None, emb, **{k: v for k, v in batch.items() if k not in ("inputs", "labels")}, bp_steps=4)
        hidden = hidden.reshape(-1, hidden.shape[-1])
        labels = batch["labels"].reshape(-1)
        mask = labels != -100
        logits = model.tied_vocab(hidden)
        F.cross_entropy(logits[mask], labels[mask].to(torch.long)).backward()
        hidden_rows.append(captured["hidden"].reshape(-1, captured["hidden"].shape[-1])[mask].cpu().float())
        for n in fit_targets:
            g = captured[n].reshape(-1, captured[n].shape[-1])[mask].cpu().float()
            grad_rows[n].append(g)
    for h in handles:
        h.remove()
    model.zero_grad(set_to_none=True)

    H_train = torch.cat(hidden_rows[:TRAIN_STEPS], 0)
    H_test = torch.cat(hidden_rows[TRAIN_STEPS:], 0)
    eye = torch.eye(H_train.shape[1])
    HtH_inv = torch.linalg.inv(H_train.T @ H_train + 1e-3 * eye)
    print(f"warmup {WARMUP} steps: train {H_train.shape[0]} / test {H_test.shape[0]} samples, hidden={H_train.shape[1]}",
          flush=True)
    print(f"MLP: {MLP_HIDDEN}->GELU->out, Adam lr {MLP_LR}, {MLP_EPOCHS} epochs full-batch; "
          f"wd0=literal prereg, wd{MLP_WD}=overfitting control (fair test)\n", flush=True)
    print(f"{'layer':<46} {'linear_te':>10} {'mlp_te_wd0':>11} {'mlp_te_wd':>10} {'mlp_tr_wd':>10} {'winner':>9}",
          flush=True)
    deep_qkv_lin, deep_qkv_mlp = [], []
    for name in fit_targets:
        G_train = torch.cat(grad_rows[name][:TRAIN_STEPS], 0)
        G_test = torch.cat(grad_rows[name][TRAIN_STEPS:], 0)
        M_T = HtH_inv @ (H_train.T @ G_train)  # linear ridge on train
        lin_te = float(torch.linalg.norm(H_test @ M_T - G_test).cpu() / max(1e-12, float(torch.linalg.norm(G_test).cpu())))
        _mlp_tr0, mlp_te0 = fit_mlp(H_train, G_train, H_test, G_test, device=device, weight_decay=0.0)
        mlp_tr, mlp_te = fit_mlp(H_train, G_train, H_test, G_test, device=device, weight_decay=MLP_WD)
        winner = "mlp" if mlp_te < lin_te else "linear"  # verdict on the regularized (fair) test residual
        if "attn.qkv" in name:
            deep_qkv_lin.append(lin_te); deep_qkv_mlp.append(mlp_te)
        print(f"{name:<46} {lin_te:10.3f} {mlp_te0:11.3f} {mlp_te:10.3f} {mlp_tr:10.3f} {winner:>9}", flush=True)

    assert len(deep_qkv_lin) == 4, f"expected 4 deep qkv layers, got {len(deep_qkv_lin)}"
    print(f"\ndeep qkv (4 layers): linear mean={sum(deep_qkv_lin)/4:.3f}  mlp(wd{MLP_WD}) mean={sum(deep_qkv_mlp)/4:.3f}",
          flush=True)
    improved = sum(1 for l, m in zip(deep_qkv_lin, deep_qkv_mlp) if m < l)
    verdict = (f"PROCEED to Phase 2 training run: regularized MLP beats linear on {improved}/4 deep qkv layers"
               if improved >= 2 else
               f"KILL: regularized MLP does not beat linear on deep qkv ({improved}/4); nonlinearity does not help (generalization, not capacity)")
    print(f"\nPhase 1 verdict: {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
