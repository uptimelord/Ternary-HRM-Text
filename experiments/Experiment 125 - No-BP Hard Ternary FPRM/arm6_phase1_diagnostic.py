"""Arm 6 Phase 1 diagnostic: does transpose-derived composed feedback beat the
free fit for the deep layers? FREE -- no training run.

Captures the true BP grad_output per layer during a bounded warmup (no optimizer
step), then for each layer compares two feedbacks head_error -> grad_l:
  (a) free fit M_l  -- ridge least squares (current arm-2/3 mechanism)
  (b) transpose chain T_l -- product of downstream forward weights' transposes
      along the actual forward path (linear approx of the chain rule; ignores
      softmax in attention and GELU in MLP -- the known approximation cost).
Reports per-layer residual (||pred - true|| / ||true||) for both. Gate to proceed
to a training run: transpose residual < free-fit residual for the deep qkv
layers. Kill-if transpose >= free-fit for deep layers (nonlinearity dominates).

Run: rtk python "experiments/Experiment 125 - No-BP Hard Ternary FPRM/arm6_phase1_diagnostic.py"
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch
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


EXP123 = _load("exp123_for_arm6", EXP123_DIR / "fprm_full_pretrain_then_sft.py")
EXP125 = _load("exp125_for_arm6", EXP125_DIR / "exp125_nobp_hard.py")


def _hard_weight(module) -> torch.Tensor:
    return NOBP.hard_ternary_weight(module)


def _mlp_feedback(mlp) -> torch.Tensor:
    """Linear-approx feedback through a TernaryMLP (fc1 -> gelu -> fc2), width->width.
    J ~= W_fc2 @ W_fc1 (ignore GELU); feedback = J^T = W_fc1^T @ W_fc2^T.
    """
    fc1_w = _hard_weight(mlp.fc1)  # [4w, w]
    fc2_w = _hard_weight(mlp.fc2)  # [w, 4w]
    return fc1_w.transpose(0, 1) @ fc2_w.transpose(0, 1)  # [w, w]


def _block_feedback(block) -> torch.Tensor:
    """Linear-approx feedback through one FPRMBlock (attn + mlp), width->width.
    attn: out weight (ignore softmax); mlp: fc1^T @ fc2^T. Composed (mlp after attn).
    """
    attn_out_w = _hard_weight(block.attn.out)  # [w, w]
    return _mlp_feedback(block.mlp) @ attn_out_w.transpose(0, 1)  # [w, w]


def build_transpose_chain(model, target_name: str) -> torch.Tensor:
    """head_error (width) -> grad at target module's output, via downstream transposes.

    Forward path: tape_reader -> conv -> [layers 0..3: attn(qkv,softmax,out)+mlp]
    -> tape_writer -> head. head_error is already width (grad w.r.t. head input).
    We chain the width-preserving transposes downstream of the target, ignoring
    softmax/GELU/conv/layernorm (linear approx). For qkv/fc1 targets the final
    step projects width -> out_dim via the module's own input-side structure.
    """
    fprm = model.model
    layers = fprm.resonance_core.layers
    tw = fprm.tape_writer
    # Width-preserving feedback from head_error back through tape_writer + blocks.
    # Start from head_error, apply tape_writer feedback, then blocks from top down.
    chain = _mlp_feedback(tw)  # [w, w], maps head_error -> grad at tape_writer input

    parts = target_name.split(".")
    # Find target's layer index and module type.
    # Names: model.resonance_core.layers.{k}.attn.{qkv|out} | .mlp.{fc1|fc2}
    #        model.tape_writer.{fc1|fc2} | model.tape_reader.{fc1|fc2}
    if "resonance_core" in parts:
        k = int(parts[parts.index("layers") + 1])
        # Apply blocks ABOVE layer k (layers k+1 .. end) to reach layer k's output.
        for j in range(len(layers) - 1, k, -1):
            chain = chain @ _block_feedback(layers[j])
        layer = layers[k]
        sub = parts[-2] + "." + parts[-1]  # "attn.qkv" etc
        if sub == "attn.out":
            # grad at out-output (width): chain already there (layer k's block output
            # is after out+mlp; out-output is before mlp, so chain is correct as the
            # grad at the start of layer k's block).
            return chain  # [w, w] : head_error -> grad at layer k block input = out output region
        if sub == "attn.qkv":
            # grad at qkv-output (3w): qkv output -> softmax -> out. Linear approx:
            # only the V slice carries linearly through out; Q,K affect softmax
            # (ignored). Crude: pad the out^T mapping to 3w with zeros for Q,K.
            w = layer.attn.out.weight.shape[1]
            out_t = _hard_weight(layer.attn.out).transpose(0, 1)  # [w, w]
            to_qkv = torch.zeros(3 * w, w, device=chain.device, dtype=chain.dtype)
            to_qkv[2 * w:3 * w] = out_t  # V slot only
            return to_qkv @ chain  # [3w, w]
        if sub == "mlp.fc1":
            # grad at fc1-output (4w): fc1 out -> gelu -> fc2. Linear approx via fc2^T.
            fc2_t = _hard_weight(layer.mlp.fc2).transpose(0, 1)  # [4w, w]
            return fc2_t @ chain  # [4w, w]
        if sub == "mlp.fc2":
            # grad at fc2-output (w): fc2 out -> next block. chain already reaches
            # layer k block input (before attn); fc2-output is after attn+gelu.
            # Approx: chain (reaches block input) ~ grad at fc2 output (crude).
            return chain
    elif "tape_writer" in parts:
        sub = parts[-1]
        w = tw.fc1.weight.shape[1]
        if sub == "fc1":
            fc2_t = _hard_weight(tw.fc2).transpose(0, 1)  # [4w, w]
            return fc2_t  # head_error -> grad at fc1 output
        if sub == "fc2":
            return torch.eye(w, device=chain.device, dtype=chain.dtype)  # head_error = grad at fc2 output (identity)
    elif "tape_reader" in parts:
        # tape_reader is INPUT-side; downstream is the whole core + tape_writer.
        # Already chained through all blocks + tape_writer; tape_reader.fc1/fc2
        # outputs are before the core. Approx: chain reaches tape_reader output.
        sub = parts[-1]
        w = tw.fc1.weight.shape[1]
        if sub == "fc1":
            fc2_t = _hard_weight(fprm.tape_reader.fc2).transpose(0, 1)
            return fc2_t @ chain
        if sub == "fc2":
            return chain
    return chain  # fallback


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1)
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

    # Capture true BP grads via a bounded warmup (reuse the arm-2 mechanism).
    targets = NOBP.local_update_targets(model, "nobp-dfa-full-hard")
    fit_targets = {n: m for n, m in targets.items() if n != "model.tape_writer.fc2"}
    captured = {}
    handles = []

    def hook_vocab(_m, grad_input, _grad_output):
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

    hidden_rows, grad_rows = [], {n: [] for n in fit_targets}
    WARMUP = 50
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

    H = torch.cat(hidden_rows, 0)  # [N, w]
    eye = torch.eye(H.shape[1])
    HtH_inv = torch.linalg.inv(H.T @ H + 1e-3 * eye)
    print(f"warmup {WARMUP} steps, {H.shape[0]} samples, hidden={H.shape[1]}\n", flush=True)
    print(f"{'layer':<48} {'free_fit':>9} {'transpose':>9} {'winner':>9}", flush=True)
    deep_qkv_free, deep_qkv_trans = [], []
    for name in fit_targets:
        G = torch.cat(grad_rows[name], 0)  # [N, out]
        # (a) free fit residual
        M_T = HtH_inv @ (H.T @ G)  # [w, out]
        free_pred = H @ M_T
        free_res = float(torch.linalg.norm(free_pred - G).cpu() / max(1e-12, float(torch.linalg.norm(G).cpu())))
        # (b) transpose-chain residual
        T = build_transpose_chain(model, name).cpu().float()  # [out, w]
        trans_pred = H @ T.T  # [N, out]
        trans_res = float(torch.linalg.norm(trans_pred - G).cpu() / max(1e-12, float(torch.linalg.norm(G).cpu())))
        winner = "transpose" if trans_res < free_res else "free_fit"
        is_qkv = "attn.qkv" in name
        if is_qkv:
            deep_qkv_free.append(free_res); deep_qkv_trans.append(trans_res)
        print(f"{name:<48} {free_res:9.3f} {trans_res:9.3f} {winner:>9}", flush=True)

    print(f"\ndeep qkv (4 layers): free_fit mean={sum(deep_qkv_free)/4:.3f}  "
          f"transpose mean={sum(deep_qkv_trans)/4:.3f}", flush=True)
    improved = sum(1 for f, t in zip(deep_qkv_free, deep_qkv_trans) if t < f)
    verdict = (f"PROCEED to training run: transpose beats free_fit on {improved}/4 deep qkv layers"
               if improved >= 2 else
               f"KILL: transpose does not beat free_fit on deep qkv ({improved}/4); nonlinearity dominates")
    print(f"\nPhase 1 verdict: {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
