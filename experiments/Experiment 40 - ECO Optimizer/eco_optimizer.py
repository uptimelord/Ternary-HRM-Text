"""ECO: Error-Compensated Optimizer wrapper (Rank 11).

Source: ECO (arXiv 2601.22101), quantizer per BitNet b1.58 (2402.17764).

ECO removes the high-precision master weight by keeping each quantized parameter
AT its quantized value between steps and folding the quantization error into the
optimizer's first-moment (momentum) buffer, which doubles as the error-feedback
store (zero extra state). Per step, for a quantized param theta:

    1. standard Adam-atan2 update produces theta_tilde
    2. theta_hat = q(theta_tilde)            # re-quantize (RTN ternary, groupwise)
    3. e = theta_tilde - theta_hat           # quantization error
    4. theta <- theta_hat                     # store the quantized value
    5. exp_avg <- exp_avg + e / (lr*(1-beta1))  # inject error into momentum

Non-quantized params (norms, dense vocab rows, attention) get a plain
Adam-atan2 step. The ternary quantizer q matches TernaryLinear158Init's packed
representation: groupwise mean_abs scale, threshold ternary, hard value
(ternary * scale) at every position.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT


def _ternary_quantize(weight: Tensor, group_size: int, threshold: float, eps: float) -> Tensor:
    """RTN groupwise ternary quantize -> hard value (ternary*scale), weight shape.

    Mirrors TernaryLinear158Init.quantized_weight value path (mean_abs scale).
    """
    flat = weight.reshape(-1)
    pad = (group_size - (flat.numel() % group_size)) % group_size
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    groups = flat.reshape(-1, group_size)
    scale = groups.abs().mean(dim=1, keepdim=True).clamp_min(eps)
    normalized = groups / scale
    ternary = torch.where(
        normalized > threshold,
        torch.ones_like(groups),
        torch.where(normalized < -threshold, -torch.ones_like(groups), torch.zeros_like(groups)),
    )
    hard = (ternary * scale).reshape(-1)
    if pad:
        hard = hard[:-pad]
    return hard.reshape_as(weight)


class ECOAdamAtan2(Optimizer):
    """Adam-atan2 with ECO master-weight-free error compensation on tagged params.

    Params placed in a group with ``eco=True`` are quantized in place each step
    and their quantization error is injected into momentum. All other groups run
    the unmodified Adam-atan2 update (identical to models.adam_atan2.AdamATan2).
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.0,
        eco: bool = False,
        eco_group_size: int = 128,
        eco_threshold: float = 0.5,
        eco_eps: float = 1e-6,
    ):
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[1]}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "eco": eco,
            "eco_group_size": eco_group_size,
            "eco_threshold": eco_threshold,
            "eco_eps": eco_eps,
        }
        super().__init__(params, defaults)
        self._init_state()

    @torch.no_grad()
    def _init_state(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = torch.tensor(0.0, dtype=torch.get_default_dtype())
                if group["betas"][0] > 0:
                    state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
                # ECO: snap the param to its quantized value at init so the
                # stored weight is low-precision from step 0 (no FP master).
                if group.get("eco", False):
                    p.copy_(_ternary_quantize(p, group["eco_group_size"], group["eco_threshold"], group["eco_eps"]))

    @torch.no_grad()
    def step(self, closure=None):  # pyright: ignore[reportIncompatibleMethodOverride]
        assert closure is None, "Closure is not supported"
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                grad = param.grad

                if group["weight_decay"] != 0:
                    param.mul_(1 - lr * group["weight_decay"])

                if "exp_avg" in state:
                    state["exp_avg"].lerp_(grad, 1 - beta1)
                state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = lr / bias_correction1
                denom = state["exp_avg_sq"].sqrt() / bias_correction2.sqrt()

                if "exp_avg" in state:
                    param.add_(torch.atan2(state["exp_avg"], denom), alpha=-step_size)
                else:
                    param.add_(torch.atan2(grad, denom), alpha=-lr)

                # ECO master-weight-free error compensation
                if group.get("eco", False):
                    pre = param.detach().clone()
                    quant = _ternary_quantize(param, group["eco_group_size"], group["eco_threshold"], group["eco_eps"])
                    error = pre - quant
                    param.copy_(quant)
                    if "exp_avg" in state and beta1 < 1.0:
                        state["exp_avg"].add_(error, alpha=1.0 / (lr * (1 - beta1)))


def build_eco_param_groups(model, *, eco: bool, **eco_kwargs):
    """Split model params into an ECO group (ternary layer weights) and a plain
    group. When eco=False returns a single plain group (pure Adam-atan2)."""
    from models.layers import TernaryLinear158Init

    eco_param_ids = set()
    if eco:
        for module in model.modules():
            if isinstance(module, TernaryLinear158Init):
                eco_param_ids.add(id(module.weight))

    eco_params = [p for p in model.parameters() if id(p) in eco_param_ids]
    plain_params = [p for p in model.parameters() if id(p) not in eco_param_ids]

    groups = []
    if eco_params:
        groups.append({"params": eco_params, "eco": True, **eco_kwargs})
    if plain_params:
        groups.append({"params": plain_params, "eco": False})
    return groups, len(eco_params)
