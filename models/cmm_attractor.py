"""CMM recurrence losses for Exp84.

This implements the CMM pieces we can test in this repo: NSDE noise,
equilibrium pull, Routh-Hurwitz trace pressure, and hyperspherical repulsion.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Any, Callable, Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class CMMSettings:
    equilibrium_weight: float | None = None
    stability_weight: float | None = None
    repulsion_weight: float | None = None
    lm_weight: float = 1.0
    bce_weight: float = 0.5
    equilibrium_x_weight: float = 1.0
    equilibrium_z_weight: float = 1.0
    rh_stable_z_weight: float = 1.0e4
    rh_unstable_x_weight: float = 10.0
    repulsion_x_weight: float = 1.0e3
    repulsion_z_weight: float = 1.0e3
    noise_std: float = 0.01
    stability_trace_samples: int = 1
    stability_fd_eps: float = 1e-2

    def weight_for(self, name: str) -> float:
        weights = {
            "lm": self.lm_weight,
            "bce": self.bce_weight,
            "equilibrium_x": self.equilibrium_x_weight,
            "equilibrium_z_h": self.equilibrium_z_weight,
            "rh_stable_z_h": self.rh_stable_z_weight,
            "rh_unstable_x": self.rh_unstable_x_weight,
            "repulsion_x": self.repulsion_x_weight,
            "repulsion_z_h": self.repulsion_z_weight,
        }
        return weights[name]


class CMMState:
    """Tracks last recurrent losses for reporting."""

    def __init__(self) -> None:
        self.prev_z_h: Tensor | None = None
        self.prev_z_l: Tensor | None = None
        self.residual_sum: float = 0.0
        self.steps: int = 0
        self.loss_terms: dict[str, float] = {}


def _module_zero(module: nn.Module) -> Tensor:
    param = next(module.parameters(), None)
    if param is None:
        return torch.zeros(())
    return param.new_zeros(())


def _flatten_by_sequence(states: Tensor, cu_seqlens: Tensor | None) -> Tensor:
    if cu_seqlens is None:
        return states.float().reshape(-1, states.shape[-1])
    rows: list[Tensor] = []
    cpu_lens = cu_seqlens.detach().cpu().tolist()
    max_width = 0
    for start, end in zip(cpu_lens[:-1], cpu_lens[1:]):
        row = states[int(start) : int(end)].float().reshape(-1)
        rows.append(row)
        max_width = max(max_width, int(row.numel()))
    if not rows:
        return states.new_zeros((0, states.shape[-1])).float()
    padded = []
    for row in rows:
        if row.numel() < max_width:
            row = F.pad(row, (0, max_width - row.numel()))
        padded.append(row)
    return torch.stack(padded, dim=0)


def hyperspherical_repulsion_loss(states: Tensor, eps: float = 1e-6, cu_seqlens: Tensor | None = None) -> Tensor:
    """Penalize hidden states that point in the same direction."""
    flat = _flatten_by_sequence(states, cu_seqlens)
    if flat.shape[0] < 2:
        return states.new_zeros(())
    unit = F.normalize(flat, dim=-1, eps=eps)
    sim = unit @ unit.T
    eye = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    off_diag = sim.masked_select(~eye)
    if off_diag.numel() == 0:
        return states.new_zeros(())
    return off_diag.pow(2).mean()


def routh_hurwitz_trace_loss(
    update_fn: Callable[[Tensor], Tensor],
    z: Tensor,
    *,
    samples: int = 1,
    eps: float = 1e-2,
    kind: str = "stable",
) -> Tensor:
    """Finite-difference Hutchinson trace pressure for d(F(z)-z)/dz.

    Stable continuous dynamics want negative local trace. Positive trace means
    nearby states spread out, so we penalize only the positive part. The finite
    difference form avoids second-order attention backward on CPU.
    """
    if samples <= 0 or eps <= 0 or not torch.is_grad_enabled():
        return z.new_zeros(())

    z_base = z.detach()
    field_base = update_fn(z_base) - z_base
    total = field_base.new_zeros(())
    for _ in range(samples):
        v = torch.empty_like(z_base).bernoulli_(0.5).mul_(2).sub_(1)
        z_perturbed = z_base + eps * v
        field_perturbed = update_fn(z_perturbed) - z_perturbed
        trace_mean = ((field_perturbed - field_base) * v).mean() / eps
        if kind == "stable":
            term = torch.relu(trace_mean - 1.0).pow(2)
        elif kind == "unstable":
            term = torch.relu(1.0 - trace_mean).pow(2)
        else:
            raise ValueError("kind must be stable or unstable")
        total = total + term
    return total / samples


class AlgGradNorm:
    """Algebraic GradNorm weight balancer from the CMM paper."""

    def __init__(
        self,
        names: Iterable[str],
        *,
        alpha: float = 1.0,
        rho: float = 0.9,
        eps: float = 1e-12,
        reset_interval: int = 2500,
        initial_weights: dict[str, float] | None = None,
    ) -> None:
        self.names = list(names)
        self.alpha = alpha
        self.rho = rho
        self.eps = eps
        self.reset_interval = reset_interval
        self.step = 0
        self.weights = {name: float((initial_weights or {}).get(name, 1.0)) for name in self.names}
        self._renormalize()
        self.reference_losses: dict[str, float] = {}

    def _renormalize(self) -> None:
        total = sum(self.weights.values())
        scale = len(self.weights) / max(total, self.eps)
        self.weights = {name: value * scale for name, value in self.weights.items()}

    def update_from_values(self, *, losses: dict[str, float], grad_norms: dict[str, float]) -> dict[str, float]:
        self.step += 1
        inactive = [
            name
            for name in self.names
            if name in losses and name in grad_norms and (float(losses[name]) <= self.eps or float(grad_norms[name]) <= self.eps)
        ]
        for name in inactive:
            self.weights[name] = 0.0
        active = [
            name
            for name in self.names
            if name in losses and name in grad_norms and float(losses[name]) > self.eps and float(grad_norms[name]) > self.eps
        ]
        if not active:
            self._renormalize()
            return dict(self.weights)
        if not self.reference_losses or (self.reset_interval > 0 and self.step % self.reset_interval == 0):
            for name in active:
                self.reference_losses[name] = max(float(losses[name]), self.eps)

        current_g = {name: self.weights[name] * max(float(grad_norms[name]), self.eps) for name in active}
        g_bar = sum(current_g.values()) / len(active)
        rel_losses = {
            name: max(float(losses[name]), self.eps) / max(self.reference_losses.get(name, self.eps), self.eps)
            for name in active
        }
        rel_bar = sum(rel_losses.values()) / len(active)
        temp = dict(self.weights)
        for name in active:
            rate = rel_losses[name] / max(rel_bar, self.eps)
            target = g_bar * (rate**self.alpha)
            factor = max(0.1, min(10.0, target / max(current_g[name], self.eps)))
            temp[name] = self.weights[name] * factor

        total = sum(temp[name] for name in active)
        scaled = {name: temp[name] * len(active) / max(total, self.eps) for name in active}
        for name in active:
            self.weights[name] = self.rho * self.weights[name] + (1.0 - self.rho) * scaled[name]
        self._renormalize()
        return dict(self.weights)

    def update(
        self,
        loss_terms: dict[str, Tensor],
        params: Iterable[nn.Parameter],
    ) -> dict[str, float]:
        param_list = [p for p in params if p.requires_grad]
        losses = {name: float(term.detach().cpu()) for name, term in loss_terms.items()}
        grad_norms = {name: 0.0 for name in loss_terms}
        active_items = [(name, term) for name, term in loss_terms.items() if term.requires_grad]
        if active_items and param_list:
            names = [name for name, _term in active_items]
            terms = tuple(term for _name, term in active_items)
            n_terms = len(terms)
            grad_outputs = []
            for i, term in enumerate(terms):
                eye_col = torch.zeros(n_terms, dtype=term.dtype, device=term.device)
                eye_col[i] = 1.0
                grad_outputs.append(eye_col)
            grads = torch.autograd.grad(
                terms,
                param_list,
                grad_outputs=tuple(grad_outputs),
                retain_graph=True,
                allow_unused=True,
                is_grads_batched=True,
            )
            sq_by_term = terms[0].new_zeros((n_terms,), dtype=torch.float32)
            for grad in grads:
                if grad is not None:
                    sq_by_term = sq_by_term + grad.detach().float().flatten(start_dim=1).pow(2).sum(dim=1)
            norm_by_term = torch.sqrt(sq_by_term).cpu()
            for i, name in enumerate(names):
                grad_norms[name] = float(norm_by_term[i])
        return self.update_from_values(losses=losses, grad_norms=grad_norms)


def patch_hrm_with_cmm(hrm: nn.Module, settings: CMMSettings) -> CMMState:
    """Wrap HRM forward with CMM losses + train-time noise injection.

    bp_steps gradient truncation MUST match the unpatched HRM forward
    (H_bp_steps = min(H, bp_steps-1); L_bp_steps = bp_steps - H_bp_steps),
    otherwise the CMM arm gets more backprop than the control and the
    A/B comparison is confounded (not the smallest fair diff).
    """
    state = CMMState()
    if not hasattr(hrm, "H_cycles"):
        raise TypeError("expected HRM backbone with H_cycles")
    if not hasattr(hrm, "L_level"):
        raise TypeError("expected recurrent backbone with L_level")

    def forward_with_cmm(self, carry, x: Tensor, cache=None, bp_steps: int = 2, **seq_info):
        use_state_carry = bool(getattr(self, "use_state_carry", False))
        if use_state_carry and carry is not None:
            z_H, z_L = carry
        else:
            z_H, z_L = x, self.zL_init
        total_res = x.new_zeros(())
        final_res = x.new_zeros(())
        equilibrium_z_h_step = x.new_zeros(())
        steps = 0
        h_steps = 0
        # Match original HRM truncation exactly.
        H_bp_steps = min(self.H_cycles, bp_steps - 1)
        L_bp_steps = bp_steps - H_bp_steps
        noisy = self.training and settings.noise_std > 0
        cu_seqlens = seq_info.get("cu_seqlens")
        z_L_initial = z_L

        for i in range(self.H_cycles):
            for k in range(i * self.L_cycles, (i + 1) * self.L_cycles):
                grad_on = torch.is_grad_enabled() and (k >= self.H_cycles * self.L_cycles - L_bp_steps)
                with torch.set_grad_enabled(grad_on):
                    z_in = z_L
                    if noisy:
                        z_in = z_in + settings.noise_std * torch.randn_like(z_in)
                    z_L_new = self.L_level(z_in, z_H, **seq_info, cache=cache["L"][k] if cache is not None else None)
                    step_res = (z_L_new - z_L).pow(2).mean()
                    total_res = total_res + step_res
                    final_res = step_res
                    z_L = z_L_new
                    steps += 1
            grad_on = torch.is_grad_enabled() and (i >= self.H_cycles - H_bp_steps)
            with torch.set_grad_enabled(grad_on):
                h_level = self.H_level if hasattr(self, "H_level") else self.L_level
                z_H_new = h_level(z_H, z_L, **seq_info, cache=cache["H"][i] if cache is not None else None)
                step_res = (z_H_new - z_H).pow(2).mean()
                total_res = total_res + step_res
                final_res = step_res
                equilibrium_z_h_step = equilibrium_z_h_step + (z_H_new - z_H.detach()).pow(2).mean()
                h_steps += 1
                z_H = z_H_new
                steps += 1

        aux_grad_on = torch.is_grad_enabled()
        with torch.set_grad_enabled(aux_grad_on):
            def h_update_final(z_probe: Tensor) -> Tensor:
                h_level = self.H_level if hasattr(self, "H_level") else self.L_level
                return h_level(z_probe, z_L.detach(), **seq_info, cache=None)

            def h_update_input(z_probe: Tensor) -> Tensor:
                h_level = self.H_level if hasattr(self, "H_level") else self.L_level
                return h_level(z_probe, z_L_initial.detach(), **seq_info, cache=None)

            equilibrium_z_h = (z_H.detach() - h_update_final(z_H)).pow(2).mean()
            equilibrium_x = (x.detach() - h_update_input(x)).pow(2).mean()
            repulsion_x = hyperspherical_repulsion_loss(x, cu_seqlens=cu_seqlens)
            repulsion_z_h = hyperspherical_repulsion_loss(z_H, cu_seqlens=cu_seqlens)
            rh_stable_z_h = routh_hurwitz_trace_loss(
                h_update_final,
                z_H,
                samples=settings.stability_trace_samples,
                eps=settings.stability_fd_eps,
                kind="stable",
            )
            rh_unstable_x = routh_hurwitz_trace_loss(
                h_update_input,
                x,
                samples=settings.stability_trace_samples,
                eps=settings.stability_fd_eps,
                kind="unstable",
            )

        mean_residual = total_res / max(1, steps)
        mean_equilibrium_step = equilibrium_z_h_step / max(1, h_steps)
        loss_terms = {
            "equilibrium_x": equilibrium_x,
            "equilibrium_z_h": equilibrium_z_h,
            "rh_stable_z_h": rh_stable_z_h,
            "rh_unstable_x": rh_unstable_x,
            "repulsion_x": repulsion_x,
            "repulsion_z_h": repulsion_z_h,
        }
        state.residual_sum = float(total_res.detach().cpu())
        state.steps = steps
        state.prev_z_h = z_H.detach()
        state.prev_z_l = z_L.detach()
        state.loss_terms = {name: float(value.detach().cpu()) for name, value in loss_terms.items()}
        state.loss_terms["final_residual"] = float(final_res.detach().cpu())
        hrm._cmm_residual = mean_residual  # type: ignore[attr-defined]
        hrm._cmm_final_residual = final_res  # type: ignore[attr-defined]
        hrm._cmm_equilibrium_loss = mean_equilibrium_step  # type: ignore[attr-defined]
        hrm._cmm_stability_loss = rh_stable_z_h  # type: ignore[attr-defined]
        hrm._cmm_repulsion_loss = repulsion_z_h  # type: ignore[attr-defined]
        hrm._cmm_loss_terms = loss_terms  # type: ignore[attr-defined]
        new_carry = (z_H.detach(), z_L.detach()) if use_state_carry else None
        return new_carry, z_H

    hrm.forward = MethodType(forward_with_cmm, hrm)  # type: ignore[method-assign]
    return state


def cmm_aux_loss(hrm: nn.Module, settings: CMMSettings) -> Tensor:
    """CMM auxiliary loss: equilibrium + stability + repulsion."""
    loss_terms = getattr(hrm, "_cmm_loss_terms", None)
    if loss_terms is not None:
        total = None
        for name, term in loss_terms.items():
            weighted = settings.weight_for(name) * term
            total = weighted if total is None else total + weighted
        if total is None:
            return _module_zero(hrm)
        return total

    equilibrium = getattr(hrm, "_cmm_equilibrium_loss", getattr(hrm, "_cmm_residual", None))
    if equilibrium is None:
        return _module_zero(hrm)
    eq_weight = settings.equilibrium_weight if settings.equilibrium_weight is not None else settings.equilibrium_z_weight
    loss = eq_weight * equilibrium
    stability = getattr(hrm, "_cmm_stability_loss", getattr(hrm, "_cmm_final_residual", None))
    if stability is not None:
        st_weight = settings.stability_weight if settings.stability_weight is not None else settings.rh_stable_z_weight
        loss = loss + st_weight * stability
    repulsion = getattr(hrm, "_cmm_repulsion_loss", None)
    if repulsion is not None:
        rep_weight = settings.repulsion_weight if settings.repulsion_weight is not None else settings.repulsion_z_weight
        loss = loss + rep_weight * repulsion
    return loss


def set_recurrence_depth(hrm: nn.Module, *, H_cycles: int, L_cycles: int) -> tuple[int, int]:
    prev_h, prev_l = hrm.H_cycles, hrm.L_cycles
    hrm.H_cycles = H_cycles
    hrm.L_cycles = L_cycles
    return prev_h, prev_l
