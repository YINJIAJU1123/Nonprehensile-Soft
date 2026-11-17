#!/usr/bin/env python3
"""SEER-specific future-conditioned action residual adapter.

This is deliberately separate from the legacy ``distill_wm`` gated residual.
It consumes the student ``FT_head`` prediction for the base action and only
modifies the first ``active_action_dims`` action dimensions. The residual
output is exactly zero at initialization; the gate starts near zero so the
new action path begins as the base SEER policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch as th
import torch.nn as nn


SEER_FUTURE_RESIDUAL_VERSION = "seer_future_residual_v1"


@dataclass
class SeerFutureResidualConfig:
    enabled: bool = False
    hidden_dim: int = 128
    active_action_dims: int = 6
    residual_scale: float = 0.1
    gate_init_bias: float = -4.0
    residual_reg_weight: float = 1e-4
    gate_l1_weight: float = 0.0
    detach_base_action_for_future: bool = True
    # The environment's NormalizeEnv owns the final action clipping.  Keeping
    # this disabled by default is important: with the zero-initialized
    # residual branch, the adapter must be exactly identity on the base
    # action, including during the first optimization step.
    clamp_action: bool = False


def _activation(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "elu":
        return nn.ELU()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    raise ValueError(
        f"unsupported SEER residual activation {name!r}; "
        "expected elu, relu, or gelu"
    )


class SeerFutureResidualAdapter(nn.Module):
    """Fuse a one-step student future prediction into the action mean."""

    def __init__(
        self,
        cfg: SeerFutureResidualConfig,
        *,
        state_dim: int,
        teacher_dim_state: int,
        action_dim: int,
        progress_dim: int,
        contact_dim: int,
        activation: str = "elu",
    ):
        super().__init__()
        if cfg.active_action_dims < 1 or cfg.active_action_dims > action_dim:
            raise ValueError(
                "active_action_dims must be in [1, action_dim], got "
                f"{cfg.active_action_dims} for action_dim={action_dim}"
            )
        if cfg.hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if cfg.residual_scale <= 0.0:
            raise ValueError("residual_scale must be positive")
        if not th.isfinite(th.tensor(float(cfg.gate_init_bias))):
            raise ValueError("gate_init_bias must be finite")

        self.cfg = cfg
        self.state_dim = state_dim
        self.teacher_dim_state = teacher_dim_state
        self.action_dim = action_dim
        self.active_action_dims = cfg.active_action_dims
        self.feature_dim = (
            state_dim + teacher_dim_state + progress_dim + contact_dim + action_dim
        )

        self.residual_net = nn.Sequential(
            nn.Linear(self.feature_dim, cfg.hidden_dim),
            _activation(activation),
            nn.Linear(cfg.hidden_dim, cfg.active_action_dims),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(self.feature_dim, cfg.hidden_dim),
            _activation(activation),
            nn.Linear(cfg.hidden_dim, 1),
        )

        # Exact zero residual plus a near-zero gate gives a stable base-policy
        # start while leaving a small gradient through the residual branch.
        nn.init.zeros_(self.residual_net[-1].weight)
        nn.init.zeros_(self.residual_net[-1].bias)
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias, cfg.gate_init_bias)

    def forward(
        self,
        output: th.Tensor,
        student_state: th.Tensor,
        future_head: nn.Module,
    ) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
        if output.ndim != 3 or output.shape[-2] != 2:
            raise ValueError(
                f"student output must be (B,2,A), got {tuple(output.shape)}")
        if output.shape[-1] != self.action_dim:
            raise ValueError(
                f"action dimension mismatch: {output.shape[-1]} != "
                f"{self.action_dim}")
        if student_state.ndim != 2 or student_state.shape[-1] != self.state_dim:
            raise ValueError(
                "student_state must be (B,state_dim), got "
                f"{tuple(student_state.shape)}")
        if future_head is None:
            raise ValueError("SEER residual adapter requires an FT head")

        base_action = output[..., 0, :]
        action_for_future = (
            base_action.detach()
            if self.cfg.detach_base_action_for_future else base_action
        )
        h0 = future_head.proj_to_teacher_latent(student_state)
        h1 = future_head.step(h0, action_for_future)
        structured = future_head.decode(h1)
        contact = structured["contact_logit"].sigmoid()
        features = th.cat(
            [
                student_state,
                h1,
                structured["progress"],
                contact,
                base_action,
            ],
            dim=-1,
        )
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"residual feature dimension mismatch: {features.shape[-1]} != "
                f"{self.feature_dim}")

        residual_active = th.tanh(self.residual_net(features))
        residual_active = residual_active * self.cfg.residual_scale
        residual = th.cat(
            [
                residual_active,
                th.zeros(
                    (*residual_active.shape[:-1],
                     self.action_dim - self.active_action_dims),
                    device=residual_active.device,
                    dtype=residual_active.dtype,
                ),
            ],
            dim=-1,
        )
        gate = th.sigmoid(self.gate_net(features))
        applied_residual = gate * residual
        action = base_action + applied_residual
        if self.cfg.clamp_action:
            action = action.clamp(-1.0, 1.0)

        adapted_output = th.stack([action, output[..., 1, :]], dim=-2)
        return adapted_output, {
            "seer_future_latent": h1,
            "seer_future_progress": structured["progress"],
            "seer_future_contact": contact,
            "seer_future_residual": residual,
            "seer_future_applied_residual": applied_residual,
            "seer_future_gate": gate.squeeze(-1),
            "seer_future_base_action": base_action,
        }

    def regularization_loss(
        self, aux: Dict[str, th.Tensor]
    ) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
        residual = aux.get("seer_future_residual")
        gate = aux.get("seer_future_gate")
        if residual is None or gate is None:
            raise KeyError(
                "SEER residual regularization needs adapter forward outputs")
        # Penalize the raw, learnable six-dimensional correction.  Penalizing
        # gate * delta would let a small gate hide a large latent correction,
        # while averaging the padded action vector would dilute the intended
        # regularizer with inactive zero dimensions.
        residual_loss = residual[..., :self.active_action_dims].square().mean()
        gate_loss = gate.mean()
        total = (
            self.cfg.residual_reg_weight * residual_loss
            + self.cfg.gate_l1_weight * gate_loss
        )
        return total, {
            "residual_reg": residual_loss.detach(),
            "gate_l1": gate_loss.detach(),
            "total": total.detach(),
        }
