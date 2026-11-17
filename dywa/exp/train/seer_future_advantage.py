#!/usr/bin/env python3
"""Calibrated future-value selector for the SEER-Adv exploratory arm.

The selector is identity initialized: its value head emits the same score for
all candidates, so deterministic tie breaking keeps the unmodified base
action.  It searches only the first configured action dimensions and leaves
the gain dimensions untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch as th
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SeerFutureAdvantageConfig:
    enabled: bool = False
    active_action_dims: int = 6
    candidate_eps: float = 0.05
    clamp_candidates: bool = True
    value_hidden_dim: int = 128
    ranking_weight: float = 0.25
    value_regression_weight: float = 0.10
    ranking_temperature: float = 0.10
    base_action_loss_weight: float = 0.50
    utility_progress_weight: float = 1.0
    utility_value_discount: float = 0.99
    advantage_margin: float = 0.0
    advantage_temperature: float = 0.10
    contact_gate_power: float = 1.0


def _activation(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "elu":
        return nn.ELU()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    raise ValueError(f"unsupported activation: {name}")


def build_local_candidates(base_action: th.Tensor, cfg: SeerFutureAdvantageConfig):
    if base_action.ndim != 2:
        raise ValueError(f"base_action must be (B,A), got {base_action.shape}")
    action_dim = base_action.shape[-1]
    active = min(cfg.active_action_dims, action_dim)
    if active < 1 or cfg.candidate_eps <= 0.0:
        raise ValueError("invalid future-advantage candidate configuration")
    candidates = [base_action]
    for dim in range(active):
        delta = th.zeros_like(base_action)
        delta[..., dim] = cfg.candidate_eps
        candidates.extend((base_action + delta, base_action - delta))
    out = th.stack(candidates, dim=1)
    if cfg.clamp_candidates:
        # Preserve candidate zero exactly; the environment owns final clipping.
        perturbed = out[:, 1:].clamp(-1.0, 1.0)
        out = th.cat((base_action[:, None], perturbed), dim=1)
    return out


class SeerFutureValueHead(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, activation: str = "elu"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            _activation(activation),
            nn.Linear(hidden_dim, hidden_dim),
            _activation(activation),
            nn.Linear(hidden_dim, 1),
        )
        # Equal initial scores force argmax to candidate zero (the base action).
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, latent: th.Tensor) -> th.Tensor:
        return self.net(latent).squeeze(-1)


class SeerFutureAdvantageSelector(nn.Module):
    def __init__(self, cfg: SeerFutureAdvantageConfig, *, latent_dim: int,
                 action_dim: int, activation: str = "elu"):
        super().__init__()
        if cfg.value_hidden_dim < 1:
            raise ValueError("value_hidden_dim must be positive")
        if cfg.ranking_temperature <= 0.0 or cfg.advantage_temperature <= 0.0:
            raise ValueError("selector temperatures must be positive")
        if cfg.contact_gate_power <= 0.0:
            raise ValueError("contact_gate_power must be positive")
        self.cfg = cfg
        self.action_dim = action_dim
        self.value_head = SeerFutureValueHead(
            latent_dim, cfg.value_hidden_dim, activation=activation)

    def forward(self, output: th.Tensor, student_state: th.Tensor, future_head):
        if output.ndim != 3 or output.shape[-2] != 2:
            raise ValueError(f"student output must be (B,2,A), got {output.shape}")
        base_action = output[..., 0, :]
        candidates = build_local_candidates(base_action, self.cfg)
        batch, count, action_dim = candidates.shape
        h0 = future_head.proj_to_teacher_latent(student_state)
        h0_flat = h0[:, None].expand(-1, count, -1).reshape(batch * count, -1)
        actions_flat = candidates.reshape(batch * count, action_dim)
        future = future_head.step(h0_flat, actions_flat)
        decoded = future_head.decode(future)
        values = self.value_head(future).reshape(batch, count)
        contact = decoded["contact_logit"].sigmoid().reshape(batch, count)

        best_index = values.argmax(dim=1)
        best_value = values.gather(1, best_index[:, None]).squeeze(1)
        base_value = values[:, 0]
        advantage = best_value - base_value
        switch = (best_index != 0) & (advantage > self.cfg.advantage_margin)
        selected_index = th.where(switch, best_index, th.zeros_like(best_index))
        selected = candidates.gather(
            1, selected_index[:, None, None].expand(-1, 1, action_dim)).squeeze(1)
        selected_contact = contact.gather(
            1, selected_index[:, None]).squeeze(1)
        confidence = th.sigmoid(
            (advantage - self.cfg.advantage_margin) /
            self.cfg.advantage_temperature)
        gate = selected_contact.pow(self.cfg.contact_gate_power) * confidence
        gate = gate * switch.to(gate.dtype)
        final_action = base_action + gate[:, None] * (selected - base_action)
        adapted = th.stack((final_action, output[..., 1, :]), dim=-2)
        return adapted, {
            "future_advantage_base_output": output,
            "future_advantage_candidates": candidates,
            "future_advantage_values": values,
            "future_advantage_contact": contact,
            "future_advantage_selected_index": selected_index,
            "future_advantage_advantage": advantage,
            "future_advantage_gate": gate,
            "future_advantage_switch": switch,
        }

    def ranking_loss(self, predicted: th.Tensor, target: th.Tensor):
        if predicted.shape != target.shape:
            raise ValueError(
                f"future-value shape mismatch: {predicted.shape} != {target.shape}")
        cfg = self.cfg
        target = target.detach()
        target_prob = F.softmax(target / cfg.ranking_temperature, dim=1)
        pred_log_prob = F.log_softmax(
            predicted / cfg.ranking_temperature, dim=1)
        ranking = F.kl_div(pred_log_prob, target_prob, reduction="batchmean")
        pred_centered = predicted - predicted[:, :1]
        target_centered = target - target[:, :1]
        regression = F.smooth_l1_loss(pred_centered, target_centered)
        total = (cfg.ranking_weight * ranking +
                 cfg.value_regression_weight * regression)
        top1 = (predicted.argmax(dim=1) == target.argmax(dim=1)).float().mean()
        return total, {
            "ranking": ranking.detach(),
            "value_regression": regression.detach(),
            "top1_agreement": top1.detach(),
        }
