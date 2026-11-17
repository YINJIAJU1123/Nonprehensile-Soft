#!/usr/bin/env python3
"""Teacher-side sidecar forward model F_T.

Used together with a frozen old teacher (E_T, pi_T). The action path stays
bit-identical to the old teacher; F_T is trained separately on (h_t, a_t,
h_{t+1}, structured_target) tuples to predict K-step latent dynamics in the
teacher's frozen latent space.

Architecture:
    Latent dynamics (residual MLP):
        h_next = h + MLP([LayerNorm(h), a])
    Structured decoder (separate MLP):
        D_T(h) -> {rel_goal_delta, object_delta, contact_logit, progress}

K-step rollout uses teacher-forcing actions during training; the same code
path supports autoregressive rollout for test-time imagination.

This module owns NO references to teacher modules. It is constructed,
trained, and consumed independently. The frozen-teacher invariant is enforced
by Step 0's action invariance check, not here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch as th
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FTSidecarConfig:
    # Required (must match teacher)
    dim_state: int = -1   # teacher state encoder output dim
    dim_act:   int = -1   # teacher action dim

    # Architecture
    dim_hidden: int = 128
    use_layernorm_input: bool = True
    activation: str = "elu"

    # Structured target dims (must match teacher rollout collector)
    rel_goal_dim:     int = 9
    object_state_dim: int = 13
    contact_dim:      int = 1
    progress_dim:     int = 1


def _act(name: str) -> nn.Module:
    return {
        "elu":  nn.ELU(),
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
    }[name.lower()]


class FTSidecar(nn.Module):
    """F_T(h_t, a_t) -> h_{t+1};  D_T(h_{t+1}) -> structured_future."""

    def __init__(self, cfg: FTSidecarConfig):
        super().__init__()
        if cfg.dim_state <= 0 or cfg.dim_act <= 0:
            raise ValueError(f"FTSidecar needs positive dims, got {cfg}")
        self.cfg = cfg

        # ---- Latent dynamics: residual MLP with LayerNorm ----
        self.input_norm = (nn.LayerNorm(cfg.dim_state)
                           if cfg.use_layernorm_input else nn.Identity())
        in_dim = cfg.dim_state + cfg.dim_act
        self.dyn = nn.Sequential(
            nn.Linear(in_dim, cfg.dim_hidden),
            _act(cfg.activation),
            nn.Linear(cfg.dim_hidden, cfg.dim_hidden),
            _act(cfg.activation),
            nn.Linear(cfg.dim_hidden, cfg.dim_state),
        )

        # ---- Structured decoder ----
        structured_dim = (cfg.rel_goal_dim + cfg.object_state_dim
                          + cfg.contact_dim + cfg.progress_dim)
        self.structured_dim = structured_dim
        self.decoder = nn.Sequential(
            nn.Linear(cfg.dim_state, cfg.dim_hidden),
            _act(cfg.activation),
            nn.Linear(cfg.dim_hidden, structured_dim),
        )

    # ---- Single-step ----
    def step(self, h: th.Tensor, a: th.Tensor) -> th.Tensor:
        """Predict h_{t+1} = h_t + MLP([LN(h_t), a_t])."""
        h_norm = self.input_norm(h)
        delta = self.dyn(th.cat([h_norm, a], dim=-1))
        return h + delta

    def decode(self, h: th.Tensor) -> Dict[str, th.Tensor]:
        """Structured future predictions split by field."""
        out = self.decoder(h)
        cfg = self.cfg
        i0 = 0
        i1 = i0 + cfg.rel_goal_dim
        i2 = i1 + cfg.object_state_dim
        i3 = i2 + cfg.contact_dim
        i4 = i3 + cfg.progress_dim
        return {
            "rel_goal_delta": out[..., i0:i1],
            "object_delta":   out[..., i1:i2],
            "contact_logit":  out[..., i2:i3],
            "progress":       out[..., i3:i4],
        }

    # ---- K-step rollout ----
    def rollout(
        self,
        h0: th.Tensor,
        actions: th.Tensor,
    ) -> Tuple[th.Tensor, List[Dict[str, th.Tensor]]]:
        """Teacher-forcing K-step rollout.

        Args:
            h0:      (B, dim_state) initial latent
            actions: (B, K, dim_act) action sequence (taken from teacher rollout)

        Returns:
            h_traj:          (B, K, dim_state) predicted latents h_1..h_K
            structured_traj: list of K dicts (per-step structured predictions)
        """
        if h0.dim() != 2:
            raise ValueError(f"h0 must be (B, dim_state), got {h0.shape}")
        if actions.dim() != 3:
            raise ValueError(f"actions must be (B, K, dim_act), got {actions.shape}")
        K = actions.shape[1]
        h = h0
        h_list: List[th.Tensor] = []
        s_list: List[Dict[str, th.Tensor]] = []
        for k in range(K):
            h = self.step(h, actions[:, k])
            h_list.append(h)
            s_list.append(self.decode(h))
        h_traj = th.stack(h_list, dim=1)
        return h_traj, s_list


# =====================================================================
# Loss helpers
# =====================================================================

def latent_loss(
    h_pred: th.Tensor,
    h_target: th.Tensor,
    *,
    mode: str = "cosine_mse",
    layernorm: bool = True,
) -> th.Tensor:
    """Latent prediction loss with magnitude-aware variants.

    Modes:
        "mse"          — plain MSE on raw latents
        "norm_mse"     — MSE after LayerNorm on both sides (scale-invariant)
        "cosine"       — 1 - cos_sim
        "cosine_mse"   — 0.5 * cos + 0.5 * norm_mse (default; balances both)

    h_target should be detach()'d by the caller (frozen teacher invariant).
    """
    h_target = h_target.detach()
    if h_pred.shape != h_target.shape:
        raise ValueError(f"shape mismatch: {h_pred.shape} vs {h_target.shape}")

    if mode == "mse":
        return F.mse_loss(h_pred, h_target)
    if mode in ("norm_mse", "cosine_mse"):
        if layernorm:
            norm_shape = (h_pred.shape[-1],)
            hp = F.layer_norm(h_pred, norm_shape)
            ht = F.layer_norm(h_target, norm_shape)
        else:
            hp, ht = h_pred, h_target
        norm_mse = F.mse_loss(hp, ht)
        if mode == "norm_mse":
            return norm_mse
        cos = 1.0 - F.cosine_similarity(h_pred, h_target, dim=-1).mean()
        return 0.5 * cos + 0.5 * norm_mse
    if mode == "cosine":
        return 1.0 - F.cosine_similarity(h_pred, h_target, dim=-1).mean()
    raise ValueError(f"unknown latent loss mode: {mode}")


def structured_loss(
    pred: Dict[str, th.Tensor],
    target: Dict[str, th.Tensor],
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
    """Per-field weighted loss; returns (total, per-field dict for logging)."""
    if weights is None:
        weights = {
            "rel_goal_delta": 1.0,
            "object_delta":   0.25,
            "contact_logit":  0.1,
            "progress":       0.5,
        }
    logs: Dict[str, th.Tensor] = {}
    total = 0.0
    target = {k: v.detach() for k, v in target.items()}
    rel = F.mse_loss(pred["rel_goal_delta"], target["rel_goal_delta"])
    obj = F.mse_loss(pred["object_delta"],   target["object_delta"])
    pro = F.mse_loss(pred["progress"],       target["progress"])
    con = F.binary_cross_entropy_with_logits(
        pred["contact_logit"], target["contact"].clamp(0.0, 1.0))
    logs["rel_goal_delta_mse"] = rel.detach()
    logs["object_delta_mse"]   = obj.detach()
    logs["progress_mse"]       = pro.detach()
    logs["contact_bce"]        = con.detach()
    total = (weights["rel_goal_delta"] * rel
             + weights["object_delta"] * obj
             + weights["contact_logit"] * con
             + weights["progress"]     * pro)
    return total, logs


def multistep_rollout_loss(
    f_t: FTSidecar,
    h0_target: th.Tensor,
    actions: th.Tensor,
    h_target_traj: th.Tensor,
    structured_targets: Optional[List[Dict[str, th.Tensor]]] = None,
    *,
    discount: float = 1.0,
    latent_mode: str = "cosine_mse",
    latent_weight: float = 1.0,
    structured_weight: float = 1.0,
) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
    """Multi-step rollout loss, summed over K steps with discount.

    Args:
        f_t:                 the FTSidecar
        h0_target:           (B, dim_state) initial latent (oracle, detached)
        actions:             (B, K, dim_act) teacher-forcing action sequence
        h_target_traj:       (B, K, dim_state) oracle latents h_1..h_K (detached)
        structured_targets:  list of K dicts of oracle structured targets, or None
        discount:            gamma^k weighting (1.0 = uniform)
        latent_mode:         see latent_loss()
        latent_weight:       coefficient on latent term
        structured_weight:   coefficient on structured term
    """
    if h_target_traj.shape[1] != actions.shape[1]:
        raise ValueError(
            f"K mismatch: actions K={actions.shape[1]} "
            f"vs h_target_traj K={h_target_traj.shape[1]}")
    K = actions.shape[1]

    # Frozen-teacher invariant: teacher latents/actions/structured targets are
    # supervision only. Detach here as a second line of defense even if the
    # caller forgot to wrap teacher encoding in no_grad().
    h0_target = h0_target.detach()
    actions = actions.detach()
    h_target_traj = h_target_traj.detach()
    if structured_targets is not None:
        structured_targets = [
            {kk: vv.detach() for kk, vv in target_k.items()}
            for target_k in structured_targets
        ]

    h_pred_traj, structured_pred_traj = f_t.rollout(h0_target, actions)

    logs: Dict[str, th.Tensor] = {}
    total = 0.0
    lat_total = 0.0
    str_total = 0.0
    for k in range(K):
        w = discount ** k
        lat_k = latent_loss(h_pred_traj[:, k], h_target_traj[:, k],
                            mode=latent_mode)
        lat_total = lat_total + w * lat_k
        logs[f"latent_loss_k{k+1}"] = lat_k.detach()
        if structured_targets is not None:
            s_k, s_logs = structured_loss(structured_pred_traj[k],
                                          structured_targets[k])
            str_total = str_total + w * s_k
            for kk, vv in s_logs.items():
                logs[f"{kk}_k{k+1}"] = vv

    total = latent_weight * lat_total
    if structured_targets is not None:
        total = total + structured_weight * str_total

    logs["latent_total"] = (lat_total.detach() if isinstance(lat_total, th.Tensor)
                            else th.tensor(lat_total))
    logs["structured_total"] = (
        str_total.detach() if isinstance(str_total, th.Tensor)
        else th.tensor(str_total)
    )
    return total, logs
