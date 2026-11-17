#!/usr/bin/env python3
"""Step 3 (a): student-side F_T distillation — auxiliary head + losses.

Defines the components needed to add F_T-style future representation
distillation to the existing student RMA training pipeline. Designed to be
plugged into `distill.py:StudentAgentRMA` along the same pattern as
`distill_fwd.py` (auxiliary-only — does NOT modify action path).

What student learns:

  o_t  -- student encoder -->  z_s_t                 (existing student state)
  z_s_t  -- student WM (F_S) -->  h_pred_{t+1..K}    (in TEACHER latent space)
  h_pred -- student decoder D_S -->  structured_pred  (per-step physical anchors)

  Loss:
    L_action      = existing teacher action MSE (unchanged)
    L_latent      = sum_k gamma^k latent_loss(h_pred_k, h_target_k)
    L_structured  = sum_k gamma^k structured_loss(structured_pred_k, structured_target_k)

  Targets:
    h_target_k         = frozen-teacher latent E_T(s_{t+k})  (from buffer / from
                         teacher's online state_net forward when teacher is loaded)
    structured_target_k = ground-truth physical deltas from env

NOTE: F_T (teacher-side) is NOT used as a target source by default. Two valid
target choices, swapped via `cfg.target_source`:
  "teacher_latent" : use teacher's latent at s_{t+k} directly  (default)
  "f_t_rollout"    : use F_T(h_t, a_t..a_{t+k-1}) as target    (codex's design B)

Both keep `policy(state)` unchanged. Use `f_t_rollout` to make F_T a
load-bearing method ingredient in the paper; use `teacher_latent` for the
cleanest possible ablation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch as th
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from modules_teacher_FT import (
    FTSidecar, FTSidecarConfig,
    latent_loss, structured_loss,
)


# =====================================================================
# Configs
# =====================================================================


def _activation(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "elu":
        return nn.ELU()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    raise ValueError(
        f"unsupported student FT activation {name!r}; "
        "expected elu, relu, or gelu"
    )

@dataclass
class StudentFTHeadConfig:
    """Config for the student-side F_T distillation head."""
    enabled: bool = False
    feed_to_policy: bool = False    # NOT supported (would change action)

    # Architecture (mirrors FTSidecar)
    dim_hidden: int = 128
    use_layernorm_input: bool = True
    activation: str = "elu"

    # Target dims
    rel_goal_dim:     int = 9
    object_state_dim: int = 13
    contact_dim:      int = 1
    progress_dim:     int = 1

    # Multi-step distillation
    K: int = 3
    discount: float = 1.0
    latent_mode: str = "cosine_mse"
    latent_weight: float = 1.0
    structured_weight: float = 1.0
    contrastive_weight: float = 0.0
    contrastive_temperature: float = 0.1

    # Per-field structured objective.  These defaults reproduce the legacy
    # hard-coded objective exactly, so adding the knobs does not change old
    # checkpoints or the default training protocol.  They affect only loss
    # construction and therefore do not change any checkpoint tensor shape.
    structured_rel_goal_delta_weight: float = 1.0
    structured_object_delta_weight: float = 0.25
    structured_contact_weight: float = 0.10
    structured_progress_weight: float = 0.50

    # Optional SEER-Anchor / SEER-CF-Anchor losses.  They default to zero so
    # legacy SEER graphs and objectives remain unchanged.
    current_latent_weight: float = 0.0
    current_latent_mode: str = "mse"
    counterfactual_weight: float = 0.0
    counterfactual_mode: str = "mse"
    counterfactual_eps: float = 0.05
    counterfactual_action_dims: int = 6
    counterfactual_include_teacher: bool = True
    counterfactual_clamp_action: bool = True
    # Arm-A centered action-effect calibration. Full latent prediction remains
    # supervised by the original K=1 objective; this branch isolates how an
    # action changes the future relative to the candidate set.
    centered_cf_direction_weight: float = 0.0
    centered_cf_scale_weight: float = 0.0
    centered_cf_scale_eps: float = 1e-6

    # How to get latent targets:
    #   "teacher_latent" : use teacher.state_net at s_{t+k} (clean, no F_T)
    #   "f_t_rollout"    : use F_T(h_t, a_t..a_{t+k-1})    (uses F_T as target)
    target_source: str = "teacher_latent"

    # Exploratory action-path variant. Zero initialization keeps the initial
    # action exactly equal to the loaded base SEER policy.
    future_film_enabled: bool = False
    future_film_hidden: int = 128
    future_film_scale: float = 0.1

    # Interpretable K=1/3 contact-future features directly modulate the action
    # decoder.  Kept separate from the legacy opaque-latent Future-FiLM arm.
    structured_phase_film_enabled: bool = False
    structured_phase_film_hidden: int = 128
    structured_phase_film_gamma_scale: float = 0.10
    structured_phase_film_beta_scale: float = 0.10
    structured_phase_film_max_horizon: int = 3
    structured_phase_film_detach_future: bool = True
    structured_phase_motion_threshold: float = 0.002
    structured_phase_motion_temperature: float = 0.001

    # Contact-Conditioned Future v1. The context is a goal-conditioned
    # representation produced from the student's three camera point clouds.
    # Its bridge is zero-initialized, so a Contact checkpoint initially
    # reproduces the existing future model exactly.
    contact_conditioned: bool = False
    contact_context_dim: int = 128
    contact_context_scale: float = 1.0


# =====================================================================
# Student aux head: F_S latent dynamics + structured decoder
# =====================================================================

class StudentFTHead(nn.Module):
    """Student-side WM head F_S, mirroring FTSidecar's interface.

    Predicts future TEACHER latent space from student's own current state.
    Used as auxiliary supervision; does NOT affect the action path.
    """

    def __init__(self, cfg: StudentFTHeadConfig, in_dim: int, dim_act: int,
                 teacher_dim_state: int):
        super().__init__()
        self.cfg = cfg
        self.in_dim = in_dim
        self.dim_act = dim_act
        self.teacher_dim_state = teacher_dim_state
        activation = _activation(cfg.activation)

        # Project student state to teacher latent space (so the rollout can
        # be measured directly against teacher latent targets).
        self.proj_to_teacher_latent = nn.Sequential(
            nn.Linear(in_dim, cfg.dim_hidden),
            activation,
            nn.Linear(cfg.dim_hidden, teacher_dim_state),
        )

        if cfg.contact_conditioned:
            if cfg.contact_context_dim <= 0:
                raise ValueError("contact_context_dim must be positive")
            self.contact_context_norm = nn.LayerNorm(cfg.contact_context_dim)
            self.contact_context_to_state = nn.Linear(
                cfg.contact_context_dim, in_dim)
            nn.init.zeros_(self.contact_context_to_state.weight)
            nn.init.zeros_(self.contact_context_to_state.bias)
        else:
            self.contact_context_norm = None
            self.contact_context_to_state = None

        # Latent dynamics in TEACHER latent space (residual MLP, same as F_T)
        self.input_norm = (nn.LayerNorm(teacher_dim_state)
                           if cfg.use_layernorm_input else nn.Identity())
        self.dyn = nn.Sequential(
            nn.Linear(teacher_dim_state + dim_act, cfg.dim_hidden),
            _activation(cfg.activation),
            nn.Linear(cfg.dim_hidden, cfg.dim_hidden),
            _activation(cfg.activation),
            nn.Linear(cfg.dim_hidden, teacher_dim_state),
        )

        # Structured decoder
        structured_dim = (cfg.rel_goal_dim + cfg.object_state_dim
                          + cfg.contact_dim + cfg.progress_dim)
        self.structured_dim = structured_dim
        self.decoder = nn.Sequential(
            nn.Linear(teacher_dim_state, cfg.dim_hidden),
            _activation(cfg.activation),
            nn.Linear(cfg.dim_hidden, structured_dim),
        )

    def step(self, h: th.Tensor, a: th.Tensor) -> th.Tensor:
        h_norm = self.input_norm(h)
        delta = self.dyn(th.cat([h_norm, a], dim=-1))
        return h + delta

    def decode(self, h: th.Tensor) -> Dict[str, th.Tensor]:
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

    def condition_state(
        self,
        z_s: th.Tensor,
        contact_context: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        """Fuse student-visible contact context without changing state size."""
        if self.contact_context_to_state is None:
            return z_s
        if contact_context is None:
            raise KeyError(
                "contact-conditioned F_S requires aux['contact_context']")
        if contact_context.shape[:-1] != z_s.shape[:-1]:
            raise ValueError(
                "contact context/student state batch mismatch: "
                f"{contact_context.shape} vs {z_s.shape}")
        if contact_context.shape[-1] != self.cfg.contact_context_dim:
            raise ValueError(
                "contact context feature mismatch: expected "
                f"{self.cfg.contact_context_dim}, got {contact_context.shape[-1]}")
        correction = self.contact_context_to_state(
            self.contact_context_norm(contact_context))
        return z_s + self.cfg.contact_context_scale * correction

    def project_current(
        self,
        z_s: th.Tensor,
        contact_context: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        return self.proj_to_teacher_latent(
            self.condition_state(z_s, contact_context))

    def rollout(self, z_s: th.Tensor, actions: th.Tensor,
                contact_context: Optional[th.Tensor] = None,
                ) -> Tuple[th.Tensor, List[Dict[str, th.Tensor]]]:
        """K-step rollout in teacher latent space.

        Args:
            z_s:     (B, in_dim) student state
            actions: (B, K, dim_act)
        Returns:
            h_traj_pred:        (B, K, teacher_dim_state)
            structured_pred:    list of K dicts
        """
        if z_s.dim() != 2:
            raise ValueError(f"z_s must be (B, in_dim), got {z_s.shape}")
        if actions.dim() != 3:
            raise ValueError(f"actions must be (B, K, dim_act), got {actions.shape}")
        K = actions.shape[1]
        h = self.project_current(z_s, contact_context)
        h_list: List[th.Tensor] = []
        s_list: List[Dict[str, th.Tensor]] = []
        for k in range(K):
            h = self.step(h, actions[:, k])
            h_list.append(h)
            s_list.append(self.decode(h))
        return th.stack(h_list, dim=1), s_list


# =====================================================================
# Loss builder
# =====================================================================

def compute_student_FT_loss(
    student_head: StudentFTHead,
    z_s_t: th.Tensor,
    actions: th.Tensor,                           # (B, K, dim_act)
    h_target_traj: th.Tensor,                     # (B, K, teacher_dim_state) — DETACHED upstream
    structured_targets: List[Dict[str, th.Tensor]],  # K dicts, each with target tensors
    h_negative_traj: Optional[th.Tensor] = None,
    contact_context: Optional[th.Tensor] = None,
    *,
    cfg: StudentFTHeadConfig,
) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
    """Run student rollout and compute multi-step distillation loss.

    All targets are detach()'d inside latent_loss / structured_loss helpers.
    """
    if h_target_traj.shape[1] != actions.shape[1]:
        raise ValueError(
            f"K mismatch: actions K={actions.shape[1]} "
            f"vs h_target_traj K={h_target_traj.shape[1]}"
        )
    K = actions.shape[1]

    h_pred_traj, struct_pred_traj = student_head.rollout(
        z_s_t, actions, contact_context=contact_context)

    structured_weights = {
        "rel_goal_delta": float(cfg.structured_rel_goal_delta_weight),
        "object_delta": float(cfg.structured_object_delta_weight),
        "contact_logit": float(cfg.structured_contact_weight),
        "progress": float(cfg.structured_progress_weight),
    }
    for name, value in structured_weights.items():
        if value < 0.0:
            raise ValueError(
                f"structured loss weight {name} must be non-negative, got {value}"
            )

    logs: Dict[str, th.Tensor] = {}
    lat_total = 0.0
    str_total = 0.0
    contrastive_total = 0.0
    # Raw field losses are accumulated with the same K-step discount as the
    # objective.  Their weighted counterparts make it possible to verify the
    # actual contribution of each field rather than only its native-unit MSE.
    structured_field_totals = {
        "rel_goal_delta": 0.0,
        "object_delta": 0.0,
        "contact_logit": 0.0,
        "progress": 0.0,
    }
    target_stat_totals = {
        "rel_goal_delta_abs_mean": 0.0,
        "object_delta_abs_mean": 0.0,
        "contact_rate": 0.0,
        "progress_abs_mean": 0.0,
    }
    discount_total = 0.0
    for k in range(K):
        w = cfg.discount ** k
        discount_total += w
        lat_k = latent_loss(h_pred_traj[:, k], h_target_traj[:, k],
                            mode=cfg.latent_mode)
        lat_total = lat_total + w * lat_k
        logs[f"latent_loss_k{k+1}"] = lat_k.detach()
        if structured_targets is not None and len(structured_targets) > k:
            str_k, str_logs_k = structured_loss(struct_pred_traj[k],
                                                structured_targets[k],
                                                weights=structured_weights)
            str_total = str_total + w * str_k
            for kk, vv in str_logs_k.items():
                logs[f"{kk}_k{k+1}"] = vv
            structured_field_totals["rel_goal_delta"] = (
                structured_field_totals["rel_goal_delta"]
                + w * str_logs_k["rel_goal_delta_mse"])
            structured_field_totals["object_delta"] = (
                structured_field_totals["object_delta"]
                + w * str_logs_k["object_delta_mse"])
            structured_field_totals["contact_logit"] = (
                structured_field_totals["contact_logit"]
                + w * str_logs_k["contact_bce"])
            structured_field_totals["progress"] = (
                structured_field_totals["progress"]
                + w * str_logs_k["progress_mse"])

            target_k = structured_targets[k]
            target_stat_totals["rel_goal_delta_abs_mean"] = (
                target_stat_totals["rel_goal_delta_abs_mean"]
                + w * target_k["rel_goal_delta"].detach().abs().mean())
            target_stat_totals["object_delta_abs_mean"] = (
                target_stat_totals["object_delta_abs_mean"]
                + w * target_k["object_delta"].detach().abs().mean())
            target_stat_totals["contact_rate"] = (
                target_stat_totals["contact_rate"]
                + w * target_k["contact"].detach().float().mean())
            target_stat_totals["progress_abs_mean"] = (
                target_stat_totals["progress_abs_mean"]
                + w * target_k["progress"].detach().abs().mean())
        if cfg.contrastive_weight > 0.0:
            if h_negative_traj is None:
                raise ValueError(
                    "contrastive_weight > 0 requires counterfactual latent targets"
                )
            pred = F.normalize(h_pred_traj[:, k], dim=-1)
            positive = F.normalize(h_target_traj[:, k].detach(), dim=-1)
            negative = F.normalize(h_negative_traj[:, k].detach(), dim=-1)
            positive_similarity = (pred * positive).sum(dim=-1)
            negative_similarity = (pred * negative).sum(dim=-1)
            logits = th.stack(
                [positive_similarity, negative_similarity], dim=-1
            ) / cfg.contrastive_temperature
            labels = th.zeros(logits.shape[0], dtype=th.long, device=logits.device)
            contrastive_k = F.cross_entropy(logits, labels)
            contrastive_total = contrastive_total + w * contrastive_k
            logs[f"contrastive_loss_k{k+1}"] = contrastive_k.detach()
            logs[f"positive_similarity_k{k+1}"] = positive_similarity.mean().detach()
            logs[f"negative_similarity_k{k+1}"] = negative_similarity.mean().detach()

    total = cfg.latent_weight * lat_total
    if structured_targets is not None:
        total = total + cfg.structured_weight * str_total
    total = total + cfg.contrastive_weight * contrastive_total

    logs["latent_total"]     = lat_total if isinstance(lat_total, th.Tensor) else th.tensor(lat_total)
    logs["structured_total"] = str_total if isinstance(str_total, th.Tensor) else th.tensor(str_total)
    logs["contrastive_total"] = (
        contrastive_total
        if isinstance(contrastive_total, th.Tensor)
        else th.tensor(contrastive_total)
    )
    if structured_targets is not None:
        for field, raw_total in structured_field_totals.items():
            raw_tensor = (
                raw_total.detach()
                if isinstance(raw_total, th.Tensor)
                else th.tensor(raw_total)
            )
            logs[f"structured_{field}_unweighted"] = raw_tensor
            logs[f"structured_{field}_weighted"] = (
                raw_tensor * structured_weights[field])
        normalizer = max(discount_total, 1e-12)
        for name, value in target_stat_totals.items():
            value_tensor = (
                value.detach()
                if isinstance(value, th.Tensor)
                else th.tensor(value)
            )
            logs[f"target_{name}"] = value_tensor / normalizer
    return total, logs


# =====================================================================
# Helpers for target generation at training time
# =====================================================================

@th.no_grad()
def build_FT_rollout_targets(
    f_t: FTSidecar,
    h_t: th.Tensor,
    actions: th.Tensor,
) -> Tuple[th.Tensor, List[Dict[str, th.Tensor]]]:
    """Use F_T to produce latent + structured targets (codex's design B)."""
    h_target_traj, structured_traj = f_t.rollout(h_t, actions)
    h_target_traj = h_target_traj.detach()
    structured_traj = [
        {kk: vv.detach() for kk, vv in s.items()} for s in structured_traj
    ]
    # Convert F_T's contact_logit to "contact" probability for structured_loss API
    structured_targets = []
    for s in structured_traj:
        structured_targets.append({
            "rel_goal_delta": s["rel_goal_delta"],
            "object_delta":   s["object_delta"],
            "contact":        s["contact_logit"].sigmoid(),
            "progress":       s["progress"],
        })
    return h_target_traj, structured_targets


# =====================================================================
# F_T loader (load Step 2 ckpt, freeze it for student-side use as target)
# =====================================================================

def load_FT_frozen(ckpt_path: str, device: th.device) -> FTSidecar:
    """Load F_T from Step 2 ckpt and return it in eval+frozen mode."""
    ckpt = th.load(ckpt_path, map_location=device)
    cfg_dict = ckpt["cfg"]
    cfg = FTSidecarConfig(**{k: v for k, v in cfg_dict.items()
                             if k in FTSidecarConfig.__dataclass_fields__})
    f_t = FTSidecar(cfg).to(device)
    f_t.load_state_dict(ckpt["f_t_state_dict"])
    f_t.eval()
    for p in f_t.parameters():
        p.requires_grad_(False)
    val_total = ckpt.get("val_total")
    val_text = (f"{float(val_total):.5f}"
                if isinstance(val_total, (int, float)) else "n/a")
    print(f"Loaded F_T from {ckpt_path} (epoch={ckpt.get('epoch')}, "
          f"val_total={val_text})")
    return f_t
