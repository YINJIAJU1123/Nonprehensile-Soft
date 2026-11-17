#!/usr/bin/env python3
"""Step 3 (b): student agent with F_T distillation aux head.

Mirrors `distill_fwd.py`'s integration pattern — adds an auxiliary head and
loss to the existing StudentAgentRMA, leaves the action path identical.

Differences from distill_fwd:
- The aux head is StudentFTHead (multi-step latent dynamics in teacher's
  latent space), not a one-step structured head.
- Targets come from either:
    (a) the loaded frozen old teacher's state_net forward over future obs,
        OR
    (b) the loaded F_T's rollout (codex design B; uses F_T as method anchor).
    Selectable via cfg.student_FT_head.target_source.

The optional ``seer_future_residual`` branch is a separate, disabled-by-default
action-path experiment. With it disabled, this module preserves the formal
SEER action path exactly.

The DAgger trainer below (in train_rma_FT.py) injects the K future obs +
action sequence into the obs dict via `_FT_<key>_k{1..K}` so that this
agent's update_policy() can compute targets in-place.
"""
from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, replace
from typing import Dict, List, Optional, Tuple

import torch as th
import torch.nn as nn
from torch.cuda.amp import autocast
from contextlib import nullcontext

from distill import StudentAgentRMA as BaseStudentAgentRMA
from student_FT_distill import (
    StudentFTHeadConfig,
    StudentFTHead,
    compute_student_FT_loss,
)
from modules_teacher_FT import FTSidecar, latent_loss
from seer_future_residual import (
    SeerFutureResidualAdapter,
    SeerFutureResidualConfig,
)
from seer_future_advantage import (
    SeerFutureAdvantageConfig,
    SeerFutureAdvantageSelector,
)
from seer_history import SeerHistoryEncoder


class StudentAgentRMA_FT(BaseStudentAgentRMA):
    """Student RMA + multi-step F_T-style distillation head.

    With the residual branch disabled, the action path is identical to
    distill.py and the FT head is purely auxiliary. The opt-in residual branch
    is intentionally a separate method family.
    """

    @dataclass
    class StudentAgentRMAConfig(BaseStudentAgentRMA.StudentAgentRMAConfig):
        student_FT_head: StudentFTHeadConfig = field(
            default_factory=StudentFTHeadConfig)
        seer_future_residual: SeerFutureResidualConfig = field(
            default_factory=SeerFutureResidualConfig)
        seer_future_advantage: SeerFutureAdvantageConfig = field(
            default_factory=SeerFutureAdvantageConfig)

        def __init__(self, **kwds):
            for f in fields(self):
                if f.default is not MISSING:
                    setattr(self, f.name, f.default)
                elif f.default_factory is not MISSING:
                    setattr(self, f.name, f.default_factory())
            names = set([f.name for f in fields(self)])
            for k, v in kwds.items():
                if k in names:
                    setattr(self, k, v)
            self.__post_init__()

    def __init__(self, cfg: StudentAgentRMAConfig, writer, device,
                 *, teacher_dim_state: int, teacher_dim_act: int,
                 frozen_teacher=None, frozen_FT: Optional[FTSidecar] = None):
        super().__init__(cfg, writer, device)
        ft_cfg = cfg.student_FT_head
        residual_cfg = cfg.seer_future_residual
        advantage_cfg = cfg.seer_future_advantage
        if ft_cfg.enabled and ft_cfg.feed_to_policy:
            raise NotImplementedError(
                "student_FT_head.feed_to_policy is intentionally not implemented "
                "(it would change the action path — violates frozen-teacher invariant)")
        if ft_cfg.contact_conditioned and not cfg.view_contact_transformer:
            raise ValueError(
                "student_FT_head.contact_conditioned requires "
                "student.view_contact_transformer=true")
        if (ft_cfg.contact_conditioned
                and ft_cfg.contact_context_dim != cfg.embed_size):
            raise ValueError(
                "contact_context_dim must match student embed_size: "
                f"{ft_cfg.contact_context_dim} vs {cfg.embed_size}")

        if cfg.use_history:
            # Replace only the SEER instance. The shared legacy HistoryEncoder
            # remains untouched so the DyWA baseline keeps its old behavior.
            history_cfg = replace(cfg.history_tokenizer, num_envs=cfg.batch_size)
            self.history_tokenizer = SeerHistoryEncoder(
                history_cfg,
                embed_size=cfg.embed_size,
                num_tokens=cfg.point_tokenizer.num_tokens + len(self.input_keys),
                norm=cfg.norm,
            ).to(device)
            self.history_tokenizer.safe_copy = cfg.history_safe_copy

        # Target modules are owned by the trainer, not by the student.  Keeping
        # either one as an attribute would register it as a child nn.Module.
        # That makes student.train() switch the frozen teacher back to train
        # mode and also serializes both modules into every student checkpoint.
        # The student only consumes targets already injected into `obs`, so it
        # deliberately keeps no reference to these modules.
        self.teacher_dim_state = teacher_dim_state
        self.teacher_dim_act = teacher_dim_act

        if ft_cfg.enabled:
            self.FT_head = StudentFTHead(
                ft_cfg,
                in_dim=cfg.aggregator.state_size,
                dim_act=teacher_dim_act,
                teacher_dim_state=teacher_dim_state,
            ).to(device)
            self.student_FT_losses = {
                "total": 0.0,
                "latent_total": 0.0,
                "structured_total": 0.0,
                "contrastive_total": 0.0,
                "structured_rel_goal_delta_unweighted": 0.0,
                "structured_rel_goal_delta_weighted": 0.0,
                "structured_object_delta_unweighted": 0.0,
                "structured_object_delta_weighted": 0.0,
                "structured_contact_logit_unweighted": 0.0,
                "structured_contact_logit_weighted": 0.0,
                "structured_progress_unweighted": 0.0,
                "structured_progress_weighted": 0.0,
                "target_rel_goal_delta_abs_mean": 0.0,
                "target_object_delta_abs_mean": 0.0,
                "target_contact_rate": 0.0,
                "target_progress_abs_mean": 0.0,
                "phase_gamma_abs_mean": 0.0,
                "phase_beta_abs_mean": 0.0,
                "phase_gamma_saturation": 0.0,
                "phase_beta_saturation": 0.0,
                "phase_k1_onset": 0.0,
                "phase_k1_persistence": 0.0,
                "phase_k1_stalled": 0.0,
                "phase_k1_moving": 0.0,
                "phase_k3_onset": 0.0,
                "phase_k3_persistence": 0.0,
                "phase_k3_stalled": 0.0,
                "phase_k3_moving": 0.0,
                "contact_future_correction_l1": 0.0,
                "positive_similarity_k1": 0.0,
                "negative_similarity_k1": 0.0,
                "current_anchor": 0.0,
                "counterfactual": 0.0,
                "cf_student_variance": 0.0,
                "cf_teacher_variance": 0.0,
                "cf_sensitivity_ratio": 0.0,
                "cf_change_direction_cosine": 0.0,
                "cf_centered_direction": 0.0,
                "cf_centered_scale": 0.0,
            }
            if ft_cfg.future_film_enabled:
                state_dim = int(cfg.aggregator.state_size)
                self.future_film = nn.Sequential(
                    nn.Linear(state_dim + teacher_dim_state,
                              ft_cfg.future_film_hidden),
                    nn.GELU(),
                    nn.Linear(ft_cfg.future_film_hidden, 2 * state_dim),
                ).to(device)
                nn.init.zeros_(self.future_film[-1].weight)
                nn.init.zeros_(self.future_film[-1].bias)
            else:
                self.future_film = None
            if ft_cfg.structured_phase_film_enabled:
                if ft_cfg.future_film_enabled:
                    raise ValueError(
                        "generic Future-FiLM and structured Phase-FiLM are "
                        "mutually exclusive")
                if not cfg.view_contact_transformer:
                    raise ValueError(
                        "structured Phase-FiLM requires view_contact_transformer")
                if int(ft_cfg.structured_phase_film_max_horizon) < 3:
                    raise ValueError(
                        "structured Phase-FiLM v1 requires horizon >= 3")
                state_dim = int(cfg.aggregator.state_size)
                # state + action + current contact + two horizons of:
                # 24 decoded structured values and 4 soft phase values.
                feature_dim = state_dim + teacher_dim_act + 1 + 2 * (
                    self.FT_head.structured_dim + 4)
                self.structured_phase_film = nn.Sequential(
                    nn.Linear(feature_dim,
                              ft_cfg.structured_phase_film_hidden),
                    nn.ELU(),
                    nn.Linear(ft_cfg.structured_phase_film_hidden,
                              2 * state_dim),
                ).to(device)
                nn.init.zeros_(self.structured_phase_film[-1].weight)
                nn.init.zeros_(self.structured_phase_film[-1].bias)
            else:
                self.structured_phase_film = None
        else:
            self.FT_head = None
            self.student_FT_losses = {}
            self.future_film = None
            self.structured_phase_film = None

        if residual_cfg.enabled:
            if self.FT_head is None:
                raise ValueError(
                    "seer_future_residual.enabled requires "
                    "student_FT_head.enabled=true")
            if cfg.estimate_level != "action":
                raise ValueError(
                    "seer_future_residual requires student estimate_level='action'")
            if cfg.action_size != teacher_dim_act:
                raise ValueError(
                    "student action_size must match teacher_dim_act for "
                    "the SEER residual adapter")
            self.seer_future_residual = SeerFutureResidualAdapter(
                residual_cfg,
                state_dim=cfg.aggregator.state_size,
                teacher_dim_state=teacher_dim_state,
                action_dim=teacher_dim_act,
                progress_dim=ft_cfg.progress_dim,
                contact_dim=ft_cfg.contact_dim,
                activation=ft_cfg.activation,
            ).to(device)
        else:
            self.seer_future_residual = None

        if residual_cfg.enabled and advantage_cfg.enabled:
            raise ValueError(
                "SEER residual and future-advantage selector are mutually exclusive")
        if advantage_cfg.enabled:
            if self.FT_head is None:
                raise ValueError(
                    "seer_future_advantage.enabled requires student_FT_head.enabled=true")
            if cfg.estimate_level != "action" or cfg.action_size != teacher_dim_act:
                raise ValueError(
                    "SEER future advantage requires action-level student output")
            self.seer_future_advantage = SeerFutureAdvantageSelector(
                advantage_cfg,
                latent_dim=teacher_dim_state,
                action_dim=teacher_dim_act,
                activation=ft_cfg.activation,
            ).to(device)
        else:
            self.seer_future_advantage = None

        self.seer_future_residual_losses = {
            "total": 0.0,
            "residual_reg": 0.0,
            "gate_l1": 0.0,
        }
        self.seer_future_advantage_losses = {
            "total": 0.0,
            "ranking": 0.0,
            "value_regression": 0.0,
            "top1_agreement": 0.0,
            "base_action": 0.0,
            "gate": 0.0,
            "switch_rate": 0.0,
        }
        # Recreate the optimizer after replacing history and adding all SEER
        # modules. This does not affect the DyWA optimizer construction.
        self.reset_optimizer()

    # ---- forward: capture aggregator state for aux head, no action change ----
    def reset(self, obs):
        """Start a SEER episode with an empty persistent history buffer."""
        if self.cfg.use_history:
            device = obs["partial_cloud"].device
            done = th.ones(
                self.cfg.batch_size,
                dtype=th.bool,
                device=device,
            )
            self.history_tokenizer.reset_history(done)
        return super().reset(obs)

    def _forward_impl(self, obs):
        # Reuse the parent's forward and grab the aggregator state via a hook-
        # less path: we need (output, aux) plus the raw aggregator state. The
        # simplest robust approach is to call parent forward then re-derive
        # state via the same aggregator path; here we copy the parent body so
        # we can capture `state` directly.
        cfg = self.cfg

        aux = dict()
        ctx_tokens = [
            self.tokenizer[k](obs[k].detach().clone())[:, None]
            for k in self.input_keys if k in obs
        ]
        # Keep the original merged-cloud pathway as the warm-start anchor.
        # The optional corrected multi-view branch consumes raw camera tensors
        # and adds a zero-initialized goal/contact-conditioned correction.
        self.vision_tokens = pcd_tokens = self.point_tokenizer(
            obs["partial_cloud"])
        if cfg.use_gpcd:
            gpcd_tokens = self.point_tokenizer(obs["goal_cloud"])
            if cfg.view_contact_transformer:
                pcd_tokens, contact_aux = self.fuse_view_contact_tokens(
                    obs, pcd_tokens, gpcd_tokens)
                aux.update(contact_aux)
            self.vision_tokens = th.concat([pcd_tokens, gpcd_tokens], dim=-2)
        embed_tokens = self.encoder(self.vision_tokens)
        self.pose_losses["vision"], pose_pred = self.vision_pose_predictor(
            self.vision_tokens, obs)
        if cfg.merge_pose_pred:
            pose_pred_tokens = self.tokenizer["pose_pred"](pose_pred)[:, None]
            embed_tokens = th.cat([pose_pred_tokens, embed_tokens], dim=-2)

        cond = (
            self.history_tokenizer(th.concat([pcd_tokens] + ctx_tokens, dim=-2))
            if cfg.use_history else None
        )
        aux["cond"] = cond

        decoded_tokens = self.decoder(
            th.cat(ctx_tokens + [embed_tokens], dim=-2), cond)
        decoded_tokens = decoded_tokens.reshape(*decoded_tokens.shape[:-2], -1)

        if not cfg.use_interim_goal:
            self.need_goal.fill_(0)

        agg = self.aggregator
        if agg.cfg.aggregator_type in ["gru", "deep_gru", "lstm"]:
            state, agg.aggregator.memory = agg.aggregator(
                decoded_tokens, agg.aggregator.memory)
        else:
            state = agg.aggregator(decoded_tokens)

        # Stash student aggregator state for aux head loss in update_policy
        aux["student_state"] = state

        output = agg.project(state)
        output = output.reshape(*output.shape[:-1], 2, -1)
        if self.future_film is not None:
            # Query F_S with the frozen base action, then use the predicted
            # future to modulate the state consumed by the existing decoder.
            base_action = output[..., 0, :].detach()
            h0 = self.FT_head.project_current(
                state, aux.get("contact_context"))
            h1 = self.FT_head.step(h0, base_action)
            gamma, beta = self.future_film(
                th.cat([state, h1], dim=-1)).chunk(2, dim=-1)
            gamma = th.tanh(gamma) * self.cfg.student_FT_head.future_film_scale
            mod_state = state * (1.0 + gamma) + beta
            output = agg.project(mod_state)
            output = output.reshape(*output.shape[:-1], 2, -1)
            aux["future_film_gamma"] = gamma
            aux["future_film_beta"] = beta
        if self.structured_phase_film is not None:
            if "view_contact_logit" not in aux:
                raise RuntimeError(
                    "structured Phase-FiLM requires view_contact_logit")
            ft_cfg = self.cfg.student_FT_head
            base_action = output[..., 0, :].detach()
            h = self.FT_head.project_current(
                state, aux.get("contact_context"))
            decoded = {}
            for horizon in range(
                    1, int(ft_cfg.structured_phase_film_max_horizon) + 1):
                h = self.FT_head.step(h, base_action)
                if horizon in (1, 3):
                    decoded[horizon] = self.FT_head.decode(h)

            current_contact = th.sigmoid(aux["view_contact_logit"])
            parts = [state, base_action, current_contact]
            previous_contact = current_contact
            phase_logs = {}
            for horizon in (1, 3):
                pred = decoded[horizon]
                future_contact = th.sigmoid(pred["contact_logit"])
                motion = th.linalg.vector_norm(
                    pred["object_delta"][..., :3], dim=-1, keepdim=True)
                moving_given_contact = th.sigmoid(
                    (motion - ft_cfg.structured_phase_motion_threshold)
                    / max(ft_cfg.structured_phase_motion_temperature, 1e-6))
                onset = (1.0 - previous_contact) * future_contact
                persistence = previous_contact * future_contact
                moving = future_contact * moving_given_contact
                stalled = future_contact * (1.0 - moving_given_contact)
                phases = th.cat([onset, persistence, stalled, moving], dim=-1)
                structured = th.cat([
                    pred["rel_goal_delta"], pred["object_delta"],
                    future_contact, pred["progress"], phases,
                ], dim=-1)
                if ft_cfg.structured_phase_film_detach_future:
                    structured = structured.detach()
                parts.append(structured)
                phase_logs[horizon] = phases
                previous_contact = future_contact

            gamma_raw, beta_raw = self.structured_phase_film(
                th.cat(parts, dim=-1)).chunk(2, dim=-1)
            gamma = th.tanh(gamma_raw) * (
                ft_cfg.structured_phase_film_gamma_scale)
            beta = th.tanh(beta_raw) * (
                ft_cfg.structured_phase_film_beta_scale)
            output = agg.project(state * (1.0 + gamma) + beta)
            output = output.reshape(*output.shape[:-1], 2, -1)
            aux["structured_phase_film_gamma"] = gamma
            aux["structured_phase_film_beta"] = beta
            aux["structured_phase_k1"] = phase_logs[1]
            aux["structured_phase_k3"] = phase_logs[3]
        if self.seer_future_residual is not None:
            output, residual_aux = self.seer_future_residual(
                output, state, self.FT_head)
            aux.update(residual_aux)
        if self.seer_future_advantage is not None:
            output, advantage_aux = self.seer_future_advantage(
                output, state, self.FT_head)
            aux.update(advantage_aux)
        return output, aux

    # ---- update_policy: add F_T distillation loss ----
    def update_policy(self, obs, step, output, aux):
        # FT supervision is required to be present whenever FT_head is enabled.
        # If a key is missing, that's a real bug — let it crash, do NOT swallow.
        if self.seer_future_residual is not None:
            residual_loss, residual_logs = (
                self.seer_future_residual.regularization_loss(aux))
            self.aux_losses["seer_future_residual"] = (
                self.aux_losses.get("seer_future_residual", 0)
                + residual_loss)
            for key in self.seer_future_residual_losses:
                self.seer_future_residual_losses[key] += residual_logs[key]

        if self.FT_head is not None:
            ft_cfg = self.cfg.student_FT_head
            assert "student_state" in aux, \
                "FT head enabled but aux['student_state'] missing from forward pass"
            z_s = aux["student_state"]
            contact_context = aux.get("contact_context")
            if ft_cfg.contact_conditioned:
                if contact_context is None:
                    raise KeyError(
                        "contact-conditioned F_S enabled but the Contact "
                        "Transformer emitted no contact context")
                correction = self.FT_head.contact_context_to_state(
                    self.FT_head.contact_context_norm(contact_context))
                self.student_FT_losses[
                    "contact_future_correction_l1"] += (
                        correction.abs().mean().detach())
            h0_s = None
            if (ft_cfg.current_latent_weight > 0.0 or
                    ft_cfg.counterfactual_weight > 0.0 or
                    ft_cfg.centered_cf_direction_weight > 0.0 or
                    ft_cfg.centered_cf_scale_weight > 0.0):
                h0_s = self.FT_head.project_current(
                    z_s, contact_context)

            if ft_cfg.current_latent_weight > 0.0:
                h0_t = obs.get("teacher_state")
                if h0_t is None:
                    raise KeyError(
                        "SEER-Anchor requires obs['teacher_state']")
                anchor_loss = latent_loss(
                    h0_s, h0_t.detach(), mode=ft_cfg.current_latent_mode)
                self.aux_losses["student_FT"] = (
                    self.aux_losses.get("student_FT", 0)
                    + ft_cfg.current_latent_weight * anchor_loss)
                self.student_FT_losses["current_anchor"] += (
                    anchor_loss.detach())

            if (ft_cfg.counterfactual_weight > 0.0 or
                    ft_cfg.centered_cf_direction_weight > 0.0 or
                    ft_cfg.centered_cf_scale_weight > 0.0):
                cf_actions = obs.get("_FT_cf_actions")
                cf_targets = obs.get("_FT_cf_h_targets")
                if cf_actions is None or cf_targets is None:
                    raise KeyError(
                        "SEER-CF requires _FT_cf_actions and _FT_cf_h_targets")
                batch, num_candidates, action_dim = cf_actions.shape
                if cf_targets.shape[:2] != (batch, num_candidates):
                    raise ValueError("SEER-CF candidate/target shape mismatch")
                h0_flat = h0_s[:, None, :].expand(
                    -1, num_candidates, -1).reshape(batch * num_candidates, -1)
                action_flat = cf_actions.reshape(
                    batch * num_candidates, action_dim)
                h_pred = self.FT_head.step(h0_flat, action_flat).reshape_as(
                    cf_targets)
                if ft_cfg.counterfactual_weight > 0.0:
                    cf_loss = latent_loss(
                        h_pred, cf_targets.detach(),
                        mode=ft_cfg.counterfactual_mode)
                    self.aux_losses["student_FT"] = (
                        self.aux_losses.get("student_FT", 0)
                        + ft_cfg.counterfactual_weight * cf_loss)
                    self.student_FT_losses["counterfactual"] += cf_loss.detach()

                pred_centered = h_pred - h_pred.mean(dim=1, keepdim=True)
                target_centered = (
                    cf_targets - cf_targets.mean(dim=1, keepdim=True))
                pred_variance = pred_centered.square().mean()
                target_variance = target_centered.square().mean()
                sensitivity_ratio = (
                    pred_variance / target_variance.clamp_min(1e-12))
                pred_change = h_pred[:, 1:] - h_pred[:, :1]
                target_change = cf_targets[:, 1:] - cf_targets[:, :1]
                change_cosine = th.nn.functional.cosine_similarity(
                    pred_change, target_change, dim=-1, eps=1e-8).mean()
                centered_direction = 1.0 - change_cosine
                pred_rms = pred_change.square().mean(dim=(1, 2)).sqrt()
                target_rms = target_change.square().mean(dim=(1, 2)).sqrt()
                centered_scale = th.nn.functional.smooth_l1_loss(
                    th.log(pred_rms + ft_cfg.centered_cf_scale_eps),
                    th.log(target_rms + ft_cfg.centered_cf_scale_eps),
                )
                centered_loss = (
                    ft_cfg.centered_cf_direction_weight * centered_direction
                    + ft_cfg.centered_cf_scale_weight * centered_scale)
                if (ft_cfg.centered_cf_direction_weight > 0.0 or
                        ft_cfg.centered_cf_scale_weight > 0.0):
                    self.aux_losses["student_FT"] = (
                        self.aux_losses.get("student_FT", 0) + centered_loss)
                self.student_FT_losses["cf_student_variance"] += (
                    pred_variance.detach())
                self.student_FT_losses["cf_teacher_variance"] += (
                    target_variance.detach())
                self.student_FT_losses["cf_sensitivity_ratio"] += (
                    sensitivity_ratio.detach())
                self.student_FT_losses["cf_change_direction_cosine"] += (
                    change_cosine.detach())
                self.student_FT_losses["cf_centered_direction"] += (
                    centered_direction.detach())
                self.student_FT_losses["cf_centered_scale"] += (
                    centered_scale.detach())

            FT_loss, FT_logs = self._compute_FT_loss(obs, aux, ft_cfg)
            if FT_loss is not None:
                self.aux_losses["student_FT"] = (
                    self.aux_losses.get("student_FT", 0) + FT_loss)
                self.student_FT_losses["total"] = (
                    self.student_FT_losses.get("total", 0) + FT_loss.detach())
                self.student_FT_losses["latent_total"] = (
                    self.student_FT_losses.get("latent_total", 0)
                    + FT_logs.get("latent_total", th.tensor(0.0)))
                self.student_FT_losses["structured_total"] = (
                    self.student_FT_losses.get("structured_total", 0)
                    + FT_logs.get("structured_total", th.tensor(0.0)))
                self.student_FT_losses["contrastive_total"] = (
                    self.student_FT_losses.get("contrastive_total", 0)
                    + FT_logs.get("contrastive_total", th.tensor(0.0)))
                for similarity_key in (
                    "positive_similarity_k1",
                    "negative_similarity_k1",
                ):
                    self.student_FT_losses[similarity_key] = (
                        self.student_FT_losses.get(similarity_key, 0)
                        + FT_logs.get(similarity_key, th.tensor(0.0)))
                diagnostic_keys = (
                    "structured_rel_goal_delta_unweighted",
                    "structured_rel_goal_delta_weighted",
                    "structured_object_delta_unweighted",
                    "structured_object_delta_weighted",
                    "structured_contact_logit_unweighted",
                    "structured_contact_logit_weighted",
                    "structured_progress_unweighted",
                    "structured_progress_weighted",
                    "target_rel_goal_delta_abs_mean",
                    "target_object_delta_abs_mean",
                    "target_contact_rate",
                    "target_progress_abs_mean",
                )
                for diagnostic_key in diagnostic_keys:
                    if diagnostic_key in FT_logs:
                        self.student_FT_losses[diagnostic_key] = (
                            self.student_FT_losses.get(diagnostic_key, 0)
                            + FT_logs[diagnostic_key])

            if self.structured_phase_film is not None:
                gamma = aux["structured_phase_film_gamma"].detach()
                beta = aux["structured_phase_film_beta"].detach()
                gamma_scale = abs(float(
                    ft_cfg.structured_phase_film_gamma_scale))
                beta_scale = abs(float(
                    ft_cfg.structured_phase_film_beta_scale))
                self.student_FT_losses["phase_gamma_abs_mean"] += (
                    gamma.abs().mean())
                self.student_FT_losses["phase_beta_abs_mean"] += (
                    beta.abs().mean())
                if gamma_scale > 0.0:
                    self.student_FT_losses["phase_gamma_saturation"] += (
                        (gamma.abs() >= 0.95 * gamma_scale).float().mean())
                if beta_scale > 0.0:
                    self.student_FT_losses["phase_beta_saturation"] += (
                        (beta.abs() >= 0.95 * beta_scale).float().mean())
                for horizon in (1, 3):
                    phase = aux[f"structured_phase_k{horizon}"].detach()
                    for index, phase_name in enumerate(
                            ("onset", "persistence", "stalled", "moving")):
                        self.student_FT_losses[
                            f"phase_k{horizon}_{phase_name}"] += (
                                phase[..., index].mean())

        if self.seer_future_advantage is not None:
            target = obs.get("_FT_advantage_targets")
            predicted = aux.get("future_advantage_values")
            base_output = aux.get("future_advantage_base_output")
            teacher_action = obs.get("teacher_action")
            if target is None or predicted is None or base_output is None:
                raise KeyError("SEER-Advantage supervision is missing")
            ranking_loss, ranking_logs = self.seer_future_advantage.ranking_loss(
                predicted, target)
            base_loss = th.tensor(0.0, device=predicted.device)
            if self.cfg.seer_future_advantage.base_action_loss_weight > 0.0:
                if teacher_action is None:
                    raise KeyError("SEER-Advantage requires teacher_action")
                base_loss = th.nn.functional.mse_loss(
                    base_output[..., 0, :], teacher_action[..., 0, :].detach())
            advantage_loss = (
                ranking_loss
                + self.cfg.seer_future_advantage.base_action_loss_weight * base_loss)
            self.aux_losses["seer_future_advantage"] = (
                self.aux_losses.get("seer_future_advantage", 0) + advantage_loss)
            logs = self.seer_future_advantage_losses
            logs["total"] += advantage_loss.detach()
            logs["ranking"] += ranking_logs["ranking"]
            logs["value_regression"] += ranking_logs["value_regression"]
            logs["top1_agreement"] += ranking_logs["top1_agreement"]
            logs["base_action"] += base_loss.detach()
            logs["gate"] += aux["future_advantage_gate"].mean().detach()
            logs["switch_rate"] += aux[
                "future_advantage_switch"].float().mean().detach()

        super().update_policy(obs, step, output, aux)

        # Periodic logging
        if (
            self.FT_head is not None
            and (step + 1) % self.cfg.horizon == 0
        ):
            if self.writer is not None:
                with th.no_grad():
                    for key, value in self.student_FT_losses.items():
                        scalar = value / self.cfg.horizon
                        if isinstance(scalar, th.Tensor):
                            self.writer.add_scalar(
                                f"loss/student_FT_{key}",
                                float(scalar.detach().cpu().item()),
                                global_step=step,
                            )
                    for key, value in self.seer_future_residual_losses.items():
                        scalar = value / self.cfg.horizon
                        if isinstance(scalar, th.Tensor):
                            self.writer.add_scalar(
                                f"loss/seer_future_residual_{key}",
                                float(scalar.detach().cpu().item()),
                                global_step=step,
                            )
                    for key, value in self.seer_future_advantage_losses.items():
                        scalar = value / self.cfg.horizon
                        if isinstance(scalar, th.Tensor):
                            self.writer.add_scalar(
                                f"loss/seer_future_advantage_{key}",
                                float(scalar.detach().cpu().item()),
                                global_step=step,
                            )
            for key in self.student_FT_losses:
                self.student_FT_losses[key] = 0
            for key in self.seer_future_residual_losses:
                self.seer_future_residual_losses[key] = 0
            for key in self.seer_future_advantage_losses:
                self.seer_future_advantage_losses[key] = 0

    def _compute_FT_loss(self, obs: Dict[str, th.Tensor], aux: Dict[str, th.Tensor],
                         ft_cfg: StudentFTHeadConfig
                         ) -> Tuple[Optional[th.Tensor], Dict[str, th.Tensor]]:
        """Build targets and compute student-side multi-step distillation loss.

        Required obs keys (injected by DAggerTrainerEnv_FT):
          _FT_action_k1, _FT_action_k2, ..., _FT_action_kK     (B, dim_act)
          _FT_h_target_k1, ..., _FT_h_target_kK                (B, teacher_dim_state)
          _FT_struct_rel_goal_delta_k1, ..., _kK
          _FT_struct_object_delta_k1, ..., _kK
          _FT_struct_contact_k1, ..., _kK
          _FT_struct_progress_k1, ..., _kK
          _FT_valid                                             (B,) bool
        """
        K = ft_cfg.K
        device = aux["student_state"].device
        # Action sequence (B, K, dim_act)
        actions = th.stack([obs[f"_FT_action_k{k+1}"] for k in range(K)], dim=1)
        # Latent target trajectory (B, K, teacher_dim_state)
        h_target = th.stack([obs[f"_FT_h_target_k{k+1}"] for k in range(K)], dim=1)
        h_negative = None
        if ft_cfg.contrastive_weight > 0.0:
            h_negative = th.stack(
                [obs[f"_FT_h_negative_k{k+1}"] for k in range(K)], dim=1
            )
        # Structured targets
        structured_targets = []
        for k in range(K):
            structured_targets.append({
                "rel_goal_delta": obs[f"_FT_struct_rel_goal_delta_k{k+1}"],
                "object_delta":   obs[f"_FT_struct_object_delta_k{k+1}"],
                "contact":        obs[f"_FT_struct_contact_k{k+1}"],
                "progress":       obs[f"_FT_struct_progress_k{k+1}"],
            })

        # Mask out invalid samples. Four cases must be distinguished — the
        # collapsed "if any and not all" form silently mis-handles the
        # all-false case (used to fall through to else and pollute loss
        # with ALL invalid rows).
        valid_mask = obs.get("_FT_valid", None)
        if valid_mask is None:
            # No mask — assume all rows valid
            z_s = aux["student_state"]
            contact_context = aux.get("contact_context")
        elif not bool(valid_mask.any().item()):
            # All-false: no valid samples this step. Skip aux loss entirely.
            return None, {}
        elif bool(valid_mask.all().item()):
            # All valid: full batch
            z_s = aux["student_state"]
            contact_context = aux.get("contact_context")
        else:
            # Mixed: subset to the valid rows
            idx = th.nonzero(valid_mask, as_tuple=False).squeeze(-1)
            z_s = aux["student_state"][idx]
            contact_context = aux.get("contact_context")
            if contact_context is not None:
                contact_context = contact_context[idx]
            actions = actions[idx]
            h_target = h_target[idx]
            if h_negative is not None:
                h_negative = h_negative[idx]
            structured_targets = [
                {kk: vv[idx] for kk, vv in s.items()} for s in structured_targets
            ]

        total, logs = compute_student_FT_loss(
            self.FT_head,
            z_s_t=z_s,
            actions=actions,
            h_target_traj=h_target,
            structured_targets=structured_targets,
            h_negative_traj=h_negative,
            contact_context=contact_context,
            cfg=ft_cfg,
        )
        return total, logs

    def get_output(self, obs):
        cfg = self.cfg
        with autocast() if cfg.use_amp else nullcontext():
            output, aux = self._forward_impl(obs)
        if not cfg.without_teacher and cfg.max_delay_steps > 0:
            if cfg.estimate_level == "state":
                teacher_state = obs.get("teacher_state", None)
                if teacher_state is not None:
                    teacher_state = teacher_state.detach().clone()
                output = th.where(
                    self.delay_count[..., None] >= 0,
                    output,
                    teacher_state,
                )
            elif cfg.estimate_level == "action":
                teacher_action = obs.get("teacher_action", None)
                if teacher_action is not None:
                    teacher_action = teacher_action.detach().clone()
                output = th.where(
                    self.delay_count[..., None, None] >= 0,
                    output,
                    teacher_action,
                )
        return output, aux

    @th.no_grad()
    def plan_action(self, output, aux, plan_cfg, value_net=None):
        """Run the optional SEER-Plan head without changing training forward."""
        if self.FT_head is None:
            raise RuntimeError("SEER-Plan requires a loaded student FT head")
        if "student_state" not in aux:
            raise KeyError("SEER-Plan requires aux['student_state']")
        from seer_plan import select_action
        base_action = output[..., 0, :]
        return select_action(
            base_action, aux["student_state"], self.FT_head, plan_cfg,
            value_net=value_net)
