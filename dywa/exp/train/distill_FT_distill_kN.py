#!/usr/bin/env python3
"""Step 3 K>=2 (delayed multi-step): student agent with F_T distillation.

Two real differences from distill_FT_distill.py (K=1):

1. The K=1 in-place FT loss (computed in update_policy from injected
   _FT_*_k1 keys) is DISABLED. The K-step FT loss is computed externally
   by DAggerTrainerEnv_FT_kN via compute_FT_loss_kN(), and injected into
   self.aux_losses['student_FT'] BEFORE super().update_policy is called.

2. Adds reencode_obs_with_memory(obs_t, stashed_memory) which re-encodes
   a stashed obs_t with a stashed RNN memory (memory_{t-1}), then exactly
   restores the aggregator's current memory. This lets the student get
   gradient through the encoder for delayed K-step supervision while
   keeping the RNN time-axis consistent.

Frozen-teacher invariant: unchanged. Action path: unchanged.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch as th

from distill_FT_distill import StudentAgentRMA_FT
from student_FT_distill import compute_student_FT_loss


def _snapshot_memory(mem):
    """Detach-and-clone an RNN memory tensor (Tensor for GRU, tuple for LSTM)."""
    if mem is None:
        return None
    if isinstance(mem, tuple):
        return tuple(m.detach().clone() for m in mem)
    return mem.detach().clone()


class StudentAgentRMA_FT_kN(StudentAgentRMA_FT):
    """K>=2 variant: delayed multi-step FT distillation.

    Reuses StudentAgentRMA_FT's __init__, FT_head construction, and
    update_policy's logging block. Only overrides:
      - _compute_FT_loss (K=1 path) -> no-op
      - adds standalone encoder forward + re-encode utilities
      - adds compute_FT_loss_kN() called by the K-window trainer
    """

    # ---- K=1 in-place FT loss disabled in this subclass ----
    def _compute_FT_loss(self, obs, aux, ft_cfg):
        return None, {}

    # ---- standalone encoder forward (no self.* side effects) ----
    def _encode_state_no_side_effects(
        self, obs: Dict[str, th.Tensor]
    ) -> Tuple[th.Tensor, Optional[th.Tensor]]:
        """Encoder forward returning state and contact context.

        Side-effect-free wrt:
          - self.vision_tokens     (not written)
          - self.pose_losses       (vision pose loss NOT accumulated)
          - self.need_goal         (not written)
        DOES still write self.aggregator.aggregator.memory; callers around
        re-encoding must stash + restore that field.
        """
        cfg = self.cfg
        ctx_tokens = [
            self.tokenizer[k](obs[k].detach().clone())[:, None]
            for k in self.input_keys if k in obs
        ]
        pcd_tokens = self.point_tokenizer(obs["partial_cloud"])
        vision_tokens = pcd_tokens
        contact_context = None
        if cfg.use_gpcd:
            gpcd_tokens = self.point_tokenizer(obs["goal_cloud"])
            if cfg.view_contact_transformer:
                # Delayed K-step supervision must re-encode through the exact
                # Contact student path.  The legacy implementation silently
                # bypassed this correction and therefore trained F_S from a
                # different state than the online action path.
                pcd_tokens, contact_aux = self.fuse_view_contact_tokens(
                    obs, pcd_tokens, gpcd_tokens)
                contact_context = contact_aux.get("contact_context")
            vision_tokens = th.concat([pcd_tokens, gpcd_tokens], dim=-2)
        embed_tokens = self.encoder(vision_tokens)
        if cfg.merge_pose_pred:
            # Need pose_pred to extend embed_tokens, but MUST NOT touch
            # self.pose_losses (the legitimate per-step vision pose loss is
            # written by the main get_output forward; re-encoding here would
            # overwrite it with a value derived from a stale obs).
            _pose_loss_discard, pose_pred = self.vision_pose_predictor(
                vision_tokens, obs)
            pose_pred_tokens = self.tokenizer["pose_pred"](pose_pred)[:, None]
            embed_tokens = th.cat([pose_pred_tokens, embed_tokens], dim=-2)
        cond = (
            self.history_tokenizer(th.concat([pcd_tokens] + ctx_tokens, dim=-2))
            if cfg.use_history else None
        )
        decoded_tokens = self.decoder(
            th.cat(ctx_tokens + [embed_tokens], dim=-2), cond)
        decoded_tokens = decoded_tokens.reshape(*decoded_tokens.shape[:-2], -1)
        agg = self.aggregator
        if agg.cfg.aggregator_type in ["gru", "deep_gru", "lstm"]:
            state, agg.aggregator.memory = agg.aggregator(
                decoded_tokens, agg.aggregator.memory)
        else:
            state = agg.aggregator(decoded_tokens)
        return state, contact_context

    def reencode_obs_with_memory(
        self, obs_t: Dict[str, th.Tensor], stashed_memory
    ) -> Tuple[th.Tensor, Optional[th.Tensor]]:
        """Re-encode obs_t with historical RNN memory, then restore current.

        Args:
            obs_t:           detached obs dict captured at time t
            stashed_memory:  RNN memory BEFORE encoding obs_t at time t
                             (i.e., memory_{t-1}). Allowed to be None for
                             non-RNN aggregators.

        Returns:
            state_t and the student-visible contact context with autograd
            attached to current weights.

        Invariant: aggregator memory and the SEER history buffer are restored
        exactly to what they were before the call. This keeps delayed replay
        from changing the online observation history.
        """
        agg = self.aggregator
        has_rnn = (agg.cfg.aggregator_type in ["gru", "deep_gru", "lstm"]
                   and hasattr(agg.aggregator, "memory"))
        current_memory = agg.aggregator.memory if has_rnn else None
        history_tokenizer = getattr(self, "history_tokenizer", None)
        history_buffer = None
        if history_tokenizer is not None and hasattr(history_tokenizer, "history"):
            history_buffer = history_tokenizer.history.detach().clone()

        try:
            if has_rnn:
                agg.aggregator.memory = stashed_memory
            return self._encode_state_no_side_effects(obs_t)
        finally:
            if has_rnn:
                agg.aggregator.memory = current_memory
            if history_buffer is not None:
                with th.no_grad():
                    history_tokenizer.history.copy_(history_buffer)

    # ---- K-step FT loss, called externally by the trainer ----
    def compute_FT_loss_kN(
        self,
        obs_t: Dict[str, th.Tensor],
        action_seq: th.Tensor,                       # (B, K, dim_act)
        h_target_traj: th.Tensor,                    # (B, K, teacher_dim_state)
        structured_targets: List[Dict[str, th.Tensor]],
        stashed_memory,
        valid_mask: Optional[th.Tensor],             # (B,) bool or None
        step: int,
    ) -> None:
        """Compute K-step delayed FT loss; inject into self.aux_losses['student_FT'].

        The parent update_policy's periodic logging block reads
        self.student_FT_losses, so we accumulate detached scalars there too.

        Mask handling mirrors K=1's 4-case structure (None / all-false /
        all-true / mixed). For the 'mixed' case we subset BEFORE re-encoding
        so the encoder runs on the smaller valid batch.
        """
        if self.FT_head is None:
            return
        ft_cfg = self.cfg.student_FT_head
        if ft_cfg.contrastive_weight > 0.0:
            raise ValueError(
                "K>=2 SEER supervision does not yet provide counterfactual "
                "latent trajectories for contrastive loss")

        idx = None
        if valid_mask is not None:
            if not bool(valid_mask.any().item()):
                return  # whole batch invalid this step
            if not bool(valid_mask.all().item()):
                idx = th.nonzero(valid_mask, as_tuple=False).squeeze(-1)

        # Re-encode the full batch first. `HistoryEncoder` keeps a fixed
        # num_env-sized history buffer, so re-encoding a valid-only subset would
        # mismatch its internal (B, history, token, dim) buffer. Invalid rows are
        # filtered after state_t is built.
        state_t, contact_context_t = self.reencode_obs_with_memory(
            obs_t, stashed_memory)
        if ft_cfg.contact_conditioned and contact_context_t is None:
            raise RuntimeError(
                "contact-conditioned K-step F_S re-encode produced no context")

        if idx is not None:
            state_t = state_t[idx]
            if contact_context_t is not None:
                contact_context_t = contact_context_t[idx]
            action_seq = action_seq[idx]
            h_target_traj = h_target_traj[idx]
            structured_targets = [
                {kk: vv[idx] for kk, vv in s.items()}
                for s in structured_targets
            ]

        total, logs = compute_student_FT_loss(
            self.FT_head,
            z_s_t=state_t,
            actions=action_seq,
            h_target_traj=h_target_traj,
            structured_targets=structured_targets,
            contact_context=contact_context_t,
            cfg=ft_cfg,
        )

        # Same aux_losses key as K=1; the parent update_policy will backward it
        self.aux_losses["student_FT"] = (
            self.aux_losses.get("student_FT", 0) + total)

        # Detached scalars for periodic logging (parent class reads these)
        self.student_FT_losses["total"] = (
            self.student_FT_losses.get("total", 0) + total.detach())
        self.student_FT_losses["latent_total"] = (
            self.student_FT_losses.get("latent_total", 0)
            + logs.get("latent_total", th.tensor(0.0)))
        self.student_FT_losses["structured_total"] = (
            self.student_FT_losses.get("structured_total", 0)
            + logs.get("structured_total", th.tensor(0.0)))
