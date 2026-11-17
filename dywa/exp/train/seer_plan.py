#!/usr/bin/env python3
"""Deterministic one-step candidate selection for SEER-Plan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch as th


@dataclass
class SeerPlanConfig:
    num_candidates: int = 13
    sigma: float = 0.05
    beta: float = 0.1
    switch_threshold: float = 0.01
    perturb_dims: int = 6
    candidate_mode: str = "axis"
    score_mode: str = "combined"
    value_discount: float = 0.99


def candidate_offsets(cfg: SeerPlanConfig, action_dim: int, *, device, dtype):
    """Return a fixed antithetic candidate set with the base action first."""
    if cfg.num_candidates < 1:
        raise ValueError("num_candidates must be positive")
    if cfg.perturb_dims < 1 or cfg.perturb_dims > action_dim:
        raise ValueError("perturb_dims is outside the action dimension")
    offsets = th.zeros(
        cfg.num_candidates, action_dim, device=device, dtype=dtype)
    if cfg.candidate_mode not in {"axis", "diagonal"}:
        raise ValueError(f"unknown candidate_mode={cfg.candidate_mode!r}")
    if cfg.candidate_mode == "diagonal":
        # Fixed low-discrepancy sign patterns.  Normalize every direction so
        # its L2 radius matches the axis candidates' validated sigma.
        for index in range(1, cfg.num_candidates):
            direction = th.tensor([
                1.0 if ((index * 0x9E3779B1) >> dim) & 1 else -1.0
                for dim in range(cfg.perturb_dims)
            ], device=device, dtype=dtype)
            direction.mul_(cfg.sigma / direction.norm())
            offsets[index, :cfg.perturb_dims] = direction
        return offsets
    # Candidate zero is the base action.  Fill complete +/- pairs and leave a
    # possible odd slot at zero, rather than introducing an unpaired positive
    # perturbation (the frozen protocol uses 16 total slots).
    num_pairs = (cfg.num_candidates - 1) // 2
    for index in range(1, 1 + 2 * num_pairs):
        pair = (index - 1) // 2
        dim = pair % cfg.perturb_dims
        level = pair // cfg.perturb_dims + 1
        sign = 1.0 if index % 2 else -1.0
        offsets[index, dim] = sign * cfg.sigma * level
    return offsets


def build_candidates(base_action: th.Tensor, cfg: SeerPlanConfig) -> th.Tensor:
    if base_action.ndim != 2:
        raise ValueError(f"base_action must be (B,A), got {base_action.shape}")
    offsets = candidate_offsets(
        cfg, base_action.shape[-1], device=base_action.device,
        dtype=base_action.dtype)
    perturbed = (base_action[:, None, :] + offsets[None, 1:]).clamp(-1.0, 1.0)
    # Preserve the exact base action. NormalizeEnv performs the final clipping
    # at the environment boundary; planning must not change a no-switch step.
    return th.cat([base_action[:, None, :], perturbed], dim=1)


def select_action(
    base_action: th.Tensor,
    student_state: th.Tensor,
    future_head,
    cfg: SeerPlanConfig,
    value_net=None,
) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
    """Score all candidates in one batched F_S forward and select safely.

    The score uses the learned progress head and a quadratic distance penalty.
    Candidate zero is always the unmodified student action. Ties resolve to the
    lowest index through ``argmax``, and switching requires a strict margin.
    """
    candidates = build_candidates(base_action, cfg)
    batch, count, action_dim = candidates.shape
    state = student_state[:, None, :].expand(-1, count, -1)
    flat_state = state.reshape(batch * count, -1)
    flat_action = candidates.reshape(batch * count, action_dim)
    projected = future_head.proj_to_teacher_latent(flat_state)
    future = future_head.step(projected, flat_action)
    progress = future_head.decode(future)["progress"].reshape(batch, count)
    values = None
    if cfg.score_mode in {"value", "combined"}:
        if value_net is None:
            raise ValueError(f"score_mode={cfg.score_mode!r} requires value_net")
        values = value_net(future).reshape(batch, count)
    deviation = (candidates - base_action[:, None]).square().sum(dim=-1)
    if cfg.score_mode == "progress":
        utility = progress
    elif cfg.score_mode == "value":
        utility = cfg.value_discount * values
    elif cfg.score_mode == "combined":
        utility = progress + cfg.value_discount * values
    else:
        raise ValueError(f"unknown score_mode={cfg.score_mode!r}")
    scores = utility - cfg.beta * deviation
    best_index = scores.argmax(dim=1)
    base_score = scores[:, 0]
    best_score = scores.gather(1, best_index[:, None]).squeeze(1)
    switch = (best_index != 0) & (
        best_score > base_score + cfg.switch_threshold)
    selected_index = th.where(switch, best_index, th.zeros_like(best_index))
    selected = candidates.gather(
        1, selected_index[:, None, None].expand(-1, 1, action_dim)).squeeze(1)
    return selected, {
        "candidates": candidates,
        "scores": scores,
        "progress": progress,
        "values": values,
        "selected_index": selected_index,
        "switched": switch,
        "score_gain": best_score - base_score,
    }
