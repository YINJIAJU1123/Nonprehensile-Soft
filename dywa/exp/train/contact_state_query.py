#!/usr/bin/env python3
"""State/action residual for the goal-conditioned contact query.

This module is deliberately small and dependency-light so its zero-init and
checkpoint migration contract can be tested without importing Isaac Gym.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import torch as th
import torch.nn as nn


CONTACT_STATE_QUERY_PREFIX = "contact_state_query."


class ContactStateActionQuery(nn.Module):
    """Project hand state and previous action into one query residual.

    The final projection is zero initialized.  Consequently, enabling this
    module on an old Contact Transformer checkpoint initially leaves the
    attention query, and therefore the complete policy, unchanged.
    """

    def __init__(self, *, hand_dim: int = 9, action_dim: int = 20,
                 hidden_dim: int = 128, embed_dim: int = 128):
        super().__init__()
        if min(hand_dim, action_dim, hidden_dim, embed_dim) <= 0:
            raise ValueError("contact state-query dimensions must be positive")
        self.hand_dim = int(hand_dim)
        self.action_dim = int(action_dim)
        self.net = nn.Sequential(
            nn.Linear(self.hand_dim + self.action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, hand_state: th.Tensor,
                previous_action: th.Tensor) -> th.Tensor:
        if hand_state.ndim != 2 or hand_state.shape[-1] != self.hand_dim:
            raise ValueError(
                "hand_state must have shape (B, "
                f"{self.hand_dim}), got {tuple(hand_state.shape)}")
        if (previous_action.ndim != 2 or
                previous_action.shape[-1] != self.action_dim):
            raise ValueError(
                "previous_action must have shape (B, "
                f"{self.action_dim}), got {tuple(previous_action.shape)}")
        if hand_state.shape[0] != previous_action.shape[0]:
            raise ValueError("hand_state and previous_action batch mismatch")
        inputs = th.cat(
            [hand_state.detach(), previous_action.detach()], dim=-1)
        return self.net(inputs)


def load_contact_query_warmstart(
        model: nn.Module,
        saved_state: Mapping[str, th.Tensor],
        *,
        ignored_runtime_keys: Iterable[str] = (),
) -> dict:
    """Strictly load an old Contact checkpoint into the extended model.

    Every persistent tensor shared by the checkpoint and runtime must match.
    The only checkpoint-missing tensors allowed are parameters belonging to
    ``contact_state_query``.  Runtime-only recurrent buffers may be ignored for
    loading, but they must still exist in the source checkpoint when expected.
    """

    runtime_state = model.state_dict()
    runtime_keys = set(runtime_state)
    saved_keys = set(saved_state)
    new_query_keys = {
        key for key in runtime_keys
        if key.startswith(CONTACT_STATE_QUERY_PREFIX)
    }
    expected_query_keys = {
        f"{CONTACT_STATE_QUERY_PREFIX}net.0.weight",
        f"{CONTACT_STATE_QUERY_PREFIX}net.0.bias",
        f"{CONTACT_STATE_QUERY_PREFIX}net.2.weight",
        f"{CONTACT_STATE_QUERY_PREFIX}net.2.bias",
    }
    if new_query_keys != expected_query_keys:
        raise RuntimeError(
            "unexpected contact state-query parameter schema: "
            f"{sorted(new_query_keys)}")

    missing = runtime_keys - saved_keys
    unexpected = saved_keys - runtime_keys
    if missing != new_query_keys or unexpected:
        raise RuntimeError(
            "contact-query warm-start mismatch: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}")

    for key in sorted(runtime_keys & saved_keys):
        source = saved_state[key]
        target = runtime_state[key]
        if tuple(source.shape) != tuple(target.shape):
            raise RuntimeError(
                f"contact-query warm-start shape mismatch for {key}: "
                f"checkpoint={tuple(source.shape)} runtime={tuple(target.shape)}")
        if not bool(th.isfinite(source).all().item()):
            raise RuntimeError(
                f"contact-query warm-start contains NaN/Inf in {key}")

    ignored = set(ignored_runtime_keys)
    loadable = {key: value for key, value in saved_state.items()
                if key not in ignored}
    incompatible = model.load_state_dict(loadable, strict=False)
    actual_missing = set(incompatible.missing_keys) - ignored
    if actual_missing != new_query_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "contact-query warm-start load result mismatch: "
            f"missing={sorted(actual_missing)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}")
    return {
        "allowed_missing": sorted(new_query_keys),
        "ignored_runtime_keys": sorted(ignored),
        "loaded_tensor_count": len(loadable),
    }
