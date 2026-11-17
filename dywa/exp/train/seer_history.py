#!/usr/bin/env python3
"""SEER persistent history encoding with safe snapshots for delayed supervision.

The shared DyWA encoder also defaults to persistent history in this repository.
This class preserves the SEER checkpoint/module interface and safe-copy option.
"""

from __future__ import annotations

import torch as th

from models.modules import HistoryEncoder


class SeerHistoryEncoder(HistoryEncoder):
    """History encoder with an explicit persistent rolling buffer."""

    def put_history(self, tensor: th.Tensor) -> th.Tensor:
        if tensor.ndim != 3:
            raise ValueError(
                f"history input must be (B,T,D), got {tuple(tensor.shape)}")
        if tensor.shape[1] != self.num_tokens:
            raise ValueError(
                f"history token count mismatch: {tensor.shape[1]} != "
                f"{self.num_tokens}")
        if tensor.shape[0] != self.history.shape[0]:
            raise ValueError(
                f"history batch mismatch: {tensor.shape[0]} != "
                f"{self.history.shape[0]}")

        # ``history`` is a registered, non-gradient buffer. Updating it in
        # place preserves temporal state while keeping the current observation
        # detached from the rolling deployment buffer.
        with th.no_grad():
            # Clone the source before writing into an overlapping destination.
            source = self.history[:, 1:]
            if getattr(self, "safe_copy", True):
                source = source.clone()
            self.history[:, :-1].copy_(source)
            self.history[:, -1].copy_(tensor.detach())
        # Return a snapshot so delayed replay can restore the buffer without
        # mutating a tensor retained by the current autograd graph.
        return self.history.clone()
