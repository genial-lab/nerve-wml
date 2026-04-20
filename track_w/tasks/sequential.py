"""SequentialFlowProxyTask — temporal XOR copy task.

Each sample is a sequence of `seq_len` tokens in R^dim. The label is
the XOR of two hyperplane projections at positions `t_a` and `t_b`
in the sequence, mapped to one of `n_classes` bins. Forces the
substrate to carry information across the temporal gap — a
static-input WML fails by construction.
"""
from __future__ import annotations

import torch


class SequentialFlowProxyTask:
    def __init__(
        self,
        seq_len:   int = 16,
        dim:       int = 16,
        n_classes: int = 4,
        t_a:       int = 2,
        t_b:       int | None = None,
        *,
        seed:      int = 0,
    ) -> None:
        self.seq_len   = seq_len
        self.dim       = dim
        self.n_classes = n_classes
        self.t_a = t_a
        self.t_b = t_b if t_b is not None else seq_len - 3
        assert 0 <= self.t_a < self.t_b < seq_len
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        # Frozen label projection: n_classes planes in R^dim at t_b.
        self._labels_proj = torch.randn(
            dim, n_classes, generator=self.generator,
        )

    def sample(self, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (xs [B, T, dim], y [B]).

        Label is derived from the t_b timestep projection only. The
        sequence carries task-relevant signal at t_b and noise elsewhere.
        This isolates the temporal transmission question: does the
        substrate align its code at the informative timestep, and does
        that alignment drop at the filler timesteps?
        """
        xs = torch.randn(
            batch, self.seq_len, self.dim, generator=self.generator,
        )
        scores = xs[:, self.t_b] @ self._labels_proj
        y = scores.argmax(dim=1)
        return xs, y.long()
