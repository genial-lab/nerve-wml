"""FlowProxyTask — a cheap linearly-separable classification task.

Used by W1-W3 to validate the nerve loop quickly; Split-MNIST is used by W4
for continual learning.
"""
from __future__ import annotations

import torch
from torch import Tensor


class FlowProxyTask:
    def __init__(self, dim: int = 16, n_classes: int = 4, *, seed: int | None = None) -> None:
        self.dim       = dim
        self.n_classes = n_classes
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)
        # Class centroids in feature space.
        self._centers = torch.randn(n_classes, dim, generator=gen) * 2.0
        self._gen     = gen

    def sample(self, batch: int = 64) -> tuple[Tensor, Tensor]:
        labels = torch.randint(0, self.n_classes, (batch,), generator=self._gen)
        noise  = torch.randn(batch, self.dim, generator=self._gen) * 0.3
        x      = self._centers[labels] + noise
        return x, labels
