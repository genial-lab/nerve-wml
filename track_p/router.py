"""Sparse top-K Gumbel routing between WMLs.

See spec §4.2 (routing_weight) and §4.5 (N-4). During training, τ is annealed
from ~1 to ~0.1. At inference, hard top-K is applied.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SparseRouter(nn.Module):
    """Returns a sparse {0, 1} edge matrix of shape [N, N] where each row has
    exactly K ones (and no self-loop)."""

    def __init__(self, n_wmls: int, k: int) -> None:
        super().__init__()
        assert 1 <= k < n_wmls
        self.n_wmls = n_wmls
        self.k      = k
        # Learnable logits per directed pair. Self-loops masked at sample time.
        self.logits = nn.Parameter(torch.randn(n_wmls, n_wmls) * 0.1)

    def sample_edges(self, *, tau: float = 1.0, hard: bool = True) -> Tensor:
        # Mask self-loops with -inf so softmax never selects them.
        mask       = torch.eye(self.n_wmls, dtype=torch.bool, device=self.logits.device)
        masked_log = self.logits.masked_fill(mask, float("-inf"))

        # Per-row Gumbel; then keep top-K per row.
        noise    = -torch.log(-torch.log(torch.rand_like(masked_log) + 1e-9) + 1e-9)
        noisy    = (masked_log + noise) / tau

        topk_idx = noisy.topk(self.k, dim=-1).indices            # [N, K]
        edges    = torch.zeros_like(masked_log)
        edges.scatter_(1, topk_idx, 1.0)

        if hard:
            return edges
        # Soft path: weight by softmax over the top-K logits (seldom used).
        soft_weights = F.softmax(noisy, dim=-1)
        return soft_weights * edges

    def routing_weight(self, src: int, dst: int, edges: Tensor) -> float:
        return float(edges[src, dst].item())
