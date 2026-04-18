"""Per-nerve soft transducer mapping src local code → dst local code.

See spec §4.3. Each row of the 64×64 logits matrix is a distribution over
possible target codes. Gumbel-softmax during training, argmax at inference.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Transducer(nn.Module):
    def __init__(self, alphabet_size: int = 64, init_scale: float = 0.1) -> None:
        super().__init__()
        self.alphabet_size = alphabet_size
        # Near-uniform init to avoid premature collapse.
        self.logits = nn.Parameter(torch.randn(alphabet_size, alphabet_size) * init_scale)

    def forward(self, src_code: Tensor, *, hard: bool = True, tau: float = 1.0) -> Tensor:
        """src_code: [B] long. Returns dst_code: [B] long."""
        row_logits = self.logits[src_code]                     # [B, alphabet_size]
        y = F.gumbel_softmax(row_logits, tau=tau, hard=hard)   # [B, alphabet_size]
        return y.argmax(dim=-1)

    def entropy(self) -> Tensor:
        """Row-wise Shannon entropy of the transducer distribution.

        Higher = more uniform (used as a regularizer to avoid collapse to identity).
        Returns the mean entropy across all rows.
        """
        p = F.softmax(self.logits, dim=-1)                     # [size, size]
        ent_per_row = -(p * (p + 1e-9).log()).sum(dim=-1)
        return ent_per_row.mean()
