"""Shared helpers — decoding a batch of inbound Neuroletters into a pooled
embedding that MLP cores and LIF input currents can consume.

Mean-pooling is intentional: the WML must treat a silent nerve (N-1) the same
as a pooled zero embedding, which is a valid first approximation. Future plans
can swap for attention-over-letters without changing the WML step() contracts.
"""
from __future__ import annotations

import torch
from torch import Tensor

from nerve_core.neuroletter import Neuroletter


def embed_inbound(inbound: list[Neuroletter], codebook: Tensor) -> Tensor:
    """Pool inbound code embeddings by mean.

    codebook: [size, dim]
    returns:  [dim] — zeros if inbound is empty.
    """
    if not inbound:
        return torch.zeros(codebook.shape[1])
    indices = torch.tensor([letter.code for letter in inbound], dtype=torch.long)
    return codebook[indices].mean(dim=0)
