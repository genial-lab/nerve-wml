"""MlpWML — a WML whose core is a 4-layer MLP.

Implements the WML protocol (nerve_core.protocols.WML): listens on its nerve
input, decodes inbound codes via an embed_inbound mean-pool, runs the MLP,
and emits π predictions (γ phase) and optionally ε errors (θ phase).

The step() method is defined in Task 6 (π) and Task 7 (ε).
"""
from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn

from nerve_core.protocols import Nerve


class MlpWML(nn.Module):
    """WML with a 4-layer MLP core + independent π/ε emission heads."""

    def __init__(
        self,
        id:            int,
        d_hidden:      int  = 128,
        alphabet_size: int  = 64,
        threshold_eps: float = 0.30,
        *,
        seed:          int | None = None,
    ) -> None:
        super().__init__()
        self.id            = id
        self.alphabet_size = alphabet_size
        self.threshold_eps = threshold_eps

        # Create a local generator for all random ops
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)

        # Local codebook (N-5 — each WML owns its vocabulary).
        init = torch.randn(alphabet_size, d_hidden, generator=gen) * 0.1
        self.codebook = nn.Parameter(init)

        # Save global RNG state to avoid mutation during nn.Linear creation.
        global_state = torch.get_rng_state()

        # 4-layer MLP core with local generator init.
        layers = []
        for _ in range(4):
            lin = nn.Linear(d_hidden, d_hidden)
            # Manually set weights using local generator
            with torch.no_grad():
                lin.weight.data = torch.randn(
                    lin.weight.shape, generator=gen
                ) * 0.1
                lin.bias.data.zero_()
            layers.append(lin)
            if _ < 3:  # No ReLU after last layer
                layers.append(nn.ReLU())

        self.core = nn.Sequential(*layers)

        # Init heads with local generator.
        self.emit_head_pi  = nn.Linear(d_hidden, alphabet_size)
        self.emit_head_eps = nn.Linear(d_hidden, alphabet_size)

        with torch.no_grad():
            self.emit_head_pi.weight.data = torch.randn(
                self.emit_head_pi.weight.shape, generator=gen
            ) * 0.1
            self.emit_head_pi.bias.data.zero_()
            self.emit_head_eps.weight.data = torch.randn(
                self.emit_head_eps.weight.shape, generator=gen
            ) * 0.1
            self.emit_head_eps.bias.data.zero_()

        # Restore global RNG state.
        torch.set_rng_state(global_state)

    # step() defined in Task 6/7 — intentionally left empty here.
    def step(self, nerve: Nerve, t: float) -> None:  # pragma: no cover
        raise NotImplementedError("Task 6 defines MlpWML.step()")

    def parameters(self, *args, **kwargs) -> Iterable[Tensor]:  # type: ignore[override]
        return super().parameters(*args, **kwargs)
