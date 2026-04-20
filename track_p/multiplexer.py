"""γ/θ phase-amplitude-coupling multiplexer for neuroletter transport.

Contract pinned by tests/unit/test_multiplexer.py (issue #1).

This module currently exposes the config dataclass and the module skeleton
(structure tests pass) but leaves the DSP body as NotImplementedError. A
follow-up PR makes forward/demodulate green — see issue #1 for the
Q1-Q5 design decisions that guided this contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from nerve_core.protocols import Nerve

__all__ = ["GammaThetaConfig", "GammaThetaMultiplexer"]


@dataclass(frozen=True)
class GammaThetaConfig:
    """γ/θ PAC hyperparameters. Constants sourced from Nerve.* to avoid drift.

    - symbols_per_theta: 7 default (Lisman & Idiart 1995; Colgin 2016 [5, 9] range).
    - sample_rate_hz: 1000 default (≥ 4·γ = 160 Hz Nyquist guard + margin).
    - modulation: 'psk' for phase-shift keying (learned constellation),
                  'pam' for pulse-amplitude modulation (future work).
    """

    gamma_hz: float = Nerve.GAMMA_HZ
    theta_hz: float = Nerve.THETA_HZ
    sample_rate_hz: float = 1000.0
    alphabet_size: int = Nerve.ALPHABET_SIZE
    symbols_per_theta: int = 7
    modulation: Literal["psk", "pam"] = "psk"


class GammaThetaMultiplexer(nn.Module):
    """Multiplex 64-code neuroletters on a γ carrier amplitude-gated by θ phase.

    Theta-gamma phase-amplitude coupling (Lisman & Idiart 1995; Tort et al. 2010;
    Harris & Gong 2026). Operates on code tensors — src/dst/timestamp from
    `Neuroletter` are transport metadata, rebound topologically by the caller.

    End-to-end differentiable: the [ALPHABET_SIZE, 2] constellation is an
    nn.Parameter initialized as PSK and learned via downstream loss.
    """

    def __init__(
        self, cfg: GammaThetaConfig | None = None, *, seed: int | None = None
    ) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else GammaThetaConfig()

        # Constellation init: IQ (2-dim) per code. Seeded generation keeps
        # the global torch RNG untouched (MlpWML convention, see issue #1).
        if seed is not None:
            gen = torch.Generator()
            gen.manual_seed(seed)
            const = torch.randn(self.cfg.alphabet_size, 2, generator=gen)
        else:
            const = torch.randn(self.cfg.alphabet_size, 2)
        self.constellation = nn.Parameter(const)

        # Time grid covers one θ period at the configured sample rate.
        # Registered as a buffer so it follows .to(device) but is not trained.
        n_samples = int(self.cfg.sample_rate_hz / self.cfg.theta_hz)
        t_grid = torch.linspace(
            0.0, 1.0 / self.cfg.theta_hz, n_samples, dtype=torch.float32
        )
        self.register_buffer("_t_grid", t_grid)

    def forward(self, codes: Tensor, *, theta_phase_offset: float = 0.0) -> Tensor:
        """Encode codes onto a γ/θ PAC carrier.

        Args:
            codes: [B, K] long, K ≤ symbols_per_theta.
            theta_phase_offset: phase offset in radians for the θ carrier.

        Returns:
            carrier: [B, T] float32, T = sample_rate_hz // theta_hz.
        """
        if codes.shape[-1] > self.cfg.symbols_per_theta:
            raise ValueError(
                f"K={codes.shape[-1]} exceeds symbols_per_theta="
                f"{self.cfg.symbols_per_theta} (Lisman-Idiart capacity bound)"
            )
        raise NotImplementedError(
            "γ/θ PAC encoder pending — contract pinned by tests, impl in follow-up PR"
        )

    def demodulate(self, carrier: Tensor, *, hard: bool = True) -> Tensor:
        """Recover code tensor from a γ/θ carrier.

        Args:
            carrier: [B, T] float32.
            hard: argmax when True, Gumbel-softmax when False (mirrors Transducer).

        Returns:
            codes: [B, K] long.
        """
        raise NotImplementedError(
            "γ/θ PAC demodulator pending — contract pinned by tests, impl in follow-up PR"
        )
