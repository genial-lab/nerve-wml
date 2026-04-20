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

    constellation: nn.Parameter
    _t_grid: Tensor

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
        k_active = codes.shape[-1]
        t = self._t_grid  # [n_samples]
        n_samples = t.numel()
        k_cap = self.cfg.symbols_per_theta
        window_n = n_samples // k_cap

        # Symbol constellation IQ per code: [batch, k_active, 2].
        sym = self.constellation[codes]

        # γ carrier quadratures (in-phase = cos, quadrature = sin): [n_samples].
        two_pi_gamma_t = 2.0 * torch.pi * self.cfg.gamma_hz * t
        gamma_i = torch.cos(two_pi_gamma_t)
        gamma_q = torch.sin(two_pi_gamma_t)

        # θ envelope — smooth cos, depth 0.45 so env ∈ [0.1, 1.0] never nulls.
        # Aligns with Harris & Gong 2026 (Nat Commun) on smooth traveling-wave
        # envelopes over rect θ windows (Q4 default, see issue #1). Depth 0.45
        # gives a detectable PAC signature without nulling the demod divide.
        two_pi_theta_t = 2.0 * torch.pi * self.cfg.theta_hz * t + theta_phase_offset
        theta_env = 0.55 + 0.45 * torch.cos(two_pi_theta_t)

        # Per-symbol window masks [k_active, n_samples]; only active symbols populated.
        idx = torch.arange(n_samples, device=t.device)
        symbol_idx = torch.clamp(idx // window_n, max=k_cap - 1)
        window_masks = (
            symbol_idx.unsqueeze(0)
            == torch.arange(k_cap, device=t.device).unsqueeze(1)
        ).to(t.dtype)[:k_active]

        # Sum symbol IQ contributions along active windows: [batch, n_samples].
        i_t = (sym[..., 0:1] * window_masks.unsqueeze(0)).sum(dim=1)
        q_t = (sym[..., 1:2] * window_masks.unsqueeze(0)).sum(dim=1)

        carrier = (
            i_t * gamma_i.unsqueeze(0) + q_t * gamma_q.unsqueeze(0)
        ) * theta_env.unsqueeze(0)

        return carrier.to(torch.float32)

    def demodulate(self, carrier: Tensor, *, hard: bool = True) -> Tensor:
        """Recover code tensor from a γ/θ carrier.

        Args:
            carrier: [B, T] float32.
            hard: argmax when True, Gumbel-softmax when False (mirrors Transducer).

        Returns:
            codes: [B, K] long, K = symbols_per_theta.
        """
        if not hard:
            raise NotImplementedError(
                "soft demodulation (hard=False) pending — Q2 arbitration in issue #1"
            )
        batch, n_samples = carrier.shape
        t = self._t_grid
        k_cap = self.cfg.symbols_per_theta
        window_n = n_samples // k_cap

        # γ quadratures and θ envelope (must match forward exactly).
        two_pi_gamma_t = 2.0 * torch.pi * self.cfg.gamma_hz * t
        gamma_i = torch.cos(two_pi_gamma_t)
        gamma_q = torch.sin(two_pi_gamma_t)
        two_pi_theta_t = 2.0 * torch.pi * self.cfg.theta_hz * t
        theta_env = 0.55 + 0.45 * torch.cos(two_pi_theta_t)

        # Undo θ envelope (safe: env ∈ [0.1, 1.0] never nulls).
        carrier_norm = carrier / theta_env.unsqueeze(0)

        # Per-window LSTSQ recovers (i, q) exactly at zero noise regardless
        # of γ orthogonality over short windows.
        recovered = torch.empty(
            batch, k_cap, 2, device=carrier.device, dtype=torch.float32
        )
        for k in range(k_cap):
            start = k * window_n
            end = (k + 1) * window_n if k < k_cap - 1 else n_samples
            basis = torch.stack(
                [gamma_i[start:end], gamma_q[start:end]], dim=-1
            )  # [n, 2]
            rhs = carrier_norm[:, start:end].T  # [n, batch]
            sol = torch.linalg.lstsq(basis, rhs).solution  # [2, batch]
            recovered[:, k, :] = sol.T.to(torch.float32)

        # Nearest-constellation assignment — dist [batch*k_cap, A], argmin → [batch, k_cap].
        dist = torch.cdist(recovered.reshape(-1, 2), self.constellation)
        codes = dist.argmin(dim=-1).reshape(batch, k_cap)
        return codes
