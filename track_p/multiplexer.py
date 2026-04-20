"""γ/θ phase-amplitude-coupling multiplexer for neuroletter transport.

Contract pinned by tests/unit/test_multiplexer.py (issue #1).

This module currently exposes the config dataclass and the module skeleton
(structure tests pass) but leaves the DSP body as NotImplementedError. A
follow-up PR makes forward/demodulate green — see issue #1 for the
Q1-Q5 design decisions that guided this contract.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from nerve_core.protocols import Nerve

__all__ = [
    "AWGN",
    "GammaThetaConfig",
    "GammaThetaMultiplexer",
    "HardwareJitterNoise",
    "NoiseModel",
]


class NoiseModel(ABC):
    """Channel noise model applied to the clean carrier.

    Concrete subclasses implement `apply`, taking the clean carrier and
    returning a noisy carrier of the same shape and dtype.
    """

    @abstractmethod
    def apply(self, carrier: Tensor) -> Tensor: ...


class AWGN(NoiseModel):
    """Additive white Gaussian noise with fixed σ.

    Textbook baseline per O'Shea & Hoydis 2017. σ=0 is a no-op so AWGN(0)
    can be passed in place of None without branching at the call site.
    """

    def __init__(self, sigma: float = 0.0) -> None:
        self.sigma = sigma

    def apply(self, carrier: Tensor) -> Tensor:
        if self.sigma == 0.0:
            return carrier
        return carrier + self.sigma * torch.randn_like(carrier)


class HardwareJitterNoise(NoiseModel):
    """Neuromorphic substrate phase-jitter noise — stub for future work.

    Loihi 2 (~200 ns) and SpiNNaker2 (~ms) jitter profiles per Moradi
    et al. 2025 (Nat Commun, doi:10.1038/s41467-025-65268-z). Instantiable
    so consumers can wire the hook, but `apply()` raises until the jitter
    profile impl lands — bouba_sens Sprint 4+ requirement per issue #1 Q3.
    """

    def __init__(self, substrate: Literal["loihi2", "spinnaker2"]) -> None:
        self.substrate = substrate

    def apply(self, carrier: Tensor) -> Tensor:
        raise NotImplementedError(
            f"HardwareJitterNoise({self.substrate!r}) pending — "
            f"bouba_sens Sprint 4+ scope per issue #1 Q3 arbitration"
        )


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

        # Constellation init: true PSK on the unit circle + small randn
        # perturbation for symmetry breaking. Min pairwise distance is
        # 2·sin(π/alphabet_size) ≈ 0.098 for alphabet_size=64, vs ~0.01
        # for plain randn (12% of pairs collide under PSK normalization).
        # Seeded generation keeps the global torch RNG untouched
        # (MlpWML convention, see issue #1).
        angles = (
            2.0
            * torch.pi
            * torch.arange(self.cfg.alphabet_size, dtype=torch.float32)
            / self.cfg.alphabet_size
        )
        psk_base = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        if seed is not None:
            gen = torch.Generator()
            gen.manual_seed(seed)
            noise = 0.01 * torch.randn(self.cfg.alphabet_size, 2, generator=gen)
        else:
            noise = 0.01 * torch.randn(self.cfg.alphabet_size, 2)
        self.constellation = nn.Parameter(psk_base + noise)

        # Time grid covers one effective θ period = symbols_per_theta × γ period
        # exactly. This bin-aligns γ on a rFFT bucket (no quantization leakage)
        # AND makes the γ period an integer submultiple of the θ period, giving
        # exact γ-quadrature orthogonality within each symbol window.
        # Registered as a buffer so it follows .to(device) but is not trained.
        n_per_gamma = max(1, round(self.cfg.sample_rate_hz / self.cfg.gamma_hz))
        n_samples = n_per_gamma * self.cfg.symbols_per_theta
        t_grid = (
            torch.arange(n_samples, dtype=torch.float32) / self.cfg.sample_rate_hz
        )
        self.register_buffer("_t_grid", t_grid)

    def forward(
        self,
        codes: Tensor,
        *,
        theta_phase_offset: float = 0.0,
        noise: NoiseModel | None = None,
        role: Tensor | None = None,
    ) -> Tensor:
        """Encode codes onto a γ/θ PAC carrier.

        Args:
            codes: [B, K] long, K ≤ symbols_per_theta.
            theta_phase_offset: phase offset in radians for the θ carrier.
            noise: optional NoiseModel applied to the clean carrier. None
                (default) returns the clean signal.
            role: optional [B, K] long carrying Role.PREDICTION / Role.ERROR
                per symbol. None (default) is the 1-channel case and is
                what bouba_sens v0.1 uses. When provided, returns a
                2-channel carrier (shape [B, 2, T]) — pending bouba_sens
                v1.2 per issue #1 Q5 arbitration (out-of-band not in-band
                split, to preserve the full 64-code alphabet).

        Returns:
            carrier: [B, T] float32 when role is None.
            carrier: [B, 2, T] float32 when role is provided (future).
        """
        if codes.shape[-1] > self.cfg.symbols_per_theta:
            raise ValueError(
                f"K={codes.shape[-1]} exceeds symbols_per_theta="
                f"{self.cfg.symbols_per_theta} (Lisman-Idiart capacity bound)"
            )
        if role is not None:
            raise NotImplementedError(
                "out-of-band role channel pending — bouba_sens v1.2 scope "
                "per issue #1 Q5 arbitration"
            )
        k_active = codes.shape[-1]
        t = self._t_grid  # [n_samples]
        n_samples = t.numel()
        k_cap = self.cfg.symbols_per_theta
        window_n = n_samples // k_cap

        # Symbol constellation IQ per code, normalized to unit norm (PSK).
        # Unit-amplitude symbols eliminate amplitude-modulation sidebands that
        # would otherwise compete with the γ peak at bin 7.
        raw_sym = self.constellation[codes]  # [batch, k_active, 2]
        sym = raw_sym / raw_sym.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # γ carrier quadratures (in-phase = cos, quadrature = sin): [n_samples].
        two_pi_gamma_t = 2.0 * torch.pi * self.cfg.gamma_hz * t
        gamma_i = torch.cos(two_pi_gamma_t)
        gamma_q = torch.sin(two_pi_gamma_t)

        # θ envelope — smooth cos, depth 0.45 so env ∈ [0.1, 1.0] never nulls.
        # Aligns with Harris & Gong 2026 (Nat Commun) on smooth traveling-wave
        # envelopes over rect θ windows (Q4 default, see issue #1). Depth 0.45
        # gives a detectable PAC signature without nulling the demod divide.
        # One envelope cycle spans the full carrier (effective θ = γ / k_cap,
        # so the θ-peak falls exactly on rFFT bin 1 of the γ power envelope).
        effective_theta_hz = self.cfg.sample_rate_hz / n_samples
        two_pi_theta_t = (
            2.0 * torch.pi * effective_theta_hz * t + theta_phase_offset
        )
        theta_env = 0.55 + 0.45 * torch.cos(two_pi_theta_t)

        # Gaussian per-symbol envelopes (Q4, see issue #1). Each symbol k is
        # a Gaussian packet centered at (k+0.5)·window_n with σ = window_n/4,
        # so ~95% of the packet sits inside window k. Overlapping Gaussians
        # smooth transitions, preserve differentiability, and track the
        # Harris & Gong 2026 nested traveling-wave preference over rect
        # windows (which leak spectral energy into θ sidebands).
        window_centers = (
            (torch.arange(k_cap, device=t.device, dtype=t.dtype) + 0.5) * window_n
        )
        sigma = window_n / 4.0
        idx = torch.arange(n_samples, device=t.device, dtype=t.dtype)
        gaussian_masks = torch.exp(
            -((idx.unsqueeze(0) - window_centers.unsqueeze(1)) ** 2)
            / (2.0 * sigma**2)
        )[:k_active]  # [k_active, n_samples]

        # Sum symbol IQ contributions weighted by Gaussians: [batch, n_samples].
        i_t = (sym[..., 0:1] * gaussian_masks.unsqueeze(0)).sum(dim=1)
        q_t = (sym[..., 1:2] * gaussian_masks.unsqueeze(0)).sum(dim=1)

        carrier = (
            i_t * gamma_i.unsqueeze(0) + q_t * gamma_q.unsqueeze(0)
        ) * theta_env.unsqueeze(0)
        carrier = carrier.to(torch.float32)

        if noise is not None:
            carrier = noise.apply(carrier)

        return carrier

    def demodulate(
        self,
        carrier: Tensor,
        *,
        hard: bool = True,
        tau: float = 1.0,
        theta_phase_offset: float = 0.0,
    ) -> Tensor:
        """Recover code tensor from a γ/θ carrier.

        Args:
            carrier: [B, T] float32.
            hard: argmax when True (eval), Gumbel-softmax distribution when
                False (training, mirrors Transducer convention).
            tau: Gumbel-softmax temperature (anneal downward during training).
                Ignored when hard=True.
            theta_phase_offset: must match the offset passed to `forward()`;
                a mismatch breaks the roundtrip because the demod divides by
                the θ envelope.

        Returns:
            - hard=True : [B, K] long, K = symbols_per_theta.
            - hard=False : [B, K, alphabet_size] float — soft distribution
                over codes per symbol slot. Enables gradient flow from
                downstream loss back through the channel into the constellation
                (bouba_sens CrossModalNerve.fuse θ-replay path).
        """
        batch, n_samples = carrier.shape
        t = self._t_grid
        k_cap = self.cfg.symbols_per_theta
        window_n = n_samples // k_cap

        # γ quadratures and θ envelope (must match forward exactly).
        two_pi_gamma_t = 2.0 * torch.pi * self.cfg.gamma_hz * t
        gamma_i = torch.cos(two_pi_gamma_t)
        gamma_q = torch.sin(two_pi_gamma_t)
        # Effective θ = γ / k_cap (bin-aligned, see forward).
        effective_theta_hz = self.cfg.sample_rate_hz / n_samples
        two_pi_theta_t = (
            2.0 * torch.pi * effective_theta_hz * t + theta_phase_offset
        )
        theta_env = 0.55 + 0.45 * torch.cos(two_pi_theta_t)

        # Undo θ envelope (safe: env ∈ [0.1, 1.0] never nulls).
        carrier_norm = carrier / theta_env.unsqueeze(0)

        # Rebuild the same Gaussian masks as forward.
        window_centers = (
            (torch.arange(k_cap, device=t.device, dtype=t.dtype) + 0.5) * window_n
        )
        sigma = window_n / 4.0
        idx = torch.arange(n_samples, device=t.device, dtype=t.dtype)
        gaussian_masks = torch.exp(
            -((idx.unsqueeze(0) - window_centers.unsqueeze(1)) ** 2)
            / (2.0 * sigma**2)
        )  # [k_cap, n_samples]

        # Design matrix with overlapping Gaussian × γ-quadrature basis:
        # column 2k   = gaussian_k · gamma_i
        # column 2k+1 = gaussian_k · gamma_q
        # Global LSTSQ recovers (I, Q) for all k jointly — required because
        # adjacent Gaussians overlap (per-window LSTSQ would bleed neighbours).
        basis_i = gaussian_masks * gamma_i.unsqueeze(0)  # [k_cap, n_samples]
        basis_q = gaussian_masks * gamma_q.unsqueeze(0)
        basis_matrix = torch.stack([basis_i, basis_q], dim=-1).permute(
            1, 0, 2
        ).reshape(n_samples, 2 * k_cap)

        sol = torch.linalg.lstsq(
            basis_matrix, carrier_norm.T
        ).solution  # [2*k_cap, batch]
        recovered = sol.T.reshape(batch, k_cap, 2).to(torch.float32)

        # Unit-normalize constellation to match forward-time PSK normalization.
        const_norm = self.constellation / self.constellation.norm(
            dim=-1, keepdim=True
        ).clamp(min=1e-8)

        # Negative squared distance = similarity logit (higher = closer).
        # Using squared distance keeps the gradient path smooth for soft mode.
        dist_sq = (
            (recovered.reshape(-1, 1, 2) - const_norm.unsqueeze(0)).pow(2).sum(dim=-1)
        )  # [batch*k_cap, alphabet_size]
        logits = (-dist_sq).reshape(batch, k_cap, -1)  # [batch, k_cap, alphabet]

        if hard:
            return logits.argmax(dim=-1)  # [batch, k_cap] long

        # Soft Gumbel-softmax distribution for differentiable backprop
        # through the channel (bouba_sens θ-replay loss path).
        return F.gumbel_softmax(logits, tau=tau, hard=False)
