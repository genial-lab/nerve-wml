"""TDD spec for track_p.multiplexer.GammaThetaMultiplexer (issue #1).

Scope: DSP layer only — roundtrip, shapes, PAC, differentiability, RNG.
Invariant under the α/β architectural choice (see issue #1 Q5) — i.e. tests
do NOT assume how SimNerve consumes the carrier.

All tests are expected to FAIL until track_p/multiplexer.py is implemented.
"""
from dataclasses import FrozenInstanceError

import pytest
import torch

from nerve_core.protocols import Nerve
from track_p.multiplexer import (
    AWGN,
    GammaThetaConfig,
    GammaThetaMultiplexer,
    HardwareJitterNoise,
    NoiseModel,
)

# ---------------------------------------------------------------------------
# Configuration & defaults
# ---------------------------------------------------------------------------

def test_config_defaults_reference_nerve_constants():
    """Defaults must come from Nerve.* — no hard-coded 40.0 / 6.0 / 64."""
    cfg = GammaThetaConfig()
    assert cfg.gamma_hz == Nerve.GAMMA_HZ
    assert cfg.theta_hz == Nerve.THETA_HZ
    assert cfg.alphabet_size == Nerve.ALPHABET_SIZE


def test_config_is_frozen():
    cfg = GammaThetaConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.gamma_hz = 42.0  # type: ignore[misc]


def test_config_symbols_per_theta_matches_lisman_idiart():
    """~7±2 γ symbols per θ period (Lisman & Idiart, 1995)."""
    cfg = GammaThetaConfig()
    assert 5 <= cfg.symbols_per_theta <= 9


def test_config_nyquist_guard():
    """sample_rate must comfortably exceed 2·γ (Nyquist)."""
    cfg = GammaThetaConfig()
    assert cfg.sample_rate_hz >= 4.0 * cfg.gamma_hz


# ---------------------------------------------------------------------------
# Shapes & module structure
# ---------------------------------------------------------------------------

def test_constellation_shape_matches_alphabet():
    mux = GammaThetaMultiplexer()
    assert mux.constellation.shape == (Nerve.ALPHABET_SIZE, 2)
    assert mux.constellation.requires_grad


def test_carrier_shape_is_bin_aligned():
    """Carrier length = round(sample_rate / γ) × symbols_per_theta.

    This pinning places γ exactly on a rFFT bin and makes every symbol
    window span an integer number of γ cycles → clean I/Q orthogonality
    and a clean PAC signature on rFFT bin 1 of the γ envelope.
    """
    cfg = GammaThetaConfig(sample_rate_hz=1000.0, gamma_hz=40.0, symbols_per_theta=7)
    mux = GammaThetaMultiplexer(cfg)
    codes = torch.randint(0, cfg.alphabet_size, (4, cfg.symbols_per_theta))
    carrier = mux.forward(codes)
    expected_t = round(cfg.sample_rate_hz / cfg.gamma_hz) * cfg.symbols_per_theta
    assert carrier.shape == (4, expected_t)  # (4, 175)
    assert carrier.dtype == torch.float32


def test_time_grid_registered_as_buffer():
    """_t_grid must be a buffer (follows .to(device)), not a Parameter."""
    mux = GammaThetaMultiplexer()
    assert "_t_grid" in dict(mux.named_buffers())
    assert "_t_grid" not in dict(mux.named_parameters())


def test_forward_rejects_too_many_symbols():
    """K > symbols_per_theta must raise — encoder can't fit extra γ chunks."""
    cfg = GammaThetaConfig(symbols_per_theta=7)
    mux = GammaThetaMultiplexer(cfg)
    bad = torch.zeros((1, cfg.symbols_per_theta + 1), dtype=torch.long)
    with pytest.raises((ValueError, AssertionError)):
        mux.forward(bad)


# ---------------------------------------------------------------------------
# DSP correctness — roundtrip & spectrum
# ---------------------------------------------------------------------------

def test_forward_demodulate_roundtrip_lossless_at_zero_noise():
    """Hard demodulation on a noise-free carrier must recover exact codes."""
    mux = GammaThetaMultiplexer(seed=0)
    cfg = mux.cfg
    codes = torch.randint(0, cfg.alphabet_size, (8, cfg.symbols_per_theta))
    carrier = mux.forward(codes)
    recovered = mux.demodulate(carrier, hard=True)
    assert torch.equal(recovered, codes)


def test_forward_demodulate_roundtrip_with_theta_phase_offset():
    """Roundtrip must hold for arbitrary θ phase offsets when forward and
    demod are called with the same offset. Guards against the bug where
    demod hard-codes offset=0 while forward honours the parameter."""
    mux = GammaThetaMultiplexer(seed=0)
    cfg = mux.cfg
    codes = torch.randint(0, cfg.alphabet_size, (4, cfg.symbols_per_theta))
    for offset in [0.0, 0.7, 1.5, -0.3]:
        carrier = mux.forward(codes, theta_phase_offset=offset)
        recovered = mux.demodulate(
            carrier, hard=True, theta_phase_offset=offset
        )
        assert torch.equal(recovered, codes), (
            f"roundtrip failed at theta_phase_offset={offset}"
        )


def test_soft_demodulate_returns_code_distribution():
    """hard=False must return a [B, K, alphabet_size] soft distribution,
    one row per symbol slot, each row summing to 1.0 (Gumbel-softmax output
    is a proper distribution).
    """
    mux = GammaThetaMultiplexer(seed=0)
    cfg = mux.cfg
    codes = torch.randint(0, cfg.alphabet_size, (3, cfg.symbols_per_theta))
    carrier = mux.forward(codes)
    soft = mux.demodulate(carrier, hard=False, tau=1.0)
    assert soft.shape == (3, cfg.symbols_per_theta, cfg.alphabet_size)
    assert soft.dtype in (torch.float32, torch.float64)
    row_sums = soft.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)


def test_soft_demodulate_gradient_flows_through_channel():
    """Soft demod must let a loss defined on the soft distribution backprop
    into the constellation — this is the non-negotiable Q2 requirement for
    bouba_sens CrossModalNerve.fuse θ-replay training.
    """
    mux = GammaThetaMultiplexer(seed=0)
    cfg = mux.cfg
    codes = torch.randint(0, cfg.alphabet_size, (2, cfg.symbols_per_theta))
    carrier = mux.forward(codes)
    soft = mux.demodulate(carrier, hard=False, tau=1.0)
    # Loss on the soft distribution (cross-entropy-like surrogate)
    soft.mean().backward()
    g = mux.constellation.grad
    assert g is not None
    assert g.abs().sum().item() > 0.0, (
        "constellation received no gradient via soft demod — "
        "backprop through channel is broken"
    )


# ---------------------------------------------------------------------------
# Noise models (Q3 scaffolding)
# ---------------------------------------------------------------------------

def test_noise_model_default_none_is_clean_carrier():
    """forward without noise arg must return identical carrier to AWGN(0)."""
    mux = GammaThetaMultiplexer(seed=0)
    cfg = mux.cfg
    codes = torch.randint(0, cfg.alphabet_size, (2, cfg.symbols_per_theta))
    clean = mux.forward(codes)
    with_awgn_zero = mux.forward(codes, noise=AWGN(sigma=0.0))
    assert torch.allclose(clean, with_awgn_zero, atol=1e-6)


def test_awgn_applies_additive_gaussian():
    """AWGN(σ>0) must change the carrier and scale with σ."""
    mux = GammaThetaMultiplexer(seed=0)
    cfg = mux.cfg
    codes = torch.randint(0, cfg.alphabet_size, (4, cfg.symbols_per_theta))
    clean = mux.forward(codes)
    torch.manual_seed(42)
    noisy = mux.forward(codes, noise=AWGN(sigma=0.1))
    diff = (noisy - clean).std().item()
    assert 0.05 < diff < 0.2, (
        f"AWGN(0.1) std deviation {diff:.3f} not in expected range [0.05, 0.2]"
    )


def test_hardware_jitter_is_instantiable_but_unimplemented():
    """HardwareJitterNoise can be constructed (hook wired) but apply() raises
    NotImplementedError — deferred to bouba_sens Sprint 4+ per issue #1 Q3.
    """
    jitter = HardwareJitterNoise(substrate="loihi2")
    assert isinstance(jitter, NoiseModel)
    mux = GammaThetaMultiplexer(seed=0)
    cfg = mux.cfg
    codes = torch.randint(0, cfg.alphabet_size, (1, cfg.symbols_per_theta))
    with pytest.raises(NotImplementedError, match="loihi2"):
        mux.forward(codes, noise=jitter)


# ---------------------------------------------------------------------------
# Role out-of-band channel (Q5 hook)
# ---------------------------------------------------------------------------

def test_role_parameter_default_none_is_current_behavior():
    """forward(codes) ≡ forward(codes, role=None) — no channel added.
    bouba_sens v0.1 uses the None path; the hook stays transparent."""
    mux = GammaThetaMultiplexer(seed=0)
    cfg = mux.cfg
    codes = torch.randint(0, cfg.alphabet_size, (2, cfg.symbols_per_theta))
    a = mux.forward(codes)
    b = mux.forward(codes, role=None)
    assert torch.equal(a, b)
    assert a.shape == (2, mux._t_grid.numel())  # 1-channel shape


def test_role_parameter_when_provided_raises():
    """role=<tensor> raises NotImplementedError — 2-channel carrier is
    deferred to bouba_sens v1.2 per issue #1 Q5 (additive extension, not
    in-band 32/32 alphabet split)."""
    mux = GammaThetaMultiplexer(seed=0)
    cfg = mux.cfg
    codes = torch.randint(0, cfg.alphabet_size, (1, cfg.symbols_per_theta))
    role = torch.zeros_like(codes)
    with pytest.raises(NotImplementedError, match="out-of-band role"):
        mux.forward(codes, role=role)


def test_carrier_spectrum_dominated_by_gamma_band():
    """Constant-code carrier reduces to pure γ × θ-env → peak at γ bin.

    Deterministic test: with all windows carrying the same code, the phase-
    modulation layer collapses and the carrier is exactly cos(ω_γ t - φ_0) ×
    theta_env(t). rFFT peak must fall on bin 7 (= 40 Hz) with strict
    tolerance < 2 Hz — bin-alignment guaranteed by the impl.
    """
    cfg = GammaThetaConfig(sample_rate_hz=1000.0)
    mux = GammaThetaMultiplexer(cfg, seed=0)
    codes = torch.zeros((1, cfg.symbols_per_theta), dtype=torch.long)
    carrier = mux.forward(codes)[0]
    spec = torch.fft.rfft(carrier).abs()
    freqs = torch.fft.rfftfreq(carrier.numel(), d=1.0 / cfg.sample_rate_hz)
    peak_hz = freqs[spec.argmax()].item()
    assert abs(peak_hz - cfg.gamma_hz) < 2.0, (
        f"spectral peak at {peak_hz} Hz ≠ γ ({cfg.gamma_hz} Hz)"
    )


def test_phase_amplitude_coupling_detectable():
    """γ-band power envelope on a constant-code carrier shows θ-rate peak.

    Deterministic test: constant codes isolate the θ-env modulation from
    random-symbol noise. The γ-power envelope's Fourier peak must fall in
    [θ_hz - 1, θ_hz + 1] with a ratio > 2 against other low-freq bins.
    """
    mux = GammaThetaMultiplexer(seed=0)
    cfg = mux.cfg
    codes = torch.zeros((1, cfg.symbols_per_theta), dtype=torch.long)
    carrier = mux.forward(codes)[0]
    pac_ratio = _theta_envelope_ratio(carrier, cfg)
    assert pac_ratio > 2.0, (
        f"θ-band/low-freq-other ratio on γ envelope = {pac_ratio:.2f} "
        f"— no phase-amplitude coupling detected"
    )


# ---------------------------------------------------------------------------
# Differentiability & reproducibility
# ---------------------------------------------------------------------------

def test_constellation_gradient_flows_through_carrier():
    mux = GammaThetaMultiplexer(seed=0)
    cfg = mux.cfg
    codes = torch.randint(0, cfg.alphabet_size, (2, cfg.symbols_per_theta))
    carrier = mux.forward(codes)
    carrier.pow(2).mean().backward()
    g = mux.constellation.grad
    assert g is not None
    assert g.abs().sum().item() > 0.0, "constellation received no gradient"


def test_seed_determinism_and_global_rng_isolation():
    """Same seed → same constellation. Global RNG untouched by __init__."""
    torch.manual_seed(1234)
    anchor = torch.randn(1).item()

    a = GammaThetaMultiplexer(seed=7)
    b = GammaThetaMultiplexer(seed=7)
    assert torch.equal(a.constellation, b.constellation)

    torch.manual_seed(1234)
    anchor_after = torch.randn(1).item()
    assert anchor == anchor_after, "__init__ polluted the global torch RNG"


# ---------------------------------------------------------------------------
# Helpers (pure torch.fft — no scipy dep, no torch.signal.hilbert)
# ---------------------------------------------------------------------------

def _theta_envelope_ratio(carrier: torch.Tensor, cfg: GammaThetaConfig) -> float:
    """Ratio of θ-band peak to out-of-band peak on the γ-envelope spectrum.

    Detects phase-amplitude coupling without needing a full Hilbert transform:
    1. Bandpass carrier around γ via rFFT mask.
    2. Square the result → power envelope.
    3. rFFT the zero-mean envelope.
    4. Peak in [θ_hz ± 1] vs peak elsewhere (excluding DC).
    """
    n = carrier.numel()
    spec = torch.fft.rfft(carrier)  # DSP convention X(f); lowercased per ruff N806
    freqs = torch.fft.rfftfreq(n, d=1.0 / cfg.sample_rate_hz)

    g_mask = (freqs >= cfg.gamma_hz - 10.0) & (freqs <= cfg.gamma_hz + 10.0)
    gamma_only = torch.fft.irfft(spec * g_mask.to(spec.dtype), n=n)

    power = gamma_only.pow(2)
    env_spec = torch.fft.rfft(power - power.mean()).abs()  # γ-envelope spectrum

    # θ band: around configured θ_hz. "Elsewhere" constrained to PAC-relevant
    # low-frequency range (< γ/2) to exclude 2γ harmonics — artefacts of cos²
    # in the power envelope, not PAC competitors.
    theta_band = (freqs >= cfg.theta_hz - 1.0) & (freqs <= cfg.theta_hz + 1.0)
    elsewhere = (~theta_band) & (freqs > 0.5) & (freqs < cfg.gamma_hz / 2.0)

    return (env_spec[theta_band].max() / (env_spec[elsewhere].max() + 1e-12)).item()
