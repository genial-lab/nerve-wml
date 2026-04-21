"""Tests for the v1.4 TransducerGating opt-in Gumbel-softmax path (issue #5).

Backward compat: default gating is HARD, forward returns [B] long codes,
and the gradient path is broken at argmax (the v0.3 bouba_sens B-3-PASS
behavior must not change).

Opt-in: gating=GUMBEL_SOFTMAX makes forward return a [B, alphabet_size]
differentiable soft distribution. Consumers (bouba_sens Sprint 7+) can
keep gradients flowing through the code axis to test whether hard 0/1
gating is the reason B-2 Me3 delta stays under threshold.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from track_p.transducer import Transducer, TransducerGating


_ALPHABET = 16


def _src(batch: int = 4) -> torch.Tensor:
    return torch.arange(batch, dtype=torch.long) % _ALPHABET


# ---------------------------------------------------------------------------
# Backward compatibility — HARD is the default, pre-v1.4 behavior preserved
# ---------------------------------------------------------------------------


def test_default_gating_is_hard_and_backward_compatible():
    """New kwargs are optional; existing call sites keep working."""
    t = Transducer(alphabet_size=_ALPHABET)
    assert t.gating == TransducerGating.HARD
    out = t.forward(_src())
    # Legacy contract: shape equals src_code shape, dtype is long.
    assert out.shape == (4,)
    assert out.dtype == torch.long
    assert (out >= 0).all() and (out < _ALPHABET).all()


def test_explicit_hard_matches_default():
    """Passing hard=True explicitly must not change the output shape."""
    torch.manual_seed(0)
    t = Transducer(alphabet_size=_ALPHABET)
    out_default = t.forward(_src(), hard=True)
    assert out_default.shape == (4,)
    assert out_default.dtype == torch.long


def test_hard_path_argmax_breaks_grad():
    """argmax is the v0.3 canonical path — grad does NOT flow through it."""
    t = Transducer(alphabet_size=_ALPHABET)
    out = t.forward(_src())  # default HARD
    # long tensor cannot carry grad; attempting .backward through it raises.
    assert not out.is_floating_point()
    # The logits parameter is still trainable via the entropy regularizer
    # or any other loss, but not through this forward call alone.


# ---------------------------------------------------------------------------
# Opt-in GUMBEL_SOFTMAX — soft distribution, differentiable
# ---------------------------------------------------------------------------


def test_gumbel_softmax_returns_soft_distribution():
    t = Transducer(alphabet_size=_ALPHABET, gating=TransducerGating.GUMBEL_SOFTMAX)
    out = t.forward(_src())
    assert out.shape == (4, _ALPHABET)
    assert out.dtype == torch.float32
    # Each row sums to 1 (gumbel_softmax normalizes).
    row_sums = out.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_gumbel_softmax_is_differentiable():
    """Gradient must flow back into the transducer logits via the soft path."""
    torch.manual_seed(0)
    t = Transducer(alphabet_size=_ALPHABET, gating=TransducerGating.GUMBEL_SOFTMAX)
    out = t.forward(_src())
    loss = out.sum()
    loss.backward()
    assert t.logits.grad is not None
    assert torch.isfinite(t.logits.grad).all()
    assert t.logits.grad.abs().sum() > 0.0


def test_per_call_hard_override_wins_over_gating():
    """Passing hard=True on a GUMBEL-configured instance forces the hard path."""
    t = Transducer(alphabet_size=_ALPHABET, gating=TransducerGating.GUMBEL_SOFTMAX)
    out_hard = t.forward(_src(), hard=True)
    assert out_hard.shape == (4,)
    assert out_hard.dtype == torch.long


def test_per_call_hard_false_forces_soft_on_hard_instance():
    """Passing hard=False on a HARD-configured instance forces the soft path."""
    t = Transducer(alphabet_size=_ALPHABET, gating=TransducerGating.HARD)
    out_soft = t.forward(_src(), hard=False)
    assert out_soft.shape == (4, _ALPHABET)
    assert out_soft.dtype == torch.float32


# ---------------------------------------------------------------------------
# Temperature knob
# ---------------------------------------------------------------------------


def test_gumbel_tau_is_stored_and_used():
    t = Transducer(alphabet_size=_ALPHABET, gating=TransducerGating.GUMBEL_SOFTMAX, gumbel_tau=0.3)
    assert t.gumbel_tau == pytest.approx(0.3)
    # Low tau sharpens toward the argmax — max row value closer to 1.
    torch.manual_seed(0)
    out = t.forward(_src())
    assert out.max(dim=-1).values.mean().item() > 0.5


def test_tau_override_per_call():
    torch.manual_seed(0)
    t = Transducer(alphabet_size=_ALPHABET, gating=TransducerGating.GUMBEL_SOFTMAX, gumbel_tau=5.0)
    out_low = t.forward(_src(), tau=0.1)
    # tau=0.1 concentrates; max value per row should be well above 1/alphabet.
    assert out_low.max(dim=-1).values.mean().item() > 0.5


# ---------------------------------------------------------------------------
# Serializability (enum is str-backed for YAML/JSON configs)
# ---------------------------------------------------------------------------


def test_gating_enum_is_yaml_json_serializable():
    import json
    assert json.dumps(TransducerGating.HARD) == '"hard"'
    assert json.dumps(TransducerGating.GUMBEL_SOFTMAX) == '"gumbel_softmax"'


def test_gating_accepts_string_in_init():
    """bouba_sens grid configs pass the mode as a plain string from YAML."""
    t = Transducer(alphabet_size=_ALPHABET, gating="gumbel_softmax")
    assert t.gating == TransducerGating.GUMBEL_SOFTMAX
