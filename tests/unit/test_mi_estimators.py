"""Unit tests for nerve_wml.methodology.mi_estimators.

Covers entropy, plug-in MI, Miller-Madow MI, and Kraskov-KSG MI.
Small arrays (<=500 samples) to keep the suite fast on CPU.
"""
from __future__ import annotations

import numpy as np
import pytest

from nerve_wml.methodology.mi_estimators import (
    entropy_discrete,
    mi_kraskov_ksg_continuous,
    mi_miller_madow_discrete,
    mi_plugin_discrete,
)


def test_entropy_uniform_alphabet() -> None:
    codes = np.tile(np.arange(4, dtype=np.int64), 250)
    h = entropy_discrete(codes)
    assert abs(h - np.log(4)) < 1e-6


def test_entropy_single_symbol() -> None:
    codes = np.full(100, 7, dtype=np.int64)
    assert entropy_discrete(codes) == 0.0


def test_entropy_empty_raises() -> None:
    with pytest.raises(ValueError):
        entropy_discrete(np.array([], dtype=np.int64))


def test_plugin_agrees_with_argmax_onehot() -> None:
    """Plug-in estimator matches the mi_null_model reference."""
    from nerve_wml.methodology.mi_null_model import mi_argmax_onehot

    rng = np.random.default_rng(0)
    a = rng.integers(0, 8, size=500).astype(np.int64)
    b = rng.integers(0, 8, size=500).astype(np.int64)
    assert abs(mi_plugin_discrete(a, b) - mi_argmax_onehot(a, b)) < 1e-9


def test_miller_madow_larger_than_plugin() -> None:
    """Miller-Madow adds a positive correction term."""
    rng = np.random.default_rng(42)
    a = rng.integers(0, 8, size=200).astype(np.int64)
    b = rng.integers(0, 8, size=200).astype(np.int64)
    plugin = mi_plugin_discrete(a, b)
    miller = mi_miller_madow_discrete(a, b)
    assert miller > plugin


def test_miller_madow_identical_codes_above_one() -> None:
    """For identical codes, Miller-Madow slightly over-estimates."""
    codes = np.tile(np.arange(4, dtype=np.int64), 100)
    rng = np.random.default_rng(0)
    rng.shuffle(codes)
    mm = mi_miller_madow_discrete(codes, codes)
    assert 1.0 <= mm < 1.1


def test_kraskov_independent_gaussians_near_zero() -> None:
    """Independent gaussians should give MI near 0."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 2))
    y = rng.standard_normal((200, 2))
    mi = mi_kraskov_ksg_continuous(x, y, k=3)
    assert mi < 0.10


def test_kraskov_perfectly_correlated_above_zero() -> None:
    """Y = X should give MI much above zero."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 2))
    y = x + 0.01 * rng.standard_normal((200, 2))
    mi = mi_kraskov_ksg_continuous(x, y, k=3)
    assert mi > 1.0


def test_kraskov_shape_mismatch_raises() -> None:
    x = np.zeros((50, 2))
    y = np.zeros((60, 2))
    with pytest.raises(ValueError, match="shape"):
        mi_kraskov_ksg_continuous(x, y)


def test_kraskov_too_few_samples_raises() -> None:
    x = np.zeros((3, 2))
    y = np.zeros((3, 2))
    with pytest.raises(ValueError, match="samples"):
        mi_kraskov_ksg_continuous(x, y, k=3)


def test_kraskov_monotone_in_correlation() -> None:
    """Stronger coupling -> larger MI estimate."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 2))
    y_weak = x + 0.5 * rng.standard_normal((200, 2))
    y_strong = x + 0.05 * rng.standard_normal((200, 2))
    mi_weak = mi_kraskov_ksg_continuous(x, y_weak, k=3)
    mi_strong = mi_kraskov_ksg_continuous(x, y_strong, k=3)
    assert mi_strong > mi_weak
