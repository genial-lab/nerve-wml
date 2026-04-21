"""Mutual-information estimators used for cross-estimator robustness checks.

Complements ``mi_null_model`` (significance) and ``bootstrap_ci_mi``
(uncertainty) with a third methodological primitive: multi-estimator
comparison. Per bouba_sens section 6.3, an MI-based claim should
agree across at least two independent estimators before being treated
as load-bearing.

This module provides:

* ``mi_plugin_discrete``       -- plug-in entropy-normalised MI on
                                  integer codes (equivalent to
                                  ``mi_argmax_onehot`` from the
                                  ``mi_null_model`` module).
* ``mi_miller_madow_discrete`` -- bias-corrected plug-in: adds the
                                  Miller-Madow term to mitigate the
                                  systematic under-estimation of
                                  plug-in MI at finite sample.
* ``mi_kraskov_ksg_continuous``-- Kraskov-Stogbauer-Grassberger 2004
                                  k-NN estimator on continuous
                                  embeddings (algorithm 1). Operates
                                  in nats; divide by ln(2) for bits.
* ``entropy_discrete``         -- Shannon entropy of an integer array
                                  (plug-in, in nats).

All estimators are pure numpy + scipy.special. They run on commodity
CPU; Kraskov on N=5000, d=32 takes a few seconds and ~600 MB RAM due
to the N-by-N Chebyshev distance matrix.
"""
from __future__ import annotations

import numpy as np
from scipy.special import digamma


def entropy_discrete(codes: np.ndarray) -> float:
    """Plug-in Shannon entropy of a 1-D integer code array, in nats."""
    if codes.ndim != 1:
        raise ValueError(f"expected 1-D codes, got ndim={codes.ndim}")
    if codes.size == 0:
        raise ValueError("empty codes array")
    _, counts = np.unique(codes, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p)))


def mi_plugin_discrete(codes_a: np.ndarray, codes_b: np.ndarray) -> float:
    """Entropy-normalised plug-in MI/H(a) on integer codes.

    Identical to ``mi_argmax_onehot`` in ``mi_null_model``; re-exported
    here for symmetric naming with the Miller-Madow and Kraskov
    estimators.
    """
    if codes_a.shape != codes_b.shape:
        raise ValueError(
            f"codes_a shape {codes_a.shape} != codes_b shape {codes_b.shape}"
        )
    if codes_a.ndim != 1:
        raise ValueError(f"expected 1-D codes, got ndim={codes_a.ndim}")
    if codes_a.size == 0:
        raise ValueError("empty arrays")

    n = codes_a.shape[0]
    alphabet_a = int(codes_a.max()) + 1
    alphabet_b = int(codes_b.max()) + 1
    joint = np.zeros((alphabet_a, alphabet_b), dtype=np.float64)
    np.add.at(joint, (codes_a, codes_b), 1.0)
    joint /= n
    p_a = joint.sum(axis=1)
    p_b = joint.sum(axis=0)

    mask = joint > 0
    denom = np.where(mask, p_a[:, None] * p_b[None, :], 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.where(mask, np.log(joint / denom), 0.0)
    mi = float(np.sum(joint * log_ratio))
    h_a = float(-np.sum([p * np.log(p) for p in p_a if p > 0]))
    return mi / h_a if h_a > 0 else 0.0


def mi_miller_madow_discrete(
    codes_a: np.ndarray,
    codes_b: np.ndarray,
) -> float:
    """Miller-Madow bias-corrected MI/H(a).

    Plug-in MI systematically under-estimates true MI at finite N.
    Miller & Madow 1955 showed that the bias can be partially
    corrected by adding ``(K_a - 1)(K_b - 1) / (2 N)`` to the plug-in
    estimate, where K_a and K_b are the effective alphabet sizes
    (count of observed symbols, not the nominal 64 slots).

    Returned value is normalised by H(a) to match the plug-in
    convention, so both estimators are in the same [0, 1]-ish range.
    """
    if codes_a.shape != codes_b.shape:
        raise ValueError("shape mismatch")
    if codes_a.ndim != 1:
        raise ValueError("expected 1-D")
    if codes_a.size == 0:
        raise ValueError("empty arrays")

    n = codes_a.shape[0]
    observed_a = np.unique(codes_a).size
    observed_b = np.unique(codes_b).size

    mi_hat = mi_plugin_discrete(codes_a, codes_b)
    h_a = entropy_discrete(codes_a)
    if h_a == 0:
        return 0.0

    mi_unnorm = mi_hat * h_a
    bias_correction = (observed_a - 1) * (observed_b - 1) / (2.0 * n)
    mi_corrected = mi_unnorm + bias_correction
    return float(mi_corrected / h_a)


def _chebyshev_pairwise(points: np.ndarray) -> np.ndarray:
    """Return (N, N) Chebyshev (L-inf) pairwise distance matrix."""
    diff = points[:, None, :] - points[None, :, :]
    return np.max(np.abs(diff), axis=2)


def mi_kraskov_ksg_continuous(
    x: np.ndarray,
    y: np.ndarray,
    *,
    k: int = 3,
) -> float:
    """Kraskov-Stogbauer-Grassberger MI estimator, algorithm 1.

    Estimates MI(X; Y) for two continuous variables X and Y via
    k-nearest-neighbour statistics in the joint (X, Y) space. Uses
    Chebyshev (L-inf) distance, per the original paper.

    Args:
        x: (N, d_x) continuous array.
        y: (N, d_y) continuous array, same N as x.
        k: number of neighbours (default 3 per the original paper).

    Returns:
        MI estimate in nats. Clipped to >= 0 (the estimator is known
        to go slightly negative for near-independent data at finite N).
    """
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"x.shape[0]={x.shape[0]} != y.shape[0]={y.shape[0]}"
        )
    n = x.shape[0]
    if n <= k + 1:
        raise ValueError(
            f"need more than {k + 1} samples for KSG with k={k}, got {n}"
        )

    xy = np.concatenate([x, y], axis=1)

    dist_xy = _chebyshev_pairwise(xy)
    np.fill_diagonal(dist_xy, np.inf)
    dist_xy_sorted = np.sort(dist_xy, axis=1)
    eps = dist_xy_sorted[:, k - 1]

    dist_x = _chebyshev_pairwise(x)
    np.fill_diagonal(dist_x, np.inf)
    n_x = np.sum(dist_x < eps[:, None], axis=1)

    dist_y = _chebyshev_pairwise(y)
    np.fill_diagonal(dist_y, np.inf)
    n_y = np.sum(dist_y < eps[:, None], axis=1)

    mi = (
        digamma(k)
        + digamma(n)
        - float(np.mean(digamma(n_x + 1) + digamma(n_y + 1)))
    )
    return float(max(mi, 0.0))
