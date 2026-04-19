"""Tests for scale_diagnostic: router sparsity across N ∈ {4, 8, 16, 32}."""
import torch

from scripts.scale_diagnostic import scale_diagnostic


def test_scale_diagnostic_returns_metrics_per_N():
    """Diagnostic runs at N ∈ {4, 8, 16, 32} and returns per-N metrics."""
    torch.manual_seed(0)
    report = scale_diagnostic(Ns=[4, 8, 16, 32])
    assert set(report.keys()) == {4, 8, 16, 32}
    for n, metrics in report.items():
        assert "fan_out_mean" in metrics
        assert "fan_in_mean" in metrics
        assert "fan_in_std" in metrics
        assert "fan_out_std" in metrics
        assert "is_strongly_connected" in metrics
        assert "n_components" in metrics
        # k_for_n(N) = max(2, ceil(log2(N))), so mean fan-out should equal k.
        assert metrics["fan_out_mean"] > 1
        assert metrics["fan_out_std"] >= 0


def test_scale_diagnostic_n_components_positive():
    """Component count is always >= 1."""
    torch.manual_seed(0)
    report = scale_diagnostic(Ns=[4, 16])
    for n, metrics in report.items():
        assert metrics["n_components"] >= 1  # at least 1 component
