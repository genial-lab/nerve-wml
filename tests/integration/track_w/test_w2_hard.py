"""W2 honest pilot on HardFlowProxyTask.

This pilot does NOT enforce a gap < 5 % — that assertion was shown
degenerate on the saturated 4-class task. Here we document the
HONEST measurement and pin reproducibility: the gap remains within
an observable, non-trivial band.
"""
import torch

from scripts.track_w_pilot import run_w2_hard


def test_w2_hard_produces_measurable_gap():
    """On the hard XOR task, MLP outperforms LIF enough that the gap
    is observable (> 5 %) but both beat random (1/12 ≈ 0.083)."""
    torch.manual_seed(0)
    report = run_w2_hard(steps=800)
    # Both beat random baseline.
    assert report["acc_mlp"] > 2 / 12
    assert report["acc_lif"] > 2 / 12
    # Gap is non-trivial — this is the honest finding.
    assert report["gap"] > 0.05, (
        f"gap {report['gap']:.3f} is suspiciously low on hard task — "
        "verify HardFlowProxyTask XOR wiring"
    )
    # Sanity ceiling — task is hard, no substrate saturates.
    assert report["acc_mlp"] < 0.95
    assert report["acc_lif"] < 0.95
