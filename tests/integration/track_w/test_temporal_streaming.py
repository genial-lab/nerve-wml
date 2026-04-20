"""Temporal streaming test — v1.2 track 3 Claim B."""
import pytest

from scripts.run_temporal_pilots import run_temporal_mi


@pytest.mark.slow
def test_temporal_mi_at_trained_step():
    """At the supervised timestep t_b, MI between MLP and LIF codes
    exceeds 10 % of H — Claim B holds under temporal streaming
    restricted to where training pressure exists. (We only supervise
    at t_b, not t_a, so untrained-step MI is expected to be near
    zero — it is the control condition for this test.)
    """
    r = run_temporal_mi(seeds=[0, 1, 2], steps=200)
    assert r["mean_mi_at_trained_step"] > 0.10, (
        f"trained-step MI/H {r['mean_mi_at_trained_step']:.3f} < 10 %"
    )
