import torch

from scripts.track_w_pilot import run_w4_multi_seed


def test_w4_multi_seed_returns_per_seed_forgetting():
    """Multi-seed helper returns forgetting lists for shared and rehearsal modes."""
    torch.manual_seed(0)
    report = run_w4_multi_seed(seeds=[0, 1, 2], steps=200)
    assert "forgetting_shared" in report
    assert "forgetting_rehearsal" in report
    assert len(report["forgetting_shared"]) == 3
    assert len(report["forgetting_rehearsal"]) == 3
    for v in report["forgetting_shared"] + report["forgetting_rehearsal"]:
        assert 0.0 <= v <= 1.0


def test_w4_multi_seed_rehearsal_better_than_shared():
    """Across at least 2 of 3 seeds, rehearsal forgets less than shared-head baseline."""
    torch.manual_seed(0)
    report = run_w4_multi_seed(seeds=[0, 1, 2], steps=200)
    better = sum(
        1 for s, r in zip(report["forgetting_shared"], report["forgetting_rehearsal"], strict=True)
        if r <= s  # ≤ because rehearsal might match shared if shared was already low
    )
    assert better >= 2
