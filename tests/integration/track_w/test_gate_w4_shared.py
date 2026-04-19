import torch

from scripts.track_w_pilot import run_w4_shared_head


def test_w4_shared_head_baseline_measures_forgetting():
    """Honest continual learning baseline — shared head, same lr.
    Expected to forget (spec §13.1 predicts > 20 %)."""
    torch.manual_seed(0)
    report = run_w4_shared_head(steps=400)
    assert "forgetting" in report
    assert 0.0 <= report["forgetting"] <= 1.0
    assert report["acc_task0_initial"] > 0.6
