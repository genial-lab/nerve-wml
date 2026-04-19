import torch

from scripts.track_w_pilot import run_w4_n16


def test_w4_n16_rehearsal_keeps_forgetting_under_20pct():
    """Rehearsal-based continual learning at N=16, forgetting < 20 %."""
    torch.manual_seed(0)
    report = run_w4_n16(steps=400, rehearsal_frac=0.3)
    # Negative forgetting = positive transfer (post > initial). Acceptable.
    assert report["forgetting"] <= 1.0
    assert report["acc_task0_initial"] > 0.6, (
        f"Task 0 initial accuracy {report['acc_task0_initial']:.3f} below 0.6"
    )
    assert report["forgetting"] < 0.20, (
        f"N=16 forgetting {report['forgetting']:.3f} exceeds 20 %"
    )
