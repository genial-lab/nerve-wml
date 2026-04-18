import torch

from scripts.track_w_pilot import run_w4


def test_w4_forgetting_under_20pct():
    torch.manual_seed(0)
    report = run_w4(steps=400)
    forgetting = (report["acc_task0_initial"] - report["acc_task0_after_task1"])
    forgetting_pct = forgetting / max(report["acc_task0_initial"], 1e-6)
    assert forgetting_pct < 0.20, f"forgetting={forgetting_pct:.3f}"
