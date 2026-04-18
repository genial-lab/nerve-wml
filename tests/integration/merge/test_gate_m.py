import torch

from scripts.merge_pilot import run_merge_gate


def test_gate_m_merged_perf_ratio():
    torch.manual_seed(0)
    report = run_merge_gate()
    assert report["acc_mock_baseline"]  > 0.6
    assert report["acc_merged"]         > 0.6
    ratio = report["acc_merged"] / max(report["acc_mock_baseline"], 1e-6)
    assert ratio >= 0.95, f"Gate M failed: ratio={ratio:.3f}"
    assert report["all_passed"] is True
