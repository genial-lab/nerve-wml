import torch

from scripts.track_w_pilot import run_gate_w


def test_gate_w_all_criteria_pass():
    torch.manual_seed(0)
    report = run_gate_w()
    assert report["w1_accuracy"]        > 0.6
    assert report["w2_acc_mlp"]         > 0.6
    assert report["w2_acc_lif"]         > 0.6
    assert report["w2_polymorphie_gap"] < 0.05
    assert report["w3_gain_over_baseline"] >= 0.10
    assert report["w4_forgetting"]      < 0.20
    assert report["all_passed"]         is True
