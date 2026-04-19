"""Gate-Adaptive — shrink + grow cycles both succeed."""
import torch

from scripts.adaptive_pilot import (
    run_adaptive_cycle,
    run_adaptive_grow_cycle,
    run_gate_adaptive,
)


def test_adaptive_shrink_retires_at_least_one_code():
    torch.manual_seed(0)
    rep = run_adaptive_cycle(size=16, dim=8, warmup_steps=500, post_steps=100)
    assert rep["codes_retired"] >= 1
    assert rep["size_after_shrink"] < rep["size_before"]


def test_adaptive_grow_adds_top_k_codes():
    torch.manual_seed(0)
    rep = run_adaptive_grow_cycle(size=16, dim=8, warmup_steps=500, top_k=4)
    assert rep["codes_added"] == 4
    assert rep["size_after_grow"] == rep["size_before_grow"] + 4


def test_gate_adaptive_all_passed():
    torch.manual_seed(0)
    rep = run_gate_adaptive()
    assert rep["shrink_passed"] is True
    assert rep["grow_passed"] is True
    assert rep["all_passed"] is True
