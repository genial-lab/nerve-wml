"""Tests for bridge.dream_bridge — collect, encode, apply."""
import numpy as np
import torch

from bridge.dream_bridge import DreamBridge
from bridge.sim_nerve_adapter import SimNerveAdapter


def test_bridge_disabled_by_default():
    """Without DREAM_CONSOLIDATION_ENABLED, all methods no-op."""
    bridge = DreamBridge()
    assert bridge.enabled is False


def test_collect_trace_empty_when_disabled():
    bridge = DreamBridge(enabled=False)
    nerve = SimNerveAdapter(n_wmls=4, k=2, seed=0)
    trace = bridge.collect_eps_trace(nerve, duration_ticks=10)
    assert trace == []


def test_collect_trace_returns_eps_letters_when_enabled():
    bridge = DreamBridge(enabled=True)
    nerve = SimNerveAdapter(n_wmls=4, k=2, seed=0)
    trace = bridge.collect_eps_trace(nerve, duration_ticks=50)
    # At least one ε delivery over 50 ticks.
    assert len(trace) >= 1
    from nerve_core.neuroletter import Role
    for letter in trace:
        assert letter.role is Role.ERROR


def test_to_dream_input_schema():
    bridge = DreamBridge(enabled=True)
    nerve = SimNerveAdapter(n_wmls=4, k=2, seed=0)
    trace = bridge.collect_eps_trace(nerve, duration_ticks=30)
    encoded = bridge.to_dream_input(trace)
    assert encoded.dtype == np.int32
    assert encoded.ndim == 2
    assert encoded.shape[1] == 4


def test_to_dream_input_empty_on_empty_trace():
    bridge = DreamBridge(enabled=True)
    encoded = bridge.to_dream_input([])
    assert encoded.shape == (0, 4)


def test_apply_zero_delta_leaves_transducers_unchanged():
    bridge = DreamBridge(enabled=True)
    nerve = SimNerveAdapter(n_wmls=4, k=2, seed=0)
    key = next(iter(nerve._transducers.keys()))
    snap = nerve._transducers[key].logits.data.clone()

    delta = np.zeros((12, 64, 64), dtype=np.float32)
    bridge.apply_consolidation_output(nerve, delta, alpha=0.1)

    assert torch.allclose(nerve._transducers[key].logits.data, snap)


def test_apply_nonzero_delta_mutates_transducers():
    bridge = DreamBridge(enabled=True)
    nerve = SimNerveAdapter(n_wmls=4, k=2, seed=0)
    key = next(iter(nerve._transducers.keys()))
    snap = nerve._transducers[key].logits.data.clone()

    delta = np.ones((12, 64, 64), dtype=np.float32)
    bridge.apply_consolidation_output(nerve, delta, alpha=0.1)

    # First transducer logits should have shifted by 0.1.
    diff = (nerve._transducers[key].logits.data - snap).mean().item()
    assert abs(diff - 0.1) < 1e-4


def test_apply_delta_no_op_when_disabled():
    bridge = DreamBridge(enabled=False)
    nerve = SimNerveAdapter(n_wmls=4, k=2, seed=0)
    key = next(iter(nerve._transducers.keys()))
    snap = nerve._transducers[key].logits.data.clone()

    delta = np.ones((12, 64, 64), dtype=np.float32)
    bridge.apply_consolidation_output(nerve, delta, alpha=0.1)

    assert torch.allclose(nerve._transducers[key].logits.data, snap)
