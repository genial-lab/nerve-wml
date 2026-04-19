"""Gate-Dream — end-to-end consolidation round-trip with MockConsolidator.

Collect → encode → consolidate (mock, zero-delta) → apply → verify
transducer logits only shift by the expected alpha-scaled delta.

Plan 7 Task 6.
"""
import numpy as np
import torch

from bridge.dream_bridge import DreamBridge
from bridge.eps_replay import load_eps_replay, save_eps_replay
from bridge.mock_consolidator import MockConsolidator
from bridge.sim_nerve_adapter import SimNerveAdapter


def test_gate_dream_zero_delta_leaves_transducers_unchanged():
    """With MockConsolidator (zero delta), transducers are identical after apply."""
    bridge = DreamBridge(enabled=True)
    nerve = SimNerveAdapter(n_wmls=4, k=2, seed=0)

    key = next(iter(nerve._transducers.keys()))
    snap = nerve._transducers[key].logits.data.clone()

    trace = bridge.collect_eps_trace(nerve, duration_ticks=50)
    encoded = bridge.to_dream_input(trace)
    delta = MockConsolidator.consolidate(
        encoded, n_transducers=len(nerve._transducers), alphabet_size=64,
    )
    bridge.apply_consolidation_output(nerve, delta, alpha=0.1)

    assert torch.allclose(nerve._transducers[key].logits.data, snap)


def test_gate_dream_env_gate_disabled_is_fully_noop():
    """DREAM_CONSOLIDATION_ENABLED=0 → every step is a no-op."""
    bridge = DreamBridge(enabled=False)
    nerve = SimNerveAdapter(n_wmls=4, k=2, seed=0)

    key = next(iter(nerve._transducers.keys()))
    snap = nerve._transducers[key].logits.data.clone()

    trace = bridge.collect_eps_trace(nerve, duration_ticks=50)
    assert trace == []

    encoded = bridge.to_dream_input(trace)
    assert encoded.shape == (0, 4)

    delta = np.ones((12, 64, 64), dtype=np.float32)
    bridge.apply_consolidation_output(nerve, delta, alpha=0.1)

    assert torch.allclose(nerve._transducers[key].logits.data, snap)


def test_gate_dream_trace_determinism(tmp_path):
    """Same seeds (global + nerve) → same trace.

    Transducer.forward uses gumbel_softmax which consumes global RNG,
    so we seed torch.manual_seed before each collect to get bit-stable
    output across two independent runs.
    """
    bridge = DreamBridge(enabled=True)

    torch.manual_seed(0)
    nerve_a = SimNerveAdapter(n_wmls=4, k=2, seed=0)
    torch.manual_seed(0)  # reset before collect for determinism
    trace_a = bridge.collect_eps_trace(nerve_a, duration_ticks=50)
    encoded_a = bridge.to_dream_input(trace_a)

    torch.manual_seed(0)
    nerve_b = SimNerveAdapter(n_wmls=4, k=2, seed=0)
    torch.manual_seed(0)
    trace_b = bridge.collect_eps_trace(nerve_b, duration_ticks=50)
    encoded_b = bridge.to_dream_input(trace_b)

    np.testing.assert_array_equal(encoded_a, encoded_b)

    # Also test save/load preserves the trace.
    save_eps_replay(encoded_a, {"schema_version": "v0"}, tmp_path)
    reloaded, meta = load_eps_replay(tmp_path)
    np.testing.assert_array_equal(reloaded, encoded_a)
    assert meta["schema_version"] == "v0"
