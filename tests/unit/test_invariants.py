import pytest
from nerve_core.invariants import (
    assert_n1_silence_legal,
    assert_n3_role_phase_consistent,
    assert_n4_routing_weight_valid,
)
from nerve_core.neuroletter import Neuroletter, Phase, Role


def test_n1_empty_listen_is_legal():
    assert_n1_silence_legal([])  # does not raise


def test_n3_prediction_must_be_gamma_in_strict_mode():
    ok = Neuroletter(5, Role.PREDICTION, Phase.GAMMA, 1, 2, 0.0)
    assert_n3_role_phase_consistent(ok, strict=True)

    bad = Neuroletter(5, Role.PREDICTION, Phase.THETA, 1, 2, 0.0)
    with pytest.raises(AssertionError, match="N-3"):
        assert_n3_role_phase_consistent(bad, strict=True)


def test_n3_error_must_be_theta_in_strict_mode():
    bad = Neuroletter(5, Role.ERROR, Phase.GAMMA, 1, 2, 0.0)
    with pytest.raises(AssertionError, match="N-3"):
        assert_n3_role_phase_consistent(bad, strict=True)


def test_n4_routing_weight_range():
    assert_n4_routing_weight_valid(0.0, pruned=True)
    assert_n4_routing_weight_valid(1.0, pruned=True)
    assert_n4_routing_weight_valid(0.42, pruned=False)  # continuous during training
    with pytest.raises(AssertionError, match="N-4"):
        assert_n4_routing_weight_valid(0.5, pruned=True)  # must be {0, 1} once pruned
