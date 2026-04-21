import numpy as np
import pytest

from scripts.baseline_frozen_encoder import train_frozen_baseline


def test_frozen_baseline_trains_two_substrates() -> None:
    result = train_frozen_baseline(seed=0, steps=100, d_hidden=16)
    assert "acc_mlp" in result
    assert "acc_lif" in result
    assert "codes_mlp" in result
    assert "codes_lif" in result
    assert result["codes_mlp"].shape == result["codes_lif"].shape
    assert result["codes_mlp"].dtype == np.int64


def test_frozen_baseline_encoder_is_frozen() -> None:
    """Verify that encoder parameters don't change during training."""
    import copy
    result = train_frozen_baseline(
        seed=0, steps=100, d_hidden=16, return_encoder=True,
    )
    encoder_before = copy.deepcopy(result["encoder_initial"])
    encoder_after = result["encoder_final"]
    for p_before, p_after in zip(
        encoder_before.parameters(), encoder_after.parameters(),
    ):
        assert (p_before == p_after).all(), "encoder changed during training"


def test_frozen_baseline_distinct_encoders_different_codes() -> None:
    """Distinct-encoders mode should produce different code distributions.

    When encoder_a != encoder_b (review beta stronger control), the two
    heads see different feature spaces. Their argmax codes should not
    be identical (if they were, it would mean the heads ignored the
    encoder difference, which is impossible with finite capacity heads
    + random features).
    """
    result = train_frozen_baseline(
        seed=0, steps=100, d_hidden=16, distinct_encoders=True,
    )
    same = (result["codes_mlp"] == result["codes_lif"]).sum()
    total = len(result["codes_mlp"])
    assert same < total, (
        f"distinct encoders produced identical codes: {same}/{total}"
    )
