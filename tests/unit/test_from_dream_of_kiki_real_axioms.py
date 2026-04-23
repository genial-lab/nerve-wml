"""Integration tests using real kiki_oniric.axioms.DR0..DR4 instances.

Skipped when the `axioms` extras group is not installed (D4: optional-
real CI gate). When the extra IS installed, these tests validate that
the bridge accepts real Axiom instances (not just plain dicts) and
that round-trip preserves identity.

Run: uv sync --extra axioms && uv run pytest tests/unit/test_from_dream_of_kiki_real_axioms.py
"""
from __future__ import annotations

import pytest

from nerve_core.from_dream_of_kiki import (
    DreamOfKikiAxiomError,
    DreamOfKikiNerve,
    from_dream_of_kiki,
    to_dream_of_kiki,
)

# The entire module is skipped when kiki_oniric.axioms cannot be imported.
up = pytest.importorskip(
    "kiki_oniric.axioms",
    reason="install nerve-wml[axioms] to run upstream-real tests",
)


def _real_spec() -> dict:
    """Spec built from the 5 canonical upstream Axiom instances."""
    return {
        "DR-0": up.DR0,
        "DR-1": up.DR1,
        "DR-2": up.DR2,
        "DR-3": up.DR3,
        "DR-4": up.DR4,
    }


def test_real_axioms_build_a_valid_nerve():
    nerve = from_dream_of_kiki(
        _real_spec(), modalities=("audio", "vision"), d_z=32,
    )
    assert isinstance(nerve, DreamOfKikiNerve)


def test_real_axioms_round_trip_preserves_instances():
    spec = _real_spec()
    nerve = from_dream_of_kiki(
        spec, modalities=("audio", "vision"), d_z=32,
    )
    recovered = to_dream_of_kiki(nerve)
    # The recovered axioms dict must carry the real Axiom instances.
    assert recovered["axioms"]["DR-0"] is up.DR0
    assert recovered["axioms"]["DR-2"] is up.DR2


def test_axioms_dr2_has_predicate_callable():
    """DR-2 at C-v0.8.0+PARTIAL carries _dr2_precondition as predicate."""
    assert up.DR2.predicate is not None
    assert callable(up.DR2.predicate)


def test_axioms_dr0_has_no_predicate():
    """DR-0 at C-v0.8.0+PARTIAL has predicate=None (runtime invariant)."""
    assert up.DR0.predicate is None


def test_mixed_axiom_and_dict_spec_is_accepted():
    """Bridge is permissive: each value independently may be Axiom or dict."""
    spec = _real_spec()
    spec["DR-4"] = {"seed": 7}  # Override with plain dict
    nerve = from_dream_of_kiki(
        spec, modalities=("audio", "vision"), d_z=32,
    )
    # DR-4 dict with seed=7 must be used verbatim.
    assert nerve._bridge_seed == 7


def test_missing_axiom_still_raises_with_real_instances():
    """Validation runs regardless of value type — a missing key fails."""
    spec = _real_spec()
    del spec["DR-3"]
    with pytest.raises(DreamOfKikiAxiomError, match="DR-3"):
        from_dream_of_kiki(spec, modalities=("a", "b"))


# ---------------------------------------------------------------------------
# DR-2 predicate consumption (v1.8.0)
# ---------------------------------------------------------------------------


def test_dr2_predicate_accepts_canonical_order():
    """Canonical order (replay, downscale, restructure, recombine)
    satisfies DR-2 precondition — _dr2_precondition returns True."""
    from kiki_oniric.dream.episode import Operation

    spec = _real_spec()
    spec["operation_order"] = (
        Operation.REPLAY,
        Operation.DOWNSCALE,
        Operation.RESTRUCTURE,
        Operation.RECOMBINE,
    )
    nerve = from_dream_of_kiki(
        spec, modalities=("a", "b"), d_z=32,
    )
    assert isinstance(nerve, DreamOfKikiNerve)


def test_dr2_predicate_rejects_restructure_before_replay():
    """Order with RESTRUCTURE before REPLAY violates the weakened DR-2
    precondition — the factory must raise DreamOfKikiAxiomError."""
    from kiki_oniric.dream.episode import Operation

    spec = _real_spec()
    spec["operation_order"] = (
        Operation.RESTRUCTURE,
        Operation.REPLAY,
        Operation.DOWNSCALE,
        Operation.RECOMBINE,
    )
    with pytest.raises(DreamOfKikiAxiomError, match="DR-2 precondition"):
        from_dream_of_kiki(spec, modalities=("a", "b"), d_z=32)


def test_dr2_predicate_skipped_when_order_absent():
    """No operation_order key → no predicate call → no error even if
    DR-2 has a predicate. The check is opt-in per spec."""
    spec = _real_spec()  # No operation_order key.
    nerve = from_dream_of_kiki(
        spec, modalities=("a", "b"), d_z=32,
    )
    assert isinstance(nerve, DreamOfKikiNerve)


def test_dr2_dict_value_skips_predicate_check():
    """Plain dict DR-2 (no predicate attribute) never triggers the check."""
    spec = _real_spec()
    spec["DR-2"] = {"op": "dr-2"}  # Plain dict, no predicate.
    spec["operation_order"] = ("this", "would", "fail", "if", "checked")
    nerve = from_dream_of_kiki(
        spec, modalities=("a", "b"), d_z=32,
    )
    assert isinstance(nerve, DreamOfKikiNerve)
