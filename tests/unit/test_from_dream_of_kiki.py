"""Tests for the dreamOfkiki axiom bridge scaffold (issue #6).

The runtime wiring is gated on upstream dream-of-kiki publishing a
versioned ``axioms`` public API. These tests pin the **input contract**
so that when the gate lifts, the day-1 wiring can swap in without
breaking any caller.
"""
from __future__ import annotations

import pytest

from nerve_core.from_dream_of_kiki import (
    REQUIRED_AXIOMS,
    DreamOfKikiAxiomError,
    from_dream_of_kiki,
    to_dream_of_kiki,
)


# ---------------------------------------------------------------------------
# Canonical constants
# ---------------------------------------------------------------------------


def test_required_axioms_is_canonical_dr0_through_dr4():
    """OSF pre-registration locks 5 axioms — order + names matter."""
    assert REQUIRED_AXIOMS == ("DR-0", "DR-1", "DR-2", "DR-3", "DR-4")


# ---------------------------------------------------------------------------
# Spec validation (load-bearing public contract)
# ---------------------------------------------------------------------------


def _complete_spec() -> dict[str, dict]:
    """Minimal spec with all 5 required keys present."""
    return {key: {"op": key.lower()} for key in REQUIRED_AXIOMS}


def test_complete_spec_passes_validation_then_raises_not_implemented():
    """Validation green, but actual wiring is gated upstream."""
    with pytest.raises(NotImplementedError, match="dream-of-kiki"):
        from_dream_of_kiki(
            axioms=_complete_spec(),
            modalities=("audio", "vision"),
            d_z=32,
        )


def test_missing_single_axiom_raises_clear_error():
    spec = _complete_spec()
    del spec["DR-2"]
    with pytest.raises(DreamOfKikiAxiomError, match="DR-2"):
        from_dream_of_kiki(spec, modalities=("audio",))


def test_missing_multiple_axioms_lists_all_in_error():
    spec = _complete_spec()
    del spec["DR-0"]
    del spec["DR-4"]
    with pytest.raises(DreamOfKikiAxiomError) as excinfo:
        from_dream_of_kiki(spec, modalities=("audio",))
    msg = str(excinfo.value)
    assert "DR-0" in msg and "DR-4" in msg


def test_non_mapping_axioms_raises_clear_type_error():
    with pytest.raises(DreamOfKikiAxiomError, match="Mapping"):
        from_dream_of_kiki(["DR-0", "DR-1"], modalities=("audio",))  # type: ignore[arg-type]


def test_empty_modalities_raises():
    with pytest.raises(DreamOfKikiAxiomError, match="modality"):
        from_dream_of_kiki(_complete_spec(), modalities=())


def test_non_string_modalities_raises():
    with pytest.raises(DreamOfKikiAxiomError, match="strings"):
        from_dream_of_kiki(_complete_spec(), modalities=("audio", 42))  # type: ignore[arg-type]


def test_empty_string_modality_raises():
    with pytest.raises(DreamOfKikiAxiomError, match="strings"):
        from_dream_of_kiki(_complete_spec(), modalities=("audio", ""))


# ---------------------------------------------------------------------------
# Round-trip dual — same upstream gate
# ---------------------------------------------------------------------------


def test_to_dream_of_kiki_raises_not_implemented_with_upstream_pointer():
    with pytest.raises(NotImplementedError, match="from_dream_of_kiki"):
        to_dream_of_kiki(object())


# ---------------------------------------------------------------------------
# Default args sanity (catches accidental signature drift)
# ---------------------------------------------------------------------------


def test_default_d_z_is_canonical_alphabet_size_32():
    """The default d_z must match GammaThetaMultiplexer's alphabet size.

    If a future change desyncs them, the dream-of-kiki wiring will
    misalign with the multiplexer expectation; this test catches that
    upstream of any runtime breakage.
    """
    import inspect

    sig = inspect.signature(from_dream_of_kiki)
    assert sig.parameters["d_z"].default == 32
