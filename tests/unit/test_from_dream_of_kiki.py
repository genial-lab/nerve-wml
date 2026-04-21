"""Tests for the dreamOfkiki axiom bridge (issue #6, unblocked 2026-04-21).

Upstream gate lifted : dream-of-kiki C-v0.8.0+PARTIAL ships
``kiki_oniric.axioms``. These tests pin the **input contract** (spec
validation, round-trip shape) and the **wiring invariants** (DR-4 seed
determinism, DR-3 gating extraction, per-edge transducer fabric).
"""
from __future__ import annotations

import pytest

from nerve_core.from_dream_of_kiki import (
    REQUIRED_AXIOMS,
    DreamOfKikiAxiomError,
    DreamOfKikiNerve,
    from_dream_of_kiki,
    to_dream_of_kiki,
)
from nerve_core.protocols import Nerve
from track_p.transducer import Transducer, TransducerGating


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


def test_missing_single_axiom_raises_clear_error():
    spec = _complete_spec()
    del spec["DR-2"]
    with pytest.raises(DreamOfKikiAxiomError, match="DR-2"):
        from_dream_of_kiki(spec, modalities=("audio", "vision"))


def test_missing_multiple_axioms_lists_all_in_error():
    spec = _complete_spec()
    del spec["DR-0"]
    del spec["DR-4"]
    with pytest.raises(DreamOfKikiAxiomError) as excinfo:
        from_dream_of_kiki(spec, modalities=("audio", "vision"))
    msg = str(excinfo.value)
    assert "DR-0" in msg and "DR-4" in msg


def test_non_mapping_axioms_raises_clear_type_error():
    with pytest.raises(DreamOfKikiAxiomError, match="Mapping"):
        from_dream_of_kiki(
            ["DR-0", "DR-1"],  # type: ignore[arg-type]
            modalities=("audio", "vision"),
        )


def test_empty_modalities_raises():
    with pytest.raises(DreamOfKikiAxiomError, match="modality"):
        from_dream_of_kiki(_complete_spec(), modalities=())


def test_non_string_modalities_raises():
    with pytest.raises(DreamOfKikiAxiomError, match="strings"):
        from_dream_of_kiki(
            _complete_spec(),
            modalities=("audio", 42),  # type: ignore[arg-type]
        )


def test_empty_string_modality_raises():
    with pytest.raises(DreamOfKikiAxiomError, match="strings"):
        from_dream_of_kiki(
            _complete_spec(), modalities=("audio", ""),
        )


def test_single_modality_raises_clear_error():
    """A nerve needs >=2 modalities — single-modality degenerates."""
    with pytest.raises(DreamOfKikiAxiomError, match="two modalities"):
        from_dream_of_kiki(_complete_spec(), modalities=("audio",))


# ---------------------------------------------------------------------------
# Concrete wiring (upstream gate lifted 2026-04-21)
# ---------------------------------------------------------------------------


def test_complete_spec_produces_dream_of_kiki_nerve():
    """Validation green + wiring live → returns a DreamOfKikiNerve."""
    nerve = from_dream_of_kiki(
        axioms=_complete_spec(),
        modalities=("audio", "vision"),
        d_z=32,
    )
    assert isinstance(nerve, DreamOfKikiNerve)


def test_returned_nerve_satisfies_nerve_protocol():
    """Bridge output remains a valid Nerve per the protocol contract."""
    nerve = from_dream_of_kiki(
        _complete_spec(), modalities=("audio", "vision"), d_z=32,
    )
    assert isinstance(nerve, Nerve)


def test_nerve_carries_modalities_and_dz():
    nerve = from_dream_of_kiki(
        _complete_spec(),
        modalities=("audio", "vision", "tactile"),
        d_z=48,
    )
    assert nerve.modalities == ("audio", "vision", "tactile")
    assert nerve.d_z == 48
    assert nerve.n_wmls == 3


def test_nerve_carries_axiom_spec_as_dict():
    spec = _complete_spec()
    nerve = from_dream_of_kiki(
        spec, modalities=("audio", "vision"), d_z=32,
    )
    assert nerve.axioms == spec


def test_transducers_instantiated_per_directed_edge():
    nerve = from_dream_of_kiki(
        _complete_spec(), modalities=("a", "b", "c"), d_z=16,
    )
    # 3 × 3 minus diagonal = 6 directed edges.
    assert len(nerve.transducers) == 6
    for (src, dst), t in nerve.transducers.items():
        assert src != dst
        assert isinstance(t, Transducer)
        assert t.alphabet_size == 16


def test_transducers_count_matches_directed_edge_count():
    """n_wmls × (n_wmls - 1) directed edges for n_wmls ≥ 2."""
    for n in (2, 3, 4):
        nerve = from_dream_of_kiki(
            _complete_spec(),
            modalities=tuple(f"m{i}" for i in range(n)),
            d_z=32,
        )
        assert len(nerve.transducers) == n * (n - 1)


# ---------------------------------------------------------------------------
# DR-4 bit-exact R1 — seed determinism
# ---------------------------------------------------------------------------


def test_dr4_explicit_seed_used_verbatim():
    spec = _complete_spec()
    spec["DR-4"] = {"seed": 12345}
    nerve = from_dream_of_kiki(
        spec, modalities=("audio", "vision"), d_z=32,
    )
    assert nerve._bridge_seed == 12345


def test_dr4_missing_seed_falls_back_to_hash_of_keys():
    spec1 = _complete_spec()
    spec2 = _complete_spec()
    nerve1 = from_dream_of_kiki(
        spec1, modalities=("audio", "vision"), d_z=32,
    )
    nerve2 = from_dream_of_kiki(
        spec2, modalities=("audio", "vision"), d_z=32,
    )
    # Same spec keys → same fallback seed (R1 contract).
    assert nerve1._bridge_seed == nerve2._bridge_seed


# ---------------------------------------------------------------------------
# DR-3 gating mode extraction
# ---------------------------------------------------------------------------


def test_dr3_default_gating_is_hard():
    nerve = from_dream_of_kiki(
        _complete_spec(), modalities=("a", "b"), d_z=32,
    )
    any_transducer = next(iter(nerve.transducers.values()))
    assert any_transducer.gating is TransducerGating.HARD


def test_dr3_gumbel_softmax_gating_propagates():
    spec = _complete_spec()
    spec["DR-3"] = {"gating": "gumbel_softmax"}
    nerve = from_dream_of_kiki(spec, modalities=("a", "b"), d_z=32)
    any_transducer = next(iter(nerve.transducers.values()))
    assert any_transducer.gating is TransducerGating.GUMBEL_SOFTMAX


def test_dr3_invalid_gating_falls_back_to_hard():
    spec = _complete_spec()
    spec["DR-3"] = {"gating": "nonsense"}
    nerve = from_dream_of_kiki(spec, modalities=("a", "b"), d_z=32)
    any_transducer = next(iter(nerve.transducers.values()))
    assert any_transducer.gating is TransducerGating.HARD


# ---------------------------------------------------------------------------
# Round-trip dual
# ---------------------------------------------------------------------------


def test_to_dream_of_kiki_reconstructs_spec_exactly():
    spec = _complete_spec()
    spec["DR-4"] = {"seed": 99}
    nerve = from_dream_of_kiki(
        spec, modalities=("audio", "vision"), d_z=48,
    )
    recovered = to_dream_of_kiki(nerve)
    assert recovered == {
        "axioms": spec,
        "modalities": ("audio", "vision"),
        "d_z": 48,
    }


def test_to_dream_of_kiki_rejects_non_dream_of_kiki_nerve():
    with pytest.raises(TypeError, match="DreamOfKikiNerve"):
        to_dream_of_kiki(object())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Default args sanity (catches accidental signature drift)
# ---------------------------------------------------------------------------


def test_default_d_z_is_canonical_alphabet_size_32():
    """The default d_z must match GammaThetaMultiplexer's alphabet size."""
    import inspect

    sig = inspect.signature(from_dream_of_kiki)
    assert sig.parameters["d_z"].default == 32
