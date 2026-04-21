"""dreamOfkiki axiom bridge — wires the DR-0..DR-4 formal layer to nerve_wml.

Status: **SCAFFOLD only**. Concrete wiring is gated on dream-of-kiki
publishing a versioned, importable ``axioms`` public API (see issue
hypneum-lab/nerve-wml#6 dependencies). This module ships:

1. The ``REQUIRED_AXIOMS`` constant — the canonical DR-0..DR-4 keys.
2. ``DreamOfKikiAxiomError`` — raised when a spec is malformed.
3. ``from_dream_of_kiki(axioms, modalities, d_z)`` — validates the spec
   shape *now* and raises ``NotImplementedError`` with a clear upstream
   pointer for the actual instantiation.
4. ``to_dream_of_kiki(nerve)`` — round-trip dual, same gating.

Pinning the contract today (validation + signature) lets downstream
consumers and the dream-of-kiki repo align on the protocol before any
runtime coupling, so the eventual concrete wiring is a one-shot diff.

See ``docs/integration-dream-of-kiki.md`` for the mapping table.
"""
from __future__ import annotations

from typing import Any, Mapping

#: Canonical 5 axioms from dreamOfkiki Paper 1 v0.2 (DR-0 through DR-4).
#: Order matters — downstream tests assert this exact tuple.
REQUIRED_AXIOMS: tuple[str, ...] = ("DR-0", "DR-1", "DR-2", "DR-3", "DR-4")

#: Pointer to the upstream issue gating the concrete wiring.
_UPSTREAM_GATE_URL = "https://github.com/hypneum-lab/nerve-wml/issues/6"

#: Pointer to the design doc explaining the DR-X → nerve-wml mapping.
_DESIGN_DOC_PATH = "docs/integration-dream-of-kiki.md"


class DreamOfKikiAxiomError(ValueError):
    """Raised when a dreamOfkiki axiom spec is missing required keys
    or contains an unsupported operation shape."""


def _validate_spec(axioms: Mapping[str, Any], modalities: tuple[str, ...]) -> None:
    """Shape-check a dreamOfkiki axiom spec without triggering wiring."""
    if not isinstance(axioms, Mapping):
        raise DreamOfKikiAxiomError(
            f"axioms must be a Mapping, got {type(axioms).__name__}",
        )
    missing = [k for k in REQUIRED_AXIOMS if k not in axioms]
    if missing:
        raise DreamOfKikiAxiomError(
            f"Missing required dreamOfkiki axioms: {missing}. "
            f"Expected all of {list(REQUIRED_AXIOMS)}.",
        )
    if not modalities:
        raise DreamOfKikiAxiomError(
            "at least one modality must be supplied (got empty tuple)",
        )
    if not all(isinstance(m, str) and m for m in modalities):
        raise DreamOfKikiAxiomError(
            f"all modalities must be non-empty strings, got {list(modalities)}",
        )


def from_dream_of_kiki(
    axioms: Mapping[str, Any],
    modalities: tuple[str, ...] = (),
    d_z: int = 32,
) -> Any:  # noqa: ARG001 — d_z reserved for impl
    """Instantiate a nerve-wml ``Nerve`` from a dreamOfkiki axiom spec.

    **Status: SCAFFOLD.** The spec validation below is the load-bearing
    public contract — it will not change once the concrete wiring lands.
    The actual ``Nerve`` instantiation is gated on dream-of-kiki publishing
    a versioned ``axioms`` public API; until then this function raises
    ``NotImplementedError`` with a pointer to the upstream issue.

    Parameters
    ----------
    axioms
        Mapping with the 5 required keys ``DR-0`` through ``DR-4``
        (case-sensitive). Values are per-axiom operation specs whose
        shape is defined upstream by dream-of-kiki.
    modalities
        Tuple of modality names (e.g. ``("audio", "vision", "tactile")``).
        Must be non-empty; each entry must be a non-empty string.
    d_z
        Latent dimensionality. Default 32 matches the canonical
        ``GammaThetaMultiplexer`` alphabet. Reserved for the concrete
        wiring; ignored today.

    Returns
    -------
    Nerve
        Once the upstream gate lifts.

    Raises
    ------
    DreamOfKikiAxiomError
        If ``axioms`` is missing required keys, is not a Mapping, or if
        ``modalities`` is empty / contains non-string entries.
    NotImplementedError
        Always, until dream-of-kiki ships the public ``axioms`` API. The
        message includes the upstream issue URL and the design doc path.

    See Also
    --------
    to_dream_of_kiki : round-trip dual.
    docs/integration-dream-of-kiki.md : DR-X → nerve-wml mapping table.
    """
    _validate_spec(axioms, modalities)
    raise NotImplementedError(
        "Concrete wiring is gated on dream-of-kiki publishing a versioned "
        "`axioms` public API. The spec validation above is the load-bearing "
        f"contract; see {_UPSTREAM_GATE_URL} for the dependency tracker and "
        f"{_DESIGN_DOC_PATH} for the DR-X → nerve-wml mapping.",
    )


def to_dream_of_kiki(nerve: Any) -> dict[str, Any]:  # noqa: ARG001
    """Round-trip dual of :func:`from_dream_of_kiki`.

    **Status: SCAFFOLD.** Gated on the same upstream dependency. Will
    return a ``dict[str, Any]`` axiom spec equivalent to the one passed
    to ``from_dream_of_kiki`` (modulo non-axiomatic Nerve internals).

    Raises
    ------
    NotImplementedError
        Always, until upstream gate lifts.
    """
    raise NotImplementedError(
        "Round-trip dual of from_dream_of_kiki — same upstream dependency. "
        f"See {_UPSTREAM_GATE_URL}.",
    )
