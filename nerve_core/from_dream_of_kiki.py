"""dreamOfkiki axiom bridge — wires the DR-0..DR-4 formal layer to nerve_wml.

Upstream gate lifted 2026-04-21 : dream-of-kiki C-v0.8.0+PARTIAL published
the ``kiki_oniric.axioms`` public API (DR-0..DR-4 as frozen ``Axiom``
instances). The factories below are now live and build a concrete
:class:`DreamOfKikiNerve` — a ``SimNerve`` that carries its originating
axiom spec so the round-trip dual ``to_dream_of_kiki`` is exact.

Wiring per ``docs/integration-dream-of-kiki.md`` :

- DR-0 (replay) → ``SimNerve`` event buffer (default ``n_wmls = |modalities|``)
- DR-1 (downscale) → per-edge :class:`Transducer` entropy regulariser
- DR-2 (restructure) → :func:`bridge.transducer_resize.resize_transducer` (not wired at construction ; invoked by downstream consolidation)
- DR-3 (recombine) → :class:`Transducer` gating mode read from spec
- DR-4 (bit-exact R1) → ``SimNerve(seed=...)`` + per-transducer deterministic init

The bridge remains permissive about spec value types : values may be
``kiki_oniric.axioms.Axiom`` instances, plain dicts, or anything else.
The factory only reads optional hints (e.g. ``DR-3 → "gating"``,
``DR-4 → "seed"``) ; missing hints fall back to deterministic defaults.

See ``docs/integration-dream-of-kiki.md`` for the mapping table.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from track_p.sim_nerve import SimNerve
from track_p.transducer import Transducer, TransducerGating

#: Canonical 5 axioms from dreamOfkiki Paper 1 v0.2 (DR-0 through DR-4).
#: Order matters — downstream tests assert this exact tuple.
REQUIRED_AXIOMS: tuple[str, ...] = ("DR-0", "DR-1", "DR-2", "DR-3", "DR-4")

#: Pointer to the upstream issue that gated the concrete wiring.
_UPSTREAM_GATE_URL = "https://github.com/hypneum-lab/nerve-wml/issues/6"

#: Pointer to the design doc explaining the DR-X → nerve-wml mapping.
_DESIGN_DOC_PATH = "docs/integration-dream-of-kiki.md"

#: Default top-K sparse routing cap.
_DEFAULT_K_CAP: int = 4


class DreamOfKikiAxiomError(ValueError):
    """Raised when a dreamOfkiki axiom spec is missing required keys
    or contains an unsupported operation shape."""


class DreamOfKikiNerve(SimNerve):
    """SimNerve carrying its originating dreamOfkiki axiom spec.

    Extends :class:`track_p.sim_nerve.SimNerve` with four bookkeeping
    attributes so that :func:`to_dream_of_kiki` can reconstruct the
    original spec deterministically. The underlying nerve protocol
    (send / listen / tick / routing_weight) is inherited unchanged —
    :class:`DreamOfKikiNerve` remains a valid ``Nerve`` per the
    :mod:`nerve_core.protocols` contract.
    """

    def __init__(
        self,
        *,
        axioms: Mapping[str, Any],
        modalities: tuple[str, ...],
        d_z: int,
        transducers: Mapping[tuple[int, int], Transducer],
        n_wmls: int,
        k: int,
        seed: int,
    ) -> None:
        super().__init__(n_wmls=n_wmls, k=k, seed=seed)
        self.axioms: dict[str, Any] = dict(axioms)
        self.modalities: tuple[str, ...] = tuple(modalities)
        self.d_z: int = d_z
        self.transducers: dict[tuple[int, int], Transducer] = dict(transducers)
        self._bridge_seed: int = seed


def _validate_spec(
    axioms: Mapping[str, Any], modalities: tuple[str, ...]
) -> None:
    """Shape-check a dreamOfkiki axiom spec. Raises on any violation."""
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
    if len(modalities) < 2:
        raise DreamOfKikiAxiomError(
            f"at least two modalities are required (got {len(modalities)}); "
            "a nerve models inter-WML communication and degenerates with a "
            "single modality. Supply a second modality or use a plain WML "
            "instead of a full nerve.",
        )


def _extract_seed(axioms: Mapping[str, Any]) -> int:
    """Derive a deterministic seed from the DR-4 spec value.

    Accepts:
      - a Mapping with an ``int`` "seed" key → use it verbatim
      - anything else → fall back to a stable hash of the spec keys

    The fallback is intentional : even without an explicit seed, two
    calls with the same axiom spec produce the same nerve (R1 contract).
    """
    dr4 = axioms.get("DR-4")
    if isinstance(dr4, Mapping):
        seed = dr4.get("seed")
        if isinstance(seed, int):
            return seed
    return abs(hash(tuple(sorted(axioms.keys())))) % (2**31)


def _extract_gating(axioms: Mapping[str, Any]) -> TransducerGating:
    """Derive the Transducer gating mode from the DR-3 spec value.

    Accepts:
      - a Mapping with a "gating" key ∈ {"hard", "gumbel_softmax"} → use it
      - anything else → default :attr:`TransducerGating.HARD`
    """
    dr3 = axioms.get("DR-3")
    if isinstance(dr3, Mapping):
        gating = dr3.get("gating", "hard")
        if gating in {"hard", "gumbel_softmax"}:
            return TransducerGating(gating)
    return TransducerGating.HARD


def from_dream_of_kiki(
    axioms: Mapping[str, Any],
    modalities: tuple[str, ...] = (),
    d_z: int = 32,
) -> DreamOfKikiNerve:
    """Instantiate a :class:`DreamOfKikiNerve` from a dreamOfkiki axiom spec.

    Wiring (see module docstring for the mapping table):

    - ``n_wmls = len(modalities)``
    - ``k = max(1, min(n_wmls - 1, 4))`` (top-K sparse routing)
    - seed → ``axioms["DR-4"]["seed"]`` if present and ``int``, else a
      stable hash of the spec keys
    - per-edge ``Transducer(alphabet_size=d_z, gating=...)`` for every
      ``(src, dst)`` pair with ``src != dst`` (empty if ``n_wmls < 2``)
    - gating → ``axioms["DR-3"]["gating"]`` if ∈ ``{"hard", "gumbel_softmax"}``,
      else :attr:`TransducerGating.HARD`

    Parameters
    ----------
    axioms
        Mapping with the 5 required keys ``DR-0`` through ``DR-4``
        (case-sensitive). Values may be ``kiki_oniric.axioms.Axiom``
        instances (recommended since 2026-04-21) or plain dicts.
    modalities
        Non-empty tuple of non-empty modality names.
    d_z
        Latent dimensionality; becomes each Transducer's alphabet size.
        Default 32 matches the canonical ``GammaThetaMultiplexer``.

    Returns
    -------
    DreamOfKikiNerve
        A live nerve that satisfies the :class:`nerve_core.protocols.Nerve`
        contract and carries its originating spec for round-trip dual.

    Raises
    ------
    DreamOfKikiAxiomError
        If ``axioms`` is not a Mapping, misses required keys, or
        ``modalities`` is empty / contains non-string entries.

    See Also
    --------
    to_dream_of_kiki : round-trip dual.
    docs/integration-dream-of-kiki.md : DR-X → nerve-wml mapping table.
    """
    _validate_spec(axioms, modalities)
    n_wmls = len(modalities)
    k = min(n_wmls - 1, _DEFAULT_K_CAP)
    seed = _extract_seed(axioms)
    gating = _extract_gating(axioms)

    transducers: dict[tuple[int, int], Transducer] = {
        (src, dst): Transducer(alphabet_size=d_z, gating=gating)
        for src in range(n_wmls)
        for dst in range(n_wmls)
        if src != dst
    }

    return DreamOfKikiNerve(
        axioms=axioms,
        modalities=modalities,
        d_z=d_z,
        transducers=transducers,
        n_wmls=n_wmls,
        k=k,
        seed=seed,
    )


def to_dream_of_kiki(nerve: DreamOfKikiNerve) -> dict[str, Any]:
    """Round-trip dual of :func:`from_dream_of_kiki`.

    Extracts the originating axiom spec, modalities, and ``d_z`` from a
    :class:`DreamOfKikiNerve`. The returned dict is equivalent to the
    input of :func:`from_dream_of_kiki` (the ``axioms`` value is a copy
    of the Mapping passed in).

    Parameters
    ----------
    nerve
        A nerve produced by :func:`from_dream_of_kiki`.

    Returns
    -------
    dict
        ``{"axioms": ..., "modalities": ..., "d_z": ...}``.

    Raises
    ------
    TypeError
        If ``nerve`` is not a :class:`DreamOfKikiNerve` — the round-trip
        is only defined for nerves this factory produced.
    """
    if not isinstance(nerve, DreamOfKikiNerve):
        raise TypeError(
            "to_dream_of_kiki requires a DreamOfKikiNerve instance, got "
            f"{type(nerve).__name__}. Construct one via "
            "from_dream_of_kiki(...). Upstream pointer: "
            f"{_UPSTREAM_GATE_URL}.",
        )
    return {
        "axioms": dict(nerve.axioms),
        "modalities": nerve.modalities,
        "d_z": nerve.d_z,
    }
