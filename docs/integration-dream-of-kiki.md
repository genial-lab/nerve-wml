# dreamOfkiki ↔ nerve-wml integration

**Status: LIVE (axioms axis, v1.8.0+) — `kiki_oniric.axioms` integration wired;
consolidate runtime still scaffolded, tracked for v1.9.0.** See
`nerve_core/from_dream_of_kiki.py` for the live contract and
[hypneum-lab/nerve-wml#6](https://github.com/hypneum-lab/nerve-wml/issues/6)
for the dependency tracker.

## Why this bridge exists

`dream-of-kiki` is the formal-framework sister repo of nerve-wml. Its
Paper 1 v0.2 defines **5 executable axioms** DR-0 to DR-4 over a free
episode semi-group with four canonical operations (replay, downscale,
restructure, recombine) plus the bit-exact reproducibility contract R1.

Today, anyone wanting to test dream-of-kiki claims empirically against
nerve-wml has to reconstruct the protocol piece by piece. This module
provides the canonical entry point — `Nerve.from_dream_of_kiki(...)` —
so the theory → empirical handoff is one function call.

## Mapping table (target API)

| dreamOfkiki axiom | Operation | nerve-wml component | Wiring detail |
|-------------------|-----------|---------------------|---------------|
| **DR-0** | replay | `SimNerve` event buffer | replay window length + γ/θ phase locking |
| **DR-1** | downscale | `Transducer` entropy regularizer | temperature schedule on the row-wise softmax |
| **DR-2** | restructure | `bridge.transducer_resize` | re-normalise row + column mass on lesion |
| **DR-3** | recombine | `Transducer` (gating mode) | hard / Gumbel-softmax — see issue #5 |
| **DR-4** | bit-exact R1 | `SimNerve(seed=...)` + per-WML `Generator` | deterministic seeding across the assembled subsystem |

The `from_dream_of_kiki` factory composes these into a `Nerve` instance
matching the dream-of-kiki spec, with `d_z = 32` defaulting to the
canonical `GammaThetaMultiplexer` alphabet size.

## Public contract (live today)

The **shape** of the input is pinned; only the **runtime instantiation**
is gated. Validation that runs today:

- `axioms` must be a `Mapping` containing every key in `REQUIRED_AXIOMS`
  (`DR-0`, `DR-1`, `DR-2`, `DR-3`, `DR-4`, case-sensitive).
- `modalities` must be a non-empty tuple of non-empty strings.
- `d_z` is reserved and validated downstream once wiring lands.

Failures raise `DreamOfKikiAxiomError` (a `ValueError` subclass) with a
message naming the missing keys or the malformed argument. Once a spec
passes validation, `from_dream_of_kiki` raises `NotImplementedError`
with an upstream pointer.

## Pipeline validation (pre-upstream, 2026-04-21)

While the `from_dream_of_kiki` factory itself remains gated on
`dream-of-kiki` publishing `axioms.py`, the **γ → θ → consolidate
pipeline that the bridge targets has been validated end-to-end**
without waiting for the upstream release.

`scripts/ablation_n3_predictive.py` (commit `a866b28`) exercises:

1. **γ phase** — `MlpWML.step(nerve, t)` where `emit_head_eps`
   naturally emits ε letters when internal surprise
   `(h − h_prior).norm()` exceeds `threshold_eps=0.30`.
2. **θ phase** — `DreamBridge.collect_eps_trace(nerve, 200 ticks)`
   records 200 `role=ERROR, phase=THETA` letters per 200 ticks,
   all structurally compliant with invariant N-3.
3. **Consolidation** — `DreamBridge.apply_consolidation_output(nerve, delta)`
   applies the delta to transducer logits (env-gated).

When `kiki_oniric` publishes its public API, only the **content**
of `delta` in step (3) needs to be substituted — the interface
contract is already exercised and the pipeline plumbing validated.
See `docs/research-notes/n3-gate-role.md` for the full reasoning
behind why Δ = 0 on this pipeline is the expected, correct result.

## Example usage (target API)

```python
from dream_of_kiki.axioms import DR_AXIOMS_V02   # ← upstream gate
from nerve_core.from_dream_of_kiki import from_dream_of_kiki

nerve = from_dream_of_kiki(
    axioms=DR_AXIOMS_V02,
    modalities=("audio", "vision", "tactile", "gravity", "force"),
    d_z=32,
)

# Round-trip — preserves the axiom spec modulo non-axiomatic internals.
from nerve_core.from_dream_of_kiki import to_dream_of_kiki
spec_back = to_dream_of_kiki(nerve)
assert spec_back == DR_AXIOMS_V02
```

## Roadmap

1. **Now (scaffold)** — public contract validates spec, raises
   `NotImplementedError` with upstream pointer. Tests cover the
   validation surface so the contract can't regress.
2. **Upstream gate lifts** (dream-of-kiki publishes `axioms.py`) —
   replace the `NotImplementedError` body with concrete wiring per the
   mapping table above. No public API change.
3. **Cycle 2 ablation** — `bouba_sens` consumers swap their hand-rolled
   `Nerve` construction for `Nerve.from_dream_of_kiki(...)` and re-run
   the 5-world grid; ADR-0012 records whether the formal-spec-driven
   instantiation matches the hand-rolled B-1/B-2/B-3 verdicts.

## Why pin the contract before the wiring

Three reasons:

- **Cross-repo design alignment.** dream-of-kiki publishing
  `axioms.py` doesn't need to invent a layout that fits nerve-wml; the
  layout is already defined here. Conversely, nerve-wml's wiring
  decisions don't need to wait on dream-of-kiki — both can evolve.
- **Downstream consumer plumbing.** `bouba_sens` can already accept a
  config flag `use_dream_of_kiki_factory: bool` and be ready to swap
  instantiation paths the day the gate lifts.
- **Audit trail.** The OSF pre-registration DOI
  `10.17605/OSF.IO/Q6JYN` (locked 2026-04-19) defines the axioms
  formally; the contract here mirrors that pre-registration so any
  later runtime divergence is traceable.

## References

- OSF pre-registration DOI: [`10.17605/OSF.IO/Q6JYN`](https://doi.org/10.17605/OSF.IO/Q6JYN)
- dream-of-kiki Paper 1 v0.2 (in review): https://github.com/hypneum-lab/dream-of-kiki
- Upstream issue tracker: https://github.com/hypneum-lab/nerve-wml/issues/6
- Design dependency: `nerve_core/from_dream_of_kiki.py`

## v1.8.0-axioms (2026-04-24)

Real `kiki_oniric.axioms.DR0..DR4` instances are now first-class inputs
to `from_dream_of_kiki`. Install the optional extra:

```bash
uv sync --extra axioms
```

When installed, users can pass real `Axiom` instances directly:

```python
from kiki_oniric import axioms as up
from nerve_core.from_dream_of_kiki import from_dream_of_kiki

nerve = from_dream_of_kiki(
    axioms={
        "DR-0": up.DR0,
        "DR-1": up.DR1,
        "DR-2": up.DR2,
        "DR-3": up.DR3,
        "DR-4": up.DR4,
    },
    modalities=("audio", "vision"),
    d_z=32,
)
```

The weakened DR-2 precondition (upstream amendment 2026-04-21) is
consumed automatically when the spec carries an `operation_order`
hint; orders that put RESTRUCTURE before REPLAY raise
`DreamOfKikiAxiomError`.

Pinned upstream version at this release: `C-v0.8.0+PARTIAL`. A
warning is emitted on import-time drift; strict mode is opt-in via
`nerve_core.axioms_compat.check_upstream_axioms_version(strict=True)`.

The `consolidate()` runtime wiring remains scaffolded; that bump is
tracked for v1.9.0.
