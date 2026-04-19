# Dream-of-kiki integration notes

Plan 7 ships the **bridge** layer between `nerve-wml` ε traces and `dream-of-kiki`'s `kiki_oniric` consolidation runtime. `kiki_oniric` is NOT a runtime dep — users who want real consolidation install it locally; CI uses `MockConsolidator`.

## Schema v0 — trace format

Tracks emitted by `DreamBridge.to_dream_input` produce an `np.int32` array with shape `[n_events, 4]`:

| column | meaning |
|---|---|
| 0 | `src_wml` — emitting WML id |
| 1 | `dst_wml` — receiving WML id |
| 2 | `code` — neuroletter code index (0..63) |
| 3 | `phase_clock` — `round(timestamp * GAMMA_HZ)` as int; locale-free |

**Tag v0** locked at `gate-dream-passed` (this plan). Future schema bumps (v1+) add columns without removing existing ones so existing consolidators keep working.

## Expected `kiki_oniric` surface

Once installed (`pip install -e /path/to/dreamOfkiki`), the module must expose:

```python
def consolidate(
    trace: np.ndarray,          # schema v0
    *,
    profile: str = "P_equ",     # one of {P_min, P_equ, P_max}
    n_transducers: int,
    alphabet_size: int = 64,
) -> np.ndarray:                # shape [n_transducers, 64, 64] — delta
    ...
```

`DreamBridge.apply_consolidation_output(nerve, delta, alpha=0.1)` adds `alpha * delta[i]` to each transducer's logits in place, preserving invariant N-4 (logits are continuous during training, pruned to {0,1} only at Gate P).

## Env gate

```bash
export DREAM_CONSOLIDATION_ENABLED=1
python scripts/your_run.py
```

When `DREAM_CONSOLIDATION_ENABLED != "1"` (default), every `DreamBridge` method is a no-op and returns empty artefacts. This lets downstream code call the bridge unconditionally without runtime surprises.

## Install dream-of-kiki

```bash
cd /path/to/dreamOfkiki
pip install -e .
# Verify the protocol surface:
uv run python -c "from bridge.dream_protocol import load_dream_module, assert_protocol_surface; m = load_dream_module('kiki_oniric'); assert_protocol_surface(m)"
```

The `assert_protocol_surface` helper raises a clear error if the module lacks `consolidate()`.

## Minimal usage

```python
from bridge.dream_bridge import DreamBridge
from bridge.dream_protocol import load_dream_module, assert_protocol_surface
from bridge.mock_consolidator import MockConsolidator
from bridge.sim_nerve_adapter import SimNerveAdapter

bridge = DreamBridge()  # reads DREAM_CONSOLIDATION_ENABLED
nerve = SimNerveAdapter(n_wmls=4, k=2, seed=0)

# Try to load the real dream module; fall back to mock.
kiki = load_dream_module("kiki_oniric")
consolidator = kiki if kiki is not None else MockConsolidator
if kiki is not None:
    assert_protocol_surface(kiki)

# Run a training episode → collect ε → consolidate offline → apply delta.
trace = bridge.collect_eps_trace(nerve, duration_ticks=1000)
encoded = bridge.to_dream_input(trace)
delta = consolidator.consolidate(
    encoded,
    profile="P_equ",
    n_transducers=len(nerve._transducers),
    alphabet_size=64,
)
bridge.apply_consolidation_output(nerve, delta, alpha=0.1)
```

## Current status

- **PARTIALLY RESOLVED** in spec §13 "Dream integration" because:
  - The bridge + mock + round-trip are production-ready and tested.
  - Real `kiki_oniric` (dream-of-kiki v0.4) does not yet expose a top-level `consolidate()` — its `PEquProfile` uses `DreamRuntime.register_handler` internally.
  - Prerequisite for full resolution: **dream-of-kiki v0.5+** with a public `consolidate(trace, profile) -> delta` entry point.

## Trace format per tag

| Tag | Schema |
|---|---|
| `gate-dream-passed` (2026-04-19) | v0 — 4 columns as above |

Bumping to v1 will add e.g. `timestamp_ns` and `confidence` columns without breaking v0 consumers.
