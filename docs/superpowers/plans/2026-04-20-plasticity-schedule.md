# Plasticity Schedule / Constellation Lock Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two opt-in time-varying plasticity mechanisms to `GammaThetaMultiplexer` so downstream consumers (specifically `bouba_sens`) can distinguish the congenital (T1) and late-acquired (T2) regimes of the pre-registered B-1 invariant (Amedi 2007).

**Architecture:** Wrap the existing `nn.Parameter` constellation with a per-step plasticity controller. `plasticity_schedule: Callable[[int], float]` multiplies the gradient flowing into the constellation via a backward hook; `constellation_lock_after: int | None` permanently sets `requires_grad=False` once the step counter crosses the threshold. Both defaults preserve current behaviour exactly — existing grids reproduce bit-identically.

**Tech Stack:** Python 3.12, uv, torch 2.5+, pytest. No new runtime dependencies.

**Parent spec:** `docs/superpowers/specs/2026-04-18-nerve-wml-design.md` (contracts N-1..N-5 unchanged — this adds an opt-in feature, not an invariant).

**Issue:** [hypneum-lab/nerve-wml#4](https://github.com/hypneum-lab/nerve-wml/issues/4). Motivation traced to bouba_sens ADR-0005 + ADR-0009 (B-1 directional inversion in 4/5 worlds).

---

## File structure

```
nerve-wml/
├── track_p/
│   └── multiplexer.py                       [MODIFY] add plasticity kwargs
├── tests/
│   └── unit/
│       ├── test_multiplexer.py              [MODIFY] guard existing contract
│       └── test_multiplexer_plasticity.py   [CREATE] 5 new focused tests
├── docs/
│   └── changelog/
│       └── v1.4.0.md                        [CREATE] release note
├── pyproject.toml                           [MODIFY] bump to 1.4.0
└── CITATION.cff                             [MODIFY] bump version
```

---

## Task 1: Backwards-compatible kwargs wired to stored state

**Goal:** Accept the two new kwargs without changing runtime behaviour yet. TDD ensures the existing 21 multiplexer tests stay green.

**Files:**
- Modify: `track_p/multiplexer.py:120-145` (the `__init__` signature + attribute stores)
- Test: `tests/unit/test_multiplexer_plasticity.py` (new file)

- [ ] **Step 1: Create the new test file with a defaults-preserve test**

File: `tests/unit/test_multiplexer_plasticity.py`

```python
"""Tests for the optional plasticity_schedule / constellation_lock kwargs.

These tests only exercise the plasticity controller; the canonical
multiplexer contract is still pinned by `test_multiplexer.py`. The
default behaviour of `GammaThetaMultiplexer()` must remain bit-identical
to v1.3.0, so existing consumers (bouba_sens v0.3 grids, in particular)
reproduce byte-for-byte.
"""

from __future__ import annotations

import torch

from track_p.multiplexer import GammaThetaConfig, GammaThetaMultiplexer


def test_defaults_preserve_v130_behaviour() -> None:
    """No kwargs = constellation remains a free nn.Parameter."""
    mux = GammaThetaMultiplexer(seed=0)
    assert mux.constellation.requires_grad is True
    assert mux.plasticity_step == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_multiplexer_plasticity.py::test_defaults_preserve_v130_behaviour -v`
Expected: FAIL with `AttributeError: 'GammaThetaMultiplexer' object has no attribute 'plasticity_step'`.

- [ ] **Step 3: Add the two kwargs + step counter to `__init__`**

In `track_p/multiplexer.py`, replace the `__init__` signature (currently lines 120-122) with:

```python
def __init__(
    self,
    cfg: GammaThetaConfig | None = None,
    *,
    seed: int | None = None,
    plasticity_schedule: Callable[[int], float] | None = None,
    constellation_lock_after: int | None = None,
) -> None:
    super().__init__()
    self.cfg = cfg if cfg is not None else GammaThetaConfig()
    self._plasticity_schedule = plasticity_schedule
    self._constellation_lock_after = constellation_lock_after
    self.plasticity_step: int = 0
```

Also add this import at the top of the file (after the existing `from typing import Literal`):

```python
from collections.abc import Callable
```

- [ ] **Step 4: Run the new test to verify it passes and run the full test suite to verify no regression**

Run:
```bash
uv run pytest tests/unit/test_multiplexer_plasticity.py::test_defaults_preserve_v130_behaviour -v
uv run pytest tests/unit/test_multiplexer.py -v
```
Expected: all green (1 new pass; 21 existing still pass).

- [ ] **Step 5: Commit**

```bash
git add track_p/multiplexer.py tests/unit/test_multiplexer_plasticity.py
git commit -m "feat(mux): Task 1 accept plasticity kwargs"
```

---

## Task 2: `step()` method drives the schedule + hard lock

**Goal:** Downstream consumers advance the plasticity clock by calling `mux.step()` once per training iteration. When the clock crosses `constellation_lock_after`, the constellation is frozen permanently.

**Files:**
- Modify: `track_p/multiplexer.py` (add the `step` method right after `__init__`)
- Test: `tests/unit/test_multiplexer_plasticity.py`

- [ ] **Step 1: Add two failing tests covering the step counter and the hard lock**

Append to `tests/unit/test_multiplexer_plasticity.py`:

```python
def test_step_increments_plasticity_counter() -> None:
    mux = GammaThetaMultiplexer(seed=0)
    assert mux.plasticity_step == 0
    mux.step()
    assert mux.plasticity_step == 1
    mux.step()
    mux.step()
    assert mux.plasticity_step == 3


def test_constellation_lock_after_freezes_requires_grad() -> None:
    mux = GammaThetaMultiplexer(seed=0, constellation_lock_after=2)
    assert mux.constellation.requires_grad is True
    mux.step()  # step 1, still plastic
    assert mux.constellation.requires_grad is True
    mux.step()  # step 2, crosses the threshold -> lock
    assert mux.constellation.requires_grad is False


def test_constellation_lock_is_permanent() -> None:
    """Once locked, the constellation must not unlock, even if step()
    keeps being called. A Phase-2 training loop must not accidentally
    re-enable plasticity after Phase 1 froze it."""
    mux = GammaThetaMultiplexer(seed=0, constellation_lock_after=1)
    mux.step()
    assert mux.constellation.requires_grad is False
    for _ in range(10):
        mux.step()
    assert mux.constellation.requires_grad is False
```

- [ ] **Step 2: Run the three tests to verify they fail**

Run: `uv run pytest tests/unit/test_multiplexer_plasticity.py -v`
Expected: three fails, all `AttributeError: 'GammaThetaMultiplexer' object has no attribute 'step'`.

- [ ] **Step 3: Implement `step()`**

In `track_p/multiplexer.py`, add this method immediately after the end of `__init__` (where the existing attributes end):

```python
def step(self) -> None:
    """Advance the plasticity clock by one iteration.

    Consumers (e.g. `bouba_sens.loop.AdaptationLoop`) call this
    once per training step. When `constellation_lock_after` is
    set and the counter crosses that threshold, the constellation
    is permanently frozen (`requires_grad=False`), which models
    the biological critical-period lock-in.
    """
    self.plasticity_step += 1
    if (
        self._constellation_lock_after is not None
        and self.plasticity_step >= self._constellation_lock_after
    ):
        self.constellation.requires_grad_(False)
```

- [ ] **Step 4: Run the three tests + full suite to verify all pass**

Run:
```bash
uv run pytest tests/unit/test_multiplexer_plasticity.py -v
uv run pytest tests/unit/ -v
```
Expected: all green (4 plasticity tests + 21 pinned contract tests = 25 total).

- [ ] **Step 5: Commit**

```bash
git add track_p/multiplexer.py tests/unit/test_multiplexer_plasticity.py
git commit -m "feat(mux): Task 2 step() drives hard lock"
```

---

## Task 3: `plasticity_schedule` modulates the constellation gradient via a backward hook

**Goal:** When `plasticity_schedule` is supplied, every gradient flowing into `self.constellation` during `.backward()` is multiplied by `plasticity_schedule(self.plasticity_step)`. Schedules returning 1.0 leave the gradient unchanged (identity); schedules returning 0.0 act like a soft lock.

**Files:**
- Modify: `track_p/multiplexer.py` (register the backward hook in `__init__`)
- Test: `tests/unit/test_multiplexer_plasticity.py`

- [ ] **Step 1: Add the failing hook test**

Append to `tests/unit/test_multiplexer_plasticity.py`:

```python
def test_plasticity_schedule_scales_constellation_gradient() -> None:
    """A schedule returning 0.5 must halve the gradient magnitude.

    The check is done by running a minimal forward + backward with
    a schedule returning a fixed 0.5, then comparing |grad| to the
    reference run (schedule = constant 1.0). Ratio must be exactly
    0.5 (no numerical slop since both runs share the same seed).
    """
    torch.manual_seed(42)
    codes = torch.randint(0, 64, (4, 7))

    def half_schedule(step: int) -> float:
        return 0.5

    def full_schedule(step: int) -> float:
        return 1.0

    mux_half = GammaThetaMultiplexer(seed=0, plasticity_schedule=half_schedule)
    mux_full = GammaThetaMultiplexer(seed=0, plasticity_schedule=full_schedule)

    carrier_half = mux_half.forward(codes)
    carrier_full = mux_full.forward(codes)

    loss_half = carrier_half.sum()
    loss_full = carrier_full.sum()

    loss_half.backward()
    loss_full.backward()

    grad_norm_half = mux_half.constellation.grad.abs().sum()
    grad_norm_full = mux_full.constellation.grad.abs().sum()

    ratio = (grad_norm_half / grad_norm_full).item()
    assert abs(ratio - 0.5) < 1e-6, f"expected 0.5x scaling, got {ratio}"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_multiplexer_plasticity.py::test_plasticity_schedule_scales_constellation_gradient -v`
Expected: FAIL with `AssertionError: expected 0.5x scaling, got 1.0` (the schedule is stored but not wired yet).

- [ ] **Step 3: Register the backward hook inside `__init__`**

In `track_p/multiplexer.py`, at the very end of `__init__` (after the existing `self.register_buffer("_t_grid", ...)` line, which lives around line 155; check the file) add:

```python
# Plasticity hook: when a schedule is registered, multiply every
# gradient flowing into the constellation by `schedule(step)`. A
# constant-1.0 schedule is exactly equivalent to no hook (identity).
if self._plasticity_schedule is not None:
    self.constellation.register_hook(self._apply_plasticity_schedule)
```

Then add the hook implementation right below the `step()` method from Task 2:

```python
def _apply_plasticity_schedule(self, grad: Tensor) -> Tensor:
    """Backward hook: scale `grad` by `plasticity_schedule(step)`.

    Called by autograd during `.backward()`. `grad` is the upstream
    gradient of the loss w.r.t. the constellation parameter; we
    return the scaled tensor and autograd uses that instead. Clamp
    scale to a plain Python float so autograd never tries to track
    the schedule itself.
    """
    if self._plasticity_schedule is None:  # defensive, never hit at runtime
        return grad
    scale = float(self._plasticity_schedule(self.plasticity_step))
    return grad * scale
```

- [ ] **Step 4: Run the new test + full suite**

Run:
```bash
uv run pytest tests/unit/test_multiplexer_plasticity.py -v
uv run pytest tests/unit/ -v
```
Expected: 5 plasticity tests + 21 pinned contract tests = 26 green.

- [ ] **Step 5: Commit**

```bash
git add track_p/multiplexer.py tests/unit/test_multiplexer_plasticity.py
git commit -m "feat(mux): Task 3 plasticity schedule hook"
```

---

## Task 4: `state_dict` / `load_state_dict` round-trip preserves the plasticity counter

**Goal:** `AdaptationLoop.snapshot()` in `bouba_sens` serialises `mux.state_dict()` at the end of Phase 1 and restores it at the start of Phase 2 (T2 cell). If the plasticity counter is NOT serialised, the T2 cell starts at `plasticity_step=0` with the constellation un-locked — defeating the entire point of the feature. This task fixes that by registering `plasticity_step` as a buffer, not a plain Python int.

**Files:**
- Modify: `track_p/multiplexer.py` (convert `plasticity_step` from Python int to a buffer)
- Test: `tests/unit/test_multiplexer_plasticity.py`

- [ ] **Step 1: Add the failing round-trip test**

Append to `tests/unit/test_multiplexer_plasticity.py`:

```python
def test_plasticity_step_roundtrips_through_state_dict() -> None:
    """Phase 1 snapshot must carry the plasticity counter into Phase 2."""
    src = GammaThetaMultiplexer(seed=0, constellation_lock_after=100)
    for _ in range(5):
        src.step()
    assert int(src.plasticity_step) == 5

    state = src.state_dict()

    dst = GammaThetaMultiplexer(seed=0, constellation_lock_after=100)
    dst.load_state_dict(state)
    assert int(dst.plasticity_step) == 5


def test_load_state_dict_re_applies_lock_when_threshold_already_crossed() -> None:
    """If a checkpoint was taken post-lock, restoring it must freeze the
    constellation even though the fresh instance started plastic."""
    src = GammaThetaMultiplexer(seed=0, constellation_lock_after=3)
    for _ in range(5):
        src.step()
    assert src.constellation.requires_grad is False

    state = src.state_dict()

    dst = GammaThetaMultiplexer(seed=0, constellation_lock_after=3)
    assert dst.constellation.requires_grad is True
    dst.load_state_dict(state)
    assert dst.constellation.requires_grad is False
```

- [ ] **Step 2: Run the two tests to verify they fail**

Run: `uv run pytest tests/unit/test_multiplexer_plasticity.py -v`
Expected: both fail. The first fails because `plasticity_step` is a plain Python int and `state_dict()` does not serialise it. The second fails for the same reason: the counter resets to 0, so the lock is never re-applied.

- [ ] **Step 3: Convert `plasticity_step` to a buffer + apply lock on load**

Edit `track_p/multiplexer.py` to (a) replace the `self.plasticity_step: int = 0` line in `__init__` with a buffer registration, and (b) override `load_state_dict` to re-apply the lock if the loaded counter crosses the threshold.

Replace (inside `__init__`):

```python
self.plasticity_step: int = 0
```

with:

```python
self.register_buffer("plasticity_step", torch.tensor(0, dtype=torch.long))
```

Update the `step()` method (from Task 2) to use the buffer:

```python
def step(self) -> None:
    """Advance the plasticity clock by one iteration. See docstring
    above; buffer-based so Phase 1 -> Phase 2 checkpoint round-trips
    carry the counter."""
    self.plasticity_step += 1
    if (
        self._constellation_lock_after is not None
        and int(self.plasticity_step) >= self._constellation_lock_after
    ):
        self.constellation.requires_grad_(False)
```

Update the hook method to convert the buffer to an int:

```python
def _apply_plasticity_schedule(self, grad: Tensor) -> Tensor:
    if self._plasticity_schedule is None:
        return grad
    scale = float(self._plasticity_schedule(int(self.plasticity_step)))
    return grad * scale
```

Add the `load_state_dict` override right below `_apply_plasticity_schedule`:

```python
def load_state_dict(  # type: ignore[override]
    self, state_dict, strict: bool = True, assign: bool = False
):
    """Re-apply `constellation_lock_after` if the loaded counter has
    already crossed the threshold. Without this, reloading a post-lock
    checkpoint into a fresh instance would leave the constellation
    plastic, silently breaking critical-period semantics.
    """
    result = super().load_state_dict(state_dict, strict=strict, assign=assign)
    if (
        self._constellation_lock_after is not None
        and int(self.plasticity_step) >= self._constellation_lock_after
    ):
        self.constellation.requires_grad_(False)
    return result
```

- [ ] **Step 4: Run the two round-trip tests + full suite**

Run:
```bash
uv run pytest tests/unit/test_multiplexer_plasticity.py -v
uv run pytest tests/unit/ -v
```
Expected: 7 plasticity tests + 21 contract tests = 28 green.

- [ ] **Step 5: Commit**

```bash
git add track_p/multiplexer.py tests/unit/test_multiplexer_plasticity.py
git commit -m "fix(mux): Task 4 plasticity counter is a buffer"
```

---

## Task 5: Release v1.4.0 — bump version + changelog + citation

**Goal:** Ship the feature under a new semver-minor version so downstream `bouba_sens` can pin `nerve-wml>=1.4.0` for the plasticity-aware grid. Zenodo DOI minting is a separate post-tag step.

**Files:**
- Modify: `pyproject.toml` (bump `version = "1.3.0"` to `"1.4.0"`)
- Modify: `CITATION.cff` (update `version:` field)
- Create: `docs/changelog/v1.4.0.md` (release note)

- [ ] **Step 1: Bump `pyproject.toml`**

In `pyproject.toml`, replace the exact line:

```toml
version = "1.3.0"
```

with:

```toml
version = "1.4.0"
```

Verify with `grep -n "^version" pyproject.toml` — exactly one match, value `"1.4.0"`. If the pre-edit version string is not `"1.3.0"` (e.g. the repo shipped a patch release in the meantime), abort the task: the release train has moved, and merging this plan would overwrite a newer tag. Open a follow-up issue and stop.

- [ ] **Step 2: Bump `CITATION.cff`**

Open `CITATION.cff`, find the line starting with `version:`, and update the value to `1.4.0`. Update `date-released:` to today's ISO date (`2026-04-20`).

- [ ] **Step 3: Create the changelog file**

File: `docs/changelog/v1.4.0.md`

```markdown
# nerve-wml v1.4.0 — plasticity schedule + constellation lock

**Release date:** 2026-04-20
**Issue:** [hypneum-lab/nerve-wml#4](https://github.com/hypneum-lab/nerve-wml/issues/4)

## Added

- Optional `plasticity_schedule: Callable[[int], float]` kwarg on
  `GammaThetaMultiplexer`. When set, the callable is invoked each
  `.backward()` with the current `plasticity_step`; its return value
  multiplies the gradient flowing into `self.constellation`.
- Optional `constellation_lock_after: int` kwarg. When the internal
  counter crosses the threshold, `self.constellation.requires_grad`
  is permanently set to `False` (biological critical-period lock-in).
- `GammaThetaMultiplexer.step()` method advances the plasticity clock
  by one iteration. Consumers call it once per training step.
- `plasticity_step` is a `long` buffer, so `state_dict()` /
  `load_state_dict()` round-trip it. Checkpoints taken post-lock
  correctly re-apply the lock on restore.

## Unchanged

- Default construction (`GammaThetaMultiplexer()` or
  `GammaThetaMultiplexer(seed=0)`) reproduces v1.3.0 behaviour byte-
  for-byte. Zero effect on existing consumers that do not opt in.
- The 21 pinned multiplexer contract tests in `test_multiplexer.py`
  remain load-bearing and all still pass.

## Motivation

bouba_sens v0.3 / v0.4 found the B-1 invariant (Amedi 2007
congenital-blindness gap) directionally falsified across 4 / 5 worlds.
The only architectural difference between T1 (congenital) and T2
(late-acquired) cells in v0.3 was whether Phase 1 ran; both regimes
shared identical multiplexer plasticity. This release lets
`AdaptationLoop` give T1 and T2 biologically distinct plasticity
profiles, so the next grid can test whether the B-1 gap recovers.

## Downstream validation plan

- bouba_sens will open a matching PR wiring `plasticity_schedule` +
  `constellation_lock_after` into `AdaptationLoop`, re-run the
  150-cell grid, and record the new B-1 verdict in a new ADR.
- If B-1 PASSes on >= 2 worlds with the schedule active, nerve-wml
  earns a second empirically-attested invariant (after B-3 @ 22.3x
  on real ECG in bouba_sens ADR-0009).
```

- [ ] **Step 4: Run the full test suite one last time to confirm nothing regressed**

Run: `uv run pytest`
Expected: all tests green across `tests/unit/`, `tests/info_theoretic/`, `tests/integration/`, `tests/golden/`.

- [ ] **Step 5: Commit and tag**

```bash
git add pyproject.toml CITATION.cff docs/changelog/v1.4.0.md
git commit -m "chore(release): v1.4.0 plasticity schedule"
git tag -a v1.4.0 -m "v1.4.0 plasticity schedule + constellation lock"
git push origin master
git push origin v1.4.0
```

- [ ] **Step 6: Close follow-up work on issue #4**

```bash
gh issue comment 4 --body "Shipped in v1.4.0 — tag pushed, Zenodo DOI minting auto-runs via the existing release hook. bouba_sens PR to wire the feature will reference this tag as its floor."
```

Expected: the issue stays open until bouba_sens confirms B-1 recovery; the comment is a progress signal, not a close.

---

## Exit criteria

1. 21 pinned contract tests in `test_multiplexer.py` still green.
2. 7 new tests in `test_multiplexer_plasticity.py` green.
3. `GammaThetaMultiplexer()` default behaviour unchanged (byte-identical carrier output for fixed seed).
4. `pyproject.toml` version = `1.4.0`.
5. Tag `v1.4.0` pushed to `hypneum-lab/nerve-wml`.
6. Changelog file committed.
7. Issue #4 has a progress comment linking the release tag.
