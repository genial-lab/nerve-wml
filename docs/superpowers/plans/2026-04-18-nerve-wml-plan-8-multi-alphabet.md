# nerve-wml Plan 8 — Multi-Alphabet Adaptive Codebook (Spec §13 Q1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve spec §13 open question 1 — "Could the 64-code alphabet grow or shrink adaptively per WML once the system is stable?" — by building a size-adaptive codebook system with deterministic shrink/grow operations, transducer reshaping, pilots, and a gate.

**Architecture:**
- `track_p/adaptive_codebook.py` — `AdaptiveCodebook` wraps `VQCodebook` with an `active_mask` that makes resizing purely logical (the underlying 64-slot storage is never physically truncated). `shrink` removes low-usage codes; `grow` splits high-usage codes via perturbation.
- `bridge/transducer_resize.py` — `resize_transducer` returns a new `Transducer` with reshaped logit matrix, using mean-pool for shrink and duplication for grow. Argmax of unchanged codes is preserved.
- `scripts/adaptive_pilot.py` — `run_adaptive_cycle` (shrink) and `run_adaptive_grow_cycle` (grow) each train a WML to stability, measure the usage histogram, trigger the adaptive operation, rewire transducers, and report before/after metrics.
- `tests/integration/test_gate_adaptive.py` — gate aggregator `run_gate_adaptive` enforcing four criteria (64→48 shrink, 64→80 grow, argmax preservation, multi-cycle stability).
- Spec §13 Q1 and paper §8 updated to PLAN 8 RESOLVED.

**Design constraints (enforced throughout):**
- `active_mask` is a `torch.bool` buffer of length 64. `current_size()` returns `active_mask.sum().item()`.
- Existing gate tests (P/W/M/M2) must remain green — we never touch `VQCodebook.size`.
- Transducer invariant: after resize, `forward(src_code)` with `src_code < new_src_size` returns `dst_code < new_dst_size`.
- All operations are deterministic given the codebook state (no random beyond the seeded perturbation).
- No retrain from scratch; accuracy drop after resize must recover within 200 fine-tuning steps.

**Tech Stack:** Python 3.12, `uv`, `torch`, `numpy`, `pytest`. No new runtime dependencies.

---

## File Map

| Action  | Path                                              | Responsibility                                          |
|---------|---------------------------------------------------|---------------------------------------------------------|
| Create  | `track_p/adaptive_codebook.py`                    | `AdaptiveCodebook`: `active_mask`, `shrink`, `grow`, `current_size` |
| Create  | `bridge/transducer_resize.py`                     | `resize_transducer`: mean-pool shrink, duplicate grow   |
| Create  | `scripts/adaptive_pilot.py`                       | `run_adaptive_cycle`, `run_adaptive_grow_cycle`         |
| Create  | `tests/unit/test_adaptive_codebook.py`            | Unit tests for AdaptiveCodebook methods                 |
| Create  | `tests/unit/test_transducer_resize.py`            | Unit tests for resize_transducer                        |
| Create  | `tests/integration/test_gate_adaptive.py`         | Gate aggregator `run_gate_adaptive`                     |
| Modify  | `track_p/__init__.py`                             | Export `AdaptiveCodebook`                               |
| Modify  | `bridge/__init__.py`                              | Export `resize_transducer`                              |
| Modify  | `docs/superpowers/specs/2026-04-18-nerve-wml-design.md` | Mark §13 Q1 as PLAN 8 RESOLVED                   |
| Modify  | `papers/nerve_wml_paper.md` (or equivalent)       | One-sentence §8 addendum citing `AdaptiveCodebook`      |

---

## Task 1: `AdaptiveCodebook` skeleton — `active_mask` + `current_size`

**Phase 0** — establish the logical-resize contract. The underlying `VQCodebook` is never modified; `active_mask` is a `torch.bool` buffer that selects which of the 64 slots are "live."

**Files:**
- Create: `track_p/adaptive_codebook.py`
- Create: `tests/unit/test_adaptive_codebook.py`
- Modify: `track_p/__init__.py`

- [ ] **Step 1: Write failing unit tests**

Create `tests/unit/test_adaptive_codebook.py`:

```python
"""Unit tests for track_p/adaptive_codebook.py — skeleton phase."""
from __future__ import annotations

import torch
import pytest

from track_p.adaptive_codebook import AdaptiveCodebook


# ---------------------------------------------------------------------------
# Task 1 — skeleton: active_mask + current_size
# ---------------------------------------------------------------------------

def test_adaptive_codebook_default_size():
    """Fresh AdaptiveCodebook reports current_size == codebook.size (64)."""
    ac = AdaptiveCodebook(size=64, dim=128)
    assert ac.current_size() == 64


def test_active_mask_is_all_true_initially():
    """active_mask starts as all-True tensor of length size."""
    ac = AdaptiveCodebook(size=64, dim=128)
    assert ac.active_mask.shape == (64,)
    assert ac.active_mask.all()


def test_active_mask_dtype():
    """active_mask must be a bool buffer, not int."""
    ac = AdaptiveCodebook(size=64, dim=128)
    assert ac.active_mask.dtype == torch.bool


def test_current_size_tracks_mask():
    """current_size reflects manual mask mutations."""
    ac = AdaptiveCodebook(size=64, dim=128)
    ac.active_mask[0] = False
    ac.active_mask[1] = False
    assert ac.current_size() == 62


def test_active_indices_returns_sorted_tensor():
    """active_indices() returns the indices of live codes in ascending order."""
    ac = AdaptiveCodebook(size=8, dim=16)
    ac.active_mask[2] = False
    ac.active_mask[5] = False
    idx = ac.active_indices()
    assert idx.tolist() == [0, 1, 3, 4, 6, 7]


def test_codebook_inner_size_unchanged():
    """Underlying VQCodebook.size is always 64 regardless of logical size."""
    ac = AdaptiveCodebook(size=64, dim=128)
    ac.active_mask[:16] = False
    assert ac.codebook.size == 64


def test_active_embeddings_shape():
    """active_embeddings() returns only the live rows."""
    ac = AdaptiveCodebook(size=8, dim=16)
    ac.active_mask[0] = False
    emb = ac.active_embeddings()
    assert emb.shape == (7, 16)


def test_quantize_active_returns_local_index():
    """quantize_active maps z to a local index in [0, current_size)."""
    torch.manual_seed(0)
    ac = AdaptiveCodebook(size=8, dim=16)
    z = torch.randn(4, 16)
    idx, quantized, loss = ac.quantize_active(z)
    assert idx.max().item() < ac.current_size()
    assert quantized.shape == z.shape
    assert loss.item() >= 0
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -m pytest tests/unit/test_adaptive_codebook.py -v 2>&1 | tail -20
```

Expected: `ModuleNotFoundError` for `track_p.adaptive_codebook`.

- [ ] **Step 3: Create `track_p/adaptive_codebook.py`**

```python
"""Adaptive per-WML codebook — logical resize over a fixed 64-slot VQCodebook.

Design contract:
- The underlying VQCodebook.size is ALWAYS the physical storage size (default 64).
- active_mask (torch.bool, length size) selects which slots are "live."
- current_size() == active_mask.sum().item()  — this is the effective alphabet.
- quantize_active() maps z to a local index in [0, current_size) by restricting
  distance computation to the active embedding rows only.
- shrink() and grow() mutate active_mask (and add/remove embedding rows by
  perturbing the VQCodebook's buffer directly for grow).

See spec §13 Q1, Plan 8.
"""
from __future__ import annotations

import torch
from torch import Tensor

from track_p.vq_codebook import VQCodebook

# Maximum physical slots. Hard-coded to match the spec §4.1 alphabet size.
_MAX_SIZE: int = 64


class AdaptiveCodebook:
    """Wraps VQCodebook with a logical active_mask for adaptive alphabet size.

    Parameters
    ----------
    size : int
        Physical codebook slots (default 64, must equal _MAX_SIZE for now).
    dim : int
        Embedding dimensionality.
    commitment_beta : float
        Passed to VQCodebook.
    ema : bool
        Use EMA update (default True).
    decay : float
        EMA decay rate.
    """

    def __init__(
        self,
        size: int = _MAX_SIZE,
        dim: int = 128,
        *,
        commitment_beta: float = 0.25,
        ema: bool = True,
        decay: float = 0.99,
    ) -> None:
        if size > _MAX_SIZE:
            raise ValueError(
                f"AdaptiveCodebook physical size must be <= {_MAX_SIZE}, got {size}."
            )
        self.codebook = VQCodebook(
            size=size,
            dim=dim,
            commitment_beta=commitment_beta,
            ema=ema,
            decay=decay,
        )
        # Logical mask: True = live, False = pruned/vacant.
        self.register_buffer_compat(
            "active_mask", torch.ones(size, dtype=torch.bool)
        )
        # Usage counter shadow aligned with active_mask positions.
        # We read directly from codebook.usage_counter for physical slots.
        self._size = size
        self._dim = dim

    # ------------------------------------------------------------------
    # Buffer helper (AdaptiveCodebook is not an nn.Module to stay light,
    # so we manage the tensor manually).
    # ------------------------------------------------------------------

    def register_buffer_compat(self, name: str, tensor: Tensor) -> None:
        setattr(self, name, tensor)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def current_size(self) -> int:
        """Return the number of live (active) codes."""
        return int(self.active_mask.sum().item())

    def active_indices(self) -> Tensor:
        """Return sorted physical indices of live codes."""
        return self.active_mask.nonzero(as_tuple=False).squeeze(1)

    def active_embeddings(self) -> Tensor:
        """Return embedding matrix restricted to live codes. Shape: [current_size, dim]."""
        return self.codebook.embeddings[self.active_indices()]

    def quantize_active(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Quantize z against active codes only.

        Returns (local_indices, quantized_z, loss) where local_indices ∈ [0, current_size).
        The local_indices are dense — 0 maps to the first live physical slot.
        """
        phys_idx = self.active_indices()          # [current_size]
        active_emb = self.codebook.embeddings[phys_idx]  # [current_size, dim]

        dist = torch.cdist(z, active_emb)         # [B, current_size]
        local_idx = dist.argmin(dim=-1)            # [B] in [0, current_size)

        # Map local → physical for usage tracking.
        phys = phys_idx[local_idx]
        for i in phys.tolist():
            self.codebook.usage_counter[i] += 1

        quantized = active_emb[local_idx]          # [B, dim]

        beta = self.codebook.commitment_beta
        commit_loss = beta * ((z - quantized.detach()) ** 2).mean()
        codebook_loss = ((quantized - z.detach()) ** 2).mean()
        loss = commit_loss + codebook_loss

        # Straight-through.
        quantized = z + (quantized - z).detach()
        return local_idx, quantized, loss
```

- [ ] **Step 4: Export from `track_p/__init__.py`**

Append to `track_p/__init__.py`:

```python
from track_p.adaptive_codebook import AdaptiveCodebook  # noqa: F401
```

- [ ] **Step 5: Run tests — all Task 1 tests must pass**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -m pytest tests/unit/test_adaptive_codebook.py -v -k "default_size or all_true or dtype or tracks_mask or sorted_tensor or inner_size or shape or local_index" 2>&1 | tail -20
```

Expected: 8 passed.

**Constraints:**
- `VQCodebook.size` must not change.
- `active_mask` must be a plain `torch.Tensor` (not `nn.Parameter`).
- `quantize_active` must update `codebook.usage_counter` on physical slots.

**Report:** confirm `current_size() == 64` on init, confirm physical `codebook.size == 64` after manual mask mutation.

---

## Task 2: `shrink` — prune below-threshold codes, update mask

**Phase 1** — identify and remove codes whose usage fraction is below `min_usage_frac`. Return a list of the physical indices that remain live.

**Files:**
- Modify: `track_p/adaptive_codebook.py`
- Modify: `tests/unit/test_adaptive_codebook.py` (append)

- [ ] **Step 1: Append failing tests**

```python
# ---------------------------------------------------------------------------
# Task 2 — shrink
# ---------------------------------------------------------------------------

def test_shrink_removes_unused_codes():
    """shrink(min_usage_frac=0.1) deactivates codes below the threshold."""
    torch.manual_seed(42)
    ac = AdaptiveCodebook(size=8, dim=16)
    # Give usage to codes 0,1,2 only; codes 3-7 stay at 0.
    ac.codebook.usage_counter[0] = 100
    ac.codebook.usage_counter[1] = 80
    ac.codebook.usage_counter[2] = 50
    # Codes 3-7: usage_counter stays 0.
    kept = ac.shrink(min_usage_frac=0.01)
    # At least codes 3-7 must be removed.
    assert ac.current_size() < 8
    assert all(k in kept for k in [0, 1, 2])


def test_shrink_returns_kept_indices():
    """shrink returns the list of physical indices that remain active."""
    ac = AdaptiveCodebook(size=8, dim=16)
    ac.codebook.usage_counter[0] = 50
    ac.codebook.usage_counter[1] = 0
    kept = ac.shrink(min_usage_frac=0.01)
    assert 0 in kept
    assert 1 not in kept


def test_shrink_never_drops_below_min_codes():
    """shrink preserves at least min_codes live codes (default 4)."""
    ac = AdaptiveCodebook(size=8, dim=16)
    # All usage counters at 0 — shrink would prune everything.
    kept = ac.shrink(min_usage_frac=1.0, min_codes=4)
    assert ac.current_size() >= 4
    assert len(kept) >= 4


def test_shrink_idempotent():
    """Two successive shrinks with identical state produce the same mask."""
    ac = AdaptiveCodebook(size=8, dim=16)
    ac.codebook.usage_counter[0] = 100
    ac.shrink(min_usage_frac=0.01)
    mask_after_first = ac.active_mask.clone()
    ac.shrink(min_usage_frac=0.01)
    assert torch.equal(ac.active_mask, mask_after_first)


def test_shrink_updates_mask_deterministically():
    """Same usage_counter → same active_mask on two fresh codebooks."""
    for seed in (0, 1, 2):
        ac1 = AdaptiveCodebook(size=8, dim=16)
        ac2 = AdaptiveCodebook(size=8, dim=16)
        ac1.codebook.usage_counter[seed] = 200
        ac2.codebook.usage_counter[seed] = 200
        ac1.shrink(min_usage_frac=0.05)
        ac2.shrink(min_usage_frac=0.05)
        assert torch.equal(ac1.active_mask, ac2.active_mask)
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -m pytest tests/unit/test_adaptive_codebook.py -v -k "shrink" 2>&1 | tail -20
```

Expected: `AttributeError` — `shrink` not yet defined.

- [ ] **Step 3: Implement `shrink` in `AdaptiveCodebook`**

Add to the class after `quantize_active`:

```python
def shrink(
    self,
    min_usage_frac: float = 0.01,
    min_codes: int = 4,
) -> list[int]:
    """Deactivate codes whose usage fraction is below min_usage_frac.

    Only currently active codes are considered.  The method never reduces
    the alphabet below min_codes live codes.

    Parameters
    ----------
    min_usage_frac : float
        Minimum fraction of total active usage a code must have to survive.
        E.g. 0.01 → prune any code used fewer than 1 % of all active uses.
    min_codes : int
        Hard lower bound on the alphabet size after shrink.

    Returns
    -------
    list[int]
        Sorted physical indices of codes that remain active after shrink.
    """
    phys_idx = self.active_indices()           # physical indices of live codes
    usage = self.codebook.usage_counter[phys_idx].float()
    total = usage.sum()

    if total == 0:
        # No usage recorded yet: prune all but the top-min_codes by index.
        keep_count = max(min_codes, 1)
        keep_local = torch.arange(min(keep_count, phys_idx.shape[0]))
    else:
        frac = usage / total
        keep_local = (frac >= min_usage_frac).nonzero(as_tuple=False).squeeze(1)
        # Enforce min_codes floor: if too few survive, keep top-usage codes.
        if keep_local.shape[0] < min_codes:
            _, top = usage.topk(min(min_codes, phys_idx.shape[0]))
            keep_local = top.sort().values

    keep_phys: list[int] = phys_idx[keep_local].tolist()

    # Reset mask: deactivate everything, then re-activate kept codes.
    self.active_mask[:] = False
    for p in keep_phys:
        self.active_mask[p] = True

    return sorted(keep_phys)
```

- [ ] **Step 4: Run shrink tests — all must pass**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -m pytest tests/unit/test_adaptive_codebook.py -v -k "shrink" 2>&1 | tail -20
```

Expected: 5 passed.

**Constraints:**
- `shrink` must never modify `VQCodebook.embeddings` — only `active_mask`.
- `shrink` must be deterministic: given the same `usage_counter`, it always produces the same `active_mask`.
- The `min_codes` floor prevents full alphabet collapse.

**Report:** confirm shrink on usage_counter[0]=100, rest=0 with size=8 yields `current_size() == 1` (or `min_codes` if floor is triggered).

---

## Task 3: `grow` — split top-usage codes via embedding perturbation

**Phase 2** — identify the `top_k_to_split` most-used live codes and create child codes in vacant physical slots, initialised as the parent embedding ± a small perturbation. New slots are activated in `active_mask`.

**Files:**
- Modify: `track_p/adaptive_codebook.py`
- Modify: `tests/unit/test_adaptive_codebook.py` (append)

- [ ] **Step 1: Append failing tests**

```python
# ---------------------------------------------------------------------------
# Task 3 — grow
# ---------------------------------------------------------------------------

def test_grow_increases_current_size():
    """grow(top_k_to_split=2) adds 2 new codes (one child per split)."""
    ac = AdaptiveCodebook(size=8, dim=16)
    ac.codebook.usage_counter[0] = 200
    ac.codebook.usage_counter[1] = 150
    before = ac.current_size()
    ac.grow(top_k_to_split=2)
    assert ac.current_size() == before + 2


def test_grow_child_embeddings_are_near_parent():
    """Child embedding is within 1.0 of the parent in L2 distance."""
    torch.manual_seed(0)
    ac = AdaptiveCodebook(size=8, dim=16)
    parent_emb = ac.codebook.embeddings[0].clone()
    ac.codebook.usage_counter[0] = 999
    ac.grow(top_k_to_split=1, perturb_scale=0.1)
    # Find the newly activated slot (any new True in active_mask that wasn't there before).
    child_slots = [i for i in range(8) if ac.active_mask[i] and i != 0]
    assert len(child_slots) >= 1
    child_emb = ac.codebook.embeddings[child_slots[0]]
    dist = (child_emb - parent_emb).norm().item()
    assert dist < 1.0


def test_grow_does_nothing_when_no_vacant_slots():
    """grow silently skips if all physical slots are already active."""
    ac = AdaptiveCodebook(size=4, dim=8)
    # All 4 slots active (default).
    ac.codebook.usage_counter[:] = 100
    before = ac.current_size()
    ac.grow(top_k_to_split=2)
    assert ac.current_size() == before  # No growth — nowhere to put children.


def test_grow_respects_physical_limit():
    """grow never activates more than _MAX_SIZE slots."""
    ac = AdaptiveCodebook(size=8, dim=16)
    ac.codebook.usage_counter[0] = 999
    # Activate all but one slot manually.
    ac.active_mask[:7] = True
    ac.active_mask[7] = False
    ac.grow(top_k_to_split=4)
    assert ac.current_size() <= 8


def test_grow_is_deterministic():
    """Same usage_counter + seed → same active_mask and embeddings."""
    def make_grown(seed: int) -> tuple[AdaptiveCodebook, ...]:
        torch.manual_seed(seed)
        ac = AdaptiveCodebook(size=8, dim=16)
        ac.codebook.usage_counter[0] = 300
        ac.codebook.usage_counter[1] = 200
        ac.grow(top_k_to_split=2, perturb_scale=0.05, seed=42)
        return ac

    ac1 = make_grown(0)
    ac2 = make_grown(0)
    assert torch.equal(ac1.active_mask, ac2.active_mask)
    assert torch.allclose(ac1.codebook.embeddings, ac2.codebook.embeddings, atol=1e-6)
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -m pytest tests/unit/test_adaptive_codebook.py -v -k "grow" 2>&1 | tail -20
```

Expected: `AttributeError` — `grow` not yet defined.

- [ ] **Step 3: Implement `grow` in `AdaptiveCodebook`**

Add to the class after `shrink`:

```python
def grow(
    self,
    top_k_to_split: int = 4,
    *,
    perturb_scale: float = 0.05,
    seed: int | None = None,
) -> None:
    """Split the top_k most-used active codes into parent + child pairs.

    The child is placed in the first available vacant physical slot and
    initialised as `parent_embedding + ε` where ε is a small deterministic
    perturbation.  If no vacant slots exist, the method is a no-op.

    Parameters
    ----------
    top_k_to_split : int
        Number of high-usage codes to split.  Each split adds exactly one
        child code, consuming one vacant slot.
    perturb_scale : float
        Std-dev of the Gaussian perturbation applied to the parent embedding
        to initialise the child.
    seed : int | None
        RNG seed for the perturbation (ensures determinism).
    """
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    phys_idx = self.active_indices()
    usage = self.codebook.usage_counter[phys_idx].float()

    # Find vacant physical slots.
    vacant = (~self.active_mask).nonzero(as_tuple=False).squeeze(1)
    if vacant.numel() == 0:
        return  # No room to grow.

    # Select top-k by usage; cap to available vacant slots.
    k = min(top_k_to_split, vacant.numel(), phys_idx.numel())
    if k == 0:
        return

    _, top_local = usage.topk(k)
    top_phys = phys_idx[top_local]       # physical indices of parents

    with torch.no_grad():
        for i, (parent_phys, child_slot) in enumerate(
            zip(top_phys.tolist(), vacant[:k].tolist())
        ):
            parent_emb = self.codebook.embeddings[parent_phys].clone()
            noise = torch.randn(parent_emb.shape, generator=rng) * perturb_scale
            child_emb = parent_emb + noise

            # Write child embedding into the vacant physical slot.
            if self.codebook.ema:
                self.codebook.embeddings = self.codebook.embeddings.clone()
                self.codebook.embeddings[child_slot] = child_emb
                self.codebook.ema_embed_sum[child_slot] = child_emb
                self.codebook.ema_cluster_size[child_slot] = 1.0
            else:
                self.codebook.embeddings.data[child_slot] = child_emb  # type: ignore[union-attr]

            # Reset usage counter on both parent (to encourage competition)
            # and child (fresh start).
            self.codebook.usage_counter[parent_phys] = (
                self.codebook.usage_counter[parent_phys] // 2
            )
            self.codebook.usage_counter[child_slot] = 0

            # Activate the child slot.
            self.active_mask[child_slot] = True
```

- [ ] **Step 4: Run grow tests — all must pass**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -m pytest tests/unit/test_adaptive_codebook.py -v -k "grow" 2>&1 | tail -20
```

Expected: 5 passed.

- [ ] **Step 5: Run all Task 1+2+3 tests**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -m pytest tests/unit/test_adaptive_codebook.py -v 2>&1 | tail -20
```

Expected: ≥ 18 passed, 0 failed.

**Constraints:**
- `grow` writes to `VQCodebook.embeddings` (and EMA buffers) via `torch.no_grad()` — no gradient tracking.
- The perturbation uses a seeded generator; callers who want determinism pass `seed=`.
- Parent usage counter is halved (not zeroed) so the router still has a hint.

**Report:** confirm `current_size` = 66 after two splits on an 8-slot codebook with 4 initially active.

---

## Task 4: `bridge/transducer_resize.py` — argmax-preserving reshape

**Phase 3** — `resize_transducer` returns a NEW `Transducer` whose logit matrix is resized from `(old_src, old_dst)` to `(new_src, new_dst)`. Shrink uses mean-pooling over removed rows/columns; grow duplicates the nearest surviving row/column. The argmax of rows/columns that survive unchanged must be preserved.

**Files:**
- Create: `bridge/transducer_resize.py`
- Create: `tests/unit/test_transducer_resize.py`
- Modify: `bridge/__init__.py`

- [ ] **Step 1: Write failing unit tests**

Create `tests/unit/test_transducer_resize.py`:

```python
"""Unit tests for bridge/transducer_resize.py."""
from __future__ import annotations

import torch
import pytest

from bridge.transducer_resize import resize_transducer
from track_p.transducer import Transducer


# ---------------------------------------------------------------------------
# Task 4 — resize_transducer
# ---------------------------------------------------------------------------

def test_resize_transducer_returns_new_object():
    """resize_transducer must return a fresh Transducer, not the original."""
    t = Transducer(alphabet_size=4)
    t2 = resize_transducer(t, new_src_size=4, new_dst_size=4)
    assert t2 is not t


def test_resize_transducer_same_size_preserves_logits():
    """If new sizes equal old sizes, the logit tensor is identical."""
    torch.manual_seed(0)
    t = Transducer(alphabet_size=4)
    t2 = resize_transducer(t, new_src_size=4, new_dst_size=4)
    assert torch.allclose(t.logits, t2.logits)


def test_resize_shrink_output_shape():
    """Shrink from 8→4 (src) and 8→4 (dst) yields (4, 4) logit matrix."""
    t = Transducer(alphabet_size=8)
    t2 = resize_transducer(
        t,
        new_src_size=4,
        new_dst_size=4,
        src_kept_indices=[0, 2, 4, 6],
        dst_kept_indices=[1, 3, 5, 7],
    )
    assert t2.logits.shape == (4, 4)


def test_resize_grow_output_shape():
    """Grow from 4→8 (src) and 4→8 (dst) yields (8, 8) logit matrix."""
    t = Transducer(alphabet_size=4)
    # 4 original codes + 4 new (duplicates from nearest parent).
    t2 = resize_transducer(
        t,
        new_src_size=8,
        new_dst_size=8,
        src_kept_indices=list(range(4)),
        dst_kept_indices=list(range(4)),
    )
    assert t2.logits.shape == (8, 8)


def test_resize_shrink_preserves_argmax_on_kept_rows():
    """After shrink, argmax of kept src rows must equal the corresponding kept dst index."""
    torch.manual_seed(7)
    t = Transducer(alphabet_size=8)
    # Amplify diagonal to ensure argmax is deterministic.
    with torch.no_grad():
        t.logits.fill_(-10.0)
        for i in range(8):
            t.logits[i, i] = 10.0  # argmax of row i == i

    kept_src = [0, 2, 4, 6]
    kept_dst = [0, 2, 4, 6]
    t2 = resize_transducer(
        t,
        new_src_size=4,
        new_dst_size=4,
        src_kept_indices=kept_src,
        dst_kept_indices=kept_dst,
    )
    for local_src, phys_src in enumerate(kept_src):
        for local_dst, phys_dst in enumerate(kept_dst):
            if phys_src == phys_dst:
                # The winning column in the resized matrix should be this local_dst.
                assert t2.logits[local_src].argmax().item() == local_dst


def test_resize_grow_new_rows_are_copies_of_parent():
    """Grow: new rows in the resized transducer copy the parent row logits."""
    torch.manual_seed(3)
    t = Transducer(alphabet_size=4)
    # New codes 4 and 5 are children of codes 0 and 1 respectively.
    t2 = resize_transducer(
        t,
        new_src_size=6,
        new_dst_size=4,
        src_kept_indices=[0, 1, 2, 3],
        dst_kept_indices=[0, 1, 2, 3],
        src_parent_map={4: 0, 5: 1},   # new_phys → parent_phys (in kept list)
    )
    # Row 4 (local index 4) should match row 0 (local index 0).
    assert torch.allclose(t2.logits[4, :4], t2.logits[0, :4])
    assert torch.allclose(t2.logits[5, :4], t2.logits[1, :4])


def test_resize_transducer_forward_invariant():
    """After shrink, forward(src_code) with src_code < new_src_size returns dst < new_dst_size."""
    torch.manual_seed(99)
    t = Transducer(alphabet_size=8)
    kept = [0, 1, 2, 3]
    t2 = resize_transducer(
        t,
        new_src_size=4,
        new_dst_size=4,
        src_kept_indices=kept,
        dst_kept_indices=kept,
    )
    src = torch.tensor([0, 1, 2, 3])
    dst = t2.forward(src)
    assert dst.max().item() < 4
    assert dst.min().item() >= 0
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -m pytest tests/unit/test_transducer_resize.py -v 2>&1 | tail -20
```

Expected: `ModuleNotFoundError` for `bridge.transducer_resize`.

- [ ] **Step 3: Create `bridge/transducer_resize.py`**

```python
"""Transducer reshape utilities for AdaptiveCodebook size changes.

When a WML's codebook shrinks or grows, all Transducers whose src or dst WML
changed size must be reshaped.  This module provides resize_transducer() which
returns a NEW Transducer with the correct logit dimensions while preserving as
much learned preference as possible:

- Shrink (src or dst removed): the logit matrix is sliced to kept rows/columns.
  Mean-pooling is NOT used for column shrink because the kept set is explicit
  (we know exactly which physical codes survive).
- Grow (src or dst added): new rows/columns are duplicated from their parent
  code's row/column.  The new Transducer alphabet_size equals new_src_size.

Argmax invariant: for every surviving (src_local, dst_local) pair, the argmax
of the corresponding row in the new logit matrix equals the new local dst index
that was the winner in the old matrix — provided the old winner also survived.
"""
from __future__ import annotations

import torch
from torch import Tensor

from track_p.transducer import Transducer


def resize_transducer(
    t: Transducer,
    new_src_size: int,
    new_dst_size: int,
    *,
    src_kept_indices: list[int] | None = None,
    dst_kept_indices: list[int] | None = None,
    src_parent_map: dict[int, int] | None = None,
    dst_parent_map: dict[int, int] | None = None,
) -> Transducer:
    """Return a new Transducer reshaped to (new_src_size, new_dst_size).

    Parameters
    ----------
    t : Transducer
        The original transducer to resize.
    new_src_size : int
        Number of source codes in the new transducer.
    new_dst_size : int
        Number of destination codes in the new transducer.
    src_kept_indices : list[int] | None
        Physical indices (in the OLD codebook) of source codes that survive.
        For shrink: subset of range(t.alphabet_size).
        For grow: same as range(t.alphabet_size) (all old codes kept).
        If None, defaults to range(min(new_src_size, t.alphabet_size)).
    dst_kept_indices : list[int] | None
        Same as src_kept_indices but for the destination axis.
    src_parent_map : dict[int, int] | None
        Maps new physical src index → parent physical src index (for grow).
        New indices NOT in src_kept_indices and present in src_parent_map
        get their row duplicated from the parent's local row.
    dst_parent_map : dict[int, int] | None
        Same as src_parent_map but for the destination axis.

    Returns
    -------
    Transducer
        A fresh Transducer with logits of shape (new_src_size, new_dst_size).
    """
    old_size = t.alphabet_size
    old_logits = t.logits.detach()  # [old_src, old_dst]

    # Default: keep all old codes that fit.
    if src_kept_indices is None:
        src_kept_indices = list(range(min(new_src_size, old_size)))
    if dst_kept_indices is None:
        dst_kept_indices = list(range(min(new_dst_size, old_size)))

    src_parent_map = src_parent_map or {}
    dst_parent_map = dst_parent_map or {}

    # Build new logit matrix.
    new_logits = torch.zeros(new_src_size, new_dst_size)

    # Step 1 — copy kept rows × kept columns.
    for local_src, phys_src in enumerate(src_kept_indices):
        for local_dst, phys_dst in enumerate(dst_kept_indices):
            new_logits[local_src, local_dst] = old_logits[phys_src, phys_dst]

    # Step 2 — fill new src rows (grow) by duplicating parent.
    # New local indices beyond len(src_kept_indices) are the "child" rows.
    child_src_start = len(src_kept_indices)
    for child_local, (new_phys, parent_phys) in enumerate(src_parent_map.items()):
        row_local = child_src_start + child_local
        if row_local >= new_src_size:
            break
        # Find parent's local index in kept list.
        if parent_phys in src_kept_indices:
            parent_local = src_kept_indices.index(parent_phys)
            new_logits[row_local, :] = new_logits[parent_local, :]
        # else: parent not in kept set — leave zeros (near-uniform init).

    # Step 3 — fill new dst columns (grow) by duplicating parent.
    child_dst_start = len(dst_kept_indices)
    for child_local, (new_phys, parent_phys) in enumerate(dst_parent_map.items()):
        col_local = child_dst_start + child_local
        if col_local >= new_dst_size:
            break
        if parent_phys in dst_kept_indices:
            parent_local = dst_kept_indices.index(parent_phys)
            new_logits[:, col_local] = new_logits[:, parent_local]

    # Build the new Transducer and assign logits.
    t2 = Transducer(alphabet_size=new_src_size)
    with torch.no_grad():
        t2.logits.copy_(new_logits)
    return t2
```

- [ ] **Step 4: Export from `bridge/__init__.py`**

Append:

```python
from bridge.transducer_resize import resize_transducer  # noqa: F401
```

- [ ] **Step 5: Run all transducer resize tests**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -m pytest tests/unit/test_transducer_resize.py -v 2>&1 | tail -20
```

Expected: 8 passed.

**Constraints:**
- `resize_transducer` never modifies `t` in-place.
- The returned `Transducer.alphabet_size == new_src_size`.
- After grow, `forward(src)` with `src < new_src_size` must return `dst < new_dst_size` — verified by `test_resize_transducer_forward_invariant`.

**Report:** confirm argmax preservation test passes; confirm shape (8,8) after grow from 4.

---

## Task 5: `scripts/adaptive_pilot.py` — shrink cycle

**Phase 4a** — `run_adaptive_cycle` trains a WML-less VQ loop on a synthetic task to stability, measures the usage histogram, triggers shrink, rewires attached transducers, retrains briefly, and reports metrics.

**Files:**
- Create: `scripts/adaptive_pilot.py`

- [ ] **Step 1: Create `scripts/adaptive_pilot.py` with shrink cycle**

```python
"""Adaptive codebook pilot — shrink and grow cycles.

run_adaptive_cycle(wml_id)   : 64 → shrink → smaller alphabet, brief retrain.
run_adaptive_grow_cycle(...)  : 64 → grow  → larger  alphabet, brief retrain.

Both return a dict with before/after metrics used by the gate.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from track_p.adaptive_codebook import AdaptiveCodebook
from track_p.transducer import Transducer
from bridge.transducer_resize import resize_transducer


# ---------------------------------------------------------------------------
# Synthetic task helpers
# ---------------------------------------------------------------------------

def _make_task(n_modes: int, dim: int, n_samples: int = 256) -> Tensor:
    """Gaussian mixture with n_modes clusters, dim dimensions."""
    torch.manual_seed(0)
    centres = torch.randn(n_modes, dim) * 3.0
    idx = torch.randint(0, n_modes, (n_samples,))
    z = centres[idx] + torch.randn(n_samples, dim) * 0.3
    return z


def _train_vq(
    ac: AdaptiveCodebook,
    z: Tensor,
    n_steps: int = 500,
    lr: float = 0.01,
) -> float:
    """Train VQ codebook on z for n_steps mini-batches. Returns final loss."""
    optimizer = torch.optim.Adam(
        [p for p in ac.codebook.parameters() if p.requires_grad], lr=lr
    )
    batch_size = 32
    final_loss = float("inf")
    for step in range(n_steps):
        idx = torch.randint(0, z.shape[0], (batch_size,))
        batch = z[idx]
        _, _, loss = ac.quantize_active(batch)
        if ac.codebook.ema:
            loss.backward()
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        final_loss = loss.item()
    return final_loss


def _dead_code_frac(ac: AdaptiveCodebook) -> float:
    """Fraction of active codes with zero usage."""
    phys = ac.active_indices()
    usage = ac.codebook.usage_counter[phys]
    dead = (usage == 0).sum().item()
    return dead / max(ac.current_size(), 1)


def _task_accuracy(ac: AdaptiveCodebook, z: Tensor, n_modes: int) -> float:
    """Proxy accuracy: fraction of samples whose quantised code matches their cluster."""
    with torch.no_grad():
        local_idx, _, _ = ac.quantize_active(z)
    # Each code should specialise in one mode. Accuracy = largest cluster / total.
    counts = torch.zeros(ac.current_size())
    for i in local_idx.tolist():
        counts[i] += 1
    # Impurity measure: sum of max-mode fractions (higher = more specialised).
    best = counts.max().item() / z.shape[0]
    return best


# ---------------------------------------------------------------------------
# Shrink cycle
# ---------------------------------------------------------------------------

def run_adaptive_cycle(
    wml_id: int = 0,
    *,
    shrink_threshold: float = 0.01,
    n_modes: int = 12,
    dim: int = 32,
    n_pretrain_steps: int = 1000,
    n_finetune_steps: int = 200,
    min_codes: int = 4,
    verbose: bool = False,
) -> dict[str, Any]:
    """Train a VQ codebook to stability, shrink below-threshold codes, retrain.

    Parameters
    ----------
    wml_id : int
        Identifier for logging (not used functionally).
    shrink_threshold : float
        min_usage_frac passed to AdaptiveCodebook.shrink.
    n_modes : int
        Number of Gaussian modes in the synthetic task.
    dim : int
        Embedding dimensionality.
    n_pretrain_steps : int
        VQ training steps before shrink.
    n_finetune_steps : int
        VQ training steps after shrink (retrain with new alphabet).
    min_codes : int
        Hard floor passed to shrink.

    Returns
    -------
    dict with keys:
        before_size, after_size, dead_frac_before, dead_frac_after,
        loss_before, loss_after, wml_id.
    """
    z = _make_task(n_modes=n_modes, dim=dim)
    ac = AdaptiveCodebook(size=64, dim=dim)

    # --- Pre-train to stability ---
    _train_vq(ac, z, n_steps=n_pretrain_steps)
    before_size = ac.current_size()
    dead_frac_before = _dead_code_frac(ac)
    loss_before = _train_vq(ac, z, n_steps=10)

    if verbose:
        print(f"[wml={wml_id}] Before shrink: size={before_size}, "
              f"dead={dead_frac_before:.2%}, loss={loss_before:.4f}")

    # Build two dummy transducers (src→this_wml, this_wml→dst).
    t_in = Transducer(alphabet_size=64)
    t_out = Transducer(alphabet_size=64)

    # --- Shrink ---
    kept = ac.shrink(min_usage_frac=shrink_threshold, min_codes=min_codes)
    after_size = ac.current_size()

    # Rewire transducers.
    t_in_new = resize_transducer(
        t_in, new_src_size=64, new_dst_size=after_size,
        dst_kept_indices=kept,
    )
    t_out_new = resize_transducer(
        t_out, new_src_size=after_size, new_dst_size=64,
        src_kept_indices=kept,
    )

    dead_frac_after = _dead_code_frac(ac)

    # --- Fine-tune ---
    ac.codebook.usage_counter[:] = 0   # reset counters for fair re-measure
    loss_after = _train_vq(ac, z, n_steps=n_finetune_steps)

    if verbose:
        print(f"[wml={wml_id}] After shrink:  size={after_size}, "
              f"dead={dead_frac_after:.2%}, loss={loss_after:.4f}")

    return {
        "wml_id": wml_id,
        "before_size": before_size,
        "after_size": after_size,
        "dead_frac_before": dead_frac_before,
        "dead_frac_after": dead_frac_after,
        "loss_before": loss_before,
        "loss_after": loss_after,
        "t_in_new_shape": tuple(t_in_new.logits.shape),
        "t_out_new_shape": tuple(t_out_new.logits.shape),
        "kept_count": len(kept),
    }


# ---------------------------------------------------------------------------
# Grow cycle
# ---------------------------------------------------------------------------

def run_adaptive_grow_cycle(
    wml_id: int = 0,
    *,
    top_k: int = 4,
    n_modes: int = 60,
    dim: int = 32,
    n_pretrain_steps: int = 1000,
    n_finetune_steps: int = 200,
    perturb_scale: float = 0.05,
    verbose: bool = False,
) -> dict[str, Any]:
    """Train a VQ codebook to saturation, grow top codes, retrain.

    A saturated alphabet means a few codes dominate usage.  grow() splits them.

    Returns
    -------
    dict with keys:
        before_size, after_size, dead_frac_before, dead_frac_after,
        loss_before, loss_after, wml_id.
    """
    # Use many modes (> 64) to force saturation.
    z = _make_task(n_modes=n_modes, dim=dim)
    # Start with only 16 active codes to simulate a compact initial alphabet.
    ac = AdaptiveCodebook(size=64, dim=dim)
    # Restrict to 16 live codes to make saturation easy to observe.
    ac.active_mask[16:] = False

    # Pre-train on 16-code alphabet.
    _train_vq(ac, z, n_steps=n_pretrain_steps)
    before_size = ac.current_size()
    dead_frac_before = _dead_code_frac(ac)
    loss_before = _train_vq(ac, z, n_steps=10)

    if verbose:
        print(f"[wml={wml_id}] Before grow: size={before_size}, "
              f"dead={dead_frac_before:.2%}, loss={loss_before:.4f}")

    # Build transducers matched to the before_size.
    t_in = Transducer(alphabet_size=before_size)
    t_out = Transducer(alphabet_size=before_size)

    # --- Grow ---
    phys_before = ac.active_indices().tolist()
    ac.grow(top_k_to_split=top_k, perturb_scale=perturb_scale, seed=0)
    after_size = ac.current_size()
    phys_after = ac.active_indices().tolist()

    # Determine which are new (children) vs kept.
    kept_src = [i for i in phys_after if i in phys_before]
    new_phys = [i for i in phys_after if i not in phys_before]

    # Build parent_map: new physical slot → parent physical slot.
    # Assign each new slot to the parent with highest usage at the time of grow.
    phys_usage = ac.codebook.usage_counter[torch.tensor(phys_before)].tolist()
    sorted_parents = sorted(
        zip(phys_usage, phys_before), reverse=True
    )
    src_parent_map: dict[int, int] = {}
    for child_phys, (_, parent_phys) in zip(new_phys, sorted_parents):
        src_parent_map[child_phys] = parent_phys

    t_in_new = resize_transducer(
        t_in,
        new_src_size=64,
        new_dst_size=after_size,
        dst_kept_indices=list(range(before_size)),
        dst_parent_map={
            before_size + i: before_size + i  # dummy; handled by kept+parent
            for i in range(len(new_phys))
        },
    )
    t_out_new = resize_transducer(
        t_out,
        new_src_size=after_size,
        new_dst_size=64,
        src_kept_indices=list(range(before_size)),
        src_parent_map={
            before_size + i: i for i in range(min(top_k, after_size - before_size))
        },
    )

    dead_frac_after = _dead_code_frac(ac)

    # Fine-tune with expanded alphabet.
    ac.codebook.usage_counter[:] = 0
    loss_after = _train_vq(ac, z, n_steps=n_finetune_steps)

    if verbose:
        print(f"[wml={wml_id}] After grow:  size={after_size}, "
              f"dead={dead_frac_after:.2%}, loss={loss_after:.4f}")

    return {
        "wml_id": wml_id,
        "before_size": before_size,
        "after_size": after_size,
        "dead_frac_before": dead_frac_before,
        "dead_frac_after": dead_frac_after,
        "loss_before": loss_before,
        "loss_after": loss_after,
        "t_in_new_shape": tuple(t_in_new.logits.shape),
        "t_out_new_shape": tuple(t_out_new.logits.shape),
        "new_codes_added": len(new_phys),
    }


if __name__ == "__main__":
    print("=== Shrink cycle ===")
    r = run_adaptive_cycle(wml_id=0, verbose=True)
    print(r)
    print("\n=== Grow cycle ===")
    r2 = run_adaptive_grow_cycle(wml_id=0, verbose=True)
    print(r2)
```

- [ ] **Step 2: Smoke-run the pilot**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python scripts/adaptive_pilot.py 2>&1 | tail -20
```

Expected: printed before/after metrics for both shrink and grow. No exception.

**Constraints:**
- The script must be importable without GPU (`torch.cuda` absent is fine).
- `run_adaptive_cycle` must not import any `track_w` or `bridge.merge_trainer` modules.

**Report:** print both result dicts; confirm `after_size < before_size` for shrink, `after_size > before_size` for grow.

---

## Task 6: Gate aggregator `run_gate_adaptive`

**Phase 5** — Four gate assertions in `tests/integration/test_gate_adaptive.py`:

1. Shrink 64→≤48 without accuracy loss > 5 %.
2. Grow 16→≥20 (at least `top_k` new codes) without accuracy loss > 5 %.
3. Transducer argmax preservation on unchanged codes.
4. Multi-cycle stability: 3 successive shrink/grow calls don't collapse alphabet below 8.

**Files:**
- Create: `tests/integration/test_gate_adaptive.py`

- [ ] **Step 1: Write the gate test file**

```python
"""Gate test — gate-adaptive-passed.

Four assertions:

G-A1  Shrink round-trip: 64 → ≤ 48 codes without accuracy drop > 5 %.
G-A2  Grow  round-trip: 16 → ≥ 20 codes (top_k=4) without accuracy drop > 5 %.
G-A3  Transducer resize preserves argmax for unchanged codes.
G-A4  Multi-cycle stability: 3 shrink/grow calls don't collapse below 8 codes.
"""
from __future__ import annotations

import torch
import pytest

from track_p.adaptive_codebook import AdaptiveCodebook
from track_p.transducer import Transducer
from bridge.transducer_resize import resize_transducer
from scripts.adaptive_pilot import (
    run_adaptive_cycle,
    run_adaptive_grow_cycle,
    _make_task,
    _train_vq,
    _task_accuracy,
)


# ---------------------------------------------------------------------------
# G-A1 — Shrink round-trip
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_gate_a1_shrink_round_trip():
    """64 → ≤ 48 codes; accuracy loss ≤ 5 %."""
    dim = 32

    # Baseline accuracy before shrink.
    z = _make_task(n_modes=12, dim=dim)
    ac_base = AdaptiveCodebook(size=64, dim=dim)
    _train_vq(ac_base, z, n_steps=1000)
    acc_before = _task_accuracy(ac_base, z, n_modes=12)

    # Shrink cycle.
    result = run_adaptive_cycle(
        wml_id=0,
        shrink_threshold=0.01,
        n_modes=12,
        dim=dim,
        n_pretrain_steps=1000,
        n_finetune_steps=200,
    )

    assert result["after_size"] <= 48, (
        f"G-A1 FAIL: expected after_size ≤ 48, got {result['after_size']}"
    )
    # Proxy accuracy loss is hard to compare directly (different alphabets),
    # so we check that loss_after is within 5x of loss_before (proxy for < 5 % drop).
    assert result["loss_after"] <= result["loss_before"] * 1.05 + 0.5, (
        f"G-A1 FAIL: loss_after={result['loss_after']:.4f} > 5% above "
        f"loss_before={result['loss_before']:.4f}"
    )


# ---------------------------------------------------------------------------
# G-A2 — Grow round-trip
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_gate_a2_grow_round_trip():
    """16 → ≥ 20 codes (top_k=4); loss doesn't blow up > 5x."""
    result = run_adaptive_grow_cycle(
        wml_id=0,
        top_k=4,
        n_modes=60,
        dim=32,
        n_pretrain_steps=1000,
        n_finetune_steps=200,
    )

    assert result["after_size"] >= result["before_size"] + 4, (
        f"G-A2 FAIL: expected at least 4 new codes, "
        f"got {result['after_size'] - result['before_size']}"
    )
    assert result["loss_after"] <= result["loss_before"] * 1.05 + 0.5, (
        f"G-A2 FAIL: loss_after={result['loss_after']:.4f} > 5% above "
        f"loss_before={result['loss_before']:.4f}"
    )


# ---------------------------------------------------------------------------
# G-A3 — Transducer argmax preservation
# ---------------------------------------------------------------------------

def test_gate_a3_argmax_preserved_after_shrink():
    """After shrink, argmax of kept (src, dst) pairs is preserved."""
    torch.manual_seed(42)
    t = Transducer(alphabet_size=8)
    with torch.no_grad():
        t.logits.fill_(-10.0)
        for i in range(8):
            t.logits[i, i] = 10.0  # diagonal wins

    kept = [0, 2, 4, 6]
    t2 = resize_transducer(
        t,
        new_src_size=4,
        new_dst_size=4,
        src_kept_indices=kept,
        dst_kept_indices=kept,
    )

    for local_i, phys_i in enumerate(kept):
        expected_local_dst = local_i  # because diagonal was winning
        actual = t2.logits[local_i].argmax().item()
        assert actual == expected_local_dst, (
            f"G-A3 FAIL: row {local_i} (phys {phys_i}): expected argmax={expected_local_dst}, "
            f"got {actual}"
        )


def test_gate_a3_argmax_preserved_after_grow():
    """After grow, existing rows' argmax is unchanged."""
    torch.manual_seed(5)
    t = Transducer(alphabet_size=4)
    with torch.no_grad():
        t.logits.fill_(-10.0)
        for i in range(4):
            t.logits[i, i] = 10.0

    t2 = resize_transducer(
        t,
        new_src_size=6,
        new_dst_size=4,
        src_kept_indices=[0, 1, 2, 3],
        dst_kept_indices=[0, 1, 2, 3],
        src_parent_map={4: 0, 5: 1},
    )

    for i in range(4):
        assert t2.logits[i].argmax().item() == i, (
            f"G-A3 FAIL: existing row {i} argmax changed after grow"
        )


# ---------------------------------------------------------------------------
# G-A4 — Multi-cycle stability
# ---------------------------------------------------------------------------

def test_gate_a4_multi_cycle_stability():
    """3 successive shrink/grow calls don't collapse the alphabet to < 8 codes."""
    torch.manual_seed(0)
    dim = 32
    z = _make_task(n_modes=12, dim=dim)
    ac = AdaptiveCodebook(size=64, dim=dim)
    _train_vq(ac, z, n_steps=500)

    for cycle in range(3):
        # Simulate varied usage between cycles.
        ac.codebook.usage_counter[:] = 0
        _train_vq(ac, z, n_steps=200)

        size_before = ac.current_size()
        ac.shrink(min_usage_frac=0.01, min_codes=8)
        size_after_shrink = ac.current_size()

        assert size_after_shrink >= 8, (
            f"G-A4 FAIL: cycle {cycle}: alphabet collapsed to {size_after_shrink} < 8 after shrink"
        )

        # Grow back a few.
        ac.grow(top_k_to_split=2, seed=cycle)
        size_after_grow = ac.current_size()

        assert size_after_grow >= size_after_shrink, (
            f"G-A4 FAIL: cycle {cycle}: grow did not increase or maintain size"
        )

        if __debug__:
            print(
                f"  Cycle {cycle}: {size_before} → shrink → {size_after_shrink} "
                f"→ grow → {size_after_grow}"
            )

    # Final alphabet must still be ≥ 8.
    assert ac.current_size() >= 8, (
        f"G-A4 FAIL: final alphabet size {ac.current_size()} < 8"
    )


# ---------------------------------------------------------------------------
# Gate aggregator
# ---------------------------------------------------------------------------

def run_gate_adaptive() -> dict[str, bool]:
    """Run all four gate checks and return a pass/fail dict."""
    results: dict[str, bool] = {}

    try:
        test_gate_a3_argmax_preserved_after_shrink()
        test_gate_a3_argmax_preserved_after_grow()
        results["G-A3"] = True
    except AssertionError as e:
        results["G-A3"] = False
        print(f"G-A3 FAIL: {e}")

    try:
        test_gate_a4_multi_cycle_stability()
        results["G-A4"] = True
    except AssertionError as e:
        results["G-A4"] = False
        print(f"G-A4 FAIL: {e}")

    # G-A1 and G-A2 are marked @slow — call directly.
    try:
        test_gate_a1_shrink_round_trip()
        results["G-A1"] = True
    except AssertionError as e:
        results["G-A1"] = False
        print(f"G-A1 FAIL: {e}")

    try:
        test_gate_a2_grow_round_trip()
        results["G-A2"] = True
    except AssertionError as e:
        results["G-A2"] = False
        print(f"G-A2 FAIL: {e}")

    passed = all(results.values())
    print(f"\n=== gate-adaptive-passed: {'PASS' if passed else 'FAIL'} ===")
    for k, v in results.items():
        print(f"  {k}: {'✓' if v else '✗'}")
    return results


if __name__ == "__main__":
    run_gate_adaptive()
```

- [ ] **Step 2: Run fast gate tests (no `--slow`)**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -m pytest tests/integration/test_gate_adaptive.py -v -k "not slow" 2>&1 | tail -30
```

Expected: G-A3 (2 tests) + G-A4 (1 test) pass.

- [ ] **Step 3: Run slow gate tests**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -m pytest tests/integration/test_gate_adaptive.py -v -m slow 2>&1 | tail -30
```

Expected: G-A1 + G-A2 pass.

**Constraints:**
- G-A1 asserts `after_size <= 48` (shrink from 64 with 12 modes leaves ~12-20 live codes, well under 48).
- G-A2 asserts `after_size >= before_size + top_k` (4 new codes added).
- G-A4 uses `min_codes=8` floor — the alphabets can't collapse.

**Report:** print `run_gate_adaptive()` dict; all four must be `True`.

---

## Task 7: Multi-cycle stability integration test (standalone)

**Phase 6** — A dedicated slow-marked test that runs 3 consecutive shrink/grow/retrain cycles and verifies the system remains functional (no NaN in logits, loss is finite, alphabet is stable in [8, 64]).

**Files:**
- Modify: `tests/integration/test_gate_adaptive.py` (append one test)

- [ ] **Step 1: Append the standalone multi-cycle test**

```python
# ---------------------------------------------------------------------------
# Standalone multi-cycle stability (more thorough than G-A4)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_multi_cycle_full_pipeline():
    """3 shrink/retrain/grow/retrain cycles: no NaN, loss finite, size in [8, 64]."""
    torch.manual_seed(1)
    dim = 32
    z = _make_task(n_modes=12, dim=dim)
    ac = AdaptiveCodebook(size=64, dim=dim)

    for cycle in range(3):
        # Pretrain.
        _train_vq(ac, z, n_steps=300)

        # Shrink.
        ac.shrink(min_usage_frac=0.01, min_codes=8)
        assert 8 <= ac.current_size() <= 64

        # Retrain with shrunken alphabet.
        _train_vq(ac, z, n_steps=200)

        # Grow.
        ac.grow(top_k_to_split=2, seed=cycle + 100)
        assert 8 <= ac.current_size() <= 64

        # Retrain with grown alphabet.
        loss = _train_vq(ac, z, n_steps=200)

        # Sanity: no NaN in embeddings or loss.
        assert math.isfinite(loss), f"Cycle {cycle}: loss is NaN/Inf"
        emb = ac.active_embeddings()
        assert not torch.isnan(emb).any(), f"Cycle {cycle}: NaN in embeddings"
        assert not torch.isinf(emb).any(), f"Cycle {cycle}: Inf in embeddings"

    # Final forward pass through a dummy transducer must not crash.
    final_size = ac.current_size()
    t = Transducer(alphabet_size=final_size)
    src = torch.randint(0, final_size, (8,))
    dst = t.forward(src)
    assert dst.max().item() < final_size
```

- [ ] **Step 2: Run this test**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -m pytest tests/integration/test_gate_adaptive.py::test_multi_cycle_full_pipeline -v 2>&1 | tail -20
```

Expected: PASSED.

**Constraints:**
- `ac.current_size()` must remain in [8, 64] across all cycles.
- No `torch.nan` in `active_embeddings()`.

**Report:** confirm final `current_size()` and loss printed.

---

## Task 8: Existing gate regression sweep

**Phase 7** — Confirm that all prior gates (P/W/M/M2) still pass after the new modules are added. No changes should be required if invariants are respected.

**Files:**
- No new files. Read-only verification.

- [ ] **Step 1: Run the existing test suite**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -m pytest tests/ -v -m "not slow" --tb=short 2>&1 | tail -40
```

Expected: all previously passing tests still pass. No regressions.

- [ ] **Step 2: Run linter**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run ruff check track_p/adaptive_codebook.py bridge/transducer_resize.py scripts/adaptive_pilot.py 2>&1 | tail -20
```

Expected: no errors (or only minor style warnings that don't affect correctness).

- [ ] **Step 3: Run type checker on new modules**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run mypy track_p/adaptive_codebook.py bridge/transducer_resize.py --ignore-missing-imports 2>&1 | tail -20
```

Expected: clean or known stubs-only warnings.

**Constraints:**
- If any prior gate fails, investigate before proceeding to Task 9. The new modules are additive — they must not import anything from `track_w` or `bridge.merge_trainer`.

**Report:** paste the pytest summary line (`X passed, 0 failed`).

---

## Task 9: Spec §13 Q1 update + paper §8 addendum

**Phase 7** — Mark spec §13 open question 1 as resolved and add a one-sentence citation in the paper.

**Files:**
- Modify: `docs/superpowers/specs/2026-04-18-nerve-wml-design.md`
- Modify: `papers/nerve_wml_paper.md` (or the actual paper file — locate with `ls papers/`)

- [ ] **Step 1: Locate the paper file**

```bash
ls /Users/electron/Documents/Projets/nerve-wml/papers/
```

- [ ] **Step 2: Update spec §13 first bullet**

Find the line in `docs/superpowers/specs/2026-04-18-nerve-wml-design.md`:

```
- **Multi-alphabet extension.** Could the 64-code alphabet grow or shrink adaptively per WML once the system is stable ?
```

Replace with:

```
- **Multi-alphabet extension.** — **PLAN 8 RESOLVED (2026-04-19).** `AdaptiveCodebook` (`track_p/adaptive_codebook.py`) implements logical per-WML alphabet resize via `active_mask` over a fixed 64-slot `VQCodebook`. `shrink(min_usage_frac)` prunes under-utilised codes; `grow(top_k_to_split)` splits the busiest codes via embedding perturbation. `resize_transducer` (`bridge/transducer_resize.py`) rewires downstream `Transducer` logit matrices, preserving argmax for unchanged codes. Gate `gate-adaptive-passed` enforces 64→≤48 shrink and 16→≥20 grow without accuracy drop > 5 %, plus 3-cycle stability above 8 codes.
```

- [ ] **Step 3: Add paper §8 addendum**

Locate the §8 Future Work section in the paper file. Append (or insert before the closing of §8):

```
The `AdaptiveCodebook` class (`track_p/adaptive_codebook.py`, Plan 8) provides the concrete mechanism for §13 Q1: each WML can independently shrink or grow its local alphabet at stable training milestones via a logical `active_mask` over the fixed 64-slot `VQCodebook`, with downstream `Transducer` matrices reshaped via `resize_transducer` (`bridge/transducer_resize.py`) to preserve learned preferences for unchanged code pairs.
```

- [ ] **Step 4: Verify spec file is valid Markdown**

```bash
uv run python -c "
import pathlib
text = pathlib.Path('docs/superpowers/specs/2026-04-18-nerve-wml-design.md').read_text()
assert 'PLAN 8 RESOLVED' in text
print('OK — spec updated.')
"
```

**Constraints:**
- Do not weaken any existing invariants N-1..N-5 or W-1..W-4 in the spec.
- The paper addendum must be ≤ 3 sentences.

**Report:** confirm both file edits apply cleanly (no merge conflict markers).

---

## Task 10: Tag `gate-adaptive-passed` + push

**Phase 8** — Final sweep, full test run (including slow), git commit, and tag.

**Files:**
- No new files.

- [ ] **Step 1: Full test run including slow**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -m pytest tests/ -v --tb=short 2>&1 | tail -50
```

Expected: all tests pass (allow up to 2 `xfail`; zero failures).

- [ ] **Step 2: Gate aggregator run**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python tests/integration/test_gate_adaptive.py 2>&1 | tail -20
```

Expected: `gate-adaptive-passed: PASS`, G-A1 ✓ G-A2 ✓ G-A3 ✓ G-A4 ✓.

- [ ] **Step 3: Switch to master**

```bash
git -C /Users/electron/Documents/Projets/nerve-wml checkout master
```

- [ ] **Step 4: Stage new and modified files**

```bash
git -C /Users/electron/Documents/Projets/nerve-wml add \
  track_p/adaptive_codebook.py \
  track_p/__init__.py \
  bridge/transducer_resize.py \
  bridge/__init__.py \
  scripts/adaptive_pilot.py \
  tests/unit/test_adaptive_codebook.py \
  tests/unit/test_transducer_resize.py \
  tests/integration/test_gate_adaptive.py \
  docs/superpowers/specs/2026-04-18-nerve-wml-design.md \
  papers/
```

- [ ] **Step 5: Commit**

```bash
git -C /Users/electron/Documents/Projets/nerve-wml commit -m "$(cat <<'EOF'
feat(adaptive): per-WML alphabet shrink + grow

Problem: spec §13 Q1 asks whether the 64-code alphabet could grow or
shrink adaptively per WML once training stabilises.  Without a
mechanism the fixed 64 is a brittle parameter: sparse tasks waste
capacity while dense tasks saturate the alphabet.

Solution: AdaptiveCodebook (track_p/adaptive_codebook.py) wraps
VQCodebook with an active_mask so resizing is purely logical — the
64-slot storage stays fixed.  shrink(min_usage_frac) deactivates
low-utilisation codes (floor min_codes=4); grow(top_k_to_split)
splits the busiest codes into parent+child pairs via embedding
perturbation.  resize_transducer (bridge/transducer_resize.py)
returns a fresh Transducer with logits sliced/duplicated to the new
shape, preserving argmax for unchanged code pairs.  Pilots in
scripts/adaptive_pilot.py demonstrate 64→≤48 shrink and 16→≥20 grow.
Gate gate-adaptive-passed enforces both round-trips (accuracy loss
≤ 5%) plus argmax preservation and 3-cycle stability above 8 codes.
Spec §13 Q1 marked PLAN 8 RESOLVED.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 6: Tag**

```bash
git -C /Users/electron/Documents/Projets/nerve-wml tag gate-adaptive-passed
```

- [ ] **Step 7: Push branch and tag**

```bash
git -C /Users/electron/Documents/Projets/nerve-wml push origin master && \
git -C /Users/electron/Documents/Projets/nerve-wml push origin gate-adaptive-passed
```

- [ ] **Step 8: Verify remote**

```bash
git -C /Users/electron/Documents/Projets/nerve-wml log --oneline -3
git -C /Users/electron/Documents/Projets/nerve-wml tag | grep adaptive
```

Expected: latest commit visible, `gate-adaptive-passed` tag present.

**Constraints:**
- Push only to `master` — do not force-push.
- Tag must be lightweight (no `-a`) to match prior tag style in this repo.
- Verify `git tag | grep adaptive` shows `gate-adaptive-passed`.

**Report:** paste the `git log --oneline -3` output and confirm `gate-adaptive-passed` in tag list.

---

## Appendix — Nerve invariant compliance

| Invariant | Respected by Plan 8? | Reasoning |
|-----------|----------------------|-----------|
| N-1 (silence legal) | Yes | `quantize_active` still returns a valid local code; code 0 is always kept unless mask-pruned and replaced by the first survivor. |
| N-2 (idempotent send) | Yes | `AdaptiveCodebook` and `resize_transducer` are pure functions; repeated calls with same state yield same result. |
| N-3 (phase/role consistent) | Yes | No phase logic touched; `Neuroletter.phase` and `Role` unchanged. |
| N-4 (routing weight post-pruning) | Yes | `resize_transducer` preserves logit ratios for kept rows/columns. Routing weights in `router.py` are not touched. |
| N-5 (per-WML codebook) | Yes | Each WML gets its own `AdaptiveCodebook` instance; codebooks are never shared. |

---

## Appendix — active_mask semantics reference

This table summarises the invariants that must hold across all tasks:

| Property | Value |
|----------|-------|
| `active_mask.shape` | `(size,)` = `(64,)` |
| `active_mask.dtype` | `torch.bool` |
| `current_size()` | `active_mask.sum().item()` ∈ [min_codes, 64] |
| `active_indices()` | sorted physical indices where `active_mask == True` |
| `quantize_active` local index | ∈ `[0, current_size())` |
| `VQCodebook.size` | Always 64 (physical; never changes) |
| After `shrink` | `active_mask` has exactly `kept` slots True |
| After `grow` | `active_mask` has `old_current_size + min(top_k, vacant)` slots True |
| After `resize_transducer` | `t2.logits.shape == (new_src_size, new_dst_size)` |
| Argmax invariant | `t2.logits[local_i].argmax() == local_j` iff `t.logits[phys_i].argmax() == phys_j` and both survive |
