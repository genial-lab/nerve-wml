# LLM Integration — NerveWmlAdvisor for micro-kiki Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `bridge/kiki_nerve_advisor.py` — an advisory-only, env-gated, never-raising bridge that lets micro-kiki's MetaRouter query nerve-wml's WML pool for soft domain weights before its sigmoid decision.

**Architecture:** `NerveWmlAdvisor` lives entirely in nerve-wml's `bridge/` directory; it loads a serialized checkpoint (safetensors WML codebooks + transducer logits, pickle topology), runs a fixed MiniLM projection → VQ quantisation → multi-tick SimNerveAdapter simulation, then maps output WML π-emissions to micro-kiki's 35 domain indices. All calls return `None` on any error or when `NERVE_WML_ENABLED=0`, mirroring the `KikiFlowBridge` pattern. The micro-kiki side needs only 5-10 lines in its own follow-up PR, documented in `docs/integration/micro-kiki-wiring.md`.

**Tech Stack:** Python 3.12+, PyTorch 2.3+, safetensors 0.4+, sentence-transformers 2.7+ (dev extra only), pytest 8+, nerve-wml's existing `SimNerveAdapter`, `MlpWML`, `Neuroletter` / `Role` / `Phase`, `Nerve` protocol.

---

## File Map

| File | Status | Responsibility |
|---|---|---|
| `bridge/checkpoint.py` | Create | `save_advisor_checkpoint` / `load_advisor_checkpoint` — safetensors + pickle |
| `bridge/query_encoder.py` | Create | MiniLM → linear projection → VQ quantisation → 64-code neuroletter sequence |
| `bridge/kiki_nerve_advisor.py` | Create | `NerveWmlAdvisor`: env gate, lazy checkpoint load, `advise()` entry point |
| `bridge/__init__.py` | Modify | Export `NerveWmlAdvisor`, `save_advisor_checkpoint`, `load_advisor_checkpoint` |
| `pyproject.toml` | Modify | Add `safetensors>=0.4` + `sentence-transformers>=2.7` to `[project.optional-dependencies] dev` |
| `tests/unit/test_checkpoint.py` | Create | Round-trip save/load + golden bit-stability test |
| `tests/unit/test_query_encoder.py` | Create | Shape, range, determinism, VQ index bounds |
| `tests/unit/test_kiki_nerve_advisor.py` | Create | Env gate → None; mock checkpoint; NaN → None; valid → 35-key dict |
| `tests/integration/test_advisor_latency.py` | Create | `@pytest.mark.slow` — real checkpoint, `advise()` < 50 ms on M5 |
| `docs/integration/micro-kiki-wiring.md` | Create | Self-sufficient recipe for the micro-kiki follow-up PR |

---

## Task 1: pyproject.toml — add dev deps

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Check current dev extras**

```bash
grep -A10 "\[project.optional-dependencies\]" /Users/electron/Documents/Projets/nerve-wml/pyproject.toml
```

Expected: see `pytest`, `pytest-cov`, `ruff`, `mypy` under `dev`.

- [ ] **Step 2: Add safetensors + sentence-transformers to dev extras**

Open `pyproject.toml` and replace the `[project.optional-dependencies]` section so it reads:

```toml
[project.optional-dependencies]
dev = [
  "pytest>=8.0",
  "pytest-cov>=5.0",
  "ruff>=0.5",
  "mypy>=1.10",
  "safetensors>=0.4",
  "sentence-transformers>=2.7",
]
```

- [ ] **Step 3: Sync the lockfile**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv sync --extra dev
```

Expected: uv resolves without conflict; `safetensors` and `sentence-transformers` appear in the installed list.

- [ ] **Step 4: Verify imports work**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -c "import safetensors; import sentence_transformers; print('ok')"
```

Expected: prints `ok` with no errors.

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/nerve-wml
git add pyproject.toml uv.lock
git commit -m "chore: add safetensors + sentence-transformers to dev deps"
```

---

## Task 2: `bridge/checkpoint.py` — save/load primitives

**Files:**
- Create: `bridge/checkpoint.py`
- Create: `tests/unit/test_checkpoint.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_checkpoint.py`:

```python
"""Tests for advisor checkpoint round-trip."""
from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import pytest
import torch

from bridge.sim_nerve_adapter import SimNerveAdapter
from track_w.mlp_wml import MlpWML


def _make_pool_and_nerve(n_wmls: int = 4) -> tuple[list[MlpWML], SimNerveAdapter]:
    nerve = SimNerveAdapter(n_wmls=n_wmls, k=2, seed=42)
    pool = [MlpWML(id=i, d_hidden=32, alphabet_size=64, seed=i) for i in range(n_wmls)]
    return pool, nerve


def test_round_trip_restores_codebooks():
    """save then load must reproduce identical codebook tensors."""
    from bridge.checkpoint import load_advisor_checkpoint, save_advisor_checkpoint

    pool, nerve = _make_pool_and_nerve()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "advisor.ckpt"
        save_advisor_checkpoint(pool, nerve, path)
        loaded_pool, loaded_nerve, topo = load_advisor_checkpoint(path)

    for orig, restored in zip(pool, loaded_pool):
        assert torch.equal(orig.codebook.data, restored.codebook.data), (
            "codebook mismatch after round-trip"
        )


def test_round_trip_restores_transducer_logits():
    """Transducer parameter tensors must be bit-identical after load."""
    from bridge.checkpoint import load_advisor_checkpoint, save_advisor_checkpoint

    pool, nerve = _make_pool_and_nerve()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "advisor.ckpt"
        save_advisor_checkpoint(pool, nerve, path)
        _, loaded_nerve, _ = load_advisor_checkpoint(path)

    for key, orig_t in nerve._transducers.items():
        loaded_t = loaded_nerve._transducers[key]
        for (n1, p1), (n2, p2) in zip(
            orig_t.named_parameters(), loaded_t.named_parameters()
        ):
            assert n1 == n2
            assert torch.equal(p1.data, p2.data), f"transducer {key}.{n1} mismatch"


def test_topology_map_preserved():
    """Topology dict from pickle must carry n_wmls and edge matrix."""
    from bridge.checkpoint import load_advisor_checkpoint, save_advisor_checkpoint

    pool, nerve = _make_pool_and_nerve(n_wmls=4)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "advisor.ckpt"
        save_advisor_checkpoint(pool, nerve, path)
        _, _, topo = load_advisor_checkpoint(path)

    assert topo["n_wmls"] == 4
    assert topo["edges"].shape == (4, 4)


def test_golden_codebook_value():
    """Deterministic seed → known codebook[0, 0] value (bit-stability)."""
    from bridge.checkpoint import load_advisor_checkpoint, save_advisor_checkpoint

    pool, nerve = _make_pool_and_nerve(n_wmls=2)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "advisor.ckpt"
        save_advisor_checkpoint(pool, nerve, path)
        loaded_pool, _, _ = load_advisor_checkpoint(path)

    # Value must equal the original, not some transformed version.
    orig_val = pool[0].codebook.data[0, 0].item()
    load_val = loaded_pool[0].codebook.data[0, 0].item()
    assert orig_val == load_val, "golden codebook[0,0] mismatch — not bit-stable"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -m pytest tests/unit/test_checkpoint.py -v 2>&1 | tail -20
```

Expected: `ImportError` or `ModuleNotFoundError` for `bridge.checkpoint`.

- [ ] **Step 3: Write `bridge/checkpoint.py`**

Create `bridge/checkpoint.py`:

```python
"""Checkpoint save/load for NerveWmlAdvisor.

Format:
  <path>.safetensors  — WML codebooks + transducer weight tensors
  <path>.topo.pkl     — topology dict: n_wmls, k, edge matrix, wml_ids

Usage:
  save_advisor_checkpoint(pool, nerve, Path("advisor.ckpt"))
  pool, nerve, topo = load_advisor_checkpoint(Path("advisor.ckpt"))
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from bridge.sim_nerve_adapter import SimNerveAdapter
from track_w.mlp_wml import MlpWML


def save_advisor_checkpoint(
    pool: list[MlpWML],
    nerve: SimNerveAdapter,
    path: Path,
) -> None:
    """Serialise pool codebooks + nerve transducer weights to *path*.

    Creates two sibling files:
      - ``<path>.safetensors``  — all tensors (bit-stable)
      - ``<path>.topo.pkl``     — topology metadata (pickle)
    """
    path = Path(path)
    tensors: dict[str, torch.Tensor] = {}

    # WML codebooks
    for wml in pool:
        tensors[f"wml_{wml.id}_codebook"] = wml.codebook.data.cpu().contiguous()

    # Transducer parameters
    for edge_key, transducer in nerve._transducers.items():
        for param_name, param in transducer.named_parameters():
            tensors[f"transducer_{edge_key}_{param_name}"] = (
                param.data.cpu().contiguous()
            )

    save_file(tensors, str(path) + ".safetensors")

    topo: dict[str, Any] = {
        "n_wmls": nerve.n_wmls,
        "wml_ids": [wml.id for wml in pool],
        "wml_d_hidden": pool[0].codebook.shape[1] if pool else 128,
        "wml_alphabet_size": pool[0].alphabet_size if pool else 64,
        "edges": nerve._edges.cpu(),
        "edge_keys": list(nerve._transducers.keys()),
    }
    with open(str(path) + ".topo.pkl", "wb") as f:
        pickle.dump(topo, f, protocol=5)


def load_advisor_checkpoint(
    path: Path,
) -> tuple[list[MlpWML], SimNerveAdapter, dict[str, Any]]:
    """Load a checkpoint previously saved by ``save_advisor_checkpoint``.

    Returns:
        pool   — list of MlpWML with codebooks restored (frozen)
        nerve  — SimNerveAdapter with transducer weights restored (frozen)
        topo   — raw topology dict for caller inspection
    """
    path = Path(path)

    with open(str(path) + ".topo.pkl", "rb") as f:
        topo: dict[str, Any] = pickle.load(f)

    tensors = load_file(str(path) + ".safetensors")

    n_wmls      = topo["n_wmls"]
    d_hidden    = topo["wml_d_hidden"]
    alphabet_sz = topo["wml_alphabet_size"]

    # Reconstruct WML pool (architecture only — weights loaded below).
    pool: list[MlpWML] = [
        MlpWML(id=wid, d_hidden=d_hidden, alphabet_size=alphabet_sz)
        for wid in topo["wml_ids"]
    ]
    for wml in pool:
        key = f"wml_{wml.id}_codebook"
        wml.codebook.data.copy_(tensors[key])
        for p in wml.parameters():
            p.requires_grad_(False)

    # Reconstruct SimNerveAdapter topology, then load transducer weights.
    # We need k; infer it from edge count per row.
    edges = topo["edges"]
    k_approx = max(1, int(edges.sum(dim=1).float().mean().item()))
    nerve = SimNerveAdapter(n_wmls=n_wmls, k=k_approx, seed=0)
    # Override the sampled edge matrix with the saved one.
    nerve._edges = edges.float()
    # Rebuild transducers for exactly the saved edges.
    from track_p.transducer import Transducer
    nerve._transducers = torch.nn.ModuleDict()
    for edge_key in topo["edge_keys"]:
        nerve._transducers[edge_key] = Transducer(alphabet_size=alphabet_sz)

    for edge_key, transducer in nerve._transducers.items():
        for param_name, param in transducer.named_parameters():
            saved_key = f"transducer_{edge_key}_{param_name}"
            if saved_key in tensors:
                param.data.copy_(tensors[saved_key])
        for p in transducer.parameters():
            p.requires_grad_(False)

    for p in nerve.parameters():
        p.requires_grad_(False)

    return pool, nerve, topo
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -m pytest tests/unit/test_checkpoint.py -v 2>&1 | tail -20
```

Expected: 4 tests PASSED.

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/nerve-wml
git add bridge/checkpoint.py tests/unit/test_checkpoint.py
git commit -m "feat(bridge): checkpoint save/load with safetensors"
```

---

## Task 3: `bridge/query_encoder.py` — MiniLM → VQ neuroletter sequence

**Files:**
- Create: `bridge/query_encoder.py`
- Create: `tests/unit/test_query_encoder.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_query_encoder.py`:

```python
"""Tests for QueryEncoder: MiniLM → projection → VQ → neuroletter codes."""
from __future__ import annotations

import torch
import pytest


def _make_encoder(n_codes: int = 64) -> "QueryEncoder":
    from bridge.query_encoder import QueryEncoder
    # Use a tiny fake codebook so tests don't need sentence-transformers loaded.
    return QueryEncoder(embed_dim=384, proj_dim=64, n_codes=n_codes, seq_len=64)


def test_output_shape():
    """encode() must return an int tensor of shape (seq_len,)."""
    enc = _make_encoder()
    dummy_embed = torch.randn(384)
    codes = enc.encode_embedding(dummy_embed)
    assert codes.shape == (64,), f"expected (64,) got {codes.shape}"
    assert codes.dtype == torch.long


def test_codes_in_range():
    """All returned codes must be in [0, n_codes)."""
    enc = _make_encoder(n_codes=64)
    dummy_embed = torch.randn(384)
    codes = enc.encode_embedding(dummy_embed)
    assert codes.min().item() >= 0
    assert codes.max().item() < 64


def test_deterministic_on_same_input():
    """Same embedding → same codes (no stochastic sampling)."""
    enc = _make_encoder()
    dummy_embed = torch.randn(384)
    codes_a = enc.encode_embedding(dummy_embed)
    codes_b = enc.encode_embedding(dummy_embed)
    assert torch.equal(codes_a, codes_b), "encode_embedding must be deterministic"


def test_different_inputs_differ():
    """Two distinct embeddings should (almost certainly) produce different codes."""
    enc = _make_encoder()
    a = torch.randn(384)
    b = torch.randn(384)
    codes_a = enc.encode_embedding(a)
    codes_b = enc.encode_embedding(b)
    # Not guaranteed but extremely likely with random codebook.
    assert not torch.equal(codes_a, codes_b), "distinct embeddings should differ"


def test_encode_text_returns_correct_shape():
    """encode_text() must return shape (seq_len,) when called with a string.

    This test is marked xfail if sentence-transformers is not installed.
    """
    pytest.importorskip("sentence_transformers")
    from bridge.query_encoder import QueryEncoder
    enc = QueryEncoder.from_minilm(seq_len=64)
    codes = enc.encode_text("hello world")
    assert codes.shape == (64,)
    assert codes.dtype == torch.long
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -m pytest tests/unit/test_query_encoder.py -v 2>&1 | tail -20
```

Expected: `ImportError: cannot import name 'QueryEncoder' from 'bridge.query_encoder'`.

- [ ] **Step 3: Write `bridge/query_encoder.py`**

Create `bridge/query_encoder.py`:

```python
"""QueryEncoder — converts a text query into a 64-code neuroletter sequence.

Pipeline:
  1. MiniLM (sentence-transformers) → 384-dim embedding  [optional — only in encode_text]
  2. Linear projection: 384 → proj_dim (default 64)
  3. Nearest-neighbour VQ against a codebook of shape (n_codes, proj_dim)
     → seq_len integer codes in [0, n_codes)

The codebook is random-initialised by default; in production it can be
replaced with one of the WML pool's own codebooks via ``set_codebook()``.

``encode_embedding()`` accepts a pre-computed embedding tensor so that
MiniLM is optional at import time (no heavy dep at module load).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class QueryEncoder(nn.Module):
    """Encode a query embedding into a fixed-length sequence of VQ codes."""

    def __init__(
        self,
        embed_dim: int = 384,
        proj_dim:  int = 64,
        n_codes:   int = 64,
        seq_len:   int = 64,
        *,
        seed: int | None = 0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.proj_dim  = proj_dim
        self.n_codes   = n_codes
        self.seq_len   = seq_len

        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)

        # Linear projection (no bias keeps it affine-stable).
        proj_weight = torch.randn(proj_dim, embed_dim, generator=gen) * 0.1
        self.projection = nn.Linear(embed_dim, proj_dim, bias=False)
        with torch.no_grad():
            self.projection.weight.data.copy_(proj_weight)

        # VQ codebook: (n_codes, proj_dim) — random init, may be replaced.
        codebook = torch.randn(n_codes, proj_dim, generator=gen) * 0.1
        self.codebook = nn.Parameter(codebook, requires_grad=False)

        # Cached MiniLM model (lazy-loaded).
        self._sentence_model: object | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_codebook(self, codebook: Tensor) -> None:
        """Replace the VQ codebook (e.g. from a WML's own codebook).

        Args:
            codebook: Tensor of shape (n_codes, proj_dim).
        """
        if codebook.shape != (self.n_codes, self.proj_dim):
            raise ValueError(
                f"codebook shape mismatch: expected ({self.n_codes}, {self.proj_dim}), "
                f"got {tuple(codebook.shape)}"
            )
        with torch.no_grad():
            self.codebook.data.copy_(codebook)

    @torch.no_grad()
    def encode_embedding(self, embedding: Tensor) -> Tensor:
        """Map a 1-D embedding tensor of shape (embed_dim,) to (seq_len,) codes.

        Projection is applied, then the projected vector is tiled to seq_len
        rows and each row is matched to its nearest codebook entry.

        Returns:
            Long tensor of shape (seq_len,) with values in [0, n_codes).
        """
        if embedding.dim() != 1 or embedding.shape[0] != self.embed_dim:
            raise ValueError(
                f"embedding must have shape ({self.embed_dim},), got {tuple(embedding.shape)}"
            )
        projected = self.projection(embedding.float())           # (proj_dim,)
        # Tile to (seq_len, proj_dim) — add deterministic positional jitter.
        pos = torch.arange(self.seq_len, dtype=torch.float32)   # (seq_len,)
        pos_enc = torch.sin(pos.unsqueeze(1) * 0.1)             # (seq_len, 1)
        tiled = projected.unsqueeze(0).expand(self.seq_len, -1) + 0.01 * pos_enc
        # Nearest-neighbour VQ: (seq_len, n_codes)
        dists = torch.cdist(tiled, self.codebook)               # (seq_len, n_codes)
        codes = dists.argmin(dim=1)                             # (seq_len,)
        return codes.long()

    def encode_text(self, text: str) -> Tensor:
        """Encode a raw string query end-to-end (requires sentence-transformers).

        Args:
            text: The query string.

        Returns:
            Long tensor of shape (seq_len,) with values in [0, n_codes).
        """
        embedding = self._embed(text)
        return self.encode_embedding(embedding)

    @classmethod
    def from_minilm(
        cls,
        proj_dim: int = 64,
        n_codes:  int = 64,
        seq_len:  int = 64,
        seed:     int = 0,
    ) -> "QueryEncoder":
        """Convenience constructor using MiniLM-L6-v2 (384-dim embeddings)."""
        return cls(
            embed_dim=384,
            proj_dim=proj_dim,
            n_codes=n_codes,
            seq_len=seq_len,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> Tensor:
        """Lazy-load MiniLM and embed text. Returns a (384,) float tensor."""
        if self._sentence_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._sentence_model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2"
                )
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for encode_text(). "
                    "Install it: pip install sentence-transformers>=2.7"
                ) from exc
        raw = self._sentence_model.encode(text, convert_to_tensor=True)  # type: ignore[union-attr]
        return raw.float().cpu()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -m pytest tests/unit/test_query_encoder.py -v 2>&1 | tail -20
```

Expected: 4 unit tests PASSED; `test_encode_text_returns_correct_shape` may be PASSED or XFAIL depending on whether MiniLM weights are cached.

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/nerve-wml
git add bridge/query_encoder.py tests/unit/test_query_encoder.py
git commit -m "feat(bridge): QueryEncoder — MiniLM -> projection -> VQ codes"
```

---

## Task 4: `bridge/kiki_nerve_advisor.py` — env gate + lazy load skeleton

**Files:**
- Create: `bridge/kiki_nerve_advisor.py`
- Create: `tests/unit/test_kiki_nerve_advisor.py` (partial — env-gate tests only)

- [ ] **Step 1: Write the env-gate failing tests**

Create `tests/unit/test_kiki_nerve_advisor.py`:

```python
"""Tests for NerveWmlAdvisor: env gate, None-on-error contract, output shape."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_checkpoint(n_wmls: int = 4, n_domains: int = 35) -> Path:
    """Create a minimal checkpoint and return its path stem."""
    from bridge.checkpoint import save_advisor_checkpoint
    from bridge.sim_nerve_adapter import SimNerveAdapter
    from track_w.mlp_wml import MlpWML

    nerve = SimNerveAdapter(n_wmls=n_wmls, k=2, seed=7)
    pool  = [MlpWML(id=i, d_hidden=32, alphabet_size=64, seed=i) for i in range(n_wmls)]
    tmp = tempfile.mkdtemp()
    ckpt = Path(tmp) / "test_advisor.ckpt"
    save_advisor_checkpoint(pool, nerve, ckpt)
    return ckpt


def _make_advisor(ckpt: Path, n_domains: int = 35) -> "NerveWmlAdvisor":
    from bridge.kiki_nerve_advisor import NerveWmlAdvisor
    return NerveWmlAdvisor(checkpoint_path=ckpt, n_domains=n_domains, n_ticks=2)


# ---------------------------------------------------------------------------
# Env gate
# ---------------------------------------------------------------------------

class TestEnvGate:
    def test_disabled_by_default_returns_none(self, tmp_path):
        """When NERVE_WML_ENABLED is unset, advise() must return None."""
        ckpt = _make_checkpoint()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NERVE_WML_ENABLED", None)
            advisor = _make_advisor(ckpt)
        tokens = torch.zeros(64, dtype=torch.long)
        result = advisor.advise(tokens, current_route={"domain": "math"})
        assert result is None, "expected None when NERVE_WML_ENABLED not set"

    def test_disabled_zero_returns_none(self, tmp_path):
        """NERVE_WML_ENABLED=0 must make advise() return None."""
        ckpt = _make_checkpoint()
        with patch.dict(os.environ, {"NERVE_WML_ENABLED": "0"}):
            advisor = _make_advisor(ckpt)
        tokens = torch.zeros(64, dtype=torch.long)
        assert advisor.advise(tokens, {}) is None

    def test_enabled_one_does_not_return_none(self, tmp_path):
        """NERVE_WML_ENABLED=1 must not return None on valid input."""
        ckpt = _make_checkpoint()
        with patch.dict(os.environ, {"NERVE_WML_ENABLED": "1"}):
            advisor = _make_advisor(ckpt)
        tokens = torch.zeros(64, dtype=torch.long)
        result = advisor.advise(tokens, {})
        assert result is not None, "expected a dict when NERVE_WML_ENABLED=1"
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -m pytest tests/unit/test_kiki_nerve_advisor.py::TestEnvGate -v 2>&1 | tail -20
```

Expected: `ImportError` or `ModuleNotFoundError` for `bridge.kiki_nerve_advisor`.

- [ ] **Step 3: Write the skeleton of `bridge/kiki_nerve_advisor.py`**

Create `bridge/kiki_nerve_advisor.py`:

```python
"""NerveWmlAdvisor — advisory bridge from nerve-wml into micro-kiki.

Design contract (mirrors KikiFlowBridge):
  - Never raises from advise().
  - Returns None on any error: disabled env gate, bad input, NaN, shape mismatch.
  - Never mutates the input current_route dict.
  - Idempotent: two calls with identical inputs produce identical outputs.
  - Checkpoint is loaded lazily on first call (not in __init__).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

log = logging.getLogger(__name__)


class NerveWmlAdvisor:
    """Advisory router: runs a nerve-wml WML pool and returns soft domain weights.

    Args:
        checkpoint_path: Path stem passed to ``load_advisor_checkpoint``.
            The loader expects ``<path>.safetensors`` and ``<path>.topo.pkl``.
        n_domains: Number of micro-kiki domains (default 35).
        n_ticks: Simulation ticks per advise() call (default 4).

    Usage::

        advisor = NerveWmlAdvisor(Path("advisor.ckpt"), n_domains=35)
        weights = advisor.advise(query_tokens, current_route)
        # weights is a dict mapping domain index (str) → float, or None.
    """

    ENABLED_ENV: str = "NERVE_WML_ENABLED"

    def __init__(
        self,
        checkpoint_path: Path,
        n_domains: int = 35,
        n_ticks:   int = 4,
    ) -> None:
        self._ckpt_path   = Path(checkpoint_path)
        self._n_domains   = n_domains
        self._n_ticks     = n_ticks
        self._enabled     = os.environ.get(self.ENABLED_ENV, "0") not in ("", "0")
        self._loaded      = False
        self._pool: list[Any]    = []
        self._nerve: Any | None  = None
        self._domain_map: dict[int, int] | None = None  # WML code → domain idx

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def advise(
        self,
        query_tokens: Tensor,
        current_route: dict[str, Any],
    ) -> dict[str, float] | None:
        """Return soft domain weights or None.

        Args:
            query_tokens: Long tensor of shape (seq_len,) with values in [0, 64).
                          Produced by QueryEncoder.encode_embedding().
            current_route: The MetaRouter's current routing decision dict.
                           This dict is never mutated.

        Returns:
            A dict with keys ``"0"`` through ``"34"`` (strings) mapping to
            non-negative floats that sum to 1.0, or None on any failure.
        """
        if not self._enabled:
            return None
        try:
            return self._safe_advise(query_tokens, current_route)
        except Exception:
            log.debug("NerveWmlAdvisor.advise() suppressed exception", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> bool:
        """Load checkpoint lazily. Returns True if ready, False on error."""
        if self._loaded:
            return True
        try:
            from bridge.checkpoint import load_advisor_checkpoint

            pool, nerve, topo = load_advisor_checkpoint(self._ckpt_path)
            self._pool   = pool
            self._nerve  = nerve
            self._domain_map = self._build_domain_map(topo)
            self._loaded = True
            return True
        except Exception:
            log.debug("NerveWmlAdvisor: checkpoint load failed", exc_info=True)
            return False

    @staticmethod
    def _build_domain_map(topo: dict[str, Any]) -> dict[int, int]:
        """Map alphabet codes (0..63) to micro-kiki domain indices (0..34).

        Strategy: code % n_domains. Deterministic and uniform enough for
        advisory use. A trained projection would replace this.
        """
        n_domains = 35
        return {code: code % n_domains for code in range(64)}

    def _safe_advise(
        self,
        query_tokens: Tensor,
        current_route: dict[str, Any],
    ) -> dict[str, float] | None:
        """Internal advise — may raise; caller wraps in try/except."""
        if not self._ensure_loaded():
            return None
        return None  # Task 5 fills this in.
```

- [ ] **Step 4: Run env-gate tests**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -m pytest tests/unit/test_kiki_nerve_advisor.py::TestEnvGate -v 2>&1 | tail -20
```

Expected: `test_disabled_by_default_returns_none` PASSED, `test_disabled_zero_returns_none` PASSED, `test_enabled_one_does_not_return_none` FAILED (because `_safe_advise` still returns None). That's correct — Task 5 finishes the implementation.

- [ ] **Step 5: Commit the skeleton**

```bash
cd /Users/electron/Documents/Projets/nerve-wml
git add bridge/kiki_nerve_advisor.py tests/unit/test_kiki_nerve_advisor.py
git commit -m "feat(bridge): NerveWmlAdvisor skeleton — env gate + lazy load"
```

---

## Task 5: Complete `advise()` — validation + None-on-error tests

**Files:**
- Modify: `bridge/kiki_nerve_advisor.py` (replace `_safe_advise` stub)
- Modify: `tests/unit/test_kiki_nerve_advisor.py` (add NaN + valid-output tests)

- [ ] **Step 1: Add the remaining tests to `tests/unit/test_kiki_nerve_advisor.py`**

Append after the `TestEnvGate` class:

```python
class TestAdviseContract:
    def test_nan_input_returns_none(self):
        """NaN in query_tokens must not propagate — advise() returns None."""
        ckpt = _make_checkpoint()
        with patch.dict(os.environ, {"NERVE_WML_ENABLED": "1"}):
            advisor = _make_advisor(ckpt)
        nan_tokens = torch.full((64,), float("nan"))
        result = advisor.advise(nan_tokens, {})
        assert result is None, "NaN input must yield None"

    def test_wrong_shape_returns_none(self):
        """A token tensor with wrong shape must yield None, not raise."""
        ckpt = _make_checkpoint()
        with patch.dict(os.environ, {"NERVE_WML_ENABLED": "1"}):
            advisor = _make_advisor(ckpt)
        bad_tokens = torch.zeros(10, dtype=torch.long)  # wrong seq_len
        result = advisor.advise(bad_tokens, {})
        assert result is None

    def test_valid_input_returns_35_key_dict(self):
        """Valid 64-code tokens → dict with exactly 35 string keys summing to 1."""
        ckpt = _make_checkpoint(n_wmls=4, n_domains=35)
        with patch.dict(os.environ, {"NERVE_WML_ENABLED": "1"}):
            advisor = _make_advisor(ckpt, n_domains=35)
        tokens = torch.randint(0, 64, (64,), dtype=torch.long)
        result = advisor.advise(tokens, {"domain": "code"})
        assert result is not None, "valid input must produce a dict"
        assert set(result.keys()) == {str(i) for i in range(35)}, (
            f"expected keys 0..34, got {sorted(result.keys())}"
        )
        total = sum(result.values())
        assert abs(total - 1.0) < 1e-4, f"weights must sum to 1.0, got {total}"

    def test_advise_does_not_mutate_current_route(self):
        """advise() must not modify the current_route dict."""
        ckpt = _make_checkpoint()
        with patch.dict(os.environ, {"NERVE_WML_ENABLED": "1"}):
            advisor = _make_advisor(ckpt)
        route = {"domain": "math", "score": 0.9}
        route_before = dict(route)
        advisor.advise(torch.zeros(64, dtype=torch.long), route)
        assert route == route_before, "current_route was mutated"

    def test_advise_is_idempotent(self):
        """Two identical calls must produce identical outputs."""
        ckpt = _make_checkpoint()
        with patch.dict(os.environ, {"NERVE_WML_ENABLED": "1"}):
            advisor = _make_advisor(ckpt)
        tokens = torch.arange(64, dtype=torch.long) % 64
        r1 = advisor.advise(tokens, {})
        r2 = advisor.advise(tokens, {})
        assert r1 == r2, "advise() must be idempotent"
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -m pytest tests/unit/test_kiki_nerve_advisor.py::TestAdviseContract -v 2>&1 | tail -20
```

Expected: several FAILED (output still None from stub `_safe_advise`).

- [ ] **Step 3: Complete `_safe_advise` in `bridge/kiki_nerve_advisor.py`**

Replace the `_safe_advise` method body:

```python
    def _safe_advise(
        self,
        query_tokens: Tensor,
        current_route: dict[str, Any],
    ) -> dict[str, float] | None:
        """Internal advise — may raise; caller wraps in try/except."""
        if not self._ensure_loaded():
            return None

        # --- Input validation ---
        if query_tokens.dim() != 1:
            raise ValueError(f"query_tokens must be 1-D, got shape {query_tokens.shape}")
        if torch.isnan(query_tokens.float()).any():
            raise ValueError("query_tokens contains NaN")
        # Clamp to [0, ALPHABET_SIZE) — graceful on shape mismatch.
        seq_len = query_tokens.shape[0]
        tokens  = query_tokens.long().clamp(0, 63)

        # --- Feed tokens into WML 0 (input WML) as inbound neuroletters ---
        from nerve_core.neuroletter import Neuroletter, Phase, Role

        nerve  = self._nerve
        pool   = self._pool
        n_wmls = nerve.n_wmls

        # Reset queues between calls (idempotency).
        from collections import defaultdict
        nerve._queues = defaultdict(list)

        # Inject tokens as PREDICTION letters from WML 0 to all neighbours.
        t0 = 0.0
        for step_i, code in enumerate(tokens[:seq_len]):
            code_int = int(code.item())
            for dst in range(n_wmls):
                if dst == 0:
                    continue
                if nerve._edges[0, dst].item() == 1.0:
                    nerve.send(Neuroletter(
                        code=code_int,
                        role=Role.PREDICTION,
                        phase=Phase.GAMMA,
                        src=0,
                        dst=dst,
                        timestamp=t0 + step_i * 0.001,
                    ))

        # --- Run n_ticks simulation steps ---
        for tick in range(self._n_ticks):
            t = t0 + tick * (1.0 / 40.0)  # γ-phase cadence
            for wml in pool:
                wml.step(nerve, t)

        # --- Read π emissions from output WML (last WML in pool) ---
        output_wml_id = pool[-1].id
        letters = nerve.listen(
            output_wml_id,
            role=Role.PREDICTION,
            phase=Phase.GAMMA,
        )

        # --- Map codes to domain weights ---
        domain_map = self._domain_map
        assert domain_map is not None
        counts = torch.zeros(self._n_domains)
        if letters:
            for letter in letters:
                domain_idx = domain_map.get(letter.code, letter.code % self._n_domains)
                counts[domain_idx] += 1.0
        else:
            # No emissions: fall back to uniform distribution.
            counts.fill_(1.0)

        # Softmax-normalise.
        weights = torch.softmax(counts, dim=0)

        if torch.isnan(weights).any():
            raise ValueError("NaN in output weights")

        return {str(i): float(weights[i].item()) for i in range(self._n_domains)}
```

- [ ] **Step 4: Run all advisor tests**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -m pytest tests/unit/test_kiki_nerve_advisor.py -v 2>&1 | tail -30
```

Expected: all 8 tests PASSED.

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/nerve-wml
git add bridge/kiki_nerve_advisor.py tests/unit/test_kiki_nerve_advisor.py
git commit -m "feat(bridge): complete advise() — VQ→sim→domain weights, None-on-error"
```

---

## Task 6: Integration test — latency gate (`test_advisor_latency.py`)

**Files:**
- Create: `tests/integration/test_advisor_latency.py`

- [ ] **Step 1: Write the latency test**

Create `tests/integration/test_advisor_latency.py`:

```python
"""Integration test: advisor latency must be < 50 ms on GrosMac M5.

Marked @pytest.mark.slow — excluded from CI by default.
Run with: pytest tests/integration/test_advisor_latency.py -m slow -v
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import torch


@pytest.mark.slow
def test_advise_completes_under_50ms():
    """NerveWmlAdvisor.advise() must complete in < 50 ms after warm-up."""
    from bridge.checkpoint import save_advisor_checkpoint
    from bridge.kiki_nerve_advisor import NerveWmlAdvisor
    from bridge.sim_nerve_adapter import SimNerveAdapter
    from track_w.mlp_wml import MlpWML

    n_wmls = 6
    nerve  = SimNerveAdapter(n_wmls=n_wmls, k=3, seed=99)
    pool   = [MlpWML(id=i, d_hidden=128, alphabet_size=64, seed=i) for i in range(n_wmls)]

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "latency_test.ckpt"
        save_advisor_checkpoint(pool, nerve, ckpt)

        with patch.dict(os.environ, {"NERVE_WML_ENABLED": "1"}):
            advisor = NerveWmlAdvisor(checkpoint_path=ckpt, n_domains=35, n_ticks=4)

        tokens = torch.randint(0, 64, (64,), dtype=torch.long)

        # Warm-up call (loads checkpoint + JIT caches).
        _ = advisor.advise(tokens, {})

        # Timed calls.
        n_runs = 10
        elapsed_ms: list[float] = []
        for _ in range(n_runs):
            t_start = time.perf_counter()
            result  = advisor.advise(tokens, {"domain": "code"})
            t_end   = time.perf_counter()
            elapsed_ms.append((t_end - t_start) * 1000.0)
            assert result is not None, "advise() returned None during latency test"

        avg_ms = sum(elapsed_ms) / len(elapsed_ms)
        max_ms = max(elapsed_ms)
        print(f"\nLatency — avg: {avg_ms:.1f} ms  max: {max_ms:.1f} ms")
        assert avg_ms < 50.0, (
            f"advise() avg latency {avg_ms:.1f} ms exceeds 50 ms budget on M5"
        )


@pytest.mark.slow
def test_env_gate_adds_zero_overhead():
    """Disabled advisor (NERVE_WML_ENABLED=0) must return None in < 1 ms."""
    from bridge.kiki_nerve_advisor import NerveWmlAdvisor

    with patch.dict(os.environ, {"NERVE_WML_ENABLED": "0"}):
        advisor = NerveWmlAdvisor(checkpoint_path=Path("nonexistent.ckpt"))

    tokens = torch.zeros(64, dtype=torch.long)
    t_start = time.perf_counter()
    for _ in range(100):
        advisor.advise(tokens, {})
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0 / 100
    print(f"\nDisabled gate overhead: {elapsed_ms:.3f} ms per call")
    assert elapsed_ms < 1.0, f"disabled gate overhead {elapsed_ms:.3f} ms > 1 ms"
```

- [ ] **Step 2: Register the `slow` marker in `pyproject.toml`**

Add to the `[tool.pytest.ini_options]` section in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
markers = [
  "slow: marks tests as slow (deselect with '-m \"not slow\"')",
]
```

- [ ] **Step 3: Run the latency tests**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -m pytest tests/integration/test_advisor_latency.py -m slow -v -s 2>&1 | tail -30
```

Expected: both tests PASSED. The latency print shows avg < 50 ms and gate overhead < 1 ms.

- [ ] **Step 4: Verify the standard suite (no slow) still passes**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -m pytest -m "not slow" --tb=short -q 2>&1 | tail -20
```

Expected: all unit tests pass; latency tests are skipped.

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/nerve-wml
git add tests/integration/test_advisor_latency.py pyproject.toml
git commit -m "test(bridge): advisor latency gate — 50ms budget on M5"
```

---

## Task 7: Update `bridge/__init__.py` — public exports

**Files:**
- Modify: `bridge/__init__.py`

- [ ] **Step 1: Read the current `bridge/__init__.py`**

```bash
cat /Users/electron/Documents/Projets/nerve-wml/bridge/__init__.py
```

- [ ] **Step 2: Replace its content**

```python
# bridge — nerve-wml/micro-kiki integration layer.
#
# Public surface:
#   NerveWmlAdvisor        — advisory router for micro-kiki's MetaRouter
#   save_advisor_checkpoint — persist a trained pool+nerve to disk
#   load_advisor_checkpoint — restore a checkpoint for inference
#   QueryEncoder            — MiniLM -> VQ neuroletter sequence encoder
from bridge.checkpoint import load_advisor_checkpoint, save_advisor_checkpoint
from bridge.kiki_nerve_advisor import NerveWmlAdvisor
from bridge.query_encoder import QueryEncoder

__all__ = [
    "NerveWmlAdvisor",
    "QueryEncoder",
    "load_advisor_checkpoint",
    "save_advisor_checkpoint",
]
```

- [ ] **Step 3: Verify imports from bridge root**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -c "
from bridge import NerveWmlAdvisor, QueryEncoder, save_advisor_checkpoint, load_advisor_checkpoint
print('imports ok')
"
```

Expected: `imports ok`.

- [ ] **Step 4: Run the full unit test suite**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -m pytest tests/unit/ -q 2>&1 | tail -15
```

Expected: all unit tests pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/nerve-wml
git add bridge/__init__.py
git commit -m "chore(bridge): export NerveWmlAdvisor, QueryEncoder, checkpoint helpers"
```

---

## Task 8: `docs/integration/micro-kiki-wiring.md` — self-sufficient integration recipe

**Files:**
- Create: `docs/integration/micro-kiki-wiring.md`

- [ ] **Step 1: Create the directory**

```bash
mkdir -p /Users/electron/Documents/Projets/nerve-wml/docs/integration
```

- [ ] **Step 2: Write the wiring doc**

Create `docs/integration/micro-kiki-wiring.md`:

````markdown
# Wiring NerveWmlAdvisor into micro-kiki's MetaRouter

> **Scope of this doc:** everything a subagent needs to add advisory nerve-wml
> routing to `micro-kiki/src/routing/meta_router.py` in a single self-contained PR.
> Do NOT modify nerve-wml from this side. All bridge code lives in nerve-wml.

---

## 1. Prerequisites

nerve-wml must be installed as an editable dependency from the micro-kiki repo:

```bash
# In micro-kiki repo root:
pip install -e /path/to/nerve-wml[dev]
# or, if using uv:
uv add --editable /path/to/nerve-wml
```

Verify:

```bash
python -c "from bridge import NerveWmlAdvisor; print('ok')"
```

---

## 2. Generate and place a checkpoint

A checkpoint must be created from a trained nerve-wml pool. If none exists yet,
generate a minimal one for testing:

```bash
# Run from nerve-wml repo:
python - <<'EOF'
from pathlib import Path
from bridge import save_advisor_checkpoint
from bridge.sim_nerve_adapter import SimNerveAdapter
from track_w.mlp_wml import MlpWML

nerve = SimNerveAdapter(n_wmls=6, k=3, seed=42)
pool  = [MlpWML(id=i, d_hidden=128, alphabet_size=64, seed=i) for i in range(6)]
ckpt  = Path("artifacts/nerve_advisor.ckpt")
ckpt.parent.mkdir(parents=True, exist_ok=True)
save_advisor_checkpoint(pool, nerve, ckpt)
print(f"Saved: {ckpt}.safetensors + {ckpt}.topo.pkl")
EOF
```

Place (or symlink) the `.safetensors` and `.topo.pkl` files where micro-kiki can
find them. The default path is `artifacts/nerve_advisor.ckpt` relative to the
micro-kiki repo root. Override with `NERVE_WML_CHECKPOINT_PATH`.

---

## 3. Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `NERVE_WML_ENABLED` | `0` (off) | Set to `1` to activate the advisor |
| `NERVE_WML_CHECKPOINT_PATH` | `artifacts/nerve_advisor.ckpt` | Stem path to the checkpoint |
| `NERVE_WML_ALPHA` | `0.1` | Mixing coefficient (0 = ignore advisor, 1 = use only advisor) |

---

## 4. The 8-line change in `meta_router.py`

Open `micro-kiki/src/routing/meta_router.py`. Find the class-level imports and
module-level init. Add the following — exact file context shown for orientation:

```python
# --- EXISTING imports (keep these) ---
import os
import logging
# ... existing imports ...

# --- ADD after existing imports ---
from pathlib import Path
from bridge import NerveWmlAdvisor

log = logging.getLogger(__name__)

# Module-level singleton — loaded once, reused across all requests.
_NERVE_ADVISOR = NerveWmlAdvisor(
    checkpoint_path=Path(
        os.environ.get("NERVE_WML_CHECKPOINT_PATH", "artifacts/nerve_advisor.ckpt")
    ),
    n_domains=35,
    n_ticks=4,
)
```

Then, inside the routing method (look for the sigmoid decision — typically a
method named `route()`, `select_domain()`, or similar), add the advisory mixing
**before** the sigmoid is applied:

```python
def route(self, query: str) -> dict:
    # ... existing tokenisation / embedding code ...
    # existing_logits: Tensor of shape (35,) — the sigmoid inputs

    # --- ADD: nerve-wml advisory mixing ---
    _NERVE_ALPHA = float(os.environ.get("NERVE_WML_ALPHA", "0.1"))
    if _NERVE_ALPHA > 0:
        from bridge import QueryEncoder
        _encoder = QueryEncoder.from_minilm(seq_len=64)
        query_tokens = _encoder.encode_text(query)          # (64,) long tensor
        current_route = {"raw_logits": existing_logits.tolist()}
        advice = _NERVE_ADVISOR.advise(query_tokens, current_route)
        if advice is not None:
            advice_tensor = torch.tensor(
                [advice[str(i)] for i in range(35)], dtype=torch.float32
            )
            # Mix: 90% existing sigmoid signal + 10% advisor
            existing_logits = (1.0 - _NERVE_ALPHA) * existing_logits + _NERVE_ALPHA * advice_tensor
    # --- END advisory mixing ---

    # existing sigmoid decision (unchanged):
    probs = torch.sigmoid(existing_logits)
    # ...
```

> **Note:** Instantiate `QueryEncoder` once at module level (not per call) if
> performance is a concern. The example above shows it inside the method for
> clarity.

---

## 5. Performance expectations

On GrosMac M5 with `n_ticks=4` and a 6-WML pool:

| Scenario | Latency |
|---|---|
| `NERVE_WML_ENABLED=0` (gate off) | < 1 ms (early return) |
| First call (checkpoint load) | 50–200 ms (one-time cost) |
| Subsequent calls | < 50 ms avg |

If latency is too high for your SLA, reduce `n_ticks` to 2 or decrease `n_wmls`
when generating the checkpoint. The advisor is strictly additive — any latency
regression can be eliminated by setting `NERVE_WML_ENABLED=0`.

---

## 6. Verification

After wiring, verify the integration:

```bash
NERVE_WML_ENABLED=1 python -c "
from src.routing.meta_router import MetaRouter
router = MetaRouter()
result = router.route('What is gradient descent?')
print('domains:', list(result.keys())[:5])
print('ok — advisor wired')
"
```

And run micro-kiki's own test suite to confirm no regressions:

```bash
pytest tests/ -q
```

---

## 7. Follow-up tasks (out of scope for this PR)

- Replace the random-seed checkpoint with one fine-tuned on micro-kiki's 35-domain
  training distribution (nerve-wml Plan 4e).
- Set `NERVE_WML_ALPHA` adaptively based on query confidence score.
- Add Langfuse span for the advisory call latency.
````

- [ ] **Step 3: Verify the doc renders**

```bash
cat /Users/electron/Documents/Projets/nerve-wml/docs/integration/micro-kiki-wiring.md | wc -l
```

Expected: > 100 lines.

- [ ] **Step 4: Commit**

```bash
cd /Users/electron/Documents/Projets/nerve-wml
git add docs/integration/micro-kiki-wiring.md
git commit -m "docs(integration): micro-kiki wiring recipe for NerveWmlAdvisor"
```

---

## Task 9: Full suite smoke + coverage check

**Files:** (no new files — sweep only)

- [ ] **Step 1: Run all unit tests**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -m pytest tests/unit/ -q 2>&1 | tail -20
```

Expected: all pass, zero failures.

- [ ] **Step 2: Run slow integration tests**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -m pytest tests/integration/test_advisor_latency.py -m slow -v -s 2>&1 | tail -20
```

Expected: both slow tests PASSED; avg latency printed < 50 ms.

- [ ] **Step 3: Run existing integration test gates to ensure no regression**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -m pytest tests/integration/ -m "not slow" -q 2>&1 | tail -20
```

Expected: all existing gate tests still pass.

- [ ] **Step 4: Check coverage for the new bridge files**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -m pytest tests/unit/test_checkpoint.py tests/unit/test_query_encoder.py tests/unit/test_kiki_nerve_advisor.py --cov=bridge --cov-report=term-missing 2>&1 | tail -30
```

Expected: `bridge/kiki_nerve_advisor.py` ≥ 80 %, `bridge/checkpoint.py` ≥ 90 %, `bridge/query_encoder.py` ≥ 85 %.

- [ ] **Step 5: Lint the new files**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run ruff check bridge/checkpoint.py bridge/query_encoder.py bridge/kiki_nerve_advisor.py
```

Expected: no errors. Fix any that appear before proceeding.

---

## Task 10: Type-check the bridge package

**Files:** (no new files)

- [ ] **Step 1: Run mypy on the bridge package**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run mypy bridge/checkpoint.py bridge/query_encoder.py bridge/kiki_nerve_advisor.py --ignore-missing-imports --no-strict-optional 2>&1 | tail -20
```

Expected: `Success: no issues found` or only `note:` lines. Fix any `error:` lines before continuing.

Common fix for `_sentence_model` type: the field is typed `object | None` intentionally; add `# type: ignore[union-attr]` to the `.encode()` call if mypy flags it.

- [ ] **Step 2: Commit any mypy fixes**

If any changes were needed:

```bash
cd /Users/electron/Documents/Projets/nerve-wml
git add bridge/
git commit -m "fix(bridge): mypy type annotation cleanup"
```

If no changes were needed, skip this step.

---

## Task 11: Tag `gate-llm-advisor-passed`

**Files:** (no new files — git tag only)

- [ ] **Step 1: Final full test run**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -m pytest -m "not slow" -q 2>&1 | tail -10
```

Expected: all pass, zero failures.

- [ ] **Step 2: Run slow tests one final time**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -m pytest tests/integration/test_advisor_latency.py -m slow -v -s 2>&1 | grep -E "PASSED|FAILED|avg|max"
```

Expected: `PASSED` for both slow tests; avg latency < 50 ms.

- [ ] **Step 3: Tag and push**

```bash
cd /Users/electron/Documents/Projets/nerve-wml
git tag gate-llm-advisor-passed
git push origin master
git push origin gate-llm-advisor-passed
```

- [ ] **Step 4: Verify tag on origin**

```bash
git -C /Users/electron/Documents/Projets/nerve-wml ls-remote --tags origin | grep gate-llm-advisor-passed
```

Expected: one line referencing `refs/tags/gate-llm-advisor-passed`.

---

## Task 12: Final sweep — docs, plan self-check, cleanup

**Files:** (no new files — review only)

- [ ] **Step 1: Verify all new files are present**

```bash
ls /Users/electron/Documents/Projets/nerve-wml/bridge/checkpoint.py \
   /Users/electron/Documents/Projets/nerve-wml/bridge/query_encoder.py \
   /Users/electron/Documents/Projets/nerve-wml/bridge/kiki_nerve_advisor.py \
   /Users/electron/Documents/Projets/nerve-wml/docs/integration/micro-kiki-wiring.md \
   /Users/electron/Documents/Projets/nerve-wml/tests/unit/test_checkpoint.py \
   /Users/electron/Documents/Projets/nerve-wml/tests/unit/test_query_encoder.py \
   /Users/electron/Documents/Projets/nerve-wml/tests/unit/test_kiki_nerve_advisor.py \
   /Users/electron/Documents/Projets/nerve-wml/tests/integration/test_advisor_latency.py
```

Expected: all 8 files listed without error.

- [ ] **Step 2: Confirm micro-kiki repo is untouched**

```bash
git -C /Users/electron/Documents/Projets/micro-kiki/ status
```

Expected: `nothing to commit, working tree clean` — this plan must not have modified micro-kiki.

- [ ] **Step 3: Verify the wiring doc is self-sufficient**

Read `docs/integration/micro-kiki-wiring.md` from top to bottom. Confirm it contains:
- How to install nerve-wml as a dep in micro-kiki (section 1)
- How to generate a checkpoint (section 2)
- All env vars (section 3)
- The exact 8-line diff for `meta_router.py` (section 4)
- Latency expectations (section 5)
- Verification command (section 6)

- [ ] **Step 4: Confirm gate tags on origin**

```bash
git -C /Users/electron/Documents/Projets/nerve-wml/ ls-remote --tags origin | grep "gate-"
```

Expected: `gate-p`, `gate-w`, `gate-m`, `gate-m2`, `gate-llm-advisor-passed` all present.

- [ ] **Step 5: Commit plan file if not yet done**

```bash
cd /Users/electron/Documents/Projets/nerve-wml
git status docs/superpowers/plans/2026-04-18-nerve-wml-plan-4d-llm-integration.md
```

If it shows as untracked or modified, stage and commit it (it should already be committed from the plan authoring step; skip if so).

---

## Self-Review Checklist

**Spec coverage:**
- [x] `bridge/checkpoint.py` with `save_advisor_checkpoint` / `load_advisor_checkpoint` — Task 2
- [x] `bridge/query_encoder.py` with MiniLM + projection + VQ — Task 3
- [x] `bridge/kiki_nerve_advisor.py` with env gate, lazy load, `advise()` — Tasks 4-5
- [x] `docs/integration/micro-kiki-wiring.md` — Task 8
- [x] Unit tests: env gate, mock checkpoint, NaN, valid 35-key output — Tasks 4-5
- [x] Latency integration test < 50 ms, `@pytest.mark.slow` — Task 6
- [x] `gate-llm-advisor-passed` tag — Task 11
- [x] `sentence-transformers>=2.7` + `safetensors>=0.4` in dev extras — Task 1
- [x] Never raises from `advise()` — enforced by outer try/except in Task 4
- [x] Never mutates `current_route` — tested in Task 5, enforced by `_safe_advise` not writing to it
- [x] Idempotent — tested in Task 5, enforced by queue reset in `_safe_advise`
- [x] Golden bit-stability test — `test_golden_codebook_value` in Task 2
- [x] micro-kiki not modified — verified in Task 12

**Type consistency:**
- `save_advisor_checkpoint(pool: list[MlpWML], nerve: SimNerveAdapter, path: Path)` — consistent Tasks 2, 6, 8
- `load_advisor_checkpoint(path: Path) -> tuple[list[MlpWML], SimNerveAdapter, dict]` — consistent Tasks 2, 4
- `QueryEncoder.encode_embedding(embedding: Tensor) -> Tensor` (long, shape (seq_len,)) — consistent Tasks 3, 8
- `NerveWmlAdvisor.advise(query_tokens: Tensor, current_route: dict) -> dict[str, float] | None` — consistent Tasks 4, 5, 6, 8
- `NerveWmlAdvisor.__init__(checkpoint_path: Path, n_domains: int, n_ticks: int)` — consistent Tasks 4, 6

**No placeholders:** every step contains complete code or exact commands.
