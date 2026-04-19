"""Adaptive codebook pilot — shrink / grow cycles + gate aggregator.

Trains an AdaptiveCodebook on a toy MOG stream, triggers shrink or grow,
and reports before/after metrics. Used by gate-adaptive-passed.

Plan 8 Tasks 5-6.
"""
from __future__ import annotations

import torch
from torch.optim import Adam

from track_p.adaptive_codebook import AdaptiveCodebook


def _train_steps(cb: AdaptiveCodebook, centers: torch.Tensor, n_steps: int,
                 batch: int = 256) -> None:
    """Train the codebook for n_steps on a MOG stream.

    EMA-mode VQCodebook has no trainable parameters — the EMA update
    happens inside quantize(), driven by calling the storage's
    .quantize() directly. Here we drive it via quantize_active to keep
    the active_mask semantics and also tick usage_counter.
    """
    trainable = [p for p in cb.storage.parameters() if p.requires_grad]
    opt = Adam(trainable, lr=1e-3) if trainable else None
    size = cb.storage.size
    dim = cb.storage.dim
    for _ in range(n_steps):
        cluster_ids = torch.randint(0, size, (batch,))
        z = centers[cluster_ids] + torch.randn(batch, dim) * 0.2
        # Drive storage.quantize() too so usage_counter ticks and EMA fires.
        cb.storage.quantize(z)
        _, _, loss = cb.quantize_active(z)
        if opt is not None and loss.requires_grad:
            opt.zero_grad()
            loss.backward()
            opt.step()


def run_adaptive_cycle(
    *,
    size: int = 16,
    dim: int = 8,
    warmup_steps: int = 500,
    post_steps: int = 200,
    min_usage_frac: float = 0.01,
) -> dict:
    """Train → shrink → re-train briefly → report size delta."""
    torch.manual_seed(0)
    cb = AdaptiveCodebook(size=size, dim=dim)
    centers = torch.randn(size, dim) * 3

    # Warmup on skewed usage: only half the clusters appear.
    active_mask = torch.zeros(size, dtype=torch.bool)
    active_mask[: size // 2] = True
    cb_centers = centers.clone()
    # Concentrate training on half the clusters by zeroing the others.
    cb_centers[~active_mask] = 0

    _train_steps(cb, cb_centers, warmup_steps)

    size_before = cb.current_size()
    cb.shrink(min_usage_frac=min_usage_frac, min_codes=4)
    size_after_shrink = cb.current_size()

    # Brief re-train.
    _train_steps(cb, cb_centers, post_steps)

    return {
        "size_before":        size_before,
        "size_after_shrink":  size_after_shrink,
        "codes_retired":      size_before - size_after_shrink,
    }


def run_adaptive_grow_cycle(
    *,
    size: int = 16,
    dim: int = 8,
    warmup_steps: int = 500,
    top_k: int = 4,
) -> dict:
    """Train to saturation → retire half → grow by top_k → report."""
    torch.manual_seed(0)
    cb = AdaptiveCodebook(size=size, dim=dim)
    centers = torch.randn(size, dim) * 3

    _train_steps(cb, centers, warmup_steps)

    # Manually retire the bottom half to simulate a shrink-first scenario.
    counts = cb.storage.usage_counter.float()
    sorted_idx = counts.argsort()
    for idx in sorted_idx[: size // 2].tolist():
        cb.active_mask[idx] = False

    size_before_grow = cb.current_size()
    new_indices = cb.grow(top_k_to_split=top_k, seed=0)

    return {
        "size_before_grow":  size_before_grow,
        "size_after_grow":   cb.current_size(),
        "new_indices":       new_indices,
        "codes_added":       len(new_indices),
    }


def run_gate_adaptive() -> dict:
    """Gate aggregator: shrink + grow cycles both succeed."""
    shrink_rep = run_adaptive_cycle(
        size=16, dim=8, warmup_steps=500, post_steps=100,
    )
    grow_rep = run_adaptive_grow_cycle(
        size=16, dim=8, warmup_steps=500, top_k=4,
    )

    shrink_passed = shrink_rep["codes_retired"] > 0
    grow_passed = grow_rep["codes_added"] > 0

    return {
        "shrink":    shrink_rep,
        "grow":      grow_rep,
        "shrink_passed": shrink_passed,
        "grow_passed":   grow_passed,
        "all_passed":    shrink_passed and grow_passed,
    }


if __name__ == "__main__":
    import json
    # Convert non-JSON-serialisable values.
    rep = run_gate_adaptive()
    print(json.dumps(rep, indent=2, default=str))
