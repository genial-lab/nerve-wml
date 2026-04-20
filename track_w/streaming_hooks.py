"""Rollout helpers for sequential inference.

v1.1 WMLs consume static [B, dim] input. For temporal experiments
we roll them over a sequence [B, T, dim] and emit one code per
timestep. LIF naturally carries state via v_mem; MLP and Transformer
are stateless and re-evaluated per step.
"""
from __future__ import annotations

import torch


def rollout_mlp_emit_codes(wml, xs: torch.Tensor) -> torch.Tensor:
    """xs: [B, T, dim] → codes [B, T] via wml.emit_head_pi(wml.core(x_t))."""
    batch, seq_len, _ = xs.shape
    codes = torch.empty(batch, seq_len, dtype=torch.long)
    for t in range(seq_len):
        with torch.no_grad():
            h = wml.core(xs[:, t])
            codes[:, t] = wml.emit_head_pi(h).argmax(-1)
    return codes


def rollout_lif_emit_codes(
    wml, xs: torch.Tensor, reset_each_batch: bool = True,
) -> torch.Tensor:
    """xs: [B, T, dim] → codes [B, T]. LIF's v_mem carries state between
    timesteps within a sample; reset per-sample by default."""
    from track_w._surrogate import spike_with_surrogate

    batch, seq_len, _ = xs.shape
    codes = torch.empty(batch, seq_len, dtype=torch.long)
    for b in range(batch):
        if reset_each_batch:
            wml.reset_state()
        for t in range(seq_len):
            with torch.no_grad():
                i_in = wml.input_proj(xs[b, t].unsqueeze(0))
                spikes = spike_with_surrogate(i_in, v_thr=wml.v_thr)
                codes[b, t] = wml.emit_head_pi(spikes).argmax(-1)[0]
    return codes
