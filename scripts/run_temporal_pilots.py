"""Temporal streaming measurements for Claim B under sequence input."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: N812, E402

from scripts.measure_info_transmission import mutual_info_score  # noqa: E402
from track_w._surrogate import spike_with_surrogate  # noqa: E402
from track_w.lif_wml import LifWML  # noqa: E402
from track_w.mlp_wml import MlpWML  # noqa: E402
from track_w.mock_nerve import MockNerve  # noqa: E402
from track_w.streaming_hooks import (  # noqa: E402
    rollout_lif_emit_codes,
    rollout_mlp_emit_codes,
)
from track_w.tasks.sequential import SequentialFlowProxyTask  # noqa: E402


def _train_pair_on_sequence(seed: int, steps: int) -> tuple:
    torch.manual_seed(seed)
    nerve = MockNerve(n_wmls=2, k=1, seed=seed)
    nerve.set_phase_active(gamma=True, theta=False)
    task = SequentialFlowProxyTask(seq_len=16, dim=16, n_classes=4, seed=seed)

    mlp = MlpWML(id=0, input_dim=16, d_hidden=32, alphabet_size=64, seed=seed)
    opt_m = torch.optim.Adam(mlp.parameters(), lr=5e-3)
    for _ in range(steps):
        xs, y = task.sample(batch=32)
        x_label = xs[:, task.t_b]
        h = mlp.core(x_label)
        logits = mlp.emit_head_pi(h)[:, : task.n_classes]
        loss = F.cross_entropy(logits, y)
        opt_m.zero_grad()
        loss.backward()
        opt_m.step()

    lif = LifWML(
        id=0, input_dim=16, n_neurons=32, alphabet_size=64, seed=seed + 10,
    )
    opt_l = torch.optim.Adam(lif.parameters(), lr=5e-3)
    for _ in range(steps):
        xs, y = task.sample(batch=32)
        x_label = xs[:, task.t_b]
        i_in = lif.input_proj(x_label)
        spikes = spike_with_surrogate(i_in, v_thr=lif.v_thr)
        logits = lif.emit_head_pi(spikes)[:, : task.n_classes]
        loss = F.cross_entropy(logits, y)
        opt_l.zero_grad()
        loss.backward()
        opt_l.step()
    return mlp, lif, task


def run_temporal_mi(
    seeds: list[int] | None = None, steps: int = 200, batch: int = 512,
) -> dict:
    if seeds is None:
        seeds = [0, 1, 2]
    trained_step_mis, filler_mis = [], []
    for s in seeds:
        mlp, lif, task = _train_pair_on_sequence(seed=s, steps=steps)
        xs, _ = task.sample(batch=batch)
        codes_mlp = rollout_mlp_emit_codes(mlp, xs)
        codes_lif = rollout_lif_emit_codes(lif, xs)
        t_mid = (task.t_a + task.t_b) // 2
        # Only t_b is supervised during training (see _train_pair_on_sequence);
        # t_a provides the XOR bit but is never seen by either substrate head
        # as a training target. Honest measurement focuses on the trained
        # step.
        mi_trained = mutual_info_score(
            codes_mlp[:, task.t_b].numpy(), codes_lif[:, task.t_b].numpy(),
        )
        _, c = np.unique(codes_mlp[:, task.t_b].numpy(), return_counts=True)
        h = float(-(c / c.sum() * np.log(c / c.sum())).sum())
        trained_step_mis.append(mi_trained / max(h, 1e-9))

        mi_filler = mutual_info_score(
            codes_mlp[:, t_mid].numpy(), codes_lif[:, t_mid].numpy(),
        )
        _, c = np.unique(codes_mlp[:, t_mid].numpy(), return_counts=True)
        h_f = float(-(c / c.sum() * np.log(c / c.sum())).sum())
        filler_mis.append(mi_filler / max(h_f, 1e-9))
    return {
        "seeds":                     list(seeds),
        "trained_step_ratios":       trained_step_mis,
        "filler_step_ratios":        filler_mis,
        "mean_mi_at_trained_step":   float(np.mean(trained_step_mis)),
        "mean_mi_at_filler_steps":   float(np.mean(filler_mis)),
    }
