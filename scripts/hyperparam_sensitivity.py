"""Hyperparameter sensitivity sweep on HardFlowProxyTask (review m3).

Sweeps d_hidden and lr at matched capacity (MlpWML.d_hidden ==
LifWML.n_neurons) to validate robustness of Claim A's ~2-3%
plateau gap. Complements the Sleep-EDF matched-scale sweep
(figures/eeg_matched_scale_sweep.json) on the synthetic benchmark
side.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from track_w._surrogate import spike_with_surrogate
from track_w.lif_wml import LifWML
from track_w.mlp_wml import MlpWML
from track_w.mock_nerve import MockNerve
from track_w.tasks.hard_flow_proxy import HardFlowProxyTask
from track_w.training import train_wml_on_task


def _one_config(
    d_hidden: int, lr: float, seed: int, steps: int = 400,
) -> dict[str, float]:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    nerve = MockNerve(n_wmls=2, k=1, seed=seed)
    nerve.set_phase_active(gamma=True, theta=False)

    task_mlp = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)
    mlp = MlpWML(id=0, d_hidden=d_hidden, input_dim=16, seed=seed)
    train_wml_on_task(mlp, nerve, task_mlp, steps=steps, lr=lr)

    task_lif = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)
    lif = LifWML(id=0, n_neurons=d_hidden, seed=seed + 10)
    input_encoder = torch.nn.Linear(16, lif.n_neurons)
    opt = torch.optim.Adam(
        list(lif.parameters()) + list(input_encoder.parameters()),
        lr=lr,
    )
    for _ in range(steps):
        x, y = task_lif.sample(batch=64)
        i_in = lif.input_proj(input_encoder(x))
        spikes = spike_with_surrogate(i_in, v_thr=lif.v_thr)
        logits = lif.emit_head_pi(spikes)[:, : task_lif.n_classes]
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    eval_task = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)
    x, y = eval_task.sample(batch=512)
    with torch.no_grad():
        pred_mlp = mlp.emit_head_pi(mlp.core(x))[:, : eval_task.n_classes].argmax(-1)
        acc_mlp = (pred_mlp == y).float().mean().item()
        i_in = lif.input_proj(input_encoder(x))
        spikes = spike_with_surrogate(i_in, v_thr=lif.v_thr)
        pred_lif = lif.emit_head_pi(spikes)[:, : eval_task.n_classes].argmax(-1)
        acc_lif = (pred_lif == y).float().mean().item()

    gap = abs(acc_mlp - acc_lif) / max(acc_mlp, 1e-6)
    return {"acc_mlp": acc_mlp, "acc_lif": acc_lif, "gap": gap}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("papers/paper1/figures/hyperparam_sensitivity.json"),
    )
    args = parser.parse_args()

    d_hidden_values = [8, 16, 32, 64]
    lr_values = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]

    print("d_hidden sweep (lr=1e-2):")
    d_sweep = []
    for d in d_hidden_values:
        per_seed = [
            _one_config(d, 1e-2, s, args.steps) for s in args.seeds
        ]
        gap_med = float(np.median([r["gap"] for r in per_seed]))
        acc_mlp_mean = float(np.mean([r["acc_mlp"] for r in per_seed]))
        acc_lif_mean = float(np.mean([r["acc_lif"] for r in per_seed]))
        d_sweep.append({
            "d_hidden":    d,
            "gap_median":  gap_med,
            "acc_mlp":     acc_mlp_mean,
            "acc_lif":     acc_lif_mean,
        })
        print(
            f"  d={d:>3d}: gap={gap_med:.4f}  "
            f"MLP={acc_mlp_mean:.4f}  LIF={acc_lif_mean:.4f}"
        )

    print("lr sweep (d_hidden=16):")
    lr_sweep = []
    for lr in lr_values:
        per_seed = [
            _one_config(16, lr, s, args.steps) for s in args.seeds
        ]
        gap_med = float(np.median([r["gap"] for r in per_seed]))
        acc_mlp_mean = float(np.mean([r["acc_mlp"] for r in per_seed]))
        acc_lif_mean = float(np.mean([r["acc_lif"] for r in per_seed]))
        lr_sweep.append({
            "lr":          lr,
            "gap_median":  gap_med,
            "acc_mlp":     acc_mlp_mean,
            "acc_lif":     acc_lif_mean,
        })
        print(
            f"  lr={lr:.0e}: gap={gap_med:.4f}  "
            f"MLP={acc_mlp_mean:.4f}  LIF={acc_lif_mean:.4f}"
        )

    out = {
        "d_hidden_sweep": d_sweep,
        "lr_sweep":       lr_sweep,
        "config": {
            "n_seeds": len(args.seeds),
            "steps":   args.steps,
            "task":    "HardFlowProxyTask N=2 matched-capacity",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
