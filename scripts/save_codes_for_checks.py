"""Generate MLP/LIF emitted codes for methodology cross-checks.

Replicates the canonical run_w2_hard pipeline
(scripts/track_w_pilot.py:429) over N seeds and saves the argmax
codes emitted by each substrate on a held-out eval batch.

The resulting NPZ artefact is consumed by:
  * scripts/measure_mi_null_model.py     (permutation significance)
  * scripts/measure_mi_bootstrap_ci.py   (IQR / 95% CI)

This is Phase A "Generate" stage. Training is light
(3 seeds x ~30s each on CPU) but by policy runs on Tower or
kxkm-ai, not on the light-client laptop.

Usage:
    uv run python scripts/save_codes_for_checks.py \\
        --seeds 0 1 2 --n-eval 5000 --steps 800 \\
        --out tests/golden/codes_mlp_lif.npz
"""
from __future__ import annotations

import argparse
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


def _train_pair(seed: int, steps: int) -> tuple[MlpWML, LifWML, torch.nn.Linear]:
    torch.manual_seed(seed)
    nerve = MockNerve(n_wmls=2, k=1, seed=seed)
    nerve.set_phase_active(gamma=True, theta=False)

    task_mlp = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)
    mlp = MlpWML(id=0, d_hidden=16, seed=seed)
    train_wml_on_task(mlp, nerve, task_mlp, steps=steps, lr=1e-2)

    task_lif = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)
    lif = LifWML(id=0, n_neurons=16, seed=seed + 10)
    input_encoder = torch.nn.Linear(16, lif.n_neurons)
    opt = torch.optim.Adam(
        list(lif.parameters()) + list(input_encoder.parameters()),
        lr=1e-2,
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

    return mlp, lif, input_encoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-eval", type=int, default=5000)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tests/golden/codes_mlp_lif.npz"),
    )
    args = parser.parse_args()

    all_mlp_codes = []
    all_lif_codes = []
    all_mlp_emb = []
    all_lif_emb = []

    for seed in args.seeds:
        print(f"seed {seed}: training MLP + LIF ({args.steps} steps)...")
        mlp, lif, lif_encoder = _train_pair(seed, args.steps)

        eval_task = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)
        x_eval, _ = eval_task.sample(batch=args.n_eval)
        with torch.no_grad():
            mlp_emb = mlp.core(x_eval)
            mlp_codes = mlp.emit_head_pi(mlp_emb).argmax(-1)
            lif_emb = lif.input_proj(lif_encoder(x_eval))
            spikes = spike_with_surrogate(lif_emb, v_thr=lif.v_thr)
            lif_codes = lif.emit_head_pi(spikes).argmax(-1)

        all_mlp_codes.append(mlp_codes.cpu().numpy().astype(np.int64))
        all_lif_codes.append(lif_codes.cpu().numpy().astype(np.int64))
        all_mlp_emb.append(mlp_emb.cpu().numpy().astype(np.float32))
        all_lif_emb.append(lif_emb.cpu().numpy().astype(np.float32))
        print(
            f"  mlp alphabet used: {len(np.unique(all_mlp_codes[-1]))}/64, "
            f"lif: {len(np.unique(all_lif_codes[-1]))}/64, "
            f"emb shape mlp={mlp_emb.shape}, lif={lif_emb.shape}"
        )

    mlp_stack = np.stack(all_mlp_codes)
    lif_stack = np.stack(all_lif_codes)
    mlp_emb_stack = np.stack(all_mlp_emb)
    lif_emb_stack = np.stack(all_lif_emb)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        mlp_codes=mlp_stack,
        lif_codes=lif_stack,
        mlp_embeddings=mlp_emb_stack,
        lif_embeddings=lif_emb_stack,
        seeds=np.asarray(args.seeds, dtype=np.int64),
        n_eval=args.n_eval,
        steps=args.steps,
    )
    print()
    print(f"Saved: {args.out}")
    print(f"  mlp_codes shape: {mlp_stack.shape}")
    print(f"  lif_codes shape: {lif_stack.shape}")
    print(f"  mlp_embeddings shape: {mlp_emb_stack.shape}")
    print(f"  lif_embeddings shape: {lif_emb_stack.shape}")


if __name__ == "__main__":
    main()
