"""Platonic Representation Hypothesis alignment (Huh et al. 2024).

Reproduces the canonical run_w2_hard pipeline (MLP + LIF trained on
HardFlowProxyTask) from scripts/track_w_pilot.py:429, then instead of
reporting accuracy gap, extracts the continuous pre-VQ embeddings of
both substrates and computes the mutual_knn alignment kernel from
Huh 2024 (arXiv:2405.07987).

Kernel: for each sample in a batch of N=1024, find the k=10 nearest
neighbors in substrate-A's embedding space and in substrate-B's
embedding space (cosine similarity), count the intersection size,
average over the batch. Score in [0, 1], where:
  * 1.0 = identical neighbor structure (self-alignment sanity)
  * k/N ~ 0.01 = chance level (random baseline sanity)
  * 0.1-0.5 = typical cross-model pairs in Huh 2024

Decision threshold:
  alignment > max(3 x random_baseline, 0.05) => substrates converge
  => local confirmation of Platonic RH on heterogeneous substrates.

Does NOT modify any runner or substrate -- uses public APIs directly
and replicates the run_w2_hard training recipe verbatim.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from track_w.mock_nerve import MockNerve
from track_w.mlp_wml import MlpWML
from track_w.lif_wml import LifWML
from track_w._surrogate import spike_with_surrogate
from track_w.tasks.hard_flow_proxy import HardFlowProxyTask
from track_w.training import train_wml_on_task


def mutual_knn(A: torch.Tensor, B: torch.Tensor, k: int = 10) -> float:
    """Mutual k-nearest-neighbor overlap (Huh et al. 2024)."""
    n = A.shape[0]
    assert B.shape[0] == n
    a = F.normalize(A.float(), dim=-1)
    b = F.normalize(B.float(), dim=-1)
    sim_a = a @ a.T
    sim_b = b @ b.T
    mask = torch.eye(n, dtype=torch.bool, device=A.device)
    sim_a.masked_fill_(mask, float("-inf"))
    sim_b.masked_fill_(mask, float("-inf"))
    _, knn_a = sim_a.topk(k, dim=-1)
    _, knn_b = sim_b.topk(k, dim=-1)
    has_a = torch.zeros(n, n, dtype=torch.bool, device=A.device)
    has_b = torch.zeros(n, n, dtype=torch.bool, device=A.device)
    has_a.scatter_(1, knn_a, True)
    has_b.scatter_(1, knn_b, True)
    overlap = (has_a & has_b).sum(dim=-1).float()
    return (overlap / k).mean().item()


def _train_mlp(steps: int, seed: int) -> MlpWML:
    """Replicates run_w2_hard MLP path at scripts/track_w_pilot.py:451-454."""
    torch.manual_seed(seed)
    nerve = MockNerve(n_wmls=2, k=1, seed=seed)
    nerve.set_phase_active(gamma=True, theta=False)
    task = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)
    mlp = MlpWML(id=0, d_hidden=16, seed=seed)
    train_wml_on_task(mlp, nerve, task, steps=steps, lr=1e-2)
    return mlp


def _train_lif(steps: int, seed: int) -> tuple[LifWML, torch.nn.Linear]:
    """Replicates run_w2_hard LIF path at scripts/track_w_pilot.py:456-477."""
    torch.manual_seed(seed)
    task = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)
    lif = LifWML(id=0, n_neurons=16, seed=seed + 10)
    input_encoder = torch.nn.Linear(16, lif.n_neurons)
    opt = torch.optim.Adam(
        list(lif.parameters()) + list(input_encoder.parameters()),
        lr=1e-2,
    )
    for _ in range(steps):
        x, y = task.sample(batch=64)
        i_in = lif.input_proj(input_encoder(x))
        spikes = spike_with_surrogate(i_in, v_thr=lif.v_thr)
        logits = lif.emit_head_pi(spikes)[:, : task.n_classes]
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return lif, input_encoder


def main() -> None:
    import numpy as np

    n_eval = 1024
    ks = [5, 10, 20, 50]
    steps = 800
    seeds = [0, 1, 2]

    per_seed: dict = {}
    for seed in seeds:
        print(f"Seed {seed}: training MLP + LIF...")
        mlp = _train_mlp(steps, seed)
        lif, lif_encoder = _train_lif(steps, seed)

        eval_task = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)
        x_eval, _ = eval_task.sample(batch=n_eval)
        with torch.no_grad():
            emb_mlp = mlp.core(x_eval)
            emb_lif = lif.input_proj(lif_encoder(x_eval))

        torch.manual_seed(seed + 1000)
        rand_a = torch.randn(n_eval, 16)
        rand_b = torch.randn(n_eval, 16)

        per_k: dict = {}
        for k in ks:
            per_k[str(k)] = {
                "self_mlp": mutual_knn(emb_mlp, emb_mlp, k=k),
                "self_lif": mutual_knn(emb_lif, emb_lif, k=k),
                "random":   mutual_knn(rand_a, rand_b, k=k),
                "mlp_lif":  mutual_knn(emb_mlp, emb_lif, k=k),
                "chance":   k / n_eval,
            }
        per_seed[str(seed)] = per_k

    aggregated: dict = {}
    for k in ks:
        mlp_lif_vals = [per_seed[str(s)][str(k)]["mlp_lif"] for s in seeds]
        random_vals  = [per_seed[str(s)][str(k)]["random"]  for s in seeds]
        aggregated[str(k)] = {
            "mlp_lif_median": float(np.median(mlp_lif_vals)),
            "mlp_lif_p25":    float(np.percentile(mlp_lif_vals, 25)),
            "mlp_lif_p75":    float(np.percentile(mlp_lif_vals, 75)),
            "mlp_lif_values": mlp_lif_vals,
            "random_median":  float(np.median(random_vals)),
            "random_values":  random_vals,
            "chance":         k / n_eval,
        }

    results = {
        "config": {"n_eval": n_eval, "ks": ks, "steps": steps, "seeds": seeds, "d": 16},
        "per_seed": per_seed,
        "aggregated": aggregated,
    }

    out_dir = Path("papers/paper1/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "platonic_rh_alignment.json"
    out_path.write_text(json.dumps(results, indent=2))

    print()
    print(f"=== Platonic RH mutual_knn (N={n_eval}, {len(seeds)} seeds) ===")
    print()
    header = f"{'k':>4}{'chance':>10}{'random':>10}{'mlp<->lif median':>20}{'IQR':>20}{'xRandom':>10}"
    print(header)
    print("-" * len(header))
    for k in ks:
        a = aggregated[str(k)]
        iqr_str = f"[{a['mlp_lif_p25']:.4f}, {a['mlp_lif_p75']:.4f}]"
        x_rand = a["mlp_lif_median"] / max(a["random_median"], 1e-9)
        print(
            f"{k:>4}{a['chance']:>10.4f}{a['random_median']:>10.4f}"
            f"{a['mlp_lif_median']:>20.4f}{iqr_str:>20}{x_rand:>10.1f}"
        )

    k_focus = 10
    ref = aggregated[str(k_focus)]
    threshold = max(3.0 * ref["random_median"], 0.05)
    sig_focus = ref["mlp_lif_median"] > threshold
    sig_all = all(
        aggregated[str(k)]["mlp_lif_median"] > max(3.0 * aggregated[str(k)]["random_median"], 0.05)
        for k in ks
    )

    print()
    print(f"Focus k={k_focus}: median={ref['mlp_lif_median']:.4f}, threshold={threshold:.4f}")
    print(f"Verdict@k=10:     {'SIGNAL' if sig_focus else 'NO SIGNAL'}")
    print(f"Robust across k: {'YES -- signal stable across k in [5,10,20,50]' if sig_all else 'NO'}")
    print()
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
