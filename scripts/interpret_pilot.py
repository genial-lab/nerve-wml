"""Interp pilot — end-to-end gate-interp-passed runner.

Trains a MlpWML briefly, extracts semantics, clusters, renders HTML.
Returns a dict with the metrics gate-interp checks.
"""
from __future__ import annotations

import torch

from interpret.clustering import cluster_codes_by_activation
from interpret.code_semantics import build_semantics_table
from interpret.visualise import render_html_report
from track_w.mlp_wml import MlpWML
from track_w.mock_nerve import MockNerve
from track_w.tasks.flow_proxy import FlowProxyTask
from track_w.training import train_wml_on_task


def run_interp_pilot(
    *,
    output_path: str = "reports/interp/w2_true_lif_semantics.html",
    steps: int = 200,
    n_inputs: int = 512,
    n_clusters: int = 8,
    wml_id: int = 0,
) -> dict:
    """Train → extract → cluster → render. Return gate-interp metrics."""
    torch.manual_seed(0)

    nerve = MockNerve(n_wmls=2, k=1, seed=0)
    nerve.set_phase_active(gamma=True, theta=False)
    # Use 32 classes to activate 32 codes and create diverse, non-degenerate clusters.
    task = FlowProxyTask(dim=16, n_classes=32, seed=0)
    wml = MlpWML(id=wml_id, d_hidden=16, seed=0)
    train_wml_on_task(wml, nerve, task, steps=steps, lr=1e-2)

    # Extract semantics from a fresh input batch.
    inputs = torch.randn(n_inputs, 16)
    table = build_semantics_table(wml, inputs, top_k_inputs=3)

    centroids = torch.stack([table[c]["activation_centroid"] for c in range(64)])
    clusters  = cluster_codes_by_activation(
        centroids, n_clusters=n_clusters, seed=0,
    )

    # Cluster assignment entropy (in bits).
    counts = torch.bincount(clusters, minlength=n_clusters).float()
    p = counts / counts.sum()
    entropy_bits = float(-(p * (p + 1e-12).log2()).sum().item())

    # Render.
    render_html_report(table, clusters, output_path=output_path, wml_id=wml_id)

    n_active = sum(1 for c in range(64) if table[c]["n_samples_mapped"] > 0)

    return {
        "entropy_bits":   entropy_bits,
        "n_active_codes": n_active,
        "output_path":    output_path,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run_interp_pilot(), indent=2))
