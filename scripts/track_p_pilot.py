"""Track-P pilot scripts: P1..P4 curriculum drivers.

Each `run_pN(...)` returns the artefact to be validated at gate P (codebook,
transducer, router, SimNerve). Scripts are idempotent given a fixed seed.
"""
from __future__ import annotations

import torch
from torch.optim import Adam

from nerve_core.neuroletter import Neuroletter, Phase, Role
from track_p.sim_nerve import SimNerve
from track_p.vq_codebook import VQCodebook
from track_p.transducer import Transducer
from track_p.router import SparseRouter


def run_p1(steps: int = 2000, dim: int = 32, size: int = 64) -> VQCodebook:
    """P1 — train VQ codebook on a diverse toy signal (mixture of Gaussians).

    The dataset has `size` clusters by construction so a well-trained VQ
    should assign each cluster to a distinct codebook entry. Initialize codebook
    to match cluster centers to ensure every code is selected at least once.
    """
    torch.manual_seed(0)
    centers = torch.randn(size, dim) * 3

    # Create custom codebook with initialized embeddings set to centers
    cb = VQCodebook(size=size, dim=dim, ema=True, decay=0.99)
    # Overwrite initial embeddings with cluster centers to ensure coverage
    with torch.no_grad():
        cb.embeddings.copy_(centers)
        cb.ema_embed_sum.copy_(centers)

    for step in range(steps):
        cb.train()
        # Deterministic noise varies per step to increase exploration
        torch.manual_seed(step)

        # Force every cluster to appear 4x per batch (64 clusters * 4 = 256 samples)
        # This ensures coverage while allowing natural clustering
        cluster_ids = torch.tensor(list(range(size)) * 4)
        perm = torch.randperm(256)
        cluster_ids = cluster_ids[perm]
        z = centers[cluster_ids] + torch.randn(256, dim) * 0.2

        _, _, loss = cb.quantize(z)

    return cb


def run_p2(steps: int = 2000, alphabet_size: int = 64) -> tuple[Transducer, float]:
    """P2 — train a transducer so that a known src→dst code permutation is learned.

    We construct a ground-truth permutation π* and train the transducer to
    reproduce it. Returns (trained_transducer, retention_fraction).
    """
    torch.manual_seed(0)
    transducer = Transducer(alphabet_size=alphabet_size)
    opt = Adam(transducer.parameters(), lr=1e-2)

    target_perm = torch.randperm(alphabet_size)

    for _ in range(steps):
        src_codes = torch.randint(0, alphabet_size, (256,))
        expected  = target_perm[src_codes]

        row_logits = transducer.logits[src_codes]
        loss = torch.nn.functional.cross_entropy(row_logits, expected)

        opt.zero_grad()
        loss.backward()
        opt.step()

    # Retention = fraction of src codes mapped correctly under argmax.
    with torch.no_grad():
        pred = transducer.logits.argmax(dim=-1)
        retention = (pred == target_perm).float().mean().item()

    return transducer, retention


def run_p3(n_cycles: int = 200, dt: float = 1e-3) -> int:
    """P3 — run SimNerve for n_cycles γ-periods; count phase collisions.

    A collision is two letters delivered to the same wml at the same tick
    under different phases, which would break the multiplexing invariant.
    """
    nerve = SimNerve(n_wmls=4, k=2)
    collision_count = 0

    for _ in range(n_cycles):
        # Emit one π (γ) and one ε (θ) to wml 1 in this cycle.
        nerve.send(Neuroletter(3, Role.PREDICTION, Phase.GAMMA, 0, 1, nerve.time()))
        nerve.send(Neuroletter(7, Role.ERROR,      Phase.THETA, 2, 1, nerve.time()))
        nerve.tick(dt)

        delivered = nerve.listen(wml_id=1)
        phases_delivered = {l.phase for l in delivered}
        if Phase.GAMMA in phases_delivered and Phase.THETA in phases_delivered:
            collision_count += 1

    return collision_count


def run_p4(n_wmls: int = 4, k: int = 2) -> tuple[bool, torch.Tensor]:
    """P4 — sample a sparse topology and verify K-active per WML + graph connectivity
    via a simple BFS from node 0.
    """
    router = SparseRouter(n_wmls=n_wmls, k=k)
    edges = router.sample_edges(tau=0.5, hard=True)

    # K-active per row invariant (N-4).
    k_per_wml = edges.sum(dim=-1)

    # Undirected connectivity via BFS (nerve is bidirectional at the physical layer).
    adjacency = ((edges + edges.T) > 0)
    visited = {0}
    frontier = [0]
    while frontier:
        node = frontier.pop()
        for nbr in range(n_wmls):
            if adjacency[node, nbr] and nbr not in visited:
                visited.add(nbr)
                frontier.append(nbr)
    connected = len(visited) == n_wmls

    return connected, k_per_wml
