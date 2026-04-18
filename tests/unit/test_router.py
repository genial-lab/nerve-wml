import torch

from track_p.router import SparseRouter


def test_router_edge_count_equals_k_per_wml():
    r = SparseRouter(n_wmls=4, k=2)
    edges = r.sample_edges(tau=0.5, hard=True)
    # edges[i, j] = 1 if i → j active. Each row must have exactly k ones.
    assert edges.shape == (4, 4)
    assert (edges.sum(dim=-1) == 2).all()


def test_router_no_self_loops():
    r = SparseRouter(n_wmls=4, k=2)
    edges = r.sample_edges(tau=0.5, hard=True)
    assert (edges.diagonal() == 0).all()


def test_routing_weight_lookup():
    r = SparseRouter(n_wmls=4, k=2)
    edges = r.sample_edges(tau=0.5, hard=True)
    for i in range(4):
        for j in range(4):
            assert r.routing_weight(i, j, edges) == float(edges[i, j])
