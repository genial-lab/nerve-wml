"""Router sparsity diagnostic across N ∈ {4, 8, 16, 32}.

Measures:
  - Fan-in and fan-out per WML (mean, std, min, max).
  - Connectivity: is the directed graph strongly connected?
    If not, how many strongly connected components?

Plan 4c §Task 6. Used by the paper §5.1 Scaling subsection.
"""
from __future__ import annotations

import numpy as np
import torch

from track_w.mock_nerve import MockNerve
from track_w.pool_factory import k_for_n


def _strongly_connected_components(edges: np.ndarray) -> int:
    """Tarjan's SCC count on an N×N binary adjacency matrix."""
    n = edges.shape[0]
    index_counter = [0]
    stack: list[int] = []
    lowlinks = [0] * n
    index = [0] * n
    on_stack = [False] * n
    visited = [False] * n
    scc_count = [0]

    def strongconnect(v: int) -> None:
        index[v] = index_counter[0]
        lowlinks[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack[v] = True
        visited[v] = True

        for w in range(n):
            if edges[v, w] == 0:
                continue
            if not visited[w]:
                strongconnect(w)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elif on_stack[w]:
                lowlinks[v] = min(lowlinks[v], index[w])

        if lowlinks[v] == index[v]:
            while stack:
                w = stack.pop()
                on_stack[w] = False
                if w == v:
                    break
            scc_count[0] += 1

    for v in range(n):
        if not visited[v]:
            strongconnect(v)
    return scc_count[0]


def scale_diagnostic(Ns: list[int] | None = None) -> dict[int, dict]:  # noqa: N803
    """Run diagnostic for each N in Ns. Returns per-N metrics dict."""
    if Ns is None:
        Ns = [4, 8, 16, 32]  # noqa: N806

    report: dict[int, dict] = {}
    for n in Ns:
        torch.manual_seed(0)
        nerve = MockNerve(n_wmls=n, k=k_for_n(n), seed=0)
        edges = nerve._edges.detach().numpy().astype(int)

        fan_out = edges.sum(axis=1)   # outgoing per row
        fan_in  = edges.sum(axis=0)   # incoming per column

        n_components = _strongly_connected_components(edges)
        report[n] = {
            "N":                     n,
            "k":                     int(k_for_n(n)),
            "fan_out_mean":          float(fan_out.mean()),
            "fan_out_std":           float(fan_out.std()),
            "fan_out_min":           int(fan_out.min()),
            "fan_out_max":           int(fan_out.max()),
            "fan_in_mean":           float(fan_in.mean()),
            "fan_in_std":            float(fan_in.std()),
            "fan_in_min":            int(fan_in.min()),
            "fan_in_max":            int(fan_in.max()),
            "is_strongly_connected": n_components == 1,
            "n_components":          n_components,
        }
    return report


if __name__ == "__main__":
    import json

    rep = scale_diagnostic()
    # Convert int keys to str for JSON.
    print(json.dumps({str(k): v for k, v in rep.items()}, indent=2))
