"""HardFlowProxyTask — a non-linearly-separable variant that exposes real variance.

Spec §Threats to Validity (paper v0.2) flags FlowProxyTask 4-class as too easy:
MLP and LIF both saturate to 1.0, making the 0 % polymorphie gap a degenerate
best case. This test pins the harder variant's contract: non-saturated accuracy.
"""
import torch

from track_w.tasks.hard_flow_proxy import HardFlowProxyTask


def test_hard_flow_proxy_sample_shapes():
    task = HardFlowProxyTask(dim=16, n_classes=12, seed=0)
    x, y = task.sample(batch=32)
    assert x.shape == (32, 16)
    assert y.shape == (32,)
    assert (y >= 0).all() and (y < task.n_classes).all()


def test_hard_flow_proxy_linear_probe_does_not_saturate():
    """A linear probe should beat random but NOT reach 1.0 — the point is
    that MLP/LIF divergence can be observed here."""
    task = HardFlowProxyTask(dim=16, n_classes=12, seed=0)
    probe = torch.nn.Linear(16, task.n_classes)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2)

    for _ in range(400):
        x, y = task.sample(batch=64)
        logits = probe(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    x, y = task.sample(batch=512)
    acc = (probe(x).argmax(-1) == y).float().mean().item()

    # Beats random (1/12 ≈ 0.083) but does NOT saturate.
    assert acc > 2 / task.n_classes, f"linear probe {acc:.3f} barely beats random"
    assert acc < 0.85, (
        f"linear probe {acc:.3f} saturates — task is not hard enough "
        "to expose polymorphie variance."
    )


def test_hard_flow_proxy_seed_is_local():
    torch.manual_seed(42)
    expected = torch.rand(1).item()
    torch.manual_seed(42)
    _ = HardFlowProxyTask(dim=16, n_classes=12, seed=99)
    observed = torch.rand(1).item()
    assert expected == observed
