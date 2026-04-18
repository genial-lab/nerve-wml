import torch

from track_w.tasks.flow_proxy import FlowProxyTask


def test_flow_proxy_sample_shapes():
    task = FlowProxyTask(dim=16, seed=0)
    x, y = task.sample(batch=32)
    assert x.shape == (32, 16)
    assert y.shape == (32,)
    assert (y >= 0).all() and (y < task.n_classes).all()


def test_flow_proxy_is_learnable():
    """A linear probe should outperform random on the task."""
    task = FlowProxyTask(dim=16, seed=0)
    probe = torch.nn.Linear(16, task.n_classes)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2)

    for _ in range(200):
        x, y = task.sample(batch=64)
        logits = probe(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()

    x, y = task.sample(batch=256)
    acc = (probe(x).argmax(-1) == y).float().mean().item()
    # Random baseline is 1/n_classes. A learnable task should easily beat it.
    assert acc > 1.5 / task.n_classes
