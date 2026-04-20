import torch

from track_w.tasks.sequential import SequentialFlowProxyTask


def test_sequential_task_shape():
    task = SequentialFlowProxyTask(seq_len=16, dim=16, n_classes=4, seed=0)
    xs, ys = task.sample(batch=8)
    assert xs.shape == (8, 16, 16)
    assert ys.shape == (8,)


def test_sequential_task_seed_stable():
    a = SequentialFlowProxyTask(seq_len=16, seed=7)
    b = SequentialFlowProxyTask(seq_len=16, seed=7)
    xa, ya = a.sample(batch=4)
    xb, yb = b.sample(batch=4)
    assert torch.equal(xa, xb)
    assert torch.equal(ya, yb)
