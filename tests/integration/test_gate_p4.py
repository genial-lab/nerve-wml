import torch

from scripts.track_p_pilot import run_p4


def test_p4_topology_is_connected():
    """After sparse K-active sampling, every WML must be reachable from every other."""
    torch.manual_seed(0)
    connected, k_per_wml = run_p4(n_wmls=4, k=2)
    assert connected is True
    assert (k_per_wml == 2).all()
