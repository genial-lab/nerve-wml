from nerve_core.neuroletter import Neuroletter, Phase, Role
from track_w.mock_nerve import MockNerve


def _letter(src: int, dst: int, role: Role, phase: Phase, t: float = 0.0) -> Neuroletter:
    return Neuroletter(code=5, role=role, phase=phase, src=src, dst=dst, timestamp=t)


def test_mock_nerve_round_trip():
    nerve = MockNerve(n_wmls=4, k=2, seed=0)
    # MockNerve starts with γ=True, θ=False by default, so γ messages deliver.
    nerve.send(_letter(0, 1, Role.PREDICTION, Phase.GAMMA))
    received = nerve.listen(wml_id=1)
    assert len(received) == 1
    assert received[0].code == 5


def test_mock_nerve_seed_is_local():
    """Constructing a MockNerve must NOT mutate the global torch RNG."""
    import torch
    torch.manual_seed(42)
    expected = torch.rand(1).item()

    torch.manual_seed(42)
    _ = MockNerve(n_wmls=4, k=2, seed=99)
    observed = torch.rand(1).item()

    assert expected == observed


def test_mock_nerve_routing_weight_count():
    nerve = MockNerve(n_wmls=4, k=2, seed=0)
    active = sum(
        1
        for i in range(4)
        for j in range(4)
        if nerve.routing_weight(i, j) == 1.0
    )
    assert active == 4 * 2


def test_mock_nerve_gamma_priority_holds_theta():
    """When γ and θ are both active, γ delivers and θ is held."""
    nerve = MockNerve(n_wmls=4, k=2, seed=0)
    nerve.set_phase_active(gamma=True, theta=True)
    nerve.send(_letter(0, 1, Role.PREDICTION, Phase.GAMMA))
    nerve.send(_letter(2, 1, Role.ERROR,      Phase.THETA))

    delivered = nerve.listen(wml_id=1)
    assert [l.role for l in delivered] == [Role.PREDICTION]

    # Now turn γ off — θ should deliver.
    nerve.set_phase_active(gamma=False, theta=True)
    delivered = nerve.listen(wml_id=1)
    assert [l.role for l in delivered] == [Role.ERROR]


def test_mock_nerve_silence_when_inactive():
    """When both phases are inactive, listen() returns []."""
    nerve = MockNerve(n_wmls=4, k=2, seed=0)
    nerve.set_phase_active(gamma=False, theta=False)
    nerve.send(_letter(0, 1, Role.PREDICTION, Phase.GAMMA))
    assert nerve.listen(wml_id=1) == []
